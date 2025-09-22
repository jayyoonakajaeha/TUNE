import torch
import torch.nn as nn
import librosa
import numpy as np
import time
import psutil
import os
from typing import Optional, List

from marble.core.base_encoder import BaseEncoder

try:
    from muq import MuQ
except ImportError:
    raise ImportError("MuQ library is required. Please install it first.")


class SegmentTransformerMLM(nn.Module):
    """Pretrained Transformer model for segment embeddings"""

    def __init__(self, emb_dim, heads=8, hidden=1024, layers=4, max_len=100000, dropout=0.1):
        super().__init__()
        # Very large max_len for learnable positional embedding
        self.pos = nn.Embedding(max_len, emb_dim)
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(emb_dim)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=hidden,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        # MLM head (not used in inference, but needed for checkpoint loading)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim)
        )
        self.max_len = max_len

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, emb_dim)
        b, l, d = x.shape

        if l > self.max_len:
            print(f"Warning: 시퀀스 길이 {l}이 max_len {self.max_len}을 초과합니다. 잘라냅니다.")
            x = x[:, :self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_len]
            l = self.max_len

        # Positional embedding 추가
        pos_idx = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        x = self.input_norm(x + self.pos(pos_idx))

        # Attention mask 처리 (패딩된 부분은 무시)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True for padded positions

        # Transformer encoding
        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # MLM prediction (not used in inference)
        logits = self.head(y)

        # Track-level embedding (attention mask를 고려한 평균)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(y)
            track_emb = (y * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            track_emb = y.mean(dim=1)

        return logits, track_emb


class TransformerSegmentEncoder(BaseEncoder):
    """
    Transformer-based encoder that processes audio by:
    1. Dividing audio into 60-second segments
    2. Encoding each 60-second segment directly with MuQ model → [~250, 1024]
    3. Concatenating all segment embeddings along time axis → [total_frames, 1024]
    4. Processing through pretrained Transformer
    5. Generating track-level embedding via attention-weighted averaging

    This exactly follows the training procedure where 60-second segments are
    directly encoded with MuQ and concatenated.
    """

    def __init__(
        self,
        muq_model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        transformer_checkpoint_path: str = "/home/jay/MusicAI/transformer_10sec_mlm.pth",
        target_sr: int = 24000,
        segment_seconds: int = 10,  # 60초 세그먼트
        min_samples_threshold: int = 2048,
        emb_dim: int = 1024,
        heads: int = 8,
        hidden: int = 1024,
        layers: int = 4,
        max_len: int = 100000,
        dropout: float = 0.1,
        device: str = None,
        **kwargs
    ):
        """
        Initialize Transformer Segment Encoder

        Args:
            muq_model_name: MuQ model name to load from pretrained
            transformer_checkpoint_path: Path to pretrained transformer checkpoint
            target_sr: Target sample rate for audio processing
            segment_seconds: Length of each segment in seconds (60)
            min_samples_threshold: Minimum samples required for a valid segment
            emb_dim: Embedding dimension
            heads: Number of attention heads
            hidden: Hidden dimension in transformer
            layers: Number of transformer layers
            max_len: Maximum sequence length for positional embedding
            dropout: Dropout rate
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__(**kwargs)

        self.target_sr = target_sr
        self.segment_seconds = segment_seconds  # 60초 세그먼트
        self.segment_samples = segment_seconds * target_sr
        self.min_samples_threshold = min_samples_threshold
        self.transformer_checkpoint_path = transformer_checkpoint_path

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 성능 통계 수집을 위한 변수들
        self.inference_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.batch_count = 0

        # Load MuQ model
        print(f"Loading MuQ model: {muq_model_name}")
        try:
            self.muq_model = MuQ.from_pretrained(muq_model_name)
            self.muq_model = self.muq_model.to(self.device).eval()
            print("MuQ model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuQ model: {e}")

        # Initialize Transformer model
        print(f"Initializing Transformer model...")
        self.transformer = SegmentTransformerMLM(
            emb_dim=emb_dim,
            heads=heads,
            hidden=hidden,
            layers=layers,
            max_len=max_len,
            dropout=dropout
        ).to(self.device)

        # Load pretrained transformer weights
        self._load_transformer_checkpoint()

        print("Transformer-based encoder initialized successfully")

    def _load_transformer_checkpoint(self):
        """Load pretrained transformer checkpoint"""
        try:
            print(f"Loading transformer checkpoint from: {self.transformer_checkpoint_path}")
            checkpoint = torch.load(self.transformer_checkpoint_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                self.transformer.load_state_dict(checkpoint['model_state_dict'])
                print("Transformer weights loaded successfully from checkpoint")
            else:
                # Try loading directly as state dict
                self.transformer.load_state_dict(checkpoint)
                print("Transformer weights loaded successfully")

            # Set to evaluation mode
            self.transformer.eval()

        except Exception as e:
            print(f"Warning: Failed to load transformer checkpoint: {e}")
            print("Using randomly initialized transformer weights")

    def _encode_60sec_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Encode a single 60-second segment using MuQ model

        Args:
            segment: Audio segment tensor [time] (60초 또는 더 짧음)

        Returns:
            Segment embedding tensor [~250, 1024] - MuQ의 자연스러운 출력
        """
        # Add batch dimension and move to device
        segment_tensor = segment.unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                output = self.muq_model(segment_tensor)
                # Extract last hidden state and remove batch dimension
                embedding = output.last_hidden_state.squeeze(0)  # [~250, 1024]

                # Clean up GPU memory
                del segment_tensor, output
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                return embedding.cpu()

            except Exception as e:
                print(f"Error encoding segment: {e}")
                # Return minimal embedding if encoding fails
                return torch.zeros(1, 1024)

    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: process audio through transformer-based segment encoding

        Exact process matching training:
        1. Divide into 60-second segments
        2. Each segment → MuQ encoding → [~250, 1024]
        3. Concatenate all segment embeddings → [total_frames, 1024]
        4. Process through Transformer → track-level embedding

        Args:
            input_tensor: Raw audio waveform [batch, time]
            input_len: Optional sequence lengths (not used in this implementation)

        Returns:
            Encoded features [batch, num_layers, time_steps, hidden_dim] - MARBLE 4D format
        """
        # 추론 시간 및 메모리 측정 시작
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        batch_size = input_tensor.shape[0]

        batch_embeddings = []
        batch_attention_masks = []

        for i in range(batch_size):
            waveform = input_tensor[i]  # [time]

            # Validate audio data
            if not torch.isfinite(waveform).all():
                print(f"Warning: Invalid audio data (inf/nan) in batch item {i}")
                batch_embeddings.append(torch.zeros(1, 1024))
                batch_attention_masks.append(torch.ones(1))
                continue

            # Step 1: Divide into 60-second segments
            wav_length = waveform.shape[0]
            segment_embeddings = []

            if wav_length <= self.segment_samples:
                # Audio is 60 seconds or shorter - process as single segment
                segment_emb = self._encode_60sec_segment(waveform)  # [~250, 1024]
                segment_embeddings.append(segment_emb)
                num_segments = 1
            else:
                # Audio is longer than 60 seconds - divide into 60-sec segments
                num_segments = int(np.ceil(wav_length / self.segment_samples))
                for j in range(num_segments):
                    start = j * self.segment_samples
                    end = start + self.segment_samples
                    segment_60sec = waveform[start:end]  # 마지막 세그먼트는 60초보다 짧을 수 있음

                    # Step 2: Encode each 60-second segment with MuQ directly
                    segment_emb = self._encode_60sec_segment(segment_60sec)  # [~250, 1024]
                    segment_embeddings.append(segment_emb)

            # Step 3: Concatenate all segment embeddings (훈련과 동일)
            # torch.cat(segment_list, dim=0) from training code
            track_embedding = torch.cat(segment_embeddings, dim=0)  # [total_frames, 1024]

            # Create attention mask (all frames are valid - no padding like training)
            attention_mask = torch.ones(track_embedding.shape[0])

            batch_embeddings.append(track_embedding)
            batch_attention_masks.append(attention_mask)

        # 배치 내에서 시간 길이 통일 (가장 긴 것에 맞춰 패딩)
        # 하지만 훈련에서 batch_size=1이었으므로 실제로는 패딩 없음
        max_time_len = max(emb.shape[0] for emb in batch_embeddings)

        batch_padded = []
        batch_masks_padded = []

        for track_emb, attention_mask in zip(batch_embeddings, batch_attention_masks):
            seq_len = track_emb.shape[0]
            pad_len = max_time_len - seq_len

            if pad_len > 0:
                # 임베딩 패딩 (0으로)
                padding = torch.zeros(pad_len, track_emb.shape[1])
                padded_track = torch.cat([track_emb, padding], dim=0)
                # Attention mask 패딩 (0으로)
                mask_padding = torch.zeros(pad_len)
                padded_mask = torch.cat([attention_mask, mask_padding], dim=0)
            else:
                padded_track = track_emb
                padded_mask = attention_mask

            batch_padded.append(padded_track)
            batch_masks_padded.append(padded_mask)

        # Stack all batch embeddings
        batch_tensor = torch.stack(batch_padded).to(self.device)  # [batch, time_steps, hidden_dim]
        attention_masks = torch.stack(batch_masks_padded).to(self.device)  # [batch, time_steps]

        # Step 4: Process through pretrained Transformer
        with torch.no_grad():
            logits, track_level_embeddings = self.transformer(batch_tensor, attention_masks)

        # track_level_embeddings: [batch, hidden_dim]
        # Convert to MARBLE 4D format: [batch, num_layers, time_steps, hidden_dim]
        batch_size, hidden_dim = track_level_embeddings.shape

        # Since we have track-level embeddings, we'll create a single time step representation
        final_embeddings = track_level_embeddings.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, hidden_dim]

        # Keep on the same device as input (don't move to CPU yet)
        # MARBLE will handle device management

        # 추론 시간 및 메모리 측정 종료
        end_time = time.time()
        inference_time = end_time - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_memory_delta = final_memory - initial_memory

        # 통계 수집
        self.batch_count += 1
        self.inference_times.append(inference_time)
        self.cpu_memory_usage.append(cpu_memory_delta)

        # 세그먼트 정보 계산
        total_60sec_segments = sum(
            1 if input_tensor[i].shape[0] <= self.segment_samples
            else int(np.ceil(input_tensor[i].shape[0] / self.segment_samples))
            for i in range(batch_size)
        )
        avg_60sec_segments = total_60sec_segments / batch_size

        if self.device == 'cuda':
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_usage.append(peak_gpu_memory)

            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"Peak GPU: {peak_gpu_memory:.1f}MB, "
                  f"Current GPU: {current_gpu_memory:.1f}MB, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Avg 60s segments: {avg_60sec_segments:.1f}, "
                  f"Final seq_len: {max_time_len}")
        else:
            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Avg 60s segments: {avg_60sec_segments:.1f}, "
                  f"Final seq_len: {max_time_len}")

        # 매 50배치마다 중간 통계 출력
        if self.batch_count % 50 == 0:
            self.print_intermediate_stats()

        return final_embeddings

    def print_intermediate_stats(self):
        """중간 통계 출력"""
        if not self.inference_times:
            return

        print(f"\n=== Intermediate Stats (After {self.batch_count} batches) ===")
        print(f"Inference Time - Min: {min(self.inference_times):.3f}s, "
              f"Max: {max(self.inference_times):.3f}s, "
              f"Avg: {sum(self.inference_times)/len(self.inference_times):.3f}s")

        print(f"CPU Memory Delta - Min: {min(self.cpu_memory_usage):.1f}MB, "
              f"Max: {max(self.cpu_memory_usage):.1f}MB, "
              f"Avg: {sum(self.cpu_memory_usage)/len(self.cpu_memory_usage):.1f}MB")

        if self.gpu_memory_usage:
            print(f"GPU Memory Peak - Min: {min(self.gpu_memory_usage):.1f}MB, "
                  f"Max: {max(self.gpu_memory_usage):.1f}MB, "
                  f"Avg: {sum(self.gpu_memory_usage)/len(self.gpu_memory_usage):.1f}MB")
        print("="*60)

    def print_final_stats(self):
        """최종 통계 출력"""
        if not self.inference_times:
            print("No inference statistics collected.")
            return

        print(f"\n{'='*80}")
        print(f"🎯 FINAL INFERENCE STATISTICS (Total batches: {self.batch_count})")
        print(f"{'='*80}")

        # 추론 시간 통계
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        avg_time = sum(self.inference_times) / len(self.inference_times)

        print(f"⏱️  INFERENCE TIME:")
        print(f"    Min: {min_time:.3f}s")
        print(f"    Max: {max_time:.3f}s")
        print(f"    Avg: {avg_time:.3f}s")
        print(f"    Total: {sum(self.inference_times):.3f}s")

        # CPU 메모리 통계
        min_cpu = min(self.cpu_memory_usage)
        max_cpu = max(self.cpu_memory_usage)
        avg_cpu = sum(self.cpu_memory_usage) / len(self.cpu_memory_usage)

        print(f"\n💾 CPU MEMORY DELTA:")
        print(f"    Min: {min_cpu:.1f}MB")
        print(f"    Max: {max_cpu:.1f}MB")
        print(f"    Avg: {avg_cpu:.1f}MB")

        # GPU 메모리 통계 (사용 가능한 경우)
        if self.gpu_memory_usage:
            min_gpu = min(self.gpu_memory_usage)
            max_gpu = max(self.gpu_memory_usage)
            avg_gpu = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage)

            print(f"\n🎮 GPU MEMORY PEAK:")
            print(f"    Min: {min_gpu:.1f}MB")
            print(f"    Max: {max_gpu:.1f}MB")
            print(f"    Avg: {avg_gpu:.1f}MB")

        print(f"{'='*80}")

        # 결과를 파일로도 저장
        self.save_stats_to_file()

    def save_stats_to_file(self):
        """통계를 파일로 저장"""
        import json
        from datetime import datetime

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_batches": self.batch_count,
            "model_type": "TransformerSegmentEncoder",
            "segment_seconds": self.segment_seconds,
            "inference_time": {
                "min": min(self.inference_times),
                "max": max(self.inference_times),
                "avg": sum(self.inference_times) / len(self.inference_times),
                "total": sum(self.inference_times),
                "all_values": self.inference_times
            },
            "cpu_memory_delta_mb": {
                "min": min(self.cpu_memory_usage),
                "max": max(self.cpu_memory_usage),
                "avg": sum(self.cpu_memory_usage) / len(self.cpu_memory_usage),
                "all_values": self.cpu_memory_usage
            }
        }

        if self.gpu_memory_usage:
            stats["gpu_memory_peak_mb"] = {
                "min": min(self.gpu_memory_usage),
                "max": max(self.gpu_memory_usage),
                "avg": sum(self.gpu_memory_usage) / len(self.gpu_memory_usage),
                "all_values": self.gpu_memory_usage
            }

        filename = f"transformer_inference_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📁 Statistics saved to: {filename}")

    def reset_stats(self):
        """통계 초기화"""
        self.inference_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.batch_count = 0

    def get_output_dim(self) -> int:
        """
        Get the output feature dimension

        Returns:
            Feature dimension (1024 for this model)
        """
        return 1024

    def get_num_layers(self) -> int:
        """
        Get number of layers (for compatibility with MARBLE)

        Returns:
            Number of layers (1 for this track-level representation)
        """
        return 1


# 통계 콜백
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class TransformerStatsCallback(Callback):
    """
    학습/검증/테스트 완료 후 Transformer encoder의 통계를 출력하는 콜백
    """

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """학습 완료 후 통계 출력"""
        encoder = pl_module.encoder
        if hasattr(encoder, 'print_final_stats'):
            print("\n🎓 TRAINING COMPLETED - INFERENCE STATISTICS:")
            encoder.print_final_stats()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """테스트 완료 후 통계 출력"""
        encoder = pl_module.encoder
        if hasattr(encoder, 'print_final_stats'):
            print("\n🧪 TESTING COMPLETED - INFERENCE STATISTICS:")
            encoder.print_final_stats()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """검증 완료 후 통계 출력 (매번 출력하지 않고 마지막에만)"""
        if trainer.state.stage == "test" and hasattr(pl_module.encoder, 'print_final_stats'):
            print("\n✅ VALIDATION COMPLETED - INFERENCE STATISTICS:")
            pl_module.encoder.print_final_stats()