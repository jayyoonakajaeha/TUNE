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


class MuQMean10SecEncoder(BaseEncoder):
    """
    MuQ-based encoder that processes audio by:
    1. Dividing audio into 10-second segments (no padding)
    2. Encoding each segment with MuQ model
    3. Average pooling all segment embeddings to create final representation

    This encoder follows the preprocessing approach from the provided MuQ embedding code
    but uses mean pooling instead of concatenation for the final representation.
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        target_sr: int = 24000,
        segment_seconds: int = 10,
        min_samples_threshold: int = 2048,
        device: str = None,
        **kwargs
    ):
        """
        Initialize MuQ Mean 10-Second Encoder

        Args:
            model_name: MuQ model name to load from pretrained
            target_sr: Target sample rate for audio processing
            segment_seconds: Length of each segment in seconds
            min_samples_threshold: Minimum samples required for a valid segment
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__(**kwargs)

        self.target_sr = target_sr
        self.segment_seconds = segment_seconds
        self.segment_samples = segment_seconds * target_sr
        self.min_samples_threshold = min_samples_threshold

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
        print(f"Loading MuQ model: {model_name}")
        try:
            self.muq_model = MuQ.from_pretrained(model_name)
            self.muq_model = self.muq_model.to(self.device).eval()
            print("MuQ model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuQ model: {e}")

    def _resample_audio(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate if needed

        Args:
            waveform: Input waveform tensor [batch, time]
            original_sr: Original sample rate

        Returns:
            Resampled waveform tensor
        """
        if original_sr != self.target_sr:
            waveform_np = waveform.cpu().numpy()
            resampled_list = []
            for i in range(waveform_np.shape[0]):
                resampled = librosa.resample(
                    waveform_np[i],
                    orig_sr=original_sr,
                    target_sr=self.target_sr
                )
                resampled_list.append(resampled)
            waveform = torch.tensor(np.stack(resampled_list), dtype=torch.float32)
        return waveform

    def _create_segments_flexible(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Divide waveform into segments with more flexible handling for short audio

        Args:
            waveform: Input waveform [time]

        Returns:
            List of segment tensors
        """
        wav_length = waveform.shape[0]

        # If audio is too short, use the whole audio
        if wav_length <= self.min_samples_threshold:
            if wav_length > 0:
                # Pad short audio to minimum threshold
                padding_needed = self.min_samples_threshold - wav_length
                padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
                return [padded_waveform]
            else:
                return []

        # If audio is shorter than one segment, use the whole audio
        if wav_length < self.segment_samples:
            return [waveform]

        # Normal segmentation for longer audio
        num_segments = int(np.ceil(wav_length / self.segment_samples))

        if num_segments == 0 and wav_length > 0:
            num_segments = 1

        segments = []
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = waveform[start:end]
            # Accept shorter segments for the last segment
            if len(segment) > self.min_samples_threshold // 4:  # More flexible threshold
                segments.append(segment)

        return segments if segments else [waveform]

    def _encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Encode a single segment using MuQ model and return mean-pooled embedding

        Args:
            segment: Audio segment tensor [time]

        Returns:
            Mean-pooled segment embedding tensor [hidden_dim] on the same device as model
        """
        segment_tensor = segment.unsqueeze(0).to(self.device)
        with torch.no_grad():
            try:
                output = self.muq_model(segment_tensor)
                # Extract last hidden state and remove batch dimension
                embedding = output.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
                # Mean pooling across time dimension
                mean_embedding = embedding.mean(dim=0)  # [hidden_dim]
                # Clean up GPU memory but keep embedding on the same device as model
                del segment_tensor, output, embedding
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                # Return mean-pooled embedding on the same device (GPU if using GPU)
                return mean_embedding  # Keep on GPU instead of moving to CPU
            except Exception as e:
                print(f"Error encoding segment: {e}")
                # Return zero embedding on the correct device
                return torch.zeros(768, device=self.device)

    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: process audio through segment-based MuQ encoding with mean pooling

        Args:
            input_tensor: Raw audio waveform [batch, time]
            input_len: Optional sequence lengths (not used in this implementation)

        Returns:
            Encoded features [batch, num_layers, time_steps, hidden_dim] - MARBLE 4D format
            Note: For mean pooling, time_steps will be 1 since we average across all segments
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

        for i in range(batch_size):
            waveform = input_tensor[i]  # [time]

            # Validate audio data
            if not torch.isfinite(waveform).all():
                print(f"Warning: Invalid audio data (inf/nan) in batch item {i}")
                # Return zero embedding for invalid audio on correct device
                batch_embeddings.append(torch.zeros(768, device=self.device))
                continue

            # Create segments with more flexible thresholds
            segments = self._create_segments_flexible(waveform)

            if not segments:
                print(f"Warning: No valid segments found for batch item {i}, using whole audio")
                # Use the whole audio if no segments found
                if len(waveform) > 0:
                    segments = [waveform]
                else:
                    batch_embeddings.append(torch.zeros(768, device=self.device))
                    continue

            # Encode each segment and collect mean-pooled embeddings
            segment_embeddings = []
            for segment in segments:
                segment_emb = self._encode_segment(segment)  # [hidden_dim]
                segment_embeddings.append(segment_emb)

            # Average pooling across all segments
            if segment_embeddings:
                # Stack and mean across segments: [num_segments, hidden_dim] -> [hidden_dim]
                segments_tensor = torch.stack(segment_embeddings)
                track_embedding = segments_tensor.mean(dim=0)  # [hidden_dim]
            else:
                track_embedding = torch.zeros(768, device=self.device)

            batch_embeddings.append(track_embedding)

        # Stack all batch embeddings: [batch, hidden_dim]
        batch_tensor = torch.stack(batch_embeddings)

        # Convert to MARBLE 4D format: [batch, num_layers, time_steps, hidden_dim]
        # For mean pooling, time_steps = 1 since we have a single averaged representation
        batch_size, hidden_dim = batch_tensor.shape
        num_layers = 1  # We only have one layer representation
        time_steps = 1  # Single time step for averaged representation

        # Reshape to [batch, num_layers, time_steps, hidden_dim]
        final_embeddings = batch_tensor.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, hidden_dim]

        # 추론 시간 및 메모리 측정 종료
        end_time = time.time()
        inference_time = end_time - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_memory_delta = final_memory - initial_memory

        # 통계 수집
        self.batch_count += 1
        self.inference_times.append(inference_time)
        self.cpu_memory_usage.append(cpu_memory_delta)

        if self.device == 'cuda':
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_usage.append(peak_gpu_memory)

            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"Peak GPU: {peak_gpu_memory:.1f}MB, "
                  f"Current GPU: {current_gpu_memory:.1f}MB, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Segments: {len(segments) if 'segments' in locals() else 0}, "
                  f"Final shape: {final_embeddings.shape}")
        else:
            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Segments: {len(segments) if 'segments' in locals() else 0}, "
                  f"Final shape: {final_embeddings.shape}")

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

        filename = f"muq_mean_inference_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
            Feature dimension (typically 768 for MuQ-large)
        """
        return 768  # MuQ-large hidden dimension

    def get_num_layers(self) -> int:
        """
        Get number of layers (for compatibility with MARBLE)

        Returns:
            Number of layers (1 for this averaged representation)
        """
        return 1


import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class MuQMeanStatsCallback(Callback):
    """
    학습/검증/테스트 완료 후 MuQ mean encoder의 통계를 출력하는 콜백
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