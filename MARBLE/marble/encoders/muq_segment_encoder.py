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


class MuQSegmentEncoder(BaseEncoder):
    """
    MuQ-based encoder that processes audio by:
    1. Dividing audio into segments (no padding)
    2. Encoding each segment with MuQ model
    3. Max pooling each segment's time dimension to get a fixed-size vector
    4. Max pooling all segment vectors to create a final single-vector track representation
    5. Projecting the final vector to a target output dimension if necessary
    """
    
    def __init__(self, 
                 model_name: str = "OpenMuQ/MuQ-large-msd-iter",
                 target_sr: int = 24000,
                 segment_seconds: int = 10,
                 min_samples_threshold: int = 2048,
                 output_dim: int = 768, # Decoders' expected input dimension
                 device: str = None,
                 **kwargs):
        """
        Initialize MuQ Segment Encoder
        
        Args:
            model_name: MuQ model name to load from pretrained
            target_sr: Target sample rate for audio processing
            segment_seconds: Length of each segment in seconds
            min_samples_threshold: Minimum samples required for a valid segment
            output_dim: The final output dimension required by the downstream model.
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__(**kwargs)
        
        self.target_sr = target_sr
        self.segment_seconds = segment_seconds
        self.segment_samples = segment_seconds * target_sr
        self.min_samples_threshold = min_samples_threshold
        self.output_dim = output_dim
        self.projection = None # Projection layer, initialized lazily
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Performance stats variables
        self.inference_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.batch_count = 0
            
        # Load MuQ model
        print(f"Loading MuQ model: {model_name}")
        try:
            self.muq_model = MuQ.from_pretrained(model_name)
            self.muq_model = self.muq_model.to(self.device).eval()
            
            # === FIX START: Dynamically determine output dimension with a dry run ===
            # This is more robust than relying on config attributes that might change.
            with torch.no_grad():
                # Create a small, silent dummy tensor that meets the minimum length
                dummy_input = torch.zeros(1, self.min_samples_threshold, device=self.device)
                output = self.muq_model(dummy_input)
                # Get the last dimension of the output tensor, which is the feature size
                self._actual_output_dim = output.last_hidden_state.shape[-1]
            print(f"MuQ model loaded. Dynamically determined output dimension: {self._actual_output_dim}")
            # === FIX END ===

        except Exception as e:
            raise RuntimeError(f"Failed to load MuQ model: {e}")
    
    def _resample_audio(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate if needed"""
        if original_sr != self.target_sr:
            waveform_np = waveform.cpu().numpy()
            resampled_list = [
                librosa.resample(wav, orig_sr=original_sr, target_sr=self.target_sr)
                for wav in waveform_np
            ]
            waveform = torch.tensor(np.stack(resampled_list), dtype=torch.float32)
        return waveform
    
    def _create_segments_flexible(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Divide waveform into segments with flexible handling for short audio"""
        wav_length = waveform.shape[0]
        
        if wav_length <= self.min_samples_threshold:
            if wav_length > 0:
                padding_needed = self.min_samples_threshold - wav_length
                padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
                return [padded_waveform]
            else:
                return []
        
        if wav_length < self.segment_samples:
            return [waveform]
        
        num_segments = int(np.ceil(wav_length / self.segment_samples))
        
        segments = []
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = waveform[start:end]
            
            if len(segment) > self.min_samples_threshold // 4:
                segments.append(segment)
        
        return segments if segments else [waveform]
    
    def _encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Encode a single segment using MuQ model"""
        segment_tensor = segment.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            try:
                output = self.muq_model(segment_tensor)
                embedding = output.last_hidden_state.squeeze(0)
                del segment_tensor, output
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return embedding
            except Exception as e:
                print(f"Error encoding segment: {e}")
                return torch.zeros(1, self._actual_output_dim, device=self.device)
    
    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass: process audio through segment-based MuQ encoding"""
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        batch_size = input_tensor.shape[0]
        batch_track_embeddings = []
        
        for i in range(batch_size):
            waveform = input_tensor[i]
            
            if not torch.isfinite(waveform).all():
                print(f"Warning: Invalid audio data (inf/nan) in batch item {i}")
                batch_track_embeddings.append(torch.zeros(self._actual_output_dim, device=self.device))
                continue
            
            segments = self._create_segments_flexible(waveform)
            
            if not segments:
                batch_track_embeddings.append(torch.zeros(self._actual_output_dim, device=self.device))
                continue
            
            # === MODIFICATION START: Two-stage max pooling ===
            segment_pooled_embeddings = []
            for segment in segments:
                # segment_emb shape: [seq_len, hidden_dim]
                segment_emb = self._encode_segment(segment)
                # 1. Pool the time axis (dim=0) of the segment embedding using max pooling
                pooled_emb = torch.max(segment_emb, dim=0)[0] # Result shape: [hidden_dim]
                segment_pooled_embeddings.append(pooled_emb)
            
            if segment_pooled_embeddings:
                # Stack all pooled segment embeddings for the track
                track_segments_tensor = torch.stack(segment_pooled_embeddings, dim=0) # Shape: [num_segments, hidden_dim]
                # 2. Pool across the segments to get the final single track embedding using max pooling
                final_track_embedding = torch.max(track_segments_tensor, dim=0)[0] # Shape: [hidden_dim]
            else:
                final_track_embedding = torch.zeros(self._actual_output_dim, device=self.device)
            
            batch_track_embeddings.append(final_track_embedding)
            # === MODIFICATION END ===

        batch_tensor = torch.stack(batch_track_embeddings) # Shape: [batch, hidden_dim]
        
        # FIX: Add projection layer to match decoder's expected input size
        if self.projection is None and self._actual_output_dim != self.output_dim:
            print(f"Initializing projection layer to map {self._actual_output_dim} -> {self.output_dim}")
            self.projection = nn.Linear(self._actual_output_dim, self.output_dim).to(self.device)

        if self.projection is not None:
            batch_tensor = self.projection(batch_tensor)

        # Convert to MARBLE 4D format: [batch, num_layers, time_steps, hidden_dim]
        # For a single vector output, num_layers and time_steps are 1.
        final_embeddings = batch_tensor.unsqueeze(1).unsqueeze(1)
        
        # --- Statistics Collection ---
        end_time = time.time()
        inference_time = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        cpu_memory_delta = final_memory - initial_memory
        
        self.batch_count += 1
        self.inference_times.append(inference_time)
        self.cpu_memory_usage.append(cpu_memory_delta)
        
        if self.device == 'cuda':
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            self.gpu_memory_usage.append(peak_gpu_memory)
            
            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"Peak GPU: {peak_gpu_memory:.1f}MB, "
                  f"Current GPU: {current_gpu_memory:.1f}MB, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Segments: {len(segments) if 'segments' in locals() else 0}")
        else:
            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Segments: {len(segments) if 'segments' in locals() else 0}")
        
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
        
        filename = f"muq_inference_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        """Get the output feature dimension"""
        return self.output_dim
    
    def get_num_layers(self) -> int:
        """Get number of layers (for compatibility with MARBLE)"""
        return 1


class MuQSegmentEncoderWithLayers(MuQSegmentEncoder):
    """
    Extended version that can output multiple layer representations
    This version stores intermediate segment embeddings for analysis
    """
    
    def __init__(self, return_all_segments: bool = False, **kwargs):
        """
        Args:
            return_all_segments: If True, return embeddings for all segments instead of averaging
        """
        super().__init__(**kwargs)
        self.return_all_segments = return_all_segments
    
    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with option to return all segment embeddings
        
        Returns:
            If return_all_segments=False: [batch, 1, 1, hidden_dim] - final averaged representation
            If return_all_segments=True: [batch, max_segments, hidden_dim] - all segments (padded)
        """
        if not self.return_all_segments:
            # The base class now correctly returns the final averaged representation
            return super().forward(input_tensor, input_len)
        
        batch_size = input_tensor.shape[0]
        all_batch_segments = []
        max_segments = 0
        
        # First pass: collect all segments and find max number
        for i in range(batch_size):
            waveform = input_tensor[i]
            segments = self._create_segments_flexible(waveform)
            
            if not torch.isfinite(waveform).all() or not segments:
                all_batch_segments.append([])
                continue
            
            segment_embeddings = []
            for segment in segments:
                segment_emb = self._encode_segment(segment)
                # MODIFICATION: Use max pooling instead of mean pooling
                pooled_emb = torch.max(segment_emb, dim=0)[0] # [hidden_dim]
                segment_embeddings.append(pooled_emb)
            
            all_batch_segments.append(segment_embeddings)
            max_segments = max(max_segments, len(segment_embeddings))
        
        # Second pass: pad and stack
        padded_embeddings = []
        for segment_list in all_batch_segments:
            if not segment_list:
                # Create zero embeddings for invalid audio on correct device
                padded = torch.zeros(max_segments, self.get_output_dim(), device=self.device)
            else:
                # Pad with zeros to max_segments
                segment_tensor = torch.stack(segment_list) # [num_segments, hidden_dim]

                # Apply projection if needed
                if self.projection is None and self._actual_output_dim != self.output_dim:
                    self.projection = nn.Linear(self._actual_output_dim, self.output_dim).to(self.device)
                if self.projection:
                    segment_tensor = self.projection(segment_tensor)

                padding_needed = max_segments - len(segment_list)
                if padding_needed > 0:
                    padding = torch.zeros(padding_needed, segment_tensor.shape[1], device=segment_tensor.device)
                    padded = torch.cat([segment_tensor, padding], dim=0)
                else:
                    padded = segment_tensor
            
            padded_embeddings.append(padded)
        
        return torch.stack(padded_embeddings) # [batch, max_segments, hidden_dim]
