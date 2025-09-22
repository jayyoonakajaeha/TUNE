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


class MuQMax10SecEncoder(BaseEncoder):
    """
    MuQ-based encoder that processes audio by:
    1. Dividing audio into 10-second segments (no padding)
    2. Encoding each segment with MuQ model
    3. Max pooling all segment embeddings to create final representation
    """
    
    def __init__(self, 
                 model_name: str = "OpenMuQ/MuQ-large-msd-iter",
                 target_sr: int = 24000,
                 segment_seconds: int = 10,
                 min_samples_threshold: int = 2048,
                 device: str = None,
                 **kwargs):
        """
        Initialize MuQ Max 10-Second Encoder
        
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
            print("MuQ model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuQ model: {e}")
    
    def _resample_audio(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate if needed
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
        """
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
        
        if num_segments == 0 and wav_length > 0:
            num_segments = 1
        
        segments = []
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = waveform[start:end]
            
            if len(segment) > self.min_samples_threshold // 4:
                segments.append(segment)
        
        return segments if segments else [waveform]
    
    def _encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Encode a single segment using MuQ model and return max-pooled embedding
        """
        segment_tensor = segment.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            try:
                output = self.muq_model(segment_tensor)
                embedding = output.last_hidden_state.squeeze(0)
                
                # MODIFIED: Max pooling across time dimension
                max_embedding = torch.max(embedding, dim=0)[0]
                
                del segment_tensor, output, embedding
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                return max_embedding
                
            except Exception as e:
                print(f"Error encoding segment: {e}")
                return torch.zeros(1024, device=self.device) # Assuming output dim is 1024
    
    def forward(self, input_tensor: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: process audio through segment-based MuQ encoding with max pooling
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        batch_size = input_tensor.shape[0]
        
        batch_embeddings = []
        
        for i in range(batch_size):
            waveform = input_tensor[i]
            
            if not torch.isfinite(waveform).all():
                print(f"Warning: Invalid audio data (inf/nan) in batch item {i}")
                batch_embeddings.append(torch.zeros(1024, device=self.device))
                continue
            
            segments = self._create_segments_flexible(waveform)
            
            if not segments:
                if len(waveform) > 0:
                    segments = [waveform]
                else:
                    batch_embeddings.append(torch.zeros(1024, device=self.device))
                    continue
            
            segment_embeddings = [self._encode_segment(seg) for seg in segments]
            
            if segment_embeddings:
                segments_tensor = torch.stack(segment_embeddings)
                # MODIFIED: Max pooling across all segments
                track_embedding = torch.max(segments_tensor, dim=0)[0]
            else:
                track_embedding = torch.zeros(1024, device=self.device)
            
            batch_embeddings.append(track_embedding)
        
        batch_tensor = torch.stack(batch_embeddings)
        
        # Reshape to MARBLE 4D format [batch, num_layers, time_steps, hidden_dim]
        final_embeddings = batch_tensor.unsqueeze(1).unsqueeze(2)
        
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
                  f"Segments: {len(segments) if 'segments' in locals() else 0}, "
                  f"Final shape: {final_embeddings.shape}")
        else:
            print(f"Batch {self.batch_count} - Time: {inference_time:.3f}s, "
                  f"CPU Delta: {cpu_memory_delta:.1f}MB, "
                  f"Segments: {len(segments) if 'segments' in locals() else 0}, "
                  f"Final shape: {final_embeddings.shape}")
        
        if self.batch_count % 50 == 0:
            self.print_intermediate_stats()
        
        return final_embeddings
    
    def print_intermediate_stats(self):
        """Prints intermediate statistics."""
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
        """Prints final statistics."""
        if not self.inference_times:
            print("No inference statistics collected.")
            return
            
        print(f"\n{'='*80}")
        print(f"🎯 FINAL INFERENCE STATISTICS (Total batches: {self.batch_count})")
        print(f"{'='*80}")
        
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        avg_time = sum(self.inference_times) / len(self.inference_times)
        
        print(f"⏱️  INFERENCE TIME:")
        print(f"    Min: {min_time:.3f}s")
        print(f"    Max: {max_time:.3f}s") 
        print(f"    Avg: {avg_time:.3f}s")
        print(f"    Total: {sum(self.inference_times):.3f}s")
        
        min_cpu = min(self.cpu_memory_usage)
        max_cpu = max(self.cpu_memory_usage) 
        avg_cpu = sum(self.cpu_memory_usage) / len(self.cpu_memory_usage)
        
        print(f"\n💾 CPU MEMORY DELTA:")
        print(f"    Min: {min_cpu:.1f}MB")
        print(f"    Max: {max_cpu:.1f}MB")
        print(f"    Avg: {avg_cpu:.1f}MB")
        
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
        """Saves statistics to a file."""
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
        
        filename = f"muq_max_inference_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📁 Statistics saved to: {filename}")
    
    def reset_stats(self):
        """Resets statistics."""
        self.inference_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.batch_count = 0
    
    def get_output_dim(self) -> int:
        """
        Get the output feature dimension
        """
        return 1024
    
    def get_num_layers(self) -> int:
        """
        Get number of layers (for compatibility with MARBLE)
        """
        return 1


import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class MuQMaxStatsCallback(Callback):
    """
    Callback to print statistics for the MuQ max encoder after training/testing.
    """
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prints stats after training."""
        encoder = pl_module.encoder
        if hasattr(encoder, 'print_final_stats'):
            print("\n🎓 TRAINING COMPLETED - INFERENCE STATISTICS:")
            encoder.print_final_stats()
    
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prints stats after testing."""
        encoder = pl_module.encoder
        if hasattr(encoder, 'print_final_stats'):
            print("\n🧪 TESTING COMPLETED - INFERENCE STATISTICS:")
            encoder.print_final_stats()
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prints stats after validation (only at the end of testing)."""
        if trainer.state.stage == "test" and hasattr(pl_module.encoder, 'print_final_stats'):
            print("\n✅ VALIDATION COMPLETED - INFERENCE STATISTICS:")
            pl_module.encoder.print_final_stats()
