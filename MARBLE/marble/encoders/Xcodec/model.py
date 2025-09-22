# marble/encoders/xcodec/model.py
import os
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from marble.core.base_encoder import BaseEncoder
from marble.encoders.Xcodec.models.soundstream_hubert_new import SoundStream


def build_codec_model(config: OmegaConf) -> nn.Module:
    model = SoundStream(**config.generator.config)
    return model


class Xcodec_Encoder(BaseEncoder):
    """
    An XCodec (SoundStream) based encoder wrapper that automatically downloads
    model weights from the Hugging Face Hub and runs in frozen mode for feature extraction.
    """

    NAME = "Xcodec"
    # The Hugging Face repository ID from which to download the model.
    HUGGINGFACE_MODEL_NAME = "m-a-p/xcodec"
    TOKEN_RATE = 50  # Typical frame rate for XCodec at 16kHz (16000 / 320)
    SAMPLING_RATE = 16000  # Audio sampling rate expected by the model
    
    NUM_FEATURES = 1024
    N_TRANSFORMER_LAYERS = 1  # The XCodec encoder is a CNN, not a Transformer

    def __init__(
        self,
        pre_trained_folder: str = None,
        mode: str = "vq_emb",
        target_bw: float = 6.0,
    ) -> None:
        """
        Initialize the XCodec encoder in frozen mode.

        If the model files (config and checkpoint) are not found in the specified
        `pre_trained_folder` or the default cache, they will be automatically
        downloaded from the Hugging Face Hub.

        Args:
            pre_trained_folder (str, optional): Path to a directory containing
                                                config.yaml and a .pth checkpoint.
                                                If None, uses a default cache path
                                                (e.g., ~/.cache/xcodec).
        """
        super().__init__()
        self.mode = mode
        # removed support for "indices" mode in marble
        assert self.mode in ["vq_emb", "pre_vq_emb"], "Mode must be in ['vq_emb', 'pre_vq_emb']."
        
        self.target_bw = target_bw
        assert self.target_bw in [i*0.5 for i in range(1, 13)], "target_bw must be one of [0.5, 1.0, ..., 6.0]"
        print(f"Initializing XCodec Encoder in '{self.mode}' mode with target_bw={self.target_bw} kbps.")

        # Determine the root directory for model files
        if pre_trained_folder:
            ckpt_root = Path(pre_trained_folder)
        else:
            # Use a dedicated cache directory for this model
            ckpt_root = Path.home() / ".cache" / "xcodec"
        
        # Ensure the cache directory exists
        ckpt_root.mkdir(parents=True, exist_ok=True)

        config_path = ckpt_root / "config.yaml"
        # The specific checkpoint file required by this model
        ckpt_filename = "ckpt_00360000.pth"
        ckpt_path = ckpt_root / ckpt_filename

        # If config file is missing, download it from the Hub
        if not config_path.is_file():
            print(f"Config file not found. Downloading 'config.yaml' from '{self.HUGGINGFACE_MODEL_NAME}'...")
            hf_hub_download(
                repo_id=self.HUGGINGFACE_MODEL_NAME,
                filename="config.yaml",
                local_dir=ckpt_root,
                local_dir_use_symlinks=False,  # Make a direct copy
            )
            print("Download complete.")

        # If checkpoint file is missing, download it from the Hub
        if not ckpt_path.is_file():
            print(f"Checkpoint not found. Downloading '{ckpt_filename}' from '{self.HUGGINGFACE_MODEL_NAME}'...")
            hf_hub_download(
                repo_id=self.HUGGINGFACE_MODEL_NAME,
                filename=ckpt_filename,
                local_dir=ckpt_root,
                local_dir_use_symlinks=False,  # Make a direct copy
            )
            print("Download complete.")

        # Load model configuration and build the model
        config = OmegaConf.load(config_path)
        self.model = build_codec_model(config)

        # Load model weights
        parameter_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(parameter_dict['codec_model'])

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set the model to evaluation mode
        self.model.eval()

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Perform a forward pass through the XCodec encoder to extract continuous embeddings.

        Args:
            x (torch.Tensor): Waveform tensor, shape (batch_size, num_samples), values in [-1, 1].
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            hidden_states (Tuple[torch.Tensor]): A tuple containing a single tensor, which is the
                  embedding from the final layer. The embedding tensor has a shape of 
                  (batch_size, seq_len, NUM_FEATURES).
        """
        # Ensure input dtype matches model parameters
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=model_device, dtype=model_dtype)

        # SoundStream/EnCodec models expect an input shape of [B, C, T], where C=1
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        embeddings_b_d_t = self.model.encode(x, mode=self.mode, target_bw=self.target_bw, **kwargs) # [B, H, T]
        embeddings_b_t_d = embeddings_b_d_t.permute(0, 2, 1) # [B, T, H]

        return (embeddings_b_t_d,)
