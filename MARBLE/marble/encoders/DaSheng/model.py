# marble/encoders/dasheng/model.py
from typing import Sequence, Dict, Optional, Union, Tuple, List

import torch

# Assuming the pretrained models are available from this path
from marble.encoders.DaSheng.pretrained.pretrained import dasheng_base, dasheng_06B, dasheng_12B
from marble.core.base_encoder import BaseEncoder


class DaSheng_Encoder(BaseEncoder):
    """
    A wrapper for the DaSheng audio representation model.
    Supports different model sizes and training modes (freeze, full fine-tuning, LoRA).
    """

    NAME = "DaSheng"
    SAMPLING_RATE = 16000
    TOKEN_RATE = 50

    def __init__(
        self,
        model_size: str = "1.2B", # one of ["base", "0.6B", "1.2B"]
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = ["q_proj", "v_proj"],
    ) -> None:
        """
        Initialize the DaSheng encoder.

        Args:
            model_size (str): The size of the DaSheng model to load.
                              Options: "base", "0.6B", "1.2B".
            train_mode (str): "freeze" to freeze base parameters, "full" for full fine-tuning,
                              or "lora" to freeze base and add LoRA adapters.
            lora_r (int): LoRA adapter rank (only if train_mode="lora").
            lora_alpha (int): LoRA scaling alpha (only if train_mode="lora").
            lora_dropout (float): Dropout probability for LoRA adapters.
        """
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        self.model_size = model_size

        if self.model_size == "base":
            self.model = dasheng_base(pre_trained_folder)
            self.num_features = 768
            self.n_transformer_layers = 12
        elif self.model_size == "0.6B":
            self.model = dasheng_06B(pre_trained_folder)
            self.num_features = 1024
            self.n_transformer_layers = 24
        elif self.model_size == "1.2B":
            self.model = dasheng_12B(pre_trained_folder)
            self.num_features = 1280
            self.n_transformer_layers = 48
        else:
            raise ValueError(f"Unknown model_size: {model_size}. Available options are 'base', '0.6B', '1.2B'.")

        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
        elif train_mode == "lora":
            try:
                from peft import get_peft_model, LoraConfig
            except ImportError:
                raise ImportError("LoRA training requires the 'peft' library. Please install it with 'pip install peft'")
            for param in self.model.parameters():
                param.requires_grad = False
            peft_config = LoraConfig(
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        if train_mode in ["lora", "full"]:
            self.model.train()
        else:
            self.model.eval()

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Perform a forward pass through the DaSheng encoder.

        Args:
            x (torch.Tensor): Waveform tensor, shape (batch_size, num_samples).
            output_hidden_states (bool): Kept for API compatibility. The model only
                                         returns the final hidden state.
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            hidden_states (Tuple[torch.Tensor]): A tuple containing only the final
                  hidden state from the encoder.
        """
        # --- FIX 1: Get model device and dtype robustly ---
        # A standard nn.Module doesn't have a `.device` attribute.
        # The correct way is to check the device of a parameter.
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=model_device, dtype=model_dtype)

        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # --- FIX 2: Handle the model's direct tensor output ---
        # The underlying DaSheng model likely does not accept 'output_hidden_states'
        # and returns the final hidden state tensor directly.
        last_hidden_state = self.model(x, **kwargs)

        # The MARBLE framework expects a tuple of hidden states.
        # Since DaSheng only provides the last one, we wrap it in a tuple
        # to maintain a consistent return type with other encoders.
        return (last_hidden_state,)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for size in ["base", "1.2B"]:
        print(f"\n----- Testing DaSheng_Encoder with model_size='{size}' -----")
        
        wav = torch.randn(2, DaSheng_Encoder.SAMPLING_RATE * 10)
        wavs = wav.to(device)

        try:
            encoder = DaSheng_Encoder(model_size=size)
            # The .to(device) call is on our wrapper, which then applies it
            # to the underlying self.model
            encoder = encoder.to(device).eval() 
        except Exception as e:
            print(f"Could not instantiate model size '{size}': {e}")
            continue

        with torch.no_grad():
            try:
                # This will now work correctly
                output_states = encoder(wavs, output_hidden_states=True)
                
                # Because we fixed the forward pass to wrap the output in a tuple,
                # the length will now be 1.
                print(f"Total number of layers returned: {len(output_states)}")
                print(f"Expected number of transformer layers in model: {encoder.n_transformer_layers}")
                print(f"Output shape of the final (and only returned) layer: {output_states[0].shape}")
                
            except Exception as e:
                 print(f"Forward pass failed for model size '{size}': {e}")