# marble/encoders/CLaMP3/model.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from einops import rearrange
from transformers import BertConfig, AutoTokenizer

from marble.core.base_encoder import BaseEncoder
from marble.encoders.CLaMP3.clamp3_util import CLaMP3Model, M3Patchilizer
from marble.encoders.CLaMP3.mert_util import load_audio
from marble.encoders.MERT.model import MERT_v1_95M_Encoder, MERT_v1_95M_FeatureExtractor


# --- Constants and Configuration ---
DEFAULT_PRE_TRAINED_FOLDER = os.path.expanduser("~/.cache/clamp3/")
CLAMP3_CKPT_NAME = "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
CLAMP3_LINK = f"https://huggingface.co/sander-wood/clamp3/resolve/main/{CLAMP3_CKPT_NAME}"

class CLaMP3Config:
    """Configuration class for CLaMP3 model parameters."""
    # Text Model Config
    TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
    MAX_TEXT_LENGTH = 128

    # Audio Model Config
    AUDIO_HIDDEN_SIZE = 768
    AUDIO_NUM_LAYERS = 12
    MAX_AUDIO_LENGTH = 128
    
    # Symbolic (M3) Model Config
    M3_HIDDEN_SIZE = 768
    PATCH_SIZE = 64
    PATCH_LENGTH = 512
    PATCH_NUM_LAYERS = 12
    
    # CLaMP3 Model Config
    CLAMP3_HIDDEN_SIZE = 768
    CLAMP3_LOAD_M3 = True # Inferred from infer_test3, can be configurable
    LOGIT_SCALE = 1.0


# --- Helper Functions ---

def download_checkpoint_if_needed(folder: str, filename: str, url: str):
    """Downloads the checkpoint file if it doesn't exist."""
    if not os.path.exists(folder):
        print(f"Creating directory: {folder}")
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        print("Downloading pre-trained CLaMP3 model...")
        os.system(f"wget -O {filepath} {url}") # Use -O to ensure correct filename
        print(f"Download complete. File saved to: {filepath}")
    return filepath

def extract_mert_features_batch(
    waveforms: torch.Tensor,
    feature_extractor: 'MERT_v1_95M_Encoder',
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Extracts per-layer, time-averaged features from a batch of audio waveforms
    using a sliding window approach.

    This function processes full-length chunks and the final partial chunk in separate
    inference calls to avoid padding the last chunk, ensuring feature fidelity.

    Args:
        waveforms (torch.Tensor): A batch of audio waveforms.
            - Shape: (batch_size, num_samples).
            - Assumption: All waveforms in the batch have the same length and have
              already been resampled to the model's target sampling rate.
        feature_extractor (MERT_v1_95M_Encoder): The MERT encoder instance.
        device (Union[str, torch.device]): The device ('cuda' or 'cpu') to run computations on.

    Returns:
        torch.Tensor: A tensor containing the extracted features. Returns None if no
                      valid chunks are found.
            - Shape: (batch_size, num_layers, num_chunks, feature_dim).
    """
    # --- Configuration ---
    target_sr = 24000
    sliding_window_size_in_sec = 5.0
    
    wavs = waveforms.to(device)
    
    window_size_samples = int(target_sr * sliding_window_size_in_sec)
    
    # --- Step 1: Chunking ---
    all_chunks = list(wavs.split(window_size_samples, dim=-1))
    
    min_len_samples = int(target_sr * 1)
    if all_chunks and all_chunks[-1].shape[-1] < min_len_samples:
        all_chunks = all_chunks[:-1]

    if not all_chunks:
        return None

    # --- Step 2: Separate full chunks from the last partial chunk ---
    last_chunk = None
    if all_chunks[-1].shape[-1] < window_size_samples:
        last_chunk = all_chunks[-1]
        full_chunks = all_chunks[:-1]
    else:
        full_chunks = all_chunks
    
    all_features = []

    # --- Step 3a: Process all full-sized chunks ---
    if full_chunks:
        full_chunks_tensor = torch.cat(full_chunks, dim=0)
        o = feature_extractor(full_chunks_tensor).hidden_states
        time_averaged_features = torch.stack(o).mean(-2)
        batch_size = waveforms.size(0)
        full_features = rearrange(time_averaged_features, 'l (c b) h -> b c l h', b=batch_size)
        all_features.append(full_features)

    # --- Step 3b: Process the final, shorter chunk ---
    if last_chunk is not None:
        o = feature_extractor(last_chunk).hidden_states
        time_averaged_features = torch.stack(o, dim=0).mean(dim=-2)
        last_feature = rearrange(time_averaged_features, 'l b h -> b 1 l h')
        all_features.append(last_feature)
        
    # --- Step 4: Concatenate results ---
    if not all_features:
        return None
    
    final_features = torch.cat(all_features, dim=1)
    final_features = rearrange(final_features, 'b c l h -> b l c h')
    
    return final_features


# --- Core Encoder Class ---
class CLaMP3_FeatureExtractor(MERT_v1_95M_FeatureExtractor):
    pass
    
class CLaMP3_Encoder(BaseEncoder):
    """
    CLaMP3 Encoder for generating joint text, audio, and symbolic embeddings.
    This implementation extracts MERT features on-the-fly and supports batching.
    """
    NAME = "CLaMP3"
    SAMPLING_RATE = 24000
    NUM_FEATURES = 768
    TOKEN_RATE = 1

    def __init__(self, train_mode: str = "freeze", pre_trained_folder: str = None,) -> None:
        super().__init__()

        self.config = CLaMP3Config()
        self.sample_rate = self.SAMPLING_RATE

        # 1. Download CLaMP3 checkpoint
        pre_trained_folder = pre_trained_folder or DEFAULT_PRE_TRAINED_FOLDER
        checkpoint_path = download_checkpoint_if_needed(
            pre_trained_folder, CLAMP3_CKPT_NAME, CLAMP3_LINK
        )
        
        # 2. Initialize MERT models for feature extraction
        self.mert_preprocessor = MERT_v1_95M_FeatureExtractor()
        self.mert_encoder = MERT_v1_95M_Encoder()
        
        # 3. Initialize CLaMP3 components
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.TEXT_MODEL_NAME)
        
        audio_config = BertConfig(
            vocab_size=1, hidden_size=self.config.AUDIO_HIDDEN_SIZE,
            num_hidden_layers=self.config.AUDIO_NUM_LAYERS,
            num_attention_heads=self.config.AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=self.config.AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=self.config.MAX_AUDIO_LENGTH
        )
        symbolic_config = BertConfig(
            vocab_size=1, hidden_size=self.config.M3_HIDDEN_SIZE,
            num_hidden_layers=self.config.PATCH_NUM_LAYERS,
            num_attention_heads=self.config.M3_HIDDEN_SIZE // 64,
            intermediate_size=self.config.M3_HIDDEN_SIZE * 4,
            max_position_embeddings=self.config.PATCH_LENGTH
        )
        self.model = CLaMP3Model(
            audio_config=audio_config, symbolic_config=symbolic_config,
            hidden_size=self.config.CLAMP3_HIDDEN_SIZE,
            load_m3=self.config.CLAMP3_LOAD_M3
        )
        
        # 4. Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"Loading CLaMP3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        self.model.load_state_dict(checkpoint['model'])

        # 5. Set training mode
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.mert_encoder.parameters():
                param.requires_grad = False
            self.model.eval()
            self.mert_encoder.eval()
        else:
            raise NotImplementedError(f"Train mode '{train_mode}' is not supported for CLaMP3_Encoder.")

    def _prepare_segments(self, data: torch.Tensor, max_len: int) -> List[torch.Tensor]:
        """Replicates the segmentation strategy from the original inference script."""
        if len(data) <= max_len:
            return [data]
        
        segments = list(data.split(max_len, dim=0))
        # Ensure the last segment is also max_len by overlapping
        if len(segments) > 1 and segments[-1].shape[0] < max_len:
             segments[-1] = data[-max_len:]
        return segments

    @torch.no_grad()
    def _get_embedding_from_segments(
        self,
        input_data: torch.Tensor,
        max_len: int,
        data_type: str,
        device: torch.device
    ) -> torch.Tensor:
        """Processes segmented data to get a final global embedding."""
        segments = self._prepare_segments(input_data, max_len)
        hidden_states_list = []

        for segment in segments:
            seg_len = segment.size(0)
            mask = torch.ones(seg_len, device=device)
            pad_len = max_len - seg_len
            mask = F.pad(mask, (0, pad_len), 'constant', 0)
            
            if data_type == 'text':
                segment = F.pad(segment, (0, pad_len), 'constant', self.tokenizer.pad_token_id)
                features = self.model.get_text_features(
                    text_inputs=segment.unsqueeze(0), text_masks=mask.unsqueeze(0), get_global=True
                )
            elif data_type == 'audio':
                pad_tensor = torch.zeros(pad_len, self.config.AUDIO_HIDDEN_SIZE, device=device)
                segment = torch.cat((segment, pad_tensor), 0)
                features = self.model.get_audio_features(
                    audio_inputs=segment.unsqueeze(0), audio_masks=mask.unsqueeze(0), get_global=True
                )
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")
            
            hidden_states_list.append(features)

        # Weighted average of segment features to get the final embedding
        full_chunks = len(input_data) // max_len
        rem_len = len(input_data) % max_len
        weights = [max_len] * full_chunks
        if rem_len > 0:
            weights.append(rem_len)
        
        feature_weights = torch.tensor(weights, device=device).view(-1, 1)
        all_features = torch.cat(hidden_states_list, dim=0)
        final_embedding = (all_features * feature_weights).sum(dim=0) / feature_weights.sum()
            
        return final_embedding
        
    @torch.no_grad()
    def forward(
        self,
        wavs: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None
    ) -> tuple:
        """
        Generates embeddings for a batch of audio waveforms or texts.
        """
        # Determine device from input tensor or model parameters
        if wavs is not None:
            device = wavs.device
        elif texts is not None:
            device = next(self.model.parameters()).device
        else:
            raise ValueError("Either 'wavs' or 'texts' must be provided.")
        
        self.model.to(device)
        self.mert_encoder.to(device)

        # --- Audio Path ---
        if wavs is not None:
            batch_size = wavs.size(0)
            
            # 1. Preprocess audio batch for MERT
            processed_wavs_list = [
                self.mert_preprocessor({'input_features': wav, 'sampling_rate': self.SAMPLING_RATE})['input_features']
                for wav in wavs
            ]
            processed_wavs = torch.stack(processed_wavs_list, dim=0)

            # 2. Extract batched MERT features (B, L, C, H)
            mert_features = extract_mert_features_batch(processed_wavs, self.mert_encoder, device)
            
            # ** 3. Average over layers to align with infer_test3.py logic **
            # This is the key step for result alignment.
            # Shape becomes (B, C, H)
            mert_chunk_features = mert_features.mean(dim=1) 
            
            # 4. Process each item in the batch through CLaMP's audio tower
            embeddings = []
            for i in range(batch_size):
                item_features = mert_chunk_features[i].to(device) # Shape: (C, H)
                
                # Add zero vectors at start/end to match original script
                zero_vec = torch.zeros((1, item_features.size(-1)), device=device)
                input_data = torch.cat((zero_vec, item_features, zero_vec), 0)

                emb = self._get_embedding_from_segments(
                    input_data=input_data,
                    max_len=self.config.MAX_AUDIO_LENGTH,
                    data_type='audio',
                    device=device
                )
                embeddings.append(emb)

            output = torch.stack(embeddings, dim=0)
            return (output.unsqueeze(1),) # Return shape (B, 1, H)

        # --- Text Path ---
        if texts is not None:
            if isinstance(texts, str): texts = [texts]
            
            embeddings = []
            for text_item in texts:
                # Prepare text input
                items = list(set(text_item.split("\n")))
                items = "\n".join(items).split("\n")
                items = [c for c in items if len(c) > 0]
                item_str = self.tokenizer.sep_token.join(items)
                input_data = self.tokenizer(item_str, return_tensors="pt")['input_ids'].squeeze(0).to(device)

                emb = self._get_embedding_from_segments(
                    input_data=input_data,
                    max_len=self.config.MAX_TEXT_LENGTH,
                    data_type='text',
                    device=device
                )
                embeddings.append(emb)

            output = torch.stack(embeddings, dim=0)
            return (output.unsqueeze(1),) # Return shape (B, 1, H)


if __name__ == '__main__':
    # --- GTZAN Demo for Verification ---
    print("--- Running GTZAN Demo with CLaMP3_Encoder ---")
    
    # 1. Setup paths and audio file
    demo_dir = "tests"
    audio_path = os.path.join(demo_dir, "blues.00000.wav")
    if not os.path.exists(audio_path):
        print(f"'{audio_path}' not found. Please ensure it exists.")
        # Create a dummy file if it doesn't exist, similar to infer_test3.py
        if not os.path.exists(demo_dir): os.makedirs(demo_dir)
        import wave, struct
        sample_rate = 22050.0; duration = 30; n_samples = int(duration * sample_rate)
        with wave.open(audio_path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
            for _ in range(n_samples): wf.writeframes(struct.pack('<h', 0))
        print(f"Created a dummy silent WAV file at '{audio_path}'.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Instantiate the encoder
    # This will automatically handle model downloading and setup
    encoder = CLaMP3_Encoder().to(device)

    # 3. Load and prepare audio data (as a batch of size 2 for testing)
    waveform = load_audio(audio_path, target_sr=encoder.SAMPLING_RATE, is_mono=True)
    wavs_batch = torch.stack([waveform, waveform], dim=0).to(device)
    print(f"Audio loaded and prepared as a batch of shape: {wavs_batch.shape}")

    # 4. Get audio features (embedding)
    print("\nExtracting audio features...")
    # The forward pass handles all intermediate steps (MERT extraction, CLaMP projection)
    audio_output = encoder(wavs=wavs_batch)
    print(f"Audio features extracted with shape: {audio_output[0].shape}")  # Should be (2, 1, 768)
    
    # We only need the embedding for the first item in the batch for this demo
    audio_feature = audio_output[0][0] # Shape: (1, 768)
    print("Audio feature extracted.")

    # 5. Calculate similarities with GTZAN genres
    gtzan_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    similarities = {}

    print("\nCalculating similarities with GTZAN genres...")
    audio_feature_norm = audio_feature / audio_feature.norm(dim=-1, keepdim=True)
    
    for genre in tqdm(gtzan_genres, desc="Genres"):
        # Get text embedding for each genre
        text_output = encoder(texts=genre)
        print(f"Text feature for genre '{genre}': {text_output[0].shape}")
        text_feature = text_output[0].squeeze(1) # Shape: (1, 768)
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
        similarity = (audio_feature_norm * text_feature_norm).sum().item()
        similarities[genre] = similarity

    # 6. Print results for verification
    print(f"\n--- Similarity Results for '{os.path.basename(audio_path)}' ---")
    sorted_genres = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    for genre, score in sorted_genres:
        print(f"{genre:<10}: {score:.4f}")

    print("\n✅ Verification complete. Compare these scores with the original script's output.")