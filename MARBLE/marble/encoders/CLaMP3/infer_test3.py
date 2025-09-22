# marble/encoders/CLaMP3/infer_test3.py
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertConfig, AutoTokenizer
from typing import Union, List

# Assuming these imports are in the user's environment
from marble.encoders.CLaMP3.hf_pretrains import HuBERTFeature
from marble.encoders.CLaMP3.clamp3_util import CLaMP3Model, M3Patchilizer
from marble.encoders.CLaMP3.mert_util import load_audio


# --- Constants and Configuration ---
PRE_TRAINED_FOLDER = os.path.expanduser("~/.cache/clamp3/")
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
    CLAMP3_LOAD_M3 = True
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
        os.system(f"wget -P {folder} {url}")
        print(f"Download complete. File saved to: {filepath}")
    return filepath

def extract_mert_features(
    audio_file: str,
    feature_extractor: HuBERTFeature,
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Extracts features from an audio file using the MERT model.
    
    *** CORRECTION: This function now correctly implements the 5-second sliding window
    logic from the original script to ensure identical output. ***
    """
    target_sr = 24000
    sliding_window_size_in_sec = 5.0
    
    try:
        waveform = load_audio(
            audio_file,
            target_sr=target_sr,
            is_mono=True,
            is_normalize=False,
            device=device,
        )
    except Exception as e:
        print(f"Failed to load audio {audio_file}: {e}")
        return None

    wav = feature_extractor.process_wav(waveform).to(device)
    
    # Re-implementing the original sliding window logic
    if sliding_window_size_in_sec > 0:
        wavs = []
        # Non-overlapping window of 5 seconds
        window_size_samples = int(target_sr * sliding_window_size_in_sec)
        for i in range(0, wav.shape[-1], window_size_samples):
            wavs.append(wav[:, i : i + window_size_samples])
        
        # Original script's logic to drop the last chunk if it's less than 1 second
        if wavs and wavs[-1].shape[-1] < target_sr * 1:
            wavs = wavs[:-1]
            
        features = []
        for wav_chunk in wavs:
            # reduction='mean' is applied per-chunk here
            features.append(feature_extractor(wav_chunk, layer=None, reduction='mean'))
        
        # Features from each chunk are concatenated along the time/sequence dimension
        features = torch.cat(features, dim=1)
    else:
        features = feature_extractor(wav, layer=None, reduction='mean')
        
    return features


# --- Core Inference Class ---

class CLaMP3Inference:
    """A class to encapsulate CLaMP3 model loading and feature extraction."""

    def __init__(self, config: CLaMP3Config, checkpoint_path: str, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.patchilizer = M3Patchilizer()
        
        # Correctly set vocab_size=1 for audio and symbolic models
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=config.AUDIO_HIDDEN_SIZE,
            num_hidden_layers=config.AUDIO_NUM_LAYERS,
            num_attention_heads=config.AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=config.AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=config.MAX_AUDIO_LENGTH
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=config.M3_HIDDEN_SIZE,
            num_hidden_layers=config.PATCH_NUM_LAYERS,
            num_attention_heads=config.M3_HIDDEN_SIZE // 64,
            intermediate_size=config.M3_HIDDEN_SIZE * 4,
            max_position_embeddings=config.PATCH_LENGTH
        )
        self.model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            hidden_size=config.CLAMP3_HIDDEN_SIZE,
            load_m3=config.CLAMP3_LOAD_M3
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"Successfully Loaded CLaMP3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def _prepare_segments(self, data: torch.Tensor, max_len: int) -> List[torch.Tensor]:
        if len(data) <= max_len:
            return [data]
        
        segments = list(data.split(max_len, dim=0))
        segments[-1] = data[-max_len:]
        return segments

    @torch.no_grad()
    def get_embedding(
        self,
        data: Union[str, np.ndarray],
        data_type: str,
        get_global: bool = True
    ) -> torch.Tensor:
        if data_type == 'text':
            items = list(set(data.split("\n")))
            items = "\n".join(items).split("\n")
            items = [c for c in items if len(c) > 0]
            item_str = self.tokenizer.sep_token.join(items)
            input_data = self.tokenizer(item_str, return_tensors="pt")['input_ids'].squeeze(0)
            max_len = self.config.MAX_TEXT_LENGTH
        elif data_type == 'audio':
            input_data = torch.from_numpy(data).float()
            input_data = input_data.reshape(-1, input_data.size(-1))
            zero_vec = torch.zeros((1, input_data.size(-1)))
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
            max_len = self.config.MAX_AUDIO_LENGTH
        elif data_type == 'symbolic':
            input_data = torch.tensor(self.patchilizer.encode(data, add_special_patches=True))
            max_len = self.config.PATCH_LENGTH
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        segments = self._prepare_segments(input_data, max_len)
        hidden_states_list = []

        for segment in segments:
            seg_len = segment.size(0)
            mask = torch.ones(seg_len, device=self.device)

            pad_len = max_len - seg_len
            mask = torch.cat((mask, torch.zeros(pad_len, device=self.device)), 0)
            
            if data_type == 'text':
                pad_indices = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                segment = torch.cat((segment, pad_indices), 0)
                features = self.model.get_text_features(
                    text_inputs=segment.unsqueeze(0).to(self.device),
                    text_masks=mask.unsqueeze(0).to(self.device),
                    get_global=get_global
                )
            elif data_type == 'audio':
                pad_indices = torch.zeros((pad_len, self.config.AUDIO_HIDDEN_SIZE), dtype=torch.float)
                segment = torch.cat((segment, pad_indices), 0)
                features = self.model.get_audio_features(
                    audio_inputs=segment.unsqueeze(0).to(self.device),
                    audio_masks=mask.unsqueeze(0).to(self.device),
                    get_global=get_global
                )
            else: # symbolic
                pad_indices = torch.full((pad_len, self.config.PATCH_SIZE), self.patchilizer.pad_token_id, dtype=torch.long)
                segment = torch.cat((segment, pad_indices), 0)
                features = self.model.get_symbolic_features(
                    symbolic_inputs=segment.unsqueeze(0).to(self.device),
                    symbolic_masks=mask.unsqueeze(0).to(self.device),
                    get_global=get_global
                )
            
            if not get_global:
                features = features[:, :seg_len, :]
            hidden_states_list.append(features)

        if get_global:
            full_chunks = len(input_data) // max_len
            rem_len = len(input_data) % max_len
            weights = [max_len] * full_chunks
            if rem_len > 0:
                weights.append(rem_len)
            
            feature_weights = torch.tensor(weights, device=self.device).view(-1, 1)
            all_features = torch.cat(hidden_states_list, dim=0)
            final_embedding = (all_features * feature_weights).sum(dim=0) / feature_weights.sum()
        else:
            all_features = [fs.squeeze(0) for fs in hidden_states_list]
            if len(all_features) > 1:
                rem_len = len(input_data) % max_len
                if rem_len > 0:
                    all_features[-1] = all_features[-1][-rem_len:]
                else: 
                    all_features[-1] = all_features[-1][-max_len:]
            final_embedding = torch.cat(all_features, dim=0)
            
        return final_embedding

# --- GTZAN Demo ---

def run_gtzan_demo():
    demo_dir = "tests"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    audio_path = os.path.join(demo_dir, "blues.00000.wav")
    if not os.path.exists(audio_path):
        print(f"'{audio_path}' not found. Creating a dummy silent WAV file for demonstration.")
        import wave, struct
        sample_rate = 22050.0
        duration = 30
        n_samples = int(duration * sample_rate)
        with wave.open(audio_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for _ in range(n_samples):
                wf.writeframes(struct.pack('<h', 0))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    mert_extractor = HuBERTFeature("m-a-p/MERT-v1-95M", 24000, force_half=False).to(device)
    mert_extractor.eval()
    print('Loaded MERT model.')

    checkpoint_path = download_checkpoint_if_needed(PRE_TRAINED_FOLDER, CLAMP3_CKPT_NAME, CLAMP3_LINK)
    clamp_infer = CLaMP3Inference(CLaMP3Config(), checkpoint_path, device)
    
    # --- Step 1: Extract Audio Feature ---
    print(f"\nProcessing audio file: {audio_path}")
    mert_feature = extract_mert_features(audio_path, mert_extractor, device)
    
    # This mean reduction is applied *after* the sliding window features are concatenated
    mert_feature_np = mert_feature.mean(dim=0, keepdim=True).cpu().numpy()
    
    audio_feature = clamp_infer.get_embedding(mert_feature_np, 'audio', get_global=True).unsqueeze(0)
    print("Audio feature extracted.")

    # --- Step 2: Calculate Similarities with GTZAN Genres ---
    gtzan_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    similarities = {}

    print("\nCalculating similarities with GTZAN genres...")
    audio_feature_norm = audio_feature / audio_feature.norm(dim=-1, keepdim=True)
    
    for genre in tqdm(gtzan_genres):
        text_feature = clamp_infer.get_embedding(genre, 'text', get_global=True).unsqueeze(0)
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = (audio_feature_norm * text_feature_norm).sum().item()
        similarities[genre] = similarity

    # --- Step 3: Print Results ---
    print(f"\n--- Similarity Results for '{os.path.basename(audio_path)}' ---")
    sorted_genres = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    for genre, score in sorted_genres:
        print(f"{genre:<10}: {score:.4f}")

if __name__ == "__main__":
    run_gtzan_demo()