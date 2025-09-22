import os
import json
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertConfig, AutoTokenizer
from marble.encoders.CLaMP3.hf_pretrains import HuBERTFeature
from marble.encoders.CLaMP3.clamp3_util import CLaMP3Model, M3Patchilizer
from marble.encoders.CLaMP3.mert_util import load_audio

# https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth
pre_trained_folder = os.path.expanduser("~/.cache/clamp3/")
if not os.path.exists(pre_trained_folder):
    os.makedirs(pre_trained_folder)

CLAMP3_CKPT_NAME = "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
CLAMP3_LINK = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
if not os.path.exists(os.path.join(pre_trained_folder, CLAMP3_CKPT_NAME)):
    print("Downloading pre-trained CLaMP3 model files...")
    os.system(f"wget -P {pre_trained_folder} {CLAMP3_LINK}")
    print("Download complete. Files saved to:", pre_trained_folder)

class CLaMP3Config():
    TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
    CLAMP3_HIDDEN_SIZE = 768
    CLAMP3_LOAD_M3 = True

    PATCH_SIZE = 64  # Size of each patch
    PATCH_LENGTH = 512  # Length of the patches
    PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
    TOKEN_NUM_LAYERS = 3  # Number of layers in the decoder
    M3_HIDDEN_SIZE = 768  # Size of the hidden layer

    M3_LEARNING_RATE = 1e-4  # Learning rate for the optimizer
    M3_BATCH_SIZE = 16  # Batch size per GPU (single card) during training
    M3_MASK_RATIO = 0.45  # Ratio of masked elements during training

    M3_WEIGHTS_PATH = (
        "weights_m3"+
        "_h_size_" + str(M3_HIDDEN_SIZE) +
        "_t_layers_" + str(TOKEN_NUM_LAYERS) +
        "_p_layers_" + str(PATCH_NUM_LAYERS) +
        "_p_size_" + str(PATCH_SIZE) +
        "_p_length_" + str(PATCH_LENGTH) +
        "_lr_" + str(M3_LEARNING_RATE) +
        "_batch_" + str(M3_BATCH_SIZE) +
        "_mask_" + str(M3_MASK_RATIO) + ".pth"
    )  # Path to store the model weights

    LOGIT_SCALE = 1

    AUDIO_HIDDEN_SIZE = 768
    AUDIO_NUM_LAYERS = 12
    MAX_AUDIO_LENGTH = 128

    MAX_TEXT_LENGTH = 128

mert_feature_extractor = HuBERTFeature(
    "m-a-p/MERT-v1-95M",
    24000,
    force_half=False,
    processor_normalize=True,
).cuda()

print('loaded mert model')

mert_feature_extractor.eval()

audio_config = BertConfig(vocab_size=1,
    hidden_size=CLaMP3Config.AUDIO_HIDDEN_SIZE,
    num_hidden_layers=CLaMP3Config.AUDIO_NUM_LAYERS,
    num_attention_heads=CLaMP3Config.AUDIO_HIDDEN_SIZE//64,
    intermediate_size=CLaMP3Config.AUDIO_HIDDEN_SIZE*4,
    max_position_embeddings=CLaMP3Config.MAX_AUDIO_LENGTH)

symbolic_config = BertConfig(vocab_size=1,
    hidden_size=CLaMP3Config.M3_HIDDEN_SIZE,
    num_hidden_layers=CLaMP3Config.PATCH_NUM_LAYERS,
    num_attention_heads=CLaMP3Config.M3_HIDDEN_SIZE//64,
    intermediate_size=CLaMP3Config.M3_HIDDEN_SIZE*4,
    max_position_embeddings=CLaMP3Config.PATCH_LENGTH)

clamp3_model = CLaMP3Model(audio_config=audio_config,
    symbolic_config=symbolic_config,
    hidden_size=CLaMP3Config.CLAMP3_HIDDEN_SIZE,
    load_m3=CLaMP3Config.CLAMP3_LOAD_M3).cuda()


clamp3_tokenizer = AutoTokenizer.from_pretrained(CLaMP3Config.TEXT_MODEL_NAME)
clamp3_patchilizer = M3Patchilizer()

clamp3_model.eval()
checkpoint = torch.load(os.path.join(pre_trained_folder, CLAMP3_CKPT_NAME), map_location="cpu", weights_only=True)
print(f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
clamp3_model.load_state_dict(checkpoint['model'])

def extract_feature_clamp3(filename, get_global):
    if not filename.endswith(".npy"):
        with open(filename, "r", encoding="utf-8") as f:
            item = f.read()

    if filename.endswith(".txt"):
        item = list(set(item.split("\n")))
        item = "\n".join(item)
        item = item.split("\n")
        item = [c for c in item if len(c) > 0]
        item = clamp3_tokenizer.sep_token.join(item)
        input_data = clamp3_tokenizer(item, return_tensors="pt")
        input_data = input_data['input_ids'].squeeze(0)
        max_input_length = CLaMP3Config.MAX_TEXT_LENGTH
    elif filename.endswith(".abc") or filename.endswith(".mtf"):
        input_data = clamp3_patchilizer.encode(item, add_special_patches=True)
        input_data = torch.tensor(input_data)
        max_input_length = CLaMP3Config.PATCH_LENGTH
    elif filename.endswith(".npy"):
        input_data = np.load(filename)
        input_data = torch.tensor(input_data)
        input_data = input_data.reshape(-1, input_data.size(-1))
        zero_vec = torch.zeros((1, input_data.size(-1)))
        input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
        max_input_length = CLaMP3Config.MAX_AUDIO_LENGTH
    else:
        raise ValueError(f"Unsupported file type: {filename}, only support .txt, .abc, .mtf, .npy files")

    segment_list = []
    for i in range(0, len(input_data), max_input_length):
        segment_list.append(input_data[i:i+max_input_length])
    segment_list[-1] = input_data[-max_input_length:]

    last_hidden_states_list = []

    for input_segment in segment_list:
        input_masks = torch.tensor([1]*input_segment.size(0))
        if filename.endswith(".txt"):
            pad_indices = torch.ones(CLaMP3Config.MAX_TEXT_LENGTH - input_segment.size(0)).long() * clamp3_tokenizer.pad_token_id
        elif filename.endswith(".abc") or filename.endswith(".mtf"):
            pad_indices = torch.ones((CLaMP3Config.PATCH_LENGTH - input_segment.size(0), CLaMP3Config.PATCH_SIZE)).long() * clamp3_patchilizer.pad_token_id
        else:
            pad_indices = torch.ones((CLaMP3Config.MAX_AUDIO_LENGTH - input_segment.size(0), CLaMP3Config.AUDIO_HIDDEN_SIZE)).float() * 0.
        input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
        input_segment = torch.cat((input_segment, pad_indices), 0)

        if filename.endswith(".txt"):
            last_hidden_states = clamp3_model.get_text_features(text_inputs=input_segment.unsqueeze(0).cuda(),
                                                        text_masks=input_masks.unsqueeze(0).cuda(),
                                                        get_global=get_global)
        elif filename.endswith(".abc") or filename.endswith(".mtf"):
            last_hidden_states = clamp3_model.get_symbolic_features(symbolic_inputs=input_segment.unsqueeze(0).cuda(),
                                                        symbolic_masks=input_masks.unsqueeze(0).cuda(),
                                                        get_global=get_global)
        else:
            last_hidden_states = clamp3_model.get_audio_features(audio_inputs=input_segment.unsqueeze(0).cuda(),
                                                        audio_masks=input_masks.unsqueeze(0).cuda(),
                                                        get_global=get_global)
        if not get_global:
            last_hidden_states = last_hidden_states[:, :input_masks.sum().long().item(), :]
        last_hidden_states_list.append(last_hidden_states)

    if not get_global:
        last_hidden_states_list = [last_hidden_states[0] for last_hidden_states in last_hidden_states_list]
        last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(input_data)%max_input_length):]
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
    else:
        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt).view(-1, 1).cuda()
        else:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len]).view(-1, 1).cuda()
        
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

    return last_hidden_states_list


def mert_infr_features(audio_file, device):
    target_sr = 24000
    is_mono = True
    is_normalize = False
    crop_to_length_in_sec = None
    crop_randomly = False
    sliding_window_size_in_sec = 5
    sliding_window_overlap_in_percent = 0.0
    layer = None
    reduction = 'mean'

    try:
        waveform = load_audio(
            audio_file,
            target_sr=target_sr,
            is_mono=is_mono,
            is_normalize=is_normalize,
            crop_to_length_in_sec=crop_to_length_in_sec,
            crop_randomly=crop_randomly,
            device=device,
        )
    except Exception as e:
        print(f"Failed to load audio {audio_file}: {e}")
        return None
    
    wav = mert_feature_extractor.process_wav(waveform)
    wav = wav.to(device)
    if sliding_window_size_in_sec:
        assert sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
        overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100
        wavs = []
        for i in range(0, wav.shape[-1], int(target_sr * (sliding_window_size_in_sec - overlap_in_sec))):
            wavs.append(wav[:, i : i + int(target_sr * sliding_window_size_in_sec)])
        if wavs[-1].shape[-1] < target_sr * 1:
            wavs = wavs[:-1]
        features = []
        for wav_chunk in wavs:
            features.append(mert_feature_extractor(wav_chunk, layer=layer, reduction=reduction))
        features = torch.cat(features, dim=1)
    else:
        features = mert_feature_extractor(wav, layer=layer, reduction=reduction)
    return features

rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))


# for _, l in tqdm(data_shard):
#     item = json.loads(l)
#     audio_path = "tests/test.mp3"

#     # text = ", ".join(item['musicinfo']['instruments'])
#     # tgt_path = "tests/test.txt"

#     # if not os.path.exists(tgt_path):
#     #     tgt_file = open(tgt_path, 'w')
#     #     tgt_file.write(text)
#     #     tgt_file.close()

#     mert_feature_path = "tests/test.mert.npy"

#     if not os.path.exists(mert_feature_path):
#         try:
#             with torch.no_grad():
#                 mert_feature = mert_infr_features(audio_path, 'cuda')

#             mert_feature = mert_feature.mean(dim=0, keepdim=True)
#             mert_feature = mert_feature.cpu().numpy()
#             np.save(mert_feature_path, mert_feature)
#         except Exception as e:
#             print(e)
#             print(f"ERROR in processing mert {item['uniq_id']}")


#     # text_feature_path = "tests/test.text.npy"

#     # if not os.path.exists(text_feature_path):
#     #     try:
#     #         with torch.no_grad():
#     #             text_feature = extract_feature_clamp3(tgt_path, get_global=True).unsqueeze(0)
#     #         np.save(text_feature_path, text_feature.detach().cpu().numpy())
#     #     except Exception as e:
#     #         print(e)
#     #         print(f"ERROR in processing text clamp3 {item['uniq_id']}")


#     audio_feature_path = "tests/test.audio.npy"

#     if not os.path.exists(audio_feature_path):
#         try:
#             with torch.no_grad():
#                 audio_feature = extract_feature_clamp3(mert_feature_path, get_global=True).unsqueeze(0)
        
#             np.save(audio_feature_path, audio_feature.detach().cpu().numpy())
#         except Exception as e:
#             print(e)
#             print(f"ERROR in processing audio clamp3 {item['uniq_id']}")


audio_path = "tests/blues.00000.wav"

mert_feature_path = "tests/test.mert.npy"

with torch.no_grad():
    mert_feature = mert_infr_features(audio_path, 'cuda')

mert_feature = mert_feature.mean(dim=0, keepdim=True)
mert_feature = mert_feature.cpu().numpy()
np.save(mert_feature_path, mert_feature)

audio_feature_path = "tests/test.clamp3.npy"

with torch.no_grad():
    audio_feature = extract_feature_clamp3(mert_feature_path, get_global=True).unsqueeze(0)

np.save(audio_feature_path, audio_feature.detach().cpu().numpy())
