import torch
import librosa
import numpy as np
import os
import json
from tqdm import tqdm
from muq import MuQ

def create_emo_track_embeddings_no_padding():
    """
    EMO 데이터셋의 훈련 오디오 파일을 10초 세그먼트로 나누고 (패딩 없음),
    MuQ 모델의 가변 길이 시퀀스 임베딩을 그대로 저장합니다. (Windows 호환)
    """
    # --- 경로 설정 ---
    # 1. EMO 데이터셋의 최상위 폴더 경로를 지정해주세요.
    AUDIO_BASE_PATH = '/home/jay/MusicAI/MARBLE/data/EMO' # Windows 예시: r'C:\Users\user\MARBLE\EMO'

    # 2. 'EMO.train.jsonl' 파일의 경로를 지정해주세요.
    JSONL_FILE_PATH = '/home/jay/MusicAI/MARBLE/data/EMO/EMO.train.jsonl' # Windows 예시: r'C:\Users\user\MARBLE\EMO\EMO.train.jsonl'

    # 3. 생성된 임베딩 벡터를 저장할 폴더 경로를 지정해주세요.
    OUTPUT_EMBEDDINGS_PATH = 'emo_embeddings_train_60sec'
    # -----------------

    os.makedirs(OUTPUT_EMBEDDINGS_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    # MuQ 모델 불러오기
    print("MuQ 모델을 불러옵니다...")
    try:
        muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        muq = muq.to(device).eval()
        print("모델 불러오기 완료.")
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 모델 설정값
    TARGET_SR = 24000
    SEGMENT_SECONDS = 60
    SEGMENT_SAMPLES = SEGMENT_SECONDS * TARGET_SR
    MIN_SAMPLES_THRESHOLD = 2048

    # JSONL 파일 읽기
    try:
        with open(JSONL_FILE_PATH, 'r') as f:
            train_files_metadata = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"오류: '{JSONL_FILE_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 {len(train_files_metadata)}개의 오디오 파일 처리를 시작합니다.")

    for metadata in tqdm(train_files_metadata, desc="오디오 파일 처리 중"):
        relative_path = ""
        try:
            relative_path_from_json = metadata['audio_path']
            
            # === 수정 시작: EMO 경로 구조에 맞게 수정 ===
            # JSONL의 경로 "data/EMO/emomusic/..." 에서 "data/EMO/" 부분을 제거
            if relative_path_from_json.startswith('data/EMO/'):
                relative_path_from_json = relative_path_from_json[len('data/EMO/'):]
            
            # Linux 경로 구분자('/')를 현재 운영체제에 맞는 구분자(Windows에서는 '\')로 변경
            relative_path = relative_path_from_json.replace('/', os.path.sep)
            # === 수정 끝 ===
            
            audio_file_path = os.path.join(AUDIO_BASE_PATH, relative_path)

            output_subdir = os.path.join(OUTPUT_EMBEDDINGS_PATH, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            file_name_without_ext = os.path.splitext(os.path.basename(relative_path))[0]
            output_embedding_path = os.path.join(output_subdir, f"{file_name_without_ext}.pt")

            if os.path.exists(output_embedding_path):
                continue

            wav, sr = librosa.load(audio_file_path, sr=TARGET_SR, mono=True)
            
            if not np.isfinite(wav).all():
                tqdm.write(f"경고: 비정상적인 값(inf/nan)이 포함된 파일입니다 '{audio_file_path}'. 건너뜁니다.")
                continue

            num_segments = int(np.ceil(len(wav) / SEGMENT_SAMPLES))
            if num_segments == 0 and len(wav) > 0:
                num_segments = 1
            
            segment_embedding_list = []
            with torch.no_grad():
                for i in range(num_segments):
                    start = i * SEGMENT_SAMPLES
                    end = start + SEGMENT_SAMPLES
                    segment = wav[start:end]
                    
                    if len(segment) > MIN_SAMPLES_THRESHOLD:
                        tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
                        output = muq(tensor)
                        embedding = output.last_hidden_state.cpu()
                        segment_embedding_list.append(embedding.squeeze(0))
                        
                        del tensor, output
                        
                        if device == 'cuda':
                            torch.cuda.empty_cache()

            if not segment_embedding_list:
                continue

            torch.save(segment_embedding_list, output_embedding_path)

        except FileNotFoundError:
             tqdm.write(f"경고: 파일을 찾을 수 없습니다 '{audio_file_path}'. 건너뜁니다.")
        except Exception as e:
            tqdm.write(f"오류 발생: {relative_path or metadata} 처리 중. 오류: {e}")

    print("\n모든 오디오 파일의 임베딩 처리가 완료되었습니다.")
    print(f"임베딩 벡터는 '{OUTPUT_EMBEDDINGS_PATH}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    create_emo_track_embeddings_no_padding()
