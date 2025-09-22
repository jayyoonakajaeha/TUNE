import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random

# --- 설정 (Configuration) ---
# 이전 단계에서 생성된 세그먼트 임베딩이 저장된 디렉토리
EMB_DIR = "gtzan_embeddings_train_10sec"
# 학습된 모델이 저장될 경로
MODEL_SAVE_PATH = "track_encoder_10sec.pth"

# 학습 하이퍼파라미터
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16  # GPU 메모리에 따라 조절
LEARNING_RATE = 1e-4
MASK_PROB = 0.15  # 전체 토큰(세그먼트) 중 마스킹할 비율
ACCUM_STEPS = 4   # Gradient Accumulation Steps

# --- 데이터셋 (Dataset) ---
class TrackSegmentDataset(Dataset):
    """
    트랙별 세그먼트 임베딩(.pt 파일)을 로드하고 MLM을 위한 마스킹을 적용하는 데이터셋
    """
    def __init__(self, emb_dir, mask_prob=0.15):
        # emb_dir 내의 모든 .pt 파일을 재귀적으로 찾음
        self.track_paths = glob.glob(os.path.join(emb_dir, '**', '*.pt'), recursive=True)
        self.mask_prob = mask_prob
        # MuQ 임베딩의 평균을 내기 위한 차원
        self.muq_seq_dim = 1
        print(f"총 {len(self.track_paths)}개의 트랙 파일을 찾았습니다.")

    def __len__(self):
        return len(self.track_paths)

    def __getitem__(self, idx):
        path = self.track_paths[idx]
        # 임베딩 로드: (num_segments, muq_seq_len, emb_dim)
        segments = torch.load(path, map_location='cpu').float()
        
        # 각 세그먼트의 임베딩을 평균내어 하나의 벡터로 만듦
        # (num_segments, muq_seq_len, emb_dim) -> (num_segments, emb_dim)
        seq = segments.mean(dim=self.muq_seq_dim)

        # MLM을 위한 입력(inp)과 레이블(lbl) 생성
        inp = seq.clone()
        lbl = seq.clone()
        
        # 마스킹할 인덱스 결정
        mask_indices = torch.rand(seq.size(0)) < self.mask_prob
        
        # 마스킹된 위치의 입력은 0으로 설정
        inp[mask_indices] = 0.0
        # 마스킹되지 않은 위치의 레이블은 -100으로 설정하여 손실 계산에서 제외
        lbl[~mask_indices] = -100.0
        
        return inp, lbl

# --- 배치 처리를 위한 Collate Function ---
def collate_fn(batch):
    """
    서로 다른 길이의 시퀀스를 패딩하여 하나의 배치로 만듦
    """
    inps, lbls = zip(*batch)
    max_len = max(x.size(0) for x in inps)
    emb_dim = inps[0].size(1)

    padded_inps = torch.zeros((len(inps), max_len, emb_dim), dtype=torch.float32)
    padded_lbls = torch.full((len(lbls), max_len, emb_dim), -100.0, dtype=torch.float32)

    for i, (inp, lbl) in enumerate(zip(inps, lbls)):
        seq_len = inp.size(0)
        padded_inps[i, :seq_len] = inp
        padded_lbls[i, :seq_len] = lbl
        
    return padded_inps, padded_lbls

# --- 트랜스포머 모델 (Transformer Model) ---
class TrackTransformerMLM(nn.Module):
    """
    세그먼트 시퀀스로부터 트랙 임베딩을 학습하는 트랜스포머 인코더
    """
    def __init__(self, emb_dim, heads=4, hidden=512, layers=4, max_len=64, dropout=0.1):
        super().__init__()
        # 학습 가능한 포지셔널 임베딩
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        # 트랜스포머 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=heads, 
            dim_feedforward=hidden, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # 마스킹된 토큰을 예측하기 위한 최종 선형 레이어 (Head)
        self.mlm_head = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, emb_dim)
        b, seq_len, emb_dim = x.shape
        
        # 포지션 인덱스 생성 및 포지셔널 임베딩 추가
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, seq_len)
        x = x + self.pos_embedding(pos_ids)
        
        # 트랜스포머 인코더 통과
        encoded_output = self.encoder(x)
        
        # MLM 예측 결과
        predicted_segments = self.mlm_head(encoded_output)
        
        # 트랙 임베딩 (시퀀스 차원을 평균)
        track_embedding = encoded_output.mean(dim=1)
        
        return predicted_segments, track_embedding

# --- 학습 함수 (Training Function) ---
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for i, (inp, lbl) in enumerate(progress_bar):
        inp, lbl = inp.to(DEVICE), lbl.to(DEVICE)
        
        with autocast(): # Mixed-precision training
            logits, _ = model(inp)
            
            # 레이블이 -100이 아닌, 즉 마스킹된 위치에 대해서만 손실 계산
            mask = (lbl != -100.0)
            loss_matrix = criterion(logits, lbl)
            loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)
        
        # Gradient Accumulation
        scaler.scale(loss / ACCUM_STEPS).backward()
        
        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print(f"사용 장치: {DEVICE}")

    # 데이터셋 및 데이터로더 준비
    dataset = TrackSegmentDataset(EMB_DIR, mask_prob=MASK_PROB)
    
    if len(dataset) == 0:
        print(f"오류: '{EMB_DIR}' 디렉토리에서 임베딩 파일을 찾을 수 없습니다.")
        print("이전 단계의 임베딩 생성 코드가 올바르게 실행되었는지 확인해주세요.")
    else:
        # 임베딩 차원 확인 (샘플 파일 로드)
        sample_tensor = torch.load(dataset.track_paths[0]).mean(dim=1)
        emb_dimension = sample_tensor.shape[-1]
        print(f"감지된 임베딩 차원: {emb_dimension}")

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2, # 시스템에 맞게 조절
            collate_fn=collate_fn,
            pin_memory=True
        )

        # 모델, 옵티마이저, 손실 함수, 스케일러 초기화
        model = TrackTransformerMLM(emb_dim=emb_dimension).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss(reduction="none") # 각 원소별 손실을 계산하기 위해 none으로 설정
        scaler = GradScaler()

        # 학습 시작
        print("\n모델 학습을 시작합니다...")
        for epoch in range(EPOCHS):
            avg_loss = train_one_epoch(model, loader, optimizer, scaler, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

        # 학습된 모델 저장
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\n학습 완료! 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")
