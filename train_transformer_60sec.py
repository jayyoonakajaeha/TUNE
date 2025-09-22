import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
import time
from tqdm import tqdm

# 5개 데이터셋 경로
DATASET_PATHS = [
    "/home/jay/MusicAI/mtg_embeddings_train_60sec",
    "/home/jay/MusicAI/mtt_embeddings_train_60sec", 
    "/home/jay/MusicAI/gtzan_embeddings_train_60sec",
    "/home/jay/MusicAI/gs_embeddings_train_60sec",
    "/home/jay/MusicAI/emo_embeddings_train_60sec"
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SegmentDataset(Dataset):
    def __init__(self, dataset_paths, mask_prob=0.15):
        self.tracks = []
        self.mask_prob = mask_prob
        
        # 모든 데이터셋에서 .pt 파일 수집
        for dataset_path in dataset_paths:
            pt_files = glob.glob(os.path.join(dataset_path, "**/*.pt"), recursive=True)
            self.tracks.extend(pt_files)
        
        print(f"총 {len(self.tracks)}개의 트랙 파일을 찾았습니다.")
        
        # 임베딩 차원과 최대 시퀀스 길이 확인
        self.emb_dim = None
        self.max_seq_len = 0
        
        print("데이터셋 통계를 수집하는 중...")
        for i, track_path in enumerate(self.tracks[:100]):  # 샘플링해서 확인
            try:
                segment_list = torch.load(track_path, map_location='cpu')
                
                if isinstance(segment_list, list) and len(segment_list) > 0:
                    # 첫 번째 세그먼트에서 임베딩 차원 확인
                    first_segment = segment_list[0]
                    if isinstance(first_segment, torch.Tensor) and first_segment.dim() == 2:
                        frame_len, emb_dim = first_segment.shape
                        
                        # 전체 시퀀스 길이 계산 (세그먼트 수 * 프레임 수)
                        total_seq_len = len(segment_list) * frame_len
                        
                        if self.emb_dim is None:
                            self.emb_dim = emb_dim
                        self.max_seq_len = max(self.max_seq_len, total_seq_len)
                        
                        if i % 20 == 0:
                            print(f"  파일 {i+1}: {len(segment_list)}개 세그먼트, 총 시퀀스 길이 {total_seq_len}, 임베딩 차원 {emb_dim}")
                        
            except Exception as e:
                if i % 20 == 0:
                    print(f"파일 로드 실패 {track_path}: {e}")
                continue
                
        print(f"임베딩 차원: {self.emb_dim}")
        print(f"최대 시퀀스 길이 (샘플에서): {self.max_seq_len}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]
        
        try:
            # .pt 파일에서 세그먼트 임베딩 로드 (리스트 형태)
            segment_list = torch.load(track_path, map_location='cpu')
            
            if not isinstance(segment_list, list):
                raise ValueError(f"Expected list, got {type(segment_list)}")
            
            if len(segment_list) == 0:
                raise ValueError("Empty segment list")
            
            # 각 세그먼트는 (250, 1024) 형태
            # 모든 세그먼트를 concatenate하여 하나의 긴 시퀀스로 만들기
            # 즉, (num_segments * 250, 1024) 형태가 됨
            segments = torch.cat(segment_list, dim=0)  # (total_frames, 1024)
            
            seq_len, emb_dim = segments.shape
            
            # 마스킹을 위한 랜덤 마스크 생성
            mask = torch.rand(seq_len) < self.mask_prob
            
            # 입력과 라벨 준비
            inp = segments.clone()
            lbl = segments.clone()
            
            # 벡터 단위 마스킹 (마스크된 부분은 0으로)
            inp[mask] = 0.0
            # 마스크되지 않은 부분은 라벨에서 -100으로 (loss 계산에서 제외)
            lbl[~mask] = -100.0
            
            return inp, lbl
            
        except Exception as e:
            print(f"파일 로드 실패 {track_path}: {e}")
            # 에러 시 더미 데이터 반환
            dummy_len = 250  # 기본 세그먼트 길이
            dummy_emb_dim = self.emb_dim if self.emb_dim is not None else 1024
            dummy_inp = torch.zeros((dummy_len, dummy_emb_dim))
            dummy_lbl = torch.zeros((dummy_len, dummy_emb_dim)) - 100.0
            return dummy_inp, dummy_lbl

def collate_batch(batch):
    """서로 다른 길이의 시퀀스를 패딩 - 제한 없음"""
    inps, lbls = zip(*batch)
    
    # 배치에서 가장 긴 시퀀스 길이 찾기
    max_len = max(inp.size(0) for inp in inps)
    emb_dim = inps[0].size(1)
    
    print(f"배치 내 최대 시퀀스 길이: {max_len}")
    
    padded_inps = []
    padded_lbls = []
    attention_masks = []
    
    for inp, lbl in zip(inps, lbls):
        seq_len = inp.size(0)
        pad_len = max_len - seq_len
        
        # attention mask (실제 토큰은 1, 패딩은 0)
        attention_mask = torch.ones(seq_len)
        
        if pad_len > 0:
            # 입력 패딩 (0으로)
            pad_inp = torch.zeros((pad_len, emb_dim), dtype=inp.dtype)
            inp = torch.cat([inp, pad_inp], dim=0)
            
            # 라벨 패딩 (-100으로, loss 계산에서 제외)
            pad_lbl = torch.zeros((pad_len, emb_dim), dtype=lbl.dtype) - 100.0
            lbl = torch.cat([lbl, pad_lbl], dim=0)
            
            # attention mask 패딩 (0으로)
            pad_mask = torch.zeros(pad_len)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
            
        padded_inps.append(inp)
        padded_lbls.append(lbl)
        attention_masks.append(attention_mask)
    
    padded_inps = torch.stack(padded_inps, dim=0)
    padded_lbls = torch.stack(padded_lbls, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    
    return padded_inps, padded_lbls, attention_masks

class SegmentTransformerMLM(nn.Module):
    def __init__(self, emb_dim, heads=8, hidden=1024, layers=4, max_len=100000, dropout=0.1):
        super().__init__()
        
        # 매우 큰 max_len을 지원하는 learnable positional embedding
        self.pos = nn.Embedding(max_len, emb_dim)
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(emb_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=heads, 
            dim_feedforward=hidden, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # MLM head
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim)
        )
        
        self.max_len = max_len
        
    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, emb_dim)
        b, l, d = x.shape
        
        if l > self.max_len:
            print(f"Warning: 시퀀스 길이 {l}이 max_len {self.max_len}을 초과합니다. 잘라냅니다.")
            x = x[:, :self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_len]
            l = self.max_len
        
        # Positional embedding 추가
        pos_idx = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        x = self.input_norm(x + self.pos(pos_idx))
        
        # Attention mask 처리 (패딩된 부분은 무시)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True for padded positions
        
        # Transformer encoding
        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # MLM prediction
        logits = self.head(y)
        
        # Track-level embedding (attention mask를 고려한 평균)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(y)
            track_emb = (y * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            track_emb = y.mean(dim=1)
        
        return logits, track_emb

def train_epoch(model, loader, optimizer, scaler, accum_steps=4):
    model.train()
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0
    optimizer.zero_grad()
    accumulated_steps = 0
    
    # autocast import
    try:
        from torch.amp import autocast
    except ImportError:
        from torch.cuda.amp import autocast
    
    start_time = time.time()
    
    # tqdm으로 진행률 표시
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch_data in pbar:
        if len(batch_data) == 3:
            inp, lbl, attention_mask = batch_data
            inp, lbl, attention_mask = inp.to(DEVICE), lbl.to(DEVICE), attention_mask.to(DEVICE)
        else:
            inp, lbl = batch_data
            inp, lbl = inp.to(DEVICE), lbl.to(DEVICE)
            attention_mask = None
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            logits, _ = model(inp, attention_mask)
            loss_matrix = criterion(logits, lbl)
            
            # -100인 부분은 loss 계산에서 제외
            mask = (lbl != -100.0)
            loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1)
        
        scaler.scale(loss / accum_steps).backward()
        accumulated_steps += 1
        
        if accumulated_steps == accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulated_steps = 0
        
        total_loss += loss.item()
        
        # 진행률 업데이트
        current_loss = total_loss / (step + 1)
        pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Seq_Len': inp.shape[1]})
        
        # 예상 시간 계산 (매 100스텝마다)
        if step > 0 and step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            remaining_steps = len(loader) - step
            eta_seconds = remaining_steps / steps_per_sec
            eta_minutes = eta_seconds / 60
            pbar.set_description(f"Training (ETA: {eta_minutes:.1f}min)")
        
        torch.cuda.empty_cache()
    
    # 남은 accumulated gradients 처리
    if accumulated_steps > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    pbar.close()
    return total_loss / len(loader)

def main():
    print("데이터셋 로딩 중...")
    start_time = time.time()
    
    dataset = SegmentDataset(DATASET_PATHS)
    
    if dataset.emb_dim is None:
        print("임베딩 차원을 확인할 수 없습니다. 첫 번째 파일을 직접 확인해보겠습니다.")
        first_segments = torch.load(dataset.tracks[0], map_location='cpu')
        if isinstance(first_segments, list) and len(first_segments) > 0:
            dataset.emb_dim = first_segments[0].shape[-1]
        else:
            dataset.emb_dim = 1024
        print(f"임베딩 차원을 {dataset.emb_dim}로 설정했습니다.")
    
    # 제한 없는 max_len 설정
    MAX_SEQ_LEN = 100000  # 매우 큰 값으로 설정, 실질적으로 제한 없음
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # 시퀀스가 매우 길 수 있으므로 1로 설정
        shuffle=True,
        num_workers=2,  # 메모리 절약을 위해 줄임
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    print(f"모델 초기화 중... (emb_dim: {dataset.emb_dim}, max_len: {MAX_SEQ_LEN})")
    model = SegmentTransformerMLM(
        emb_dim=dataset.emb_dim,
        heads=8,
        hidden=1024,
        layers=4,
        max_len=MAX_SEQ_LEN,
        dropout=0.1
    ).to(DEVICE)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = GradScaler()
    
    # 전체 학습 시간 예상
    setup_time = time.time() - start_time
    print(f"설정 완료 시간: {setup_time:.1f}초")
    print(f"총 {len(dataset)}개 샘플, {len(loader)}개 배치")
    
    # 첫 번째 배치로 시간 측정
    print("\n첫 번째 배치로 시간 측정 중...")
    first_batch_start = time.time()
    first_batch = next(iter(loader))
    first_batch_load_time = time.time() - first_batch_start
    
    # 모델 forward 시간 측정
    inp, lbl, attention_mask = first_batch
    inp, lbl, attention_mask = inp.to(DEVICE), lbl.to(DEVICE), attention_mask.to(DEVICE)
    
    forward_start = time.time()
    with torch.no_grad():
        logits, _ = model(inp, attention_mask)
    forward_time = time.time() - forward_start
    
    print(f"첫 번째 배치 로드 시간: {first_batch_load_time:.2f}초")
    print(f"첫 번째 배치 forward 시간: {forward_time:.2f}초 (시퀀스 길이: {inp.shape[1]})")
    
    # 전체 학습 시간 예상
    estimated_time_per_batch = first_batch_load_time + forward_time * 2  # forward + backward
    estimated_epoch_time = estimated_time_per_batch * len(loader)
    estimated_total_time = estimated_epoch_time * 50  # 50 epochs
    
    print(f"\n=== 예상 학습 시간 ===")
    print(f"배치당 예상 시간: {estimated_time_per_batch:.2f}초")
    print(f"에포크당 예상 시간: {estimated_epoch_time/60:.1f}분")
    print(f"전체 학습 예상 시간: {estimated_total_time/3600:.1f}시간")
    print("=" * 30)
    
    print("\n학습 시작!")
    training_start_time = time.time()
    
    for epoch in range(1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/50")
        loss = train_epoch(model, loader, optimizer, scaler, accum_steps=16)  # 더 큰 accumulation
        epoch_time = time.time() - epoch_start
        
        # 남은 시간 계산
        elapsed_total = time.time() - training_start_time
        avg_epoch_time = elapsed_total / (epoch + 1)
        remaining_epochs = 50 - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_hours = eta_seconds / 3600
        
        print(f"Epoch {epoch+1}, Average Loss: {loss:.4f}, 시간: {epoch_time/60:.1f}분")
        print(f"남은 예상 시간: {eta_hours:.1f}시간")
        
        # 주기적으로 모델 저장
        if (epoch + 1) % 10 == 0:
            save_path = f"segment_transformer_mlm_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'emb_dim': dataset.emb_dim,
                'max_len': MAX_SEQ_LEN
            }, save_path)
            print(f"모델이 '{save_path}'에 저장되었습니다.")
    
    # 최종 모델 저장
    final_save_path = "transformer_60sec_mlm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'emb_dim': dataset.emb_dim,
        'max_len': MAX_SEQ_LEN
    }, final_save_path)
    
    total_time = time.time() - training_start_time
    print(f"학습 완료! 총 소요 시간: {total_time/3600:.1f}시간")
    print(f"최종 모델이 '{final_save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    main()