# 세그먼트 분할·집계 전략에 따른 음악 임베딩 품질의 비교 분석

> **A Comparative Analysis of Music Embedding Quality Based on Segment Division and Aggregation Strategies**

## 📋 Overview

본 연구는 2025년 가천x세종 연합 학술제의 연구팀 TUNE이 진행한 연구로, [MARBLE](https://github.com/a43992899/MARBLE) 벤치마크와 [MuQ](https://github.com/tencent-ailab/MuQ) 임베딩 백본을 활용하여 음악 임베딩 생성 시 세그먼트 분할 및 집계 전략이 품질과 효율성에 미치는 영향을 체계적으로 분석한 연구입니다.

최신 음악 임베딩 모델들은 트랜스포머 아키텍처를 통해 높은 성능을 달성했지만, 긴 오디오 시퀀스에 대한 **O(N²) 계산 복잡도 문제**로 실용성에 한계가 있습니다. 본 연구는 이를 해결하기 위한 다양한 세그먼트 분할·집계 전략을 정량적으로 비교하여, **성능-효율성 트레이드오프** 관계를 규명합니다.

### 🎯 Research Objectives

1. **세그먼트 길이 영향 분석**: 10초, 30초, 60초 분할이 임베딩 품질에 미치는 영향
2. **집계 방법론 평가**: Concatenate, Mean Pooling, Transformer 기반 집계의 성능-효율성 비교
3. **실용적 가이드라인 제시**: 제한된 계산 자원 환경에서의 최적 전략 도출

## 🏗️ Architecture & Models

### Base Models
- **MuQ (Music Understanding with Quantization)**: 멜 스펙트럼 기반 Residual Vector Quantization을 활용한 자가지도 음악 표현 학습 모델
- **MARBLE**: 음악 표현 학습의 범용 평가를 위한 표준화된 벤치마크

### Experimental Framework

| 세그먼트 길이 | 집계 전략 | 설명 | 특징 |
|-------------|----------|------|------|
| **10초** | Concatenate | 세그먼트 벡터 순차 연결 | 차원 증가, 높은 메모리 요구 |
| **10초** | Mean Pooling | 세그먼트 임베딩 평균 | 효율적, 안정적 성능 |
| **10초** | Transformer | Self-attention 기반 집계 | 관계 모델링, 제한된 사전학습 |
| **30초** | (위 3가지 집계 방식 동일 적용) | 중간 길이 컨텍스트 | 균형잡힌 정보량 |
| **60초** | (위 3가지 집계 방식 동일 적용) | 긴 컨텍스트 | 구조적 정보 포함 |
| **전체 트랙** | - | 분할 없이 직접 처리 | 베이스라인 (이상적 성능) |

## 📁 Project Structure

```
TUNE/
├── configs/                    # Lightning CLI 설정 파일들
│   ├── probe.EMO.MuQMax10sec.yaml      # Max pooling 기반 10초 세그먼트
│   ├── probe.EMO.MuQMean10sec.yaml     # Mean pooling 기반 10초 세그먼트  
│   ├── probe.EMO.MuQSegment.yaml       # 기본 세그먼트 처리
│   ├── probe.EMO.Transformer10sec.yaml # 트랜스포머 집계 10초
│   ├── probe.EMO.Transformer30sec.yaml # 트랜스포머 집계 30초
│   └── probe.EMO.Transformer60sec.yaml # 트랜스포머 집계 60초
├── marble/
│   └── encoders/               # 커스텀 인코더 구현
│       ├── muq_max_10sec_encoder.py
│       ├── muq_mean_10sec_encoder.py
│       ├── muq_segment_encoder.py
│       ├── transformer_segment_encoder.py          # 60초 세그먼트
│       ├── transformer_segment_encoder_10sec.py
│       └── transformer_segment_encoder_30sec.py
└── data/
    ├── EMO/                    # 감정 인식 데이터셋
    │   ├── EMO.train.jsonl
    │   ├── EMO.val.jsonl
    │   └── EMO.test.jsonl
    └── GTZAN/                  # 장르 분류 데이터셋
```

## 🔬 Methodology

### 1. 세그먼트 생성 (Segmentation Strategy)

전체 음악 트랙 X (길이 T)를 고정 길이 L ∈ {10, 30, 60}초 단위로 분할:

```
X → {x₁, x₂, ..., xₙ}, where N = ⌈T/L⌉
각 세그먼트: zᵢ = f_θ(xᵢ) ∈ ℝᵈ
```

### 2. 집계 전략 (Aggregation Strategies)

#### **Concatenation**
```
z = [z₁ | z₂ | ... | zₙ] ∈ ℝ^(Nd)
```

#### **Mean Pooling**  
```
z = (1/N) Σᵢ₌₁ⁿ zᵢ
```

#### **Transformer-based Aggregation**
```
z = TransformerAgg(Z)
```
- 4층 인코더, self-attention 메커니즘
- MLM(Masked Language Modeling) 사전학습 (1 epoch)
- [CLS] 토큰 기반 최종 집계

### 3. 백본 고정 (Frozen Backbone)

- MuQ 모델 파라미터 완전 고정
- 다운스트림 태스크용 경량 예측 헤드만 학습
- 성능 차이가 순수하게 분할·집계 전략에서 기인하도록 통제

## 📊 Experimental Results

### MARBLE 벤치마크 성능 결과

#### EMO Dataset (Emotion Recognition)
| 방식 | 세그먼트 길이 | Peak VRAM (GB) | 추론 시간 (sec) | R²(Valence) | R²(Arousal) |
|------|-------------|---------------|----------------|-------------|-------------|
| **전체 트랙** | - | **6.816** | **45.845** | **0.574** | **0.737** |
| Concatenate | 10초 | 9.730 | 136.986 | 0.547 | 0.760 |
| **Mean Pooling** | **10초** | **3.552** | **46.210** | **0.547** | **0.760** |
| Transformer | 10초 | 4.095 | 60.886 | 0.492 | 0.707 |
| Mean Pooling | 30초 | 4.124 | 62.352 | 0.542 | 0.756 |
| **Transformer** | **60초** | **4.132** | **45.707** | 0.506 | 0.726 |

#### GTZAN Dataset (Genre Classification)  
| 방식 | 세그먼트 길이 | Peak VRAM (GB) | 추론 시간 (sec) | Accuracy |
|------|-------------|---------------|----------------|----------|
| **전체 트랙** | - | **6.546** | **60.678** | **0.845** |
| Concatenate | 10초 | 9.361 | 181.427 | 0.828 |
| **Mean Pooling** | **10초** | **3.404** | **61.285** | **0.830** |
| Transformer | 10초 | 3.932 | 80.702 | 0.795 |
| Mean Pooling | 30초 | 3.928 | 82.522 | 0.835 |
| **Transformer** | **60초** | **3.968** | **55.980** | 0.805 |

## 🔍 Key Findings

### 1. **최적 균형점: Mean Pooling + 10~30초 세그먼트**
- **VRAM 절약**: 전체 트랙 대비 약 48% 메모리 사용량 감소
- **성능 보존**: 베이스라인 성능의 98% 이상 유지
- **실용성**: 제한된 자원 환경에서 가장 효율적

### 2. **Concatenate 방식의 비효율성**
- 모든 조건에서 가장 높은 메모리 사용량 (최대 12.5GB)
- 추론 시간 최대 380% 증가
- 성능 향상 없이 계산 비용만 급증

### 3. **Transformer 집계의 특수성**
- 제한된 사전학습(1 epoch)으로 성능 저조
- **60초 세그먼트에서 가장 빠른 추론 속도** (55.980초)
- 적은 수의 긴 세그먼트 처리에 특화

### 4. **세그먼트 길이 영향**
- 긴 세그먼트일수록 성능 소폭 향상 (더 넓은 시간적 문맥)
- 메모리 사용량과 추론 시간은 길이에 비례하여 증가

## ⚙️ Experimental Setup

### Hardware & Software
```bash
- GPU: NVIDIA RTX 2080 Ti
- Framework: PyTorch Lightning
- Random Seed: 1234 (재현성 보장)
```

### Datasets
- **GTZAN**: 1,000개 트랙 (30초), 10개 장르, 장르 분류
- **EMO**: 744곡 (45초), Valence-Arousal 연속값, 감정 인식

### Evaluation Metrics
- **성능**: Accuracy (분류), R² Score (회귀)
- **효율성**: Peak VRAM, 추론 시간
- **Trade-off**: 정확도-효율성 균형 분석

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch lightning torchmetrics librosa numpy psutil
# MuQ 모델 설치 필요
```

### Running Experiments

1. **Mean Pooling (권장)**
```bash
python -m lightning.pytorch.cli fit --config configs/probe.EMO.MuQMean10sec.yaml
```

2. **Transformer 기반**
```bash  
python -m lightning.pytorch.cli fit --config configs/probe.EMO.Transformer60sec.yaml
```

3. **성능 테스트**
```bash
python -m lightning.pytorch.cli test --config [CONFIG] --ckpt_path [CHECKPOINT]
```

## 📋 Practical Guidelines

본 연구 결과를 바탕으로 한 실용적 가이드라인:

### 🎯 **최고의 균형점 (권장)**
- **전략**: 10~30초 세그먼트 + Mean Pooling
- **적용**: 자원이 제한된 대부분의 환경
- **효과**: 메모리 48% 절약, 성능 98% 보존

### 🏆 **성능 최우선**  
- **전략**: 전체 트랙 직접 처리
- **적용**: 충분한 계산 자원 보장 시
- **효과**: 이론적 최고 성능

### ⚡ **속도 최우선**
- **전략**: 60초 세그먼트 + Transformer 집계  
- **적용**: 성능 저하 감수 가능한 실시간 환경
- **효과**: 가장 빠른 추론 속도

## 🔮 Future Work

1. **트랜스포머 사전학습 확장**: 1 epoch → 충분한 학습으로 성능 개선 가능성
2. **적응형 분할**: 음악적 구조(벌스-코러스) 기반 의미 단위 분할
3. **다양한 다운스트림 태스크**: MARBLE 전체 벤치마크로 확장 평가
4. **계층적 집계**: 다중 스케일 시간 정보 통합 전략

## 📚 References

본 연구는 다음 주요 연구들을 기반으로 합니다:

- **MuQ**: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization
- **MARBLE**: Music Audio Representation Benchmark for Universal Evaluation  
- **GTZAN**: Musical genre classification of audio signals
- **EMO**: 1000 songs for emotional analysis of music

## 👥 Contributors

- **공유빈** (가천대학교 인공지능학과)
- **윤재하** (세종대학교 인공지능학과)

## 📄 License

This project builds upon MARBLE and MuQ frameworks. Please refer to their respective licenses for usage guidelines.

---

**Keywords**: Music Information Retrieval, Segment Aggregation, Efficiency-Performance Trade-off, Audio Embeddings, Transformer
