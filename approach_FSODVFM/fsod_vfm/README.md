# FSOD-VFM (Few-Shot Object Detection with Vision Foundation Model)

FSOD-VFM は、Vision Foundation Model (DINOv3/CLIP) を使用した Few-Shot Object Detection システムです。

## Overview

クエリ画像から参照画像（サポートセット）に基づいて物体を検出します。プロポーザル生成に GroundingDINO、特徴抽出に DINOv3、クラス照合に CLIP を組み合わせています。

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Processing Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query Image ──┬──► GroundingDINO ──► Proposals (boxes)                │
│                │                                                       │
│                ├──► DINOv3 Vision Backbone ──► Global Features        │
│                │                                                       │
│                └──► ROI Features + Masks ──► Proposal Features        │
│                                    │                                   │
│                                    ▼                                   │
│  Refer Images ──► DINOv3 ──► Class Prototypes                          │
│                  CLIP   ──► CLIP Embeddings                            │
│                                    │                                   │
│                                    ▼                                   │
│                    Similarity Matching (cosine)                        │
│                                    │                                   │
│                                    ▼                                   │
│                    Graph Diffusion (message passing)                   │
│                                    │                                   │
│                                    ▼                                   │
│                    Multi-modal Scoring (CLIP + proto + color)          │
│                                    │                                   │
│                                    ▼                                   │
│                           Per-class NMS ──► Detections                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technical Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Vision Backbone | DINOv3-ViT-L/16 (facebook/dinov2-large) | Feature extraction |
| Proposal Generation | GroundingDINO-tiny (IDEA-Research/grounding-dino-tiny) | Generate object proposals |
| Semantic Matching | CLIP (openai/clip-vit-base-patch32) | Image-text similarity |
| Post-processing | Graph Diffusion | Feature propagation |

## Key Features

### 1. Proposal Generation
- GroundingDINO-tiny を使用して一般物体プロポザルを生成
- プロンプト: `"object . item . thing . product . toy ."`
- フォールバック: SAM2 masks → Grid proposals

### 2. Few-shot Feature Matching
参照画像からクラスプロトタイプを構築:

```python
# 参照画像から特徴を抽出して平均
mean_feat = torch.stack(support_features).mean(dim=0)
mean_feat = F.normalize(mean_feat, dim=-1)
```

ROI特徴とプロトタイプのコサイン類似度を計算。

### 3. Graph Diffusion
プロポーザル間の関係性を活用してスコアを洗練:

```
refined_scores = α × (weighted neighborhood scores) + (1-α) × initial_scores
```

- エッジ重み: バウンディングボックスのマスク重なり
- 特徴類似度: DINOv3特徴のコサイン類似度
- 30ステップの反復伝播

### 4. Multi-modal Scoring
最終スコアは複数の類似度を組み合わせ:

```python
combined_similarity = (
    0.45 × CLIP_similarity      # CLIP画像特徴照合
    + 0.20 × prototype_similarity  # DINOv3プロトタイプ
    + 0.10 × text_similarity   # GroundingDINOテキスト
    + 0.25 × color_similarity  # HSV色ヒストラム
)
```

 дополниのフィルタ:
- サイズペナルティ: 画像面積の5%~25%のみ許可
- CLIP+色ヒストグラムの閾値チェック

## Usage

```bash
python benchmark_fsod_vfm.py \
    --sample-list data/toy-91/sample_list.txt \
    --output-dir outputs/toy-91-vfm \
    --exemplar data/toy-91/exemplar.json \
    --data-root data/toy-91
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-proposals` | 500 | Maximum proposals to process |
| `--proposal-threshold` | 0.01 | Minimum proposal score |
| `--match-threshold` | 0.08 | Minimum match score |
| `--nms-threshold` | 0.45 | NMS IoU threshold |
| `--graph-diffusion-steps` | 30 | Diffusion iterations |

## Benchmark Results (toy-91)

| Metric | Value |
|--------|-------|
| Precision | 0.5515 |
| Recall | 0.7867 |
| F1 Score | 0.6484 |
| Avg Inference Time | 3.29s/image |
| GPU Memory | 3.41 GB |

### Per-class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Stuffed Bear Blue SB-1-B | 0.430 | 0.847 | 0.570 |
| Halloween Pumpkin Plastic 7-P | 0.946 | 0.761 | 0.843 |
| Pocket Tomika P060 | 0.412 | 0.745 | 0.530 |

## Comparison with GroundingDINO

| Metric | GroundingDINO | FSOD-VFM |
|--------|--------------|----------|
| Precision | 0.4811 | **0.5515** |
| Recall | **0.9668** | 0.7867 |
| F1 | 0.6425 | **0.6484** |
| Speed (s/img) | **0.535** | 3.294 |

FSOD-VFM は Precision と F1 がわずかに優れていますが、処理速度は約6倍遅いです。

## File Structure

```
fsod_vfm/
├── __init__.py
├── detector.py          # Main detector class
├── graph_diffusion.py   # Graph diffusion module
└── README.md           # This file
```

## Requirements

- PyTorch >= 2.0
- transformers
- opencv-python
- pillow
- torchvision
- DINOv3 checkpoint (optional)