# Training-free Few-shot Object Detector

クエリ画像から参照画像（サポートセット）に基づいて物体を検出します。

## 概要

数千クラスの商品を登録し、週数十クラスの入れ替えが発生する環境下で、Few-Shot Object Detection (FSOD) を Training-Free で実現するための手法選定。

| 要件 | 内容 |
|---|---|
| 登録クラス数 | 数千クラス〜数万クラス |
| 新規クラス頻度 | 週数十クラスの追加・削除 |
| 参照画像（サポートセット） | クラス毎1〜数枚 |
| 訓練方式 | **Training-Free（必須）** |
| 画像内対象 | 限定的なサブセットのみを検出対象として指定 |
| GPU環境 | Tesla T4 × 1〜2枚（16GB VRAM） |
| 優先度 | **精度 > スピード** |
| 検出対象 | おもちゃ、フィギュア、食品、ボトル飲料、日常用品など多種多様な物体 |
| 出力 | Instance数え上げのみ（pixel単位のmask不要） |

Two approaches implemented.

## Approach 1: GroundingDINO

### Technical Stack

The implementation uses:
- GroundingDINO for generic region proposals
- CLIP for visual matching between proposals and reference images
- optional text scoring from class names

### Benchmark Results (toy-91)

| Metric | Value |
|--------|-------|
| Precision | 0.982 |
| Recall | 0.7773 |
| F1 Score | 0.8677 |
| Avg Inference Time | 0.52s/image |
| GPU Memory | 3.08 GB |

### Note

- More training images aren't necessarily better; 
- Difficult examples may not be as effective as expected;

## Approach 2: FSOD-VFM

Main idea come from paper `FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion` (arXiv:2602.03137) (ICLR 2026)
Some changes added.

### Technical Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Vision Backbone | DINOv3-ViT-L/16 (facebook/dinov2-large) | Feature extraction |
| Proposal Generation | GroundingDINO-tiny (IDEA-Research/grounding-dino-tiny) | Generate object proposals |
| Semantic Matching | CLIP (openai/clip-vit-base-patch32) | Image-text similarity |
| Post-processing | Graph Diffusion | Feature propagation |

### Benchmark Results (toy-91)

| Metric | Value |
|--------|-------|
| Precision | 0.5515 |
| Recall | 0.7867 |
| F1 Score | 0.6484 |
| Avg Inference Time | 3.29s/image |
| GPU Memory | 3.41 GB |

### Note

- More training images aren't necessarily better; 
- Difficult examples may not be as effective as expected;
- SAM2 did not contribute as much to accuracy as expected, improving the F1 score by only +0.01; conversely, inference speed is more than five times slower.
