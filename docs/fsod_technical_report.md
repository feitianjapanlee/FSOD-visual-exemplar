# Few-Shot Object Detection 技術選定レポート

## 概要

数千クラスの商品を登録し、週数十クラスの入れ替えが発生する環境下で、Few-Shot Object Detection (FSOD) を Training-Free で実現するための手法選定。

---

## 要件整理

| 要件 | 内容 |
|---|---|
| 登録クラス数 | 数千万〜数千クラス |
| 新規クラス頻度 | 週数十クラスの追加・削除 |
| 訓練方式 | **Training-Free（必須）** |
| 画像内対象 | 限定的なサブセットのみを検出対象として指定 |
| GPU環境 | Tesla T4 × 1〜2枚（16GB VRAM） |
| 優先度 | **精度 > スピード** |
| 検出対象 | おもちゃ、フィギュア、食品、ボトル飲料、日常用品など多種多様な物体 |
| 出力 | Instance数え上げのみ（pixel単位のmask不要） |

---

## 選定手法: FSOD-VFM

### 手法概要

FSOD-VFM (Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion) は、VLM/LLM を用いた訓練を一切行わず、事前学習済み Vision Foundation Models (VFM) を組み合わせることで Few-Shot 物体検出を実現する手法。

**ICLR 2026 採録予定。**

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  Input: 画像 + 検出対象クラスリスト ["toy_A", "figure_B"] │
│                                                         │
│  Step 1: Support Features 取得                          │
│    - 検出対象クラスのprototype features (DINOv3-L)       │
│    - SAM2 masks for ROI pooling                         │
│                                                         │
│  Step 2: Query Proposal 生成 (UPN)                       │
│    - 画像からカテゴリ非依存のBBoxを最大500個              │
│    - スコア > 0.01 のproposalを保持                      │
│                                                         │
│  Step 3: RoI Feature Extraction (SAM2 + DINOv3)         │
│    - 各proposal regionからfeatures抽出                   │
│    - Cosine similarity で検出対象クラスと照合             │
│                                                         │
│  Step 4: Graph Diffusion                                 │
│    - 同一クラス内のproposal間で信頼度を伝播               │
│    - 断片的なBBoxを抑制、完全な物体を重視                  │
│                                                         │
│  Step 5: 出力                                            │
│    - 各クラスのBBox + 信頼度スコア                        │
│    - Top-100 proposals / 画像                            │
└─────────────────────────────────────────────────────────┘
```

### 採用コンポーネント

| コンポーネント | 選定モデル | 役割 |
|---|---|---|
| **Vision Backbone** | **DINOv3-L** | 画像全体・ROI領域双方から高次元特徴量を抽出 |
| **Mask Extraction** | **SAM2-L** | 物体領域の精密なmask取得、RoI feature poolingとGraph Diffusionのエッジ重み計算 |

### 採用理由

#### DINOv3-L 採用理由

DINOv3 は Meta が2025年8月に 발표한 自己教師ありVision Transformer。先行モデルの DINOv2-L から以下の改善がある:

- **Gram Anchoring**: 大規模学習時の特徴品質劣化を防止
- **ドメイン汎化**: COCO-O AP で DINOv2-L 比 **+33%** (31.9 → 42.5)
- **特徴量の次元**: 1024次元（DINOv2-Lと同一）
- **後方互換性**: FSOD-VFM との特徴抽出インターフェースが完全互換

#### SAM2-L 採用理由

SAM2 は FSOD-VFM において以下の2箇所で重要な役割を担う:

1. **RoI Feature Extraction (Equation 1)**

```
F_s = (1/N_mask) * Σ F_img[:, u,v] * M_down[u,v]
```

SAM2 の mask を使って物体領域のみから features を抽出する。背景を除外した、より精密な prototype の構築が可能になる。

2. **Graph Diffusion のエッジ重み (Equation 3)**

```
E_{i,j} = Area(M^i ∩ M^j) / Area(M^i)
```

Mask overlap で「同一物体内の proposal 同士」を判定する。BBox IoU 相比べて物体の境界を正確に把握できる。

#### SAM2 を省略できない理由

Pixel 単位の mask が不要であっても、SAM2 は以下の2つの重要な処理に使われる:

- RoI pooling の精度（Mask-weighted pooling vs 単純なBBox roi pooling）
- Graph Diffusion の精度（Mask overlap vs BBox IoU）

Mask による境界推定の方が、物体の断片化抑制（over-fragmentation）に有効。多種多様な商品形状（玩具・フィギュア等）において、この精度向上が検出性能に直結する。

SAM2-L は VRAM ~4GB で動作するため、Tesla T4 (16GB) でも DINOv3-L と共存可能。

---

## ベンチマーク性能

FSOD-VFM (DINOv2-L + SAM2-L) の既知ベンチマーク:

| Dataset | 設定 | FSOD-VFM | Training-based Best (NIFF) | No-Time-To-Train |
|---|---|---|---|---|
| Pascal-5i (1-shot) | nAP50 | **77.5** | 62.8 | 70.8 |
| Pascal-5i (10-shot) | nAP50 | **85.8** | 70.3 | 79.1 |
| COCO-20i (10-shot) | nAP50 | **59.4** | 34.1 | 54.1 |
| CD-FSOD (10-shot) | nAP | **31.6** | 19.0 | 21.4 |

DINOv3-L への置換により、更なる精度向上が見込める（COCO-O robustness +33%）。

---

## 処理時間の試算 (Tesla T4)

| ステップ | 推定時間 |
|---|---|
| DINOv3-L feature extraction (query画像, 624×624) | ~0.8s |
| UPN proposal generation | ~0.3s |
| SAM2-L mask extraction (max 500 proposals) | ~1.0s |
| DINOv3-L RoI pooling (max 500 proposals) | ~0.3s |
| Cosine matching (Nクラス × 500 proposals) | ~0.01s |
| Graph diffusion (30 steps) | ~0.01s |
| **合計（2クラス検出時）** | **~2.4s / 画像** |

検出対象クラス数が多い場合は比例して高速化される。精度最優先の要件であれば許容範囲。

---

## 数千クラス環境への対応

### クラス登録の動的管理

```
support_set = {
    "figure_A": {"features": [...], "image_refs": [...]},
    "toy_B":    {"features": [...], "image_refs": [...]},
    ...
}
```

- DINOv3-L features は一度抽出すれば再利用可能
- クラス削除: Dict から除去
- クラス追加: 画像から features を抽出して追加
- 数千クラスでの抽出も数百ms/クラス程度

### 検出対象を限定する構成

入力画像に対して全数千クラスをスキャンするのではなく、**検出対象を限定**することで処理効率と精度を両立:

```
Input: {
    "image": "path/to/image.jpg",
    "target_classes": ["figure_A", "toy_B"]  # この2クラスのみを検出
}
```

画像送付時にメタデータとして検出対象クラスを指定いただければ、FSOD-VFM を限定クラスのみに実行可能。

---

## VRAM 管理 (Tesla T4 16GB)

| コンポーネント | VRAM 使用量 |
|---|---|
| DINOv3-L (feature extraction) | ~10 GB |
| SAM2-L (mask extraction) | ~4 GB |
| UPN (proposal generation) | ~1 GB |
| 中間活性・ネットワーク等 | ~1 GB |
| **合計（batch=1）** | **~16 GB** |

Tesla T4 1枚で理論上は動作するが、batch size の調整や gradient checkpointing の検討が必要。

**Tesla T4 × 2枚環境**を推奨。2枚構成にすることで、DINOv3-L と SAM2-L を別GPUに分割配置でき、より安定した処理が可能。

---

## 将来拡張: SAM3 の可能性

### SAM3 の新機能

SAM3 (Meta, 2025年11月発表) は以下の新機能を搭載:

- **Promptable Concept Segmentation (PCS)**: テキスト/画像Exemplarで「全インスタンスを一括検出」
- **Perception Encoder (PE)**: Vision-Language  контрастив 学習で事前学習された Vision Backbone
- **Presence Head**: 認識（Recognition）と定位（Localization）の分離

### SAM3 Vision Backbone (PE) への置換可能性

SAM3 の Vision Backbone である **Perception Encoder (PE-L+)** は DINOv3-L を上回る精度を持つ:

| Vision Backbone | SA-Co/Gold cgF1 |
|---|---|
| PE-L+ (SAM3) | **43.2** |
| DINOv2-L | 35.3 |
| Hiera-L (SAM2) | 32.8 |

**PE-L+** を Vision Backbone として DINOv3-L と交換することで、更なる精度向上が見込める。ただし:

- PE-L+ は ~3.5GB
- DINOv3-L は ~1.2GB
- Tesla T4 (16GB) での共存は VRAM 逼迫の risk あり

**推奨:** Tesla T4 × 2枚環境が整ってから評価を開始する。

### SAM3-L 全体の採用（非推奨）

SAM3-L 全体（约3.45GB）は SAM2-L (~900MB) より大幅に大きい。SAM3 の Principal Feature である PCS (Promptable Concept Segmentation) は FSOD-VFM の prototype matching アーキテクチャと設計思想が異なり、直接的な置換は非効率。因此、SAM3-L の全体採用は推奨しない。

---

## まとめ

| 項目 | 選定 |
|---|---|
| 基本手法 | FSOD-VFM |
| Vision Backbone | **DINOv3-L** |
| Mask Extraction | **SAM2-L** |
| Graph Diffusion | 有効活用 |
| クラス管理 | Feature Store + 動的登録 |
| 検出対象指定 | 入力メタデータで限定 |
| GPU環境 | Tesla T4 × 2枚（推奨） |
|  ожидаемая 精度 | Pascal-5i 1-shot: ~78+ nAP50 |

---

## 参考情報

- **FSOD-VFM 論文**: arXiv:2602.03137 (ICLR 2026)
- **DINOv3 公式サイト**: https://ai.meta.com/dinov3/
- **SAM3 公式サイト**: https://ai.meta.com/sam3/
- **SAM2 公式サイト**: https://ai.meta.com/sam2/
