# Training-free Visual Exemplar Few-shot Object Detector

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

---
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
| Avg Inference Time | 0.524s/image |
| GPU Memory | 2.32 GB |

### Note

- More training images aren't necessarily better; 
- Difficult examples may not be as effective as expected;

---
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
| Precision | 0.9669 |
| Recall | 0.5545 |
| F1 Score | 0.7048 |
| Avg Inference Time | 2.28s/image |
| GPU Memory | 3.41 GB |

### Note

- More training images aren't necessarily better; 
- Difficult examples may not be as effective as expected;
- SAM2 did not contribute as much to accuracy as expected, improving the F1 score by only +0.01; conversely, inference speed is more than five times slower.

---
## Approach 3: OVD_DINOv2

`docs/FSOD_Design_by_Claude_Opus4.7.md` の「推奨パイプライン」を参考にした training-free の2段階検出実装。Stage 1 にテキスト駆動のopen-vocabulary検出器、Stage 2 に DINOv2-L を用いた個体照合を置き、参考画像 (exemplar) のみでクラスを定義する。

### 概要

- **入力**: クエリ画像 + `exemplar.json` (各クラスに `class_name` / `category` / `refer_image[]`)
- **出力**: `{bbox, class, score, similarity, proposal_score, consensus_views}` のリスト
- **学習**: 一切なし (全モジュールとも事前学習済み重みをそのまま利用)

クラスの追加・削除は `exemplar.json` の更新 + DINOv2 埋め込みの再計算のみで完了し、モデル再学習は不要。

### アーキテクチャ

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 0 (オフライン / 参照側)                                     │
│  refer_image ─▶ SAM3 (text-conditioned segmentation)              │
│              ─▶ 最大面積マスクでクロップ (失敗時は彩度bbox)          │
│              ─▶ TTA (原画像 + 水平反転 + 0.85中心ズーム)             │
│              ─▶ DINOv2-L 埋め込み (L2正規化済み)                    │
│              ─▶ class_db[class_name].view_embeds                  │
│                                                                  │
│  Stage 1 (オンライン)                                             │
│  query_image + category_prompt                                   │
│              ─▶ Grounding DINO Tiny                              │
│              ─▶ box候補 (proposal_score つき)                     │
│                                                                  │
│  Stage 2 (オンライン)                                             │
│  各候補boxをクロップ ─▶ DINOv2-L 埋め込み (FP16, バッチ化)            │
│                    ─▶ torch.matmul で全クラスview_embedsと類似度計算  │
│                    ─▶ max-over-views + consensus bonus           │
│                    ─▶ margin / null-prototype / 面積フィルタ        │
│                    ─▶ per-class IoU NMS                          │
└──────────────────────────────────────────────────────────────────┘
```

### 主要コンポーネント

| 役割 | モデル | 実装での出典 |
|---|---|---|
| Stage 1 OVD | `IDEA-Research/grounding-dino-tiny` (FP32) | `transformers.AutoModelForZeroShotObjectDetection` |
| 参照前景抽出 | SAM3 (text prompt) | `/home/lee/workspace/keihin-prototype-sam/sam3` リポを `sys.path` へ |
| 画像埋め込み | `facebook/dinov2-large` (FP16) | `transformers.AutoModel` |
| 類似度検索 | `torch.matmul` | FAISS ではなく小規模マトリクス積 |

Grounding DINO のテキストプロンプトは `exemplar.json` の `category` を優先利用し (`"toy ."`)、無い場合のみ汎用カテゴリ語+class_name連結にフォールバックする (`_build_category_prompt`, `detector.py:177`)。

### スコア関数

```
best  = argmax_c [ max_v cos(q, p_{c,v}) + β·1[ #{v : cos ≥ τ_cons} ≥ 2 ] ]
score = 0.75 · best.effective + 0.25 · proposal_score
```

採用判定は `(best.effective ≥ match_threshold) ∧ (best − 2nd ≥ margin) ∧ (best_view − null_view ≥ null_margin) ∧ (score ≥ min_final_score)` の合議。

### CLIパラメータ (batch_benchmark.py 経由)

- `--approach ovd_dinov2`
- `--exemplar <path>` (`category` フィールド推奨)
- `--target-class <path>` `{"target": ["class_name", ...]}` で検出対象クラスを絞り込み
- `--device cuda|cpu|mps`

`OVDDINOv2Detector.detect_from_files` は追加で `box_threshold`, `text_threshold`, `match_threshold`, `margin_threshold`, `consensus_threshold`, `consensus_bonus`, `null_margin`, `nms_threshold`, `max_box_area_ratio`, `min_box_area_ratio` 等を直接渡せる (`detector.py:89` 以降)。

### 推奨パイプラインとの差異

| 項目 | 推奨パイプライン | 本実装 | 差分の理由 |
|---|---|---|---|
| Stage 1 OVD | MM-Grounding-DINO Tiny | **Grounding DINO Tiny (IDEA-Research)** | toy-91 で MM-G-DINO の `o365v1_goldg` 系がシーン全体に広い箱しか出さず、Grounding DINO Tiny の方が GT 箱とタイトに一致したため。`detector_id` で差し替え可能。 |
| 参照前景抽出 | SAM2-B + center-point prompt (不安定時は BiRefNet / U²-Net / rembg) | **SAM3 (text prompt)** + 彩度bbox フォールバック | 設計当初は彩度法のみ。ユーザ要請により SAM3 に置換 (2秒要件は一旦無視)。最高スコアではなく**最大面積**マスクを採用 — Tomika の参考画像で車本体のみを高スコア・全体(包装+車)を低スコアで返し、部分マスクだと DINOv2 埋め込みが崩れて F1 が 0 まで落ちたため。 |
| 検索インデックス | FAISS `IndexFlatIP` | **`torch.matmul`** 直接 | toy-91 は 3 クラス規模で行列積の方が軽い。数千クラス規模では FAISS 化が必要。 |
| Adaptive per-class threshold | `τ_c = max(τ_global, s_2nd + δ)` | **グローバル `match_threshold` + `margin_threshold`** | クラスごとの履歴がまだ無いため。Bear のように単一参照で広くマッチするクラスと Pumpkin のような明瞭形状クラスを同一しきい値で捌いており、Bear の P=0.41 の主因。 |
| Back penalty α | `−α·back_penalty(v)` (0.05〜0.10) | **未実装** | toy-91 では背面の曖昧さが顕在化しない。実運用で類似背面を持つ商品が増えた段階で追加する前提。 |
| View-specific サブプロトタイプ | `classA-front` / `classA-back` を内部で分け、UIではマージ | **未実装** | 上と同じ理由。 |
| Hard negative 事前抽出 | 全プロト相互類似度の `back↔back > 0.9` を事前抽出し推論時に正面優先 | **未実装** | 上と同じ理由。 |
| Null prototype | 実画像からランダムBG パッチで事前構築 | **クエリ画像の全体+四隅**から都度生成 | 商品横断の BG 統計が揃っていない段階の簡易実装。 |
| 最終スコア融合 | コサイン類似度+consensus のみ | **0.75·effective + 0.25·proposal_score** | OVD の confidence を低proposal_scoreで減点するため導入。過検出を抑える方向に効く。 |
| TTA | flip + 回転 + 色ジッタ + multi-scale crop。対称物体は Zero-1-to-3 補助 | **flip + 0.85 中心ズームのみ** | 最小限で開始。色ジッタは商品色を変えるリスクがありあえて除外。 |
| DINOv2 入力 | 224×224, 32 バッチ | **DINOv2 既定前処理 (処理側に委任), batch=32** | HuggingFace の `AutoImageProcessor` に合わせる方が保守が容易。 |

### ベンチマーク結果 (toy-91)

| 構成 | Precision | Recall | F1 | Latency (s/img) | GPU peak |
|---|---|---|---|---|---|
| 彩度クロップ (ベースライン) | 0.630 | 0.758 | 0.688 | 0.63 | 2.3 GB |
| SAM3 最高スコア選択 | 0.624 | 0.654 | 0.639 | 0.81 | 5.6 GB |
| **SAM3 最大面積選択** | **0.663** | **0.773** | **0.713** | 0.82 | 5.6 GB |

クラス別 F1 (SAM3最大面積構成):
- Stuffed Bear Blue: P=0.410 / R=0.764 / F1=**0.534**
- Halloween Pumpkin: P=0.954 / R=0.902 / F1=**0.927**
- Pocket Tomika: P=1.000 / R=0.532 / F1=**0.694**

### 現在の実装 vs "No time to train!" (arXiv:2507.02798) の主な違い

両者とも「training-free / SAM系 + DINOv2 /
メモリバンク照合」という土台は共通しているが、Stage 1 の領域提案の思想が根本的に異なる。

**構造の比較**
| | No time to train! | 本実装 (推奨パイプライン)  |
|---|---|---|
| Stage 1 (領域提案) | SAM2 everything (class-agnostic,言語なし) | Grounding DINO Tiny (text-conditioned, category prompt) |
| 言語プロンプトの使用  | 一切なし (参考画像のみで完結)     | カテゴリ語を使用 ("toy .")   |
| 候補の表現    | セグメンテーションマスク   | 検出バウンディングボックス  |
| 参照前景抽出  | Stage 1 と同一の SAM2 マスク  | SAM3 (text-conditioned) で独立に抽出  |
| 埋め込み      | DINOv2-B のパッチ平均プーリング (マスク内のみ)  | DINOv2-L の CLS/pooler token (crop全体)  |
| 照合集約      | Semantic-aware soft merging (プロトタイプ間の意味類似度も考慮) | max-over-views + consensus bonus + margin filter |
| 最終スコア    | 類似度ベースのみ  | 0.75·類似度 + 0.25·proposal_score   |
| NMS           | マスク単位  | ボックス単位 (per-class)  |

**主な違いの本質**

1. **「言語を使うか否か」の思想の違い**

- No time to train!: 言語を完全に排除。商品コード (SB-1-B) や無意味な型番でも動作することが最大の売り。クラスは「参考画像そのもの」だけで定義される。
- 本実装: カテゴリ語 ("toy") を使って Stage 1 の recall を稼ぐ。class_name は Grounding DINO に投げるプロンプトに含まれる場合があるが、商品コードが含まれていても Grounding DINO はトークンとして無視するだけで害はない。カテゴリメタデータという「ゆるい言語手がかり」に依存する。

2. **Stage 1 の proposal メカニズム**

- No time to train!: SAM2 の everything
モードが画像内の全オブジェクトをマスク化し、その中から memory bank と一致するものを選ぶ。密集していない大きな物体は取りこぼしにくいが、T4 では 2.5〜3.5 秒かかり (設計ドキュメント §問題2)、運用には FastSAM 等への置換か候補数制限が必須。
- 本実装: Grounding DINO Tiny はカテゴリ該当領域のみに候補を絞る (約 250〜350ms)。カテゴリから外れた物体は最初から候補にならないため、カテゴリが適切に定義されている前提ではより効率的に動く。toy-91 では 0.6〜0.8s/image を維持。

3. **マスク vs ボックス**

- No time to train!: SAM2 マスクの形状に沿ってパッチ平均プーリングできるため、背景コンタミネーションが構造的に少ない。細長い物体や複雑な形状に強い。
- 本実装: ボックスクロップは背景ピクセルを含む。そのため null-prototype 抑制や SAM3 による参照前景抽出で補っているが、クエリ側のクロップは依然として rectangular。

4. **集約方式: Soft merging vs max-over-views**

- No time to train!: "Semantic-aware soft merging"
は同一クラスの複数参照プロトタイプ間の意味的類似度で重み付けして統合する (単純平均でも max でもない)。これにより「背面と正面がかけ離れている」クラスでも、一方だけ強く発火したときに他
方が希釈しすぎない。
- 本実装: max-over-views (一番よく合った1視点を採用) + consensus bonus (≥2視点合意で加点)。実装は単純だが、プロトタイプ間の意味関係は無視している。設計ドキュメントが指摘する「back↔back 類似ペア事前抽出」もこの弱点への補強だが、本実装では未導入。

5. **埋め込みモデルの容量**

- No time to train!: DINOv2-B (≈85M)。パッチ平均で表現力を稼ぐ。
- 本実装: DINOv2-L (≈300M) の CLS/pooler token。設計ドキュメントの「Lが精度と速度の最適点」推奨に従った。

**どちらが有利か (ケース別)**

シナリオ: クラス名が完全に無意味な商品コード
優位: No time to train!
理由: 言語不要
────────────────────────────────────────
シナリオ: カテゴリが明確 (飲料 / おもちゃ / 工具)
優位: 本実装
理由: Stage 1 が絞り込みで速くて recall が安定
────────────────────────────────────────
シナリオ: 密集シーン (棚一面の商品)
優位: 本実装
理由: 検出器は多数候補を出せる、SAM2 everything は候補数に上限
────────────────────────────────────────
シナリオ: T4 での 2 秒制約が厳しい
優位: 本実装
理由: 0.6〜0.8s で収まる。No time to train! はSAM2 everything の置換が必須
────────────────────────────────────────
シナリオ: 細長・非矩形な物体
優位: No time to train!
理由: マスク前提で背景混入が少ない
────────────────────────────────────────
シナリオ: 背面と正面が全く異なる商品
優位: No time to train! (soft merging)
理由: 本実装は max-over-views で view-specific な取り扱いが不足
────────────────────────────────────────
シナリオ: 公式リポジトリでの再現性
優位: No time to train!
理由: github.com/miquel-espinosa/no-time-to-train がそのまま使える

**要約**

- No time to train! = 「参考画像 + SAM2 + DINOv2」の完全ビジュアル派。言語依存ゼロで最もクリーンだが、T4 では SAM2 everything
が重い。
- 本実装 = 「カテゴリ言語 + 検出器 + DINOv2」のハイブリッド派。カテゴリ概念が使える前提で速度・スケール面で有利だが、言語前処理(category メタデータの付与) が必要。

設計ドキュメントも両者を並列の代替案として位置付けており(§代替案A)、商用・カテゴリ明確・T4-2秒制約という要件では本実装寄りの構成が現実的、一方で「商品コードしか与えられない」「FastSAM で高速化できる」状況では No time to train! がよりクリーンな選択肢になる、という棲み分けになっている。

### 今後の改善候補 (効果が期待できそうな順)

1. **Adaptive per-class threshold の導入** — `margin_threshold` をクラスごとにキャリブレート。Bear のような「単一参照で幅広くマッチするクラス」だけ mismatch に厳しくすることで現状 P=0.41 を押し上げられる見込み。実装コスト小。
2. **View-specific sub-prototype + back penalty** — 参考画像が2枚以上のクラスで view ラベル (`front`/`back`) をオプションで受け取り、back視点に `α=0.05〜0.10` のペナルティを入れる。toy-91 では Pumpkin の 2 参照で早速効く可能性あり。実装コスト中。
3. **大域 Null prototype** — 事前にクエリ画像コーパスからランダム BG パッチを数百枚サンプルしておき、起動時に DINOv2 埋め込みを計算して `null_embeds` として固定保持する。現在のクエリ依存 null を置き換えることで、稀にターゲットの一部を null に含めてしまうリスクを排除。実装コスト小。
4. **DINOv2 入力解像度の統一・パッチ平均化** — 現在は HF 既定のリサイズ任せ。設計ドキュメントは 224×224 への明示リサイズ + パッチ平均 (CLS tokenより細粒度) を推奨。実装コスト小、Bear の個体識別精度改善が見込める。
5. **TTA 強化** — 軽い回転 (±10°) と multi-scale crop (0.75, 0.85, 1.0) を追加。色ジッタは商品色を壊すのでスキップ。consensus の機能する閾値を下げられる可能性。実装コスト小。
6. **SAM3 プロンプトフォールバック順の最適化** — 現状は `[category, class_name, "object"]` で試行し最初に当たったものを採用。サブ部位を返す問題は「最大面積」で回避したが、面積とスコアの加重 (`area * score^0.5`) にする方が堅牢になる可能性。
7. **FAISS 置換 (数千クラス拡張時)** — 現状の `torch.matmul` は3クラス規模向け。`IndexFlatIP` + カテゴリメタデータ事前フィルタに置き換えることで数千クラスでもms級を維持。実装コスト中。
8. **Stage 1 の軽量化** — TensorRT FP16 化、または YOLOE/FastSAM の補完を追加して proposal recall を上げる。現状 Bear の recall 0.76 は Stage 1 で取りこぼしている可能性もあり、proposal 側のリコール計測で切り分けが要る。
9. **Hard negative マイニング** — 全クラス相互類似度行列を起動時に計算し、類似度 0.85 超のペアを記録。推論時に両方発火したら正面プロトタイプ優先、等の決定規則を加える。クラス数が増えるほど効果的。
10. **入力サイズとクロップの最小値見直し** — 現在は 12px 未満を切っているが、Tomika のような小物体で recall が 0.53 に留まっているのは min_box_area_ratio (5e-4) で落としている可能性がある。per-class に閾値を分ける余地あり。

### 既知の制約

- SAM3 により GPU メモリが 5.6GB、レイテンシが 0.82s/img に増加。T4 (16GB) の 2 秒要件を重視する場合は `enable_sam3=False` で彩度法に戻せる。
- SAM3 はローカルリポ (`/home/lee/workspace/keihin-prototype-sam`) + 依存 (`iopath`, `ftfy`, `einops`, `regex`, `timm`, `pycocotools`, `psutil`) が必要。`SAM3_REPO_PATH` 環境変数または `sam3_repo_path` 引数で差し替え可能。
- `IDEA-Research/grounding-dino-tiny` の HF 実装は FP16 で text enhancer の dtype 不一致を起こすため、Stage 1 は FP32 固定。DINOv2 のみ FP16 で高速化している。

### 関連ファイル

- `detector.py` — `OVDDINOv2Detector` 本体
- `__init__.py` — エクスポート
- `model_checkpoints/` — `sam3.pt` と `bpe_simple_vocab_16e6.txt.gz` (user配置)
- `../batch_benchmark.py` — ベンチマーク driver (`--approach ovd_dinov2`)
- `../docs/FSOD_Design_by_Claude_Opus4.7.md` — 推奨パイプラインの原典

---
## 3つのアプローチの共通点

- 3つとも training-free で、クラス追加・削除は基本的に exemplar.json の参照画像を変える方式です。
- 3つとも GroundingDINO 系で候補 bbox を生成し、その候補 crop を参照画像特徴と照合します。
- 3つとも出力は image と detections[] で、各 detection は bbox, class, score, similarity 系を返します。
- 3つとも per-class NMS を使い、同一クラス内の重複 bbox を抑制します。
- 3つとも大きすぎる箱や小さすぎる箱を閾値で落とす思想があります。
- 3つとも T4/CUDA を想定しつつ、cuda がなければ mps / cpu に落ちる実装です。
- 3つとも「クラス名だけに依存しない」方向で、参照画像の visual embedding を主要な識別根拠にしています。
- 3つとも最終的には bbox 検出で、セグメンテーション mask を最終出力にはしません。

## 3つのアプローチの差異点

| 観点 | approach_GroundingDINO | approach_FSODVFM | approach_OVD_DINOv2 |
|---|---|---|---|
| 位置づけ | 最小構成の実用 baseline | VFM 複合型の高精度狙い | 設計文書寄りの2段階 OVD + DINOv2 |
| 候補生成 | GroundingDINO Tiny | GroundingDINO Tiny、失敗時に SAM2/grid fallback | GroundingDINO Tiny、最大80件に絞る |
| 候補 prompt | 汎用語 + class_name | 固定 "object . item . thing . product . toy ." | category 優先、なければ汎用語 + class_name |
| 参照特徴 | CLIP image + CLIP text + HSV色 | DINOv3/DINOv2 prototype + CLIP + HSV色 | DINOv2-L view embeddings |
| 参照前景抽出 | なし、参照画像全体を encode | SAM2 用フックはあるが現状 use_sam2=False | SAM3 text segmentation、失敗時は彩度 bbox |
| 複数参照の扱い | CLIP image 平均 + max pair similarity | DINO prototype 平均 + CLIP pair | view ごとの max-over-views + consensus bonus |
| スコア | 0.45*CLIP pair + 0.20*proto + 0.10*text + 0.25*color | 類似式は近いが Graph Diffusion 後スコアと size penalty を追加 | DINOv2 類似度 + consensus + margin + null prototype + proposal score |
| 誤検出抑制 | 色類似、CLIP pair、面積、tiny box filter | 面積、色、CLIP pair、Graph Diffusion | margin、null prototype、proposal score、面積、min final score |
| 指定 target class 対応 | detector API にはなし。exemplar 側で絞る必要あり | detector API にはなし。exemplar 側で絞る必要あり | target_classes を直接受ける |
| 速度/メモリ傾向 | 最軽量 | DINOv3/Graph Diffusion で重い | SAM3 + DINOv2-L でメモリ重め |
| 要件「背面が他クラスと同じ」への強さ | 弱い。view 分離なし | 弱め。prototype 平均で視点差が混ざる | 3つの中では相対的に強いが、README 上も view-specific/back penalty は未実装 |

現在の既存 metrics 差分
| approach | Precision | Recall | F1 | 平均時間 | GPU peak |
|---|---:|---:|---:|---:|---:|
| GroundingDINO | 0.982 | 0.777 | 0.868 | 0.519s/img | 2.32GB |
| FSODVFM | 0.967 | 0.555 | 0.705 | 2.255s/img | 3.41GB |
| OVD_DINOv2 | 0.658 | 0.749 | 0.701 | 1.839s/img | 6.18GB |

要約すると、現状の実装では GroundingDINO が最も軽く、既存 metrics でも最良です。FSODVFM は構成が最も複雑で Graph Diffusion まで入っていますが recall が低めです。OVD_DINOv2 は要件の category と target class 指定に最も素直に対応し、参照画像の前景抽出や視点別照合もありますが、SAM3 依存とメモリ消費が大きいです。