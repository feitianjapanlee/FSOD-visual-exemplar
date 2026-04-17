# Training-free OVDで数千商品を捌く実装設計

数千クラス商品検出をtraining-freeで実現する最有力解は、**「テキスト駆動のopen-vocabulary検出器で商品領域を粗く提案し、DINOv2-Lのコサイン類似度で参考画像と個体照合する2段階パイプライン」**である。T4 GPU・FP16・640×480で概ね**0.6〜1.2秒**、1600×1200でも固定入力解像度のため**0.7〜1.3秒**に収まり、2秒制約を満たしつつ数千SKUのクラス追加/削除をインデックス更新のみで処理できる。OVD単体(OWLv2、Grounding DINO)では商品個体の識別(例: 見た目の似たペットボトル飲料)が本質的に困難である一方、DINOv2の自己教師あり画像特徴はDISC21で**CLIPの64% vs 28%**という大差でインスタンス識別に優れ、2024-2025年の「No time to train!」(arXiv:2507.02798)が同構造のtraining-free SoTAを実証している。以下、選定根拠・各ステージ設計・多視点曖昧性への対処・推論時間内訳・代替案を詳述する。

## 既存OVDモデルの整理と選定

**商用・オープン重みでvisual exemplarまで扱えるモデルは依然として限定的**である。IDEA Research系の最高精度モデル(Grounding DINO 1.5/1.6 Pro、T-Rex2、DINO-X)はいずれも**重みが非公開でDeepDataSpace APIのみ**、商用は別契約でオンプレ展開不可。YOLO-WorldはGPL-3.0、YOLOEはAGPL-3.0で、商用SaaSに組み込むには商用ライセンス取得が前提となる。Apache-2.0で完全オープンかつ重み公開なのは**OWLv2、MM-Grounding-DINO、OV-DINO、APE、CountGD、DE-ViT**の5-6モデルに限られる。

| モデル | Text | Visual exemplar | 重み | ライセンス | LVIS-minival AP | オンプレ商用 |
|---|---|---|---|---|---|---|
| **OWLv2-L/14 ensemble** | ◎ | ◎ (image-guided) | 公開 | Apache-2.0 | 44.6 (rare) | ✓ |
| **MM-Grounding-DINO** | ◎ | ✗ | 公開 | Apache-2.0 | 41.4 (T) | ✓ |
| **OV-DINO** | ◎ | ✗ | 公開 | Apache-2.0 | 40.1 | ✓ |
| **Grounding DINO 1.6 Pro** | ◎ | ✗ | API | クラウド課金 | 57.7 | △ |
| **DINO-X Pro** | ◎ | ◎ | API | クラウド課金 | 59.8 | △ |
| **T-Rex2** | ◎ | ◎ (box/point) | API | クラウド課金 | ~54 | △ |
| **YOLO-World-L** | ◎ | △ (preview) | 公開 | GPL-3.0 | 35.4 | △ (要商用契約) |
| **YOLOE** | ◎ | ◎ (box/point/mask) | 公開 | AGPL-3.0 | ~33-36 | △ |
| **CountGD** | ◎ | ◎ (box) | 公開 | 研究用 | (計数中心) | 要確認 |

**ここから読み取れる重要な帰結が一つある**: OWLv2の`image_guided_detection`を除き、オープン重みでvisual exemplarを**箱から使える**モデルは実質存在しない。OWLv2はquery画像から最顕著パッチを自動選択してターゲット画像を走査する仕組みで、**T4-FP16で640×480・OWLv2-Bが0.8〜1.1秒、OWLv2-Lは2〜3秒**。単体パイプラインとして魅力的だが、**ViT-L版は2秒制約を超えるリスク**があり、複数クラス同時クエリ時はさらに線形に伸びる。また、T-Rex2やDINO-Xの論文が示すように商品や産業パーツなど「テキストで表現しづらい対象」はOVD単体だと識別精度が頭打ちになる。したがって、**オープン重みで精度を出すには「OVDで粗く領域提案 → 別の強力な画像特徴量で個体識別」の2段階化が構造的に有利**となる。

## 基盤となる特徴量はDINOv2で確定

DINOv2を採用する根拠は3点で一貫する。第一にインスタンス識別性能で、DISC21(コピー検出150kクラス)で**DINOv2-Bが64%、CLIPは28%**、Voxel51の細粒度10,000クラス評価でも**DINOv2-B 70% vs CLIP 15%**と5倍近い差がついている。ロゴや包装の違いを見分ける能力が段違いで、これは「画像-テキスト対比」を学ぶCLIPより「画像のみの自己教師あり」を学ぶDINOv2が**形状・テクスチャ・部品構造**に敏感だからだ。第二にパッチ特徴の品質が高く、**領域クロップ平均プーリング + L2正規化 → FAISS IndexFlatIPのコサイン類似度**というごく単純な設計で動く。第三に、Matcher(ICLR 2024)やDE-ViT(CoRL 2024)、「No time to train!」(arXiv:2507.02798、COCO-FSOD 30-shotで**nAP 36.8**のtraining-free SoTA)が同じアーキで実績を積み上げている。SigLIP2はDINOv2に近づきつつあるが**細粒度instanceではなおDINOv2 > SigLIP2 > CLIP**で、必要ならensembleの補助に回す立ち位置が妥当である。

T4上の単体推論時間(FP16, batch=1, 518×518入力)は**ViT-S 20-25ms / ViT-B 40-50ms / ViT-L 100-120ms / ViT-g 250-300ms**で、ViT-Lが精度と速度の最適点。クロップ時は224×224で十分なので、**224入力ならViT-Lでも20-30ms/crop、32バッチでまとめて200-400ms**まで縮む。

## 推奨パイプライン: MM-Grounding-DINO + SAM2(軽量) + DINOv2-Lマッチング

### Stage 0: オフライン参考画像DB構築(クラス登録時のみ)

各クラスの1〜5枚の参考画像について、**SAM2-Bにcenter-point promptを与えて前景マスクを抽出**し、背景を除外した領域だけでDINOv2-L特徴を平均プーリング+L2正規化する。SAM2でマスクが不安定な場合は**U²-Net / rembg / BiRefNet**にフォールバックする。2024年時点のBiRefNetはrembgより細部精度で上回り、商品カタログに好適だ。得られる特徴は**視点別に保持**し(詳細は後述)、FAISSの`IndexFlatIP`(数千クラス×5枚=〜数万ベクトル規模なら完全ブルートフォースでms未満)に格納する。カテゴリ情報(`category='toy'`等)はメタデータとしてインデックスに持たせ、クエリ時にカテゴリフィルタでプロトタイプ集合を事前に絞り込む。

### Stage 1: Class-agnostic/text-driven領域提案

クエリ時に指定されたクラスが属する**カテゴリ名(例: "bottle drink", "toy", "daily good")をテキストプロンプト**としてMM-Grounding-DINO(Tiny、Apache-2.0、LVIS-minival 41.4 AP、T4-FP16で**250-350ms**)に投げ、画像内のカテゴリ該当領域を**10〜50個**の候補ボックスとして得る。候補が不足する場合は汎用語("object", "item")でのfallback propose、または密集シーンでは**YOLOE-S**または**FastSAM**のeverythingモード(T4で120-180ms)を併用する。ここでの目的は**Recallの最大化**であって、精度は次段で回収する。個別商品名でOVDに問い合わせる方式は、テキストが商品コードや意味のない型番の場合に機能しないため**採らない**。

### Stage 2: DINOv2-L個体マッチングとスコア統合

各候補ボックスをクロップして224×224(またはDINOv2推奨の518の縮小)にリサイズし、**バッチ化してDINOv2-Lで一括推論**(32ボックスで約200-400ms)。得られたクエリ特徴をFAISSで検索し、指定クラスのプロトタイプ群とのコサイン類似度を計算する。**複数視点・複数参考画像に対してはmax-over-exemplarsで集約**し、同時に「閾値超えた視点数」をconsensus bonusとして加算する。

スコア関数は次の形を推奨する:

```
score(box, class c) = max_v { cos(q_box, p_{c,v}) - α·back_penalty(v) }
                    + β·1[#{v: cos>τ_v} ≥ 2]   (consensus bonus)
```

α = 0.05〜0.10、β = 0.03〜0.05を初期値とする。最後に**クラスごとのadaptive threshold** τ_c = max(τ_global, s_2nd_nearest + δ)でフィルタし、標準的なIoUベースNMSで重複除去して出力する。背景Prototypeとの比較(実画像からランダムパッチで構築した「null prototype」)を挟むことで、DE-ViTが商品ドメインで報告している「未知物体を既存クラスに押し込む」誤検出を構造的に抑制できる。

## 問題1 正面・背面が全く異なり背面が他クラスと酷似するケースの処理

この問題は**プロトタイプ設計の失敗例として典型的**で、全視点を単純平均すると「どこの視点にも弱くマッチする曖昧な重心」になり、他クラス背面との区別がむしろ悪化する。DE-ViTのREADMEが自認する「retail商品で他クラスに誤割当する傾向」はまさにこの構造に由来する。

**推奨する運用は三層構えである**。クラス定義レベルでは**内部的にview-specificサブプロトタイプ(classA-front, classA-back)として保持しつつ、ユーザIFとしてはclassAにマージ**するソフトな分割が最良。完全な別クラス扱いは下流の評価・UI・業務指標を壊すため避けるが、内部で視点情報を失うのも同じく誤り。参考画像準備の指針としては**「最低でも正面+補助1視点」を揃え、曖昧な背面は除外せず`ambiguous=True`メタフラグを付与**する。除外すると実画像で背面が写った時に確実に未検出になり、運用事故を生む。推論ロジックは既述の**max-over-views + 背面視点へのscore penalty + consensus voting + hard negative post-processing**の4段防御が基本形で、オフラインで全プロトタイプ相互類似度行列を計算し「classA-back ↔ classB-back」が類似度0.9超のペアを事前抽出、推論時に両方が発火したら**正面プロトタイプのスコアが高い方を優先**する決定規則を加える。参考画像1枚しかないクラスへの対処は、**TTA(水平反転・軽い回転・色ジッタ・multi-scale crop)で参照側を擬似拡張**するのを第一選択とし、対称でロゴ依存の薄い物体に限っては**Zero-1-to-3系のnovel-view生成**を補助的に使う。Zero-1-to-3は背面色の生成精度が低く、ラベル・ロゴが識別手がかりの商品では逆効果になる点に注意が必要だ。

## 問題2 T4-FP16・640×480と1600×1200での推論時間内訳

T4はTuring世代でFlashAttention非対応、bfloat16非対応、`torch.compile`の最適化効果も限定的(20-30%程度)である点を前置きする。主要モデルの実測/推定は以下のとおり。

| モデル(batch=1, FP16, T4) | 入力 | 推論時間 |
|---|---|---|
| OWLv2-B/16 ensemble | 960² | 700-900 ms |
| OWLv2-L/14 ensemble | 1008² | 1800-2500 ms |
| Grounding DINO Swin-T | 800² | 250-350 ms (TensorRT化 150-200 ms) |
| Grounding DINO Swin-B | 800² | 500-700 ms |
| MM-Grounding-DINO Tiny | 800² | 250-350 ms |
| Grounding DINO 1.5 Edge (API等価) | 640² | 200-220 ms (TRT 60-80 ms) |
| YOLO-World-L | 640² | 40-55 ms |
| YOLOE-S | 640² | 〜30-40 ms |
| SAM2 Hiera-S single forward | 1024² | 100-150 ms |
| SAM2 Hiera-S **everything** | 1024² | **2500-3500 ms** |
| FastSAM everything | 640² | 120-180 ms |
| MobileSAM everything | 640² | 300-600 ms |
| DINOv2 ViT-B/14 (518²) | 単発 | 40-50 ms |
| DINOv2 ViT-L/14 (518²) | 単発 | 100-120 ms |
| DINOv2 ViT-L/14 (224², 32バッチ) | 総計 | 200-400 ms |

重要な観察は、**OVD/SAMの多くが内部で固定解像度にリサイズ**するため、1600×1200の入力でも前処理オーバーヘッドが+30〜100ms増えるだけで本体推論は変わらない点だ。SAM2でマスクを元解像度にアップサンプルする処理のみ+100-300ms増え得る。

推奨パイプライン(MM-Grounding-DINO + DINOv2-L)の時間内訳を示す。

| ステージ | 640×480 | 1600×1200 |
|---|---|---|
| 前処理・リサイズ | 10-15 ms | 40-80 ms |
| MM-Grounding-DINO Tiny (text: category) | 250-350 ms | 270-380 ms |
| ボックス抽出・クロップ (〜20個) | 5-10 ms | 10-20 ms |
| DINOv2-L × 20 boxes (224², 1バッチ) | 200-350 ms | 同左 |
| FAISS検索 + スコア統合 + NMS | 5-15 ms | 同左 |
| **合計** | **約 500-750 ms** | **約 600-850 ms** |

**TensorRT FP16化まで踏み込めば300-450msまで短縮可能**で、2秒制約に対して2倍以上の余裕が出る。仮にOWLv2-Bをimage-conditioned mode単体で使うと640×480で0.8-1.1秒/クエリだが、指定クラスが2つなら線形に伸び1.5〜2秒に触れる。**OWLv2-Lは単発で2秒を超えるため本タスクでは非推奨**である。SAM2 Hiera-S "everything"モードは**2.5-3.5秒かかり単独で2秒制約を突破する**ため、SAM2はbox-promptedモード(参考画像の前景切出しやStage 2の詳細マスク用)に限定するのが鉄則となる。

## 代替案2つ: 純training-freeと最シンプル

**代替案A「No time to train!構成」(arXiv:2507.02798)** は言語プロンプトを一切使わず、参考画像のみで完結する純training-free設計である。`SAM2-Tiny everything → DINOv2-B avg-pool → memory bank cosine → semantic-aware soft merging → NMS`という構造で、公式リポジトリ(github.com/miquel-espinosa/no-time-to-train)がそのまま流用できる。論文値はCOCO-FSOD 30-shotで**nAP 36.8**、設計が極めてクリーンで商品コードのような無意味なクラス名でも動く最大の利点がある。一方でSAM2 everythingはT4で2.5秒超と重く、**FastSAMへの置換またはSAM2-Tで候補を100個以下に制限**する最適化が実運用の前提となる。密集シーンでのrecallは検出器ベースより劣る可能性があり、自社データ検証が必須。

**代替案B「OWLv2 image-guided単体」** は実装がもっとも軽く、transformers公式APIの`Owlv2ForObjectDetection`に`query_image`を渡すだけで動く。T4-FP16・ViT-B・640×480で0.8-1.1秒、指定クラス1つなら2秒制約を余裕で満たす。長所はステップ数の少なさと保守容易性、完全オープン(Apache-2.0)であることだが、**複数クラス同時検索・細粒度商品識別・数千クラスのスケーラビリティで2段階案に劣る**。数百クラス程度の小規模運用・MVP検証用としては合理的な出発点となる。

## 実装チェックリストと運用上の警告

本提案を実装に落とす際、**3点の検証項目**は事前に実施してほしい。第一に、自社の商品画像でDINOv2-L vs DINOv2-B vs SigLIP2のretrieval top-1精度を比較する(DISC21の差が商品ドメインで縮小する可能性)。第二に、MM-Grounding-DINOの**カテゴリテキストプロンプトのrecall**を実測し、候補ボックスが目標物体を漏らしていないか確認する。漏らしている場合は汎用語fallbackやFastSAM補完を併用する。第三に、**T4実機での推論時間を`torch.cuda.Event`で実測**する。上記の数値の多くはA100/V100実測からのFLOPS換算を含むため、±30〜50%の誤差幅がある。

運用上の警告として、DE-ViTの既知弱点(存在しないカテゴリを既存プロトタイプに誤割当)、「No time to train!」がまだ新しく商品ドメインでの実証が限定的な点、Zero-1-to-3系のnovel-view生成が背面テクスチャで不正確である点を意識したい。また、**IDEA Research系のProモデル(Grounding DINO 1.6 Pro、DINO-X、T-Rex2)はクラウドAPI専用であり、オンプレ必須の商用プロジェクトでは選択肢から外す**のが原則。どうしても精度が足りない場合に限り、API経由で許容される工程(例: 参考画像の前景抽出バッチ処理)にだけ使う運用が現実的である。

## 結論

数千クラス・頻繁な追加削除・T4-1枚・2秒制約という条件の組み合わせは、**「OVDのオープン重み+DINOv2」という2段階構成にほぼ一意に収束する**。OWLv2 image-guided単体は魅力的だが複数クラス同時・細粒度識別で限界があり、純SAM2+DINOv2はeverythingモードの重さが足枷となる。**MM-Grounding-DINO Tiny(Apache-2.0、text-promptで商品カテゴリを粗抽出)+ DINOv2-L(個体照合)+ FAISS(数千プロトタイプ)**の組合せが、オープン重み・商用可・2秒制約達成・数千クラスの3要件を同時に満たす唯一の現実解である。多視点問題は「内部view-specific + 外部マージ」「背面penalty」「consensus voting」「hard negative後処理」の4段防御で構造的に解ける。TensorRT FP16化と背景除去パイプラインを丁寧に組み上げれば、商用環境でも堅牢に動作する。