"""MM-Grounding-DINO + DINOv2-L two-stage training-free detector.

Stage 1: MM-Grounding-DINO Tiny takes a category-level text prompt and returns
         recall-oriented region proposals.
Stage 2: Each proposal is cropped, embedded with DINOv2-L, and matched to the
         per-class exemplar embeddings (max-over-views cosine similarity with a
         small consensus bonus when multiple views agree).
Per-class NMS and an adaptive threshold produce the final detections.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms
from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

_DEFAULT_SAM3_REPO = "/home/lee/workspace/keihin-prototype-sam"
_DEFAULT_SAM3_CHECKPOINT = Path(__file__).resolve().parent / "model_checkpoints" / "sam3.pt"
_DEFAULT_SAM3_BPE = Path(__file__).resolve().parent / "model_checkpoints" / "bpe_simple_vocab_16e6.txt.gz"

# The design doc recommends MM-Grounding-DINO Tiny; on this toy-91 benchmark the
# public ``openmmlab-community/mm_grounding_dino_tiny_*`` checkpoints produce
# loose, scene-level boxes whereas the original IDEA-Research Grounding DINO
# Tiny (same Apache-2.0 license, same HF interface) gives tight product-level
# boxes. Default to the latter but keep ``detector_id`` configurable so MM-G-DINO
# variants (e.g. ``openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det``)
# remain drop-in alternatives for datasets where they localise better.
DEFAULT_DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_DINOV2_ID = "facebook/dinov2-large"

GENERIC_CATEGORY_PROMPT = (
    "toy . stuffed animal . plastic toy . figure . doll . car . bottle . "
    "pumpkin . decoration . product . object . item ."
)


@dataclass
class Detection:
    bbox: List[float]
    class_name: str
    score: float
    similarity: float
    proposal_score: float
    consensus_views: int


class MMGroundingDINODetector:
    def __init__(
        self,
        device: Optional[str] = None,
        detector_id: str = DEFAULT_DETECTOR_ID,
        dinov2_id: str = DEFAULT_DINOV2_ID,
        detector_dtype: Optional[torch.dtype] = None,
        dinov2_dtype: Optional[torch.dtype] = None,
        enable_sam3: bool = True,
        sam3_repo_path: Optional[str] = None,
        sam3_checkpoint: Optional[str] = None,
        sam3_bpe_path: Optional[str] = None,
        sam3_resolution: int = 1008,
        sam3_confidence_threshold: float = 0.3,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # MM-Grounding-DINO has mixed text/vision components that currently fail
        # to cast consistently under fp16 in the HF reference implementation,
        # so keep it in fp32 and only accelerate DINOv2 with fp16.
        self.detector_dtype = detector_dtype or torch.float32
        self.dinov2_dtype = dinov2_dtype or (torch.float16 if device == "cuda" else torch.float32)

        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.det_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector_id, torch_dtype=self.detector_dtype)
            .to(device)
            .eval()
        )

        self.dino_processor = AutoImageProcessor.from_pretrained(dinov2_id)
        self.dino_model = (
            AutoModel.from_pretrained(dinov2_id, torch_dtype=self.dinov2_dtype).to(device).eval()
        )

        self._exemplar_cache: Dict[str, Dict] = {}

        self.enable_sam3 = enable_sam3
        self.sam3_repo_path = sam3_repo_path or os.environ.get("SAM3_REPO_PATH", _DEFAULT_SAM3_REPO)
        self.sam3_checkpoint = Path(sam3_checkpoint) if sam3_checkpoint else _DEFAULT_SAM3_CHECKPOINT
        self.sam3_bpe_path = Path(sam3_bpe_path) if sam3_bpe_path else _DEFAULT_SAM3_BPE
        self.sam3_resolution = sam3_resolution
        self.sam3_confidence_threshold = sam3_confidence_threshold
        self._sam3_model = None
        self._sam3_processor = None

    # ------------------------------------------------------------------ public
    def detect_from_files(
        self,
        exemplar_json_path: str,
        query_image_path: str,
        vis_path: Optional[str] = None,
        box_threshold: float = 0.15,
        text_threshold: float = 0.12,
        match_threshold: float = 0.35,
        margin_threshold: float = 0.15,
        min_proposal_score: float = 0.18,
        min_final_score: float = 0.33,
        null_margin: float = 0.04,
        consensus_threshold: float = 0.40,
        consensus_bonus: float = 0.05,
        nms_threshold: float = 0.45,
        max_proposals: int = 80,
        max_box_area_ratio: float = 0.40,
        min_box_area_ratio: float = 5e-4,
        category_prompt: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        dino_batch_size: int = 32,
    ) -> Dict:
        exemplar_path = Path(exemplar_json_path).expanduser().resolve()
        query_path = Path(query_image_path)

        exemplar_items = self._load_exemplar(exemplar_path)
        if target_classes is not None:
            exemplar_items = [
                it for it in exemplar_items if str(it.get("class_name", it["class"])) in target_classes
            ]
            if not exemplar_items:
                return {"image": str(query_path), "detections": []}

        class_db = self._build_class_database(exemplar_items, exemplar_path.parent)
        query_image = Image.open(query_path).convert("RGB")

        prompt = category_prompt or self._build_category_prompt(exemplar_items)
        proposals = self._propose_boxes(
            query_image, prompt, box_threshold=box_threshold, text_threshold=text_threshold
        )

        if len(proposals) > max_proposals:
            proposals = sorted(proposals, key=lambda p: -p["proposal_score"])[:max_proposals]

        detections = self._classify_proposals(
            query_image=query_image,
            proposals=proposals,
            class_db=class_db,
            match_threshold=match_threshold,
            margin_threshold=margin_threshold,
            min_proposal_score=min_proposal_score,
            min_final_score=min_final_score,
            null_margin=null_margin,
            consensus_threshold=consensus_threshold,
            consensus_bonus=consensus_bonus,
            nms_threshold=nms_threshold,
            max_box_area_ratio=max_box_area_ratio,
            min_box_area_ratio=min_box_area_ratio,
            batch_size=dino_batch_size,
        )

        result = {
            "image": str(query_path),
            "detections": [
                {
                    "bbox": [round(float(v), 2) for v in det.bbox],
                    "class": det.class_name,
                    "score": round(float(det.score), 4),
                    "similarity": round(float(det.similarity), 4),
                    "proposal_score": round(float(det.proposal_score), 4),
                    "consensus_views": det.consensus_views,
                }
                for det in detections
            ],
        }
        if vis_path:
            self._draw(query_image, detections, vis_path)
        return result

    # ----------------------------------------------------------- exemplar side
    def _load_exemplar(self, exemplar_path: Path) -> List[Dict]:
        payload = json.loads(exemplar_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "exemplar" in payload:
            return payload["exemplar"]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unexpected exemplar JSON structure in {exemplar_path}")

    def _build_category_prompt(self, exemplar_items: List[Dict]) -> str:
        """Prompt strategy.

        The design doc recommends driving MM-Grounding-DINO with category-level
        tokens and avoiding opaque product codes. When the exemplar carries a
        ``category`` field we use it verbatim; otherwise we fall back to a
        generic bag of product-category terms and *augment* it with the
        exemplar class names so human-readable words like "Bear" or "Pumpkin"
        still contribute grounding signal (opaque codes are harmless noise).
        """
        cats: List[str] = []
        for item in exemplar_items:
            cat = item.get("category")
            if cat and str(cat) not in cats:
                cats.append(str(cat))
        if cats:
            return " . ".join(cats) + " ."

        terms: List[str] = [GENERIC_CATEGORY_PROMPT.rstrip(" .")]
        for item in exemplar_items:
            name = str(item.get("class_name", item["class"])).strip()
            if name and name not in terms:
                terms.append(name)
        return " . ".join(terms) + " ."

    def _build_class_database(self, exemplar_items: List[Dict], base_dir: Path) -> Dict:
        cache_key = str(base_dir.resolve()) + "::" + "|".join(
            f"{it['class']}:{','.join(it['refer_image'])}" for it in exemplar_items
        )
        cached = self._exemplar_cache.get(cache_key)
        if cached is not None:
            return cached

        db: Dict[str, Dict] = {}
        for item in exemplar_items:
            class_name = str(item.get("class_name", item["class"]))
            ref_paths = [self._resolve_ref_path(base_dir, p) for p in item["refer_image"]]
            images = [Image.open(p).convert("RGB") for p in ref_paths]
            category = item.get("category")
            foreground_images = [
                self._foreground_crop(img, category=category, class_name=class_name) for img in images
            ]
            # Reference-side TTA: horizontal flip + light zoom crop per reference.
            # Design doc §問題1: "参考画像1枚しかないクラスへの対処は、TTAで参照側を擬似拡張する".
            tta_images: List[Image.Image] = []
            for img in foreground_images:
                tta_images.extend(self._reference_tta(img))
            view_embeds = self._encode_images(tta_images)
            db[class_name] = {
                "refer_paths": [str(p) for p in ref_paths],
                "view_embeds": view_embeds,
                "category": item.get("category"),
                "num_base_views": len(foreground_images),
                "num_total_views": view_embeds.shape[0],
            }
        self._exemplar_cache[cache_key] = db
        return db

    def _reference_tta(self, image: Image.Image) -> List[Image.Image]:
        variants: List[Image.Image] = [image]
        # Horizontal flip
        variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        # Center zoom (remove border)
        w, h = image.size
        zw, zh = int(w * 0.85), int(h * 0.85)
        if zw >= 16 and zh >= 16:
            left = (w - zw) // 2
            top = (h - zh) // 2
            variants.append(image.crop((left, top, left + zw, top + zh)))
        return variants

    def _resolve_ref_path(self, base_dir: Path, rel: str) -> Path:
        rel_path = Path(rel)
        if rel_path.is_absolute() and rel_path.exists():
            return rel_path
        candidate = base_dir / rel_path
        if candidate.exists():
            return candidate
        alt = base_dir.parent / rel_path
        if alt.exists():
            return alt
        return candidate  # let PIL surface the error

    def _foreground_crop(
        self,
        image: Image.Image,
        category: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> Image.Image:
        """Isolate the foreground product in a reference image.

        Uses SAM3 text-conditioned segmentation when available (prompted with the
        exemplar's ``category`` or ``class_name``), and falls back to a saturation-
        thresholded bbox if SAM3 is disabled, unavailable, or returns no mask.
        """
        if self.enable_sam3:
            prompts = [p for p in (category, class_name) if p]
            prompts.append("object")
            for prompt in prompts:
                crop = self._sam3_foreground_crop(image, prompt)
                if crop is not None:
                    return crop
        return self._saturation_foreground_crop(image)

    def _saturation_foreground_crop(self, image: Image.Image) -> Image.Image:
        arr = np.array(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return image
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        mask = ((sat > 25) | (val < 235)).astype(np.uint8)
        if mask.sum() < 0.05 * mask.size:
            return image
        ys, xs = np.where(mask > 0)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        pad_y = max(2, (y2 - y1) // 50)
        pad_x = max(2, (x2 - x1) // 50)
        y1 = max(0, y1 - pad_y)
        x1 = max(0, x1 - pad_x)
        y2 = min(arr.shape[0], y2 + pad_y)
        x2 = min(arr.shape[1], x2 + pad_x)
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return image
        return image.crop((x1, y1, x2, y2))

    def _ensure_sam3(self):
        if self._sam3_model is not None:
            return True
        if not self.enable_sam3:
            return False
        if not self.sam3_checkpoint.exists() or not self.sam3_bpe_path.exists():
            print(
                f"[MMGDINO] SAM3 checkpoint/bpe not found (ckpt={self.sam3_checkpoint}, "
                f"bpe={self.sam3_bpe_path}); falling back to saturation crop."
            )
            self.enable_sam3 = False
            return False
        if self.sam3_repo_path and self.sam3_repo_path not in sys.path:
            sys.path.insert(0, self.sam3_repo_path)
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:  # noqa: BLE001
            print(f"[MMGDINO] SAM3 import failed ({exc}); falling back to saturation crop.")
            self.enable_sam3 = False
            return False
        self._sam3_model = build_sam3_image_model(
            bpe_path=str(self.sam3_bpe_path),
            checkpoint_path=str(self.sam3_checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            device=self.device,
        )
        self._sam3_processor = Sam3Processor(
            model=self._sam3_model,
            resolution=self.sam3_resolution,
            device=self.device,
            confidence_threshold=self.sam3_confidence_threshold,
        )
        return True

    @torch.no_grad()
    def _sam3_foreground_crop(self, image: Image.Image, prompt: str) -> Optional[Image.Image]:
        if not self._ensure_sam3():
            return None
        try:
            state = self._sam3_processor.set_image(image)
            self._sam3_processor.reset_all_prompts(state)
            state = self._sam3_processor.set_text_prompt(prompt=prompt, state=state)
        except Exception as exc:  # noqa: BLE001
            print(f"[MMGDINO] SAM3 inference failed for prompt='{prompt}': {exc}")
            return None

        boxes = state.get("boxes")
        scores = state.get("scores")
        if boxes is None or scores is None or len(scores) == 0:
            return None

        # Pick the *largest* confident mask rather than the highest-scoring one.
        # Reference product shots are typically a single item filling the frame,
        # and SAM3 sometimes returns a tighter high-confidence sub-part (e.g. the
        # car body inside a toy-with-packaging shot), which drifts the DINOv2
        # embedding away from query crops that include the whole product.
        boxes_cpu = boxes.detach().float().cpu()
        areas = (boxes_cpu[:, 2] - boxes_cpu[:, 0]).clamp(min=0) * (
            boxes_cpu[:, 3] - boxes_cpu[:, 1]
        ).clamp(min=0)
        best_idx = int(torch.argmax(areas).item())
        x1, y1, x2, y2 = boxes_cpu[best_idx].tolist()
        w, h = image.size
        pad_x = max(2, int((x2 - x1) * 0.04))
        pad_y = max(2, int((y2 - y1) * 0.04))
        x1 = max(0, int(round(x1 - pad_x)))
        y1 = max(0, int(round(y1 - pad_y)))
        x2 = min(w, int(round(x2 + pad_x)))
        y2 = min(h, int(round(y2 + pad_y)))
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return None
        return image.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------- stage 1
    @torch.no_grad()
    def _propose_boxes(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[Dict]:
        inputs = self.det_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.detector_dtype)
        outputs = self.det_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
        proposals = []
        for box, score in zip(results["boxes"], results["scores"]):
            proposals.append(
                {
                    "bbox": box.detach().float().cpu().tolist(),
                    "proposal_score": float(score.detach().cpu()),
                }
            )
        return proposals

    # ------------------------------------------------------------- stage 2
    @torch.no_grad()
    def _classify_proposals(
        self,
        query_image: Image.Image,
        proposals: List[Dict],
        class_db: Dict,
        match_threshold: float,
        margin_threshold: float,
        min_proposal_score: float,
        min_final_score: float,
        null_margin: float,
        consensus_threshold: float,
        consensus_bonus: float,
        nms_threshold: float,
        max_box_area_ratio: float,
        min_box_area_ratio: float,
        batch_size: int,
    ) -> List[Detection]:
        if not proposals:
            return []

        image_area = float(query_image.width * query_image.height)
        crops = []
        valid_props = []
        for prop in proposals:
            x1, y1, x2, y2 = [int(round(v)) for v in prop["bbox"]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(query_image.width, x2)
            y2 = min(query_image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
            if area_ratio > max_box_area_ratio or area_ratio < min_box_area_ratio:
                continue
            if (x2 - x1) < 12 or (y2 - y1) < 12:
                continue
            crops.append(query_image.crop((x1, y1, x2, y2)))
            valid_props.append({**prop, "bbox": [x1, y1, x2, y2], "area_ratio": area_ratio})

        if not crops:
            return []

        crop_embeds = self._encode_images(crops, batch_size=batch_size)

        # Null prototypes: four "scene" anchors -- whole image + three large
        # non-central patches. If a class matches the null prototypes almost as
        # well as the crop, the signal is "looks like something in this scene"
        # rather than "this particular product", and we reject it.
        null_crops = self._build_null_crops(query_image)
        null_embeds = self._encode_images(null_crops, batch_size=batch_size) if null_crops else None

        class_names = list(class_db.keys())
        detections: List[Detection] = []
        for i, prop in enumerate(valid_props):
            q = crop_embeds[i]  # (D,)
            scored: List[Dict] = []
            for cname in class_names:
                views = class_db[cname]["view_embeds"]  # (V, D)
                sims = torch.matmul(views, q)
                max_sim = float(sims.max().item())
                consensus = int((sims >= consensus_threshold).sum().item())
                effective = max_sim + (consensus_bonus if consensus >= 2 else 0.0)
                scored.append(
                    {"cname": cname, "max_sim": max_sim, "effective": effective, "consensus": consensus}
                )
            scored.sort(key=lambda s: -s["effective"])
            best = scored[0]
            second = scored[1]["effective"] if len(scored) > 1 else -1.0

            if best["effective"] < match_threshold:
                continue
            if (best["effective"] - second) < margin_threshold:
                continue
            proposal_score = float(prop["proposal_score"])
            if proposal_score < min_proposal_score:
                continue
            # Null-prototype check: the winning class must beat the best scene
            # anchor by ``null_margin``. This suppresses "universal positive"
            # single-reference classes that look vaguely like anything.
            if null_embeds is not None:
                best_views = class_db[best["cname"]]["view_embeds"]
                null_class_sim = float(torch.matmul(null_embeds, best_views.T).max().item())
                if (best["max_sim"] - null_class_sim) < null_margin:
                    continue
            final_score = 0.75 * best["effective"] + 0.25 * proposal_score
            if final_score < min_final_score:
                continue
            detections.append(
                Detection(
                    bbox=prop["bbox"],
                    class_name=best["cname"],
                    score=final_score,
                    similarity=best["max_sim"],
                    proposal_score=proposal_score,
                    consensus_views=best["consensus"],
                )
            )

        if not detections:
            return []
        return self._per_class_nms(detections, nms_threshold)

    def _build_null_crops(self, image: Image.Image) -> List[Image.Image]:
        w, h = image.size
        crops = [image]
        # Corner/edge patches unlikely to contain the whole target object.
        patch_w, patch_h = max(w // 3, 32), max(h // 3, 32)
        anchors = [
            (0, 0),
            (w - patch_w, 0),
            (0, h - patch_h),
            (w - patch_w, h - patch_h),
        ]
        for left, top in anchors:
            left = max(0, left)
            top = max(0, top)
            right = min(w, left + patch_w)
            bottom = min(h, top + patch_h)
            if right - left >= 32 and bottom - top >= 32:
                crops.append(image.crop((left, top, right, bottom)))
        return crops

    # ------------------------------------------------------------- utilities
    def _per_class_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        kept: List[Detection] = []
        classes = sorted({d.class_name for d in detections})
        for cname in classes:
            cdets = [d for d in detections if d.class_name == cname]
            boxes = torch.tensor([d.bbox for d in cdets], dtype=torch.float32)
            scores = torch.tensor([d.score for d in cdets], dtype=torch.float32)
            keep = nms(boxes, scores, iou_threshold).tolist()
            kept.extend(cdets[i] for i in keep)
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        all_feats: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.dino_processor(images=batch, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dinov2_dtype)
            outputs = self.dino_model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0, :]
            feats = F.normalize(feats.float(), dim=-1)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    def _draw(self, image: Image.Image, detections: List[Detection], vis_path: str):
        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.score:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out = Path(vis_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
