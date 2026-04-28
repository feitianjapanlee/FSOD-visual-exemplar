"""Text-prompted OVD + DINOv2-L two-stage training-free detector.

Stage 1: An open-vocabulary detector (default: Grounding DINO Tiny) takes a
         category-level text prompt and returns recall-oriented region proposals.
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

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}


@dataclass
class Detection:
    bbox: List[float]
    class_name: str
    score: float
    similarity: float
    proposal_score: float
    consensus_views: int


class OVDDINOv2Detector:
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
        match_threshold: float = 0.19,
        margin_threshold: float = 0.12,
        min_proposal_score: float = 0.16,
        min_final_score: float = 0.31,
        null_margin: float = 0.035,
        consensus_threshold: float = 0.37,
        consensus_bonus: float = 0.06,
        nms_threshold: float = 0.48,
        max_proposals: int = 80,
        max_box_area_ratio: float = 0.40,
        min_box_area_ratio: float = 5e-4,
        color_match_threshold: float = 0.35,
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
            color_match_threshold=color_match_threshold,
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

        Use a recall-oriented prompt for Stage 1 proposals:
        generic product terms + class names + category labels. Opaque product
        codes are harmless noise, while human-readable class/category words
        can recover objects that a category-only prompt misses.
        """
        terms: List[str] = [GENERIC_CATEGORY_PROMPT.rstrip(" .")]
        for item in exemplar_items:
            for value in (item.get("class_name", item["class"]), item.get("category")):
                term = str(value).strip() if value is not None else ""
                if term and term not in terms:
                    terms.append(term)
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
            color_hists = [self._color_hist(img) for img in foreground_images]
            db[class_name] = {
                "refer_paths": [str(p) for p in ref_paths],
                "view_embeds": view_embeds,
                "color_hists": color_hists,
                "shape_prior": self._shape_prior(foreground_images),
                "color_sensitive": self._is_color_sensitive_class(class_name),
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

    def _is_color_sensitive_class(self, class_name: str) -> bool:
        tokens = {
            token.strip(" -_/().,").lower()
            for token in class_name.replace("-", " ").replace("_", " ").split()
        }
        return bool(tokens & COLOR_WORDS)

    def _color_hist(self, image: Image.Image, bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        arr = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    def _hist_intersection(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.minimum(a, b).sum())

    def _ensure_sam3(self):
        if self._sam3_model is not None:
            return True
        if not self.enable_sam3:
            return False
        if not self.sam3_checkpoint.exists() or not self.sam3_bpe_path.exists():
            print(
                f"[OVD_DINOv2] SAM3 checkpoint/bpe not found (ckpt={self.sam3_checkpoint}, "
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
            print(f"[OVD_DINOv2] SAM3 import failed ({exc}); falling back to saturation crop.")
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
            print(f"[OVD_DINOv2] SAM3 inference failed for prompt='{prompt}': {exc}")
            return None

        boxes = state.get("boxes")
        masks = state.get("masks")
        scores = state.get("scores")
        if boxes is None or scores is None or len(scores) == 0:
            return None

        # Pick the *largest* confident mask rather than the highest-scoring one.
        # Reference product shots are typically a single item filling the frame,
        # and SAM3 sometimes returns a tighter high-confidence sub-part (e.g. the
        # car body inside a toy-with-packaging shot), which drifts the DINOv2
        # embedding away from query crops that include the whole product.
        boxes_cpu = boxes.detach().float().cpu()
        masks_cpu = None
        if masks is not None:
            masks_cpu = masks.detach().bool().cpu()
            if masks_cpu.ndim == 4 and masks_cpu.shape[1] == 1:
                masks_cpu = masks_cpu[:, 0]
            if masks_cpu.ndim != 3 or masks_cpu.shape[0] != boxes_cpu.shape[0]:
                masks_cpu = None

        if masks_cpu is not None:
            areas = masks_cpu.flatten(1).sum(dim=1).float()
        else:
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
        if masks_cpu is None:
            return image.crop((x1, y1, x2, y2))

        mask_np = masks_cpu[best_idx].numpy()
        if mask_np.shape != (h, w):
            return image.crop((x1, y1, x2, y2))
        crop_arr = np.array(image.convert("RGB"))[y1:y2, x1:x2].copy()
        crop_mask = mask_np[y1:y2, x1:x2]
        if crop_mask.sum() < 16:
            return None
        segmented = np.full_like(crop_arr, 255)
        segmented[crop_mask] = crop_arr[crop_mask]
        return Image.fromarray(segmented)

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
        color_match_threshold: float,
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
        crop_hists = [self._color_hist(crop) for crop in crops]

        # Null prototypes are built from patches that avoid Stage-1 proposals.
        # Using the whole image as a null anchor is brittle in target-dominant
        # scenes: the "background" embedding can become more target-like than a
        # tight object crop and reject valid detections.
        null_crops = self._build_null_crops(query_image, valid_props)
        null_embeds = self._encode_images(null_crops, batch_size=batch_size) if null_crops else None

        class_names = list(class_db.keys())
        detections: List[Detection] = []
        for i, prop in enumerate(valid_props):
            q = crop_embeds[i]  # (D,)
            crop_hist = crop_hists[i]
            scored: List[Dict] = []
            for cname in class_names:
                views = class_db[cname]["view_embeds"]  # (V, D)
                sims = torch.matmul(views, q)
                max_sim = float(sims.max().item())
                consensus = int((sims >= consensus_threshold).sum().item())
                effective = max_sim + (consensus_bonus if consensus >= 2 else 0.0)
                color_hists = class_db[cname].get("color_hists") or []
                color_sim = (
                    max(self._hist_intersection(crop_hist, ref_hist) for ref_hist in color_hists)
                    if color_hists
                    else 1.0
                )
                similarity = 0.75 * effective + 0.25 * color_sim
                scored.append(
                    {
                        "cname": cname,
                        "max_sim": max_sim,
                        "effective": effective,
                        "similarity": similarity,
                        "consensus": consensus,
                        "color_sim": color_sim,
                    }
                )
            scored.sort(key=lambda s: -s["effective"])
            best = scored[0]
            second = scored[1]["effective"] if len(scored) > 1 else -1.0
            second_effective = max((s["effective"] for s in scored[1:]), default=-1.0)

            if best["effective"] < match_threshold:
                continue
            if (best["effective"] - second) < margin_threshold:
                continue
            proposal_score = float(prop["proposal_score"])
            if proposal_score < min_proposal_score:
                continue
            if best["color_sim"] < color_match_threshold:
                continue
            shape_penalty = self._shape_prior_penalty(
                crop=crops[i],
                bbox=prop["bbox"],
                image_area=image_area,
                prior=class_db[best["cname"]].get("shape_prior"),
            )
            # Null-prototype check: the winning class must beat proposal-free
            # background anchors by ``null_margin``. Use a top-2 average rather
            # than max so one contaminated patch does not veto a good crop.
            if null_embeds is not None:
                best_views = class_db[best["cname"]]["view_embeds"]
                strong_visual_match = (
                    best["max_sim"] >= 0.65
                    and best["consensus"] >= 8
                    and (best["effective"] - second_effective) >= 0.45
                )
                if not strong_visual_match:
                    null_class_sim = self._robust_null_similarity(null_embeds, best_views)
                    if (best["max_sim"] - null_class_sim) < null_margin:
                        continue
            final_score = 0.70 * best["similarity"] + 0.30 * proposal_score - shape_penalty
            if final_score < min_final_score:
                continue
            detections.append(
                Detection(
                    bbox=prop["bbox"],
                    class_name=best["cname"],
                    score=final_score,
                    similarity=best["similarity"],
                    proposal_score=proposal_score,
                    consensus_views=best["consensus"],
                )
            )

        if not detections:
            return []
        detections = self._suppress_multi_object_same_class_boxes(detections)
        detections = self._suppress_same_class_parts(detections)
        detections = self._per_class_nms(detections, nms_threshold)
        detections = self._suppress_multi_object_same_class_boxes(detections)
        return self._suppress_same_class_parts(detections)

    def _build_null_crops(
        self,
        image: Image.Image,
        proposals: Optional[List[Dict]] = None,
        max_candidate_overlap: float = 0.05,
    ) -> List[Image.Image]:
        w, h = image.size
        crops: List[Image.Image] = []
        proposal_boxes = [p["bbox"] for p in proposals or []]
        patch_sizes = [
            (max(w // 3, 48), max(h // 3, 48)),
            (max(w // 4, 48), max(h // 4, 48)),
        ]
        candidates = []
        seen = set()
        for patch_w, patch_h in patch_sizes:
            xs = sorted({0, max(0, (w - patch_w) // 2), max(0, w - patch_w)})
            ys = sorted({0, max(0, (h - patch_h) // 2), max(0, h - patch_h)})
            for top in ys:
                for left in xs:
                    right = min(w, left + patch_w)
                    bottom = min(h, top + patch_h)
                    box = (left, top, right, bottom)
                    if box in seen or right - left < 32 or bottom - top < 32:
                        continue
                    seen.add(box)
                    candidates.append(box)

        used_boxes = set()
        for box in candidates:
            if self._max_candidate_overlap(box, proposal_boxes, max_candidate_overlap) > max_candidate_overlap:
                continue
            crops.append(image.crop(box))
            used_boxes.add(box)
            if len(crops) >= 4:
                break

        # Add bounded context patches, but never the whole image. These recover
        # some of the old null prototype's ability to reject object-like
        # background while the robust statistic below avoids a single
        # target-containing patch dominating target-heavy images.
        for box in candidates:
            if len(crops) >= 6:
                return crops
            if box in used_boxes:
                continue
            if self._max_candidate_overlap(box, proposal_boxes, 0.60) > 0.60:
                continue
            crops.append(image.crop(box))
            used_boxes.add(box)
        if len(crops) >= 2:
            return crops

        # Some scenes are proposal-dense. Add lightly overlapping patches only
        # when strict proposal-free patches are unavailable, so null rejection is
        # not silently disabled for most images.
        for box in candidates:
            if box in used_boxes:
                continue
            if self._max_candidate_overlap(box, proposal_boxes, 0.20) > 0.20:
                continue
            crops.append(image.crop(box))
            if len(crops) >= 2:
                return crops

        # If proposals cover nearly the full frame, no reliable query-local null
        # patch exists. In that case skip null rejection instead of fabricating a
        # contaminated null prototype.
        return crops

    def _robust_null_similarity(self, null_embeds: torch.Tensor, view_embeds: torch.Tensor) -> float:
        sims = torch.matmul(null_embeds, view_embeds.T).max(dim=1).values
        if sims.numel() == 0:
            return -1.0
        top_k = min(2, sims.numel())
        top_values = torch.topk(sims, k=top_k).values
        top2_avg = top_values.mean()
        peak = top_values[0]
        return float((0.65 * peak + 0.35 * top2_avg).item())

    def _max_candidate_overlap(
        self,
        box: tuple[int, int, int, int],
        proposal_boxes: List[List[float]],
        stop_threshold: float,
    ) -> float:
        box_area = self._box_area(box)
        if box_area <= 0:
            return 1.0
        max_overlap = 0.0
        for prop in proposal_boxes:
            max_overlap = max(max_overlap, self._intersection_area(box, prop) / box_area)
            if max_overlap > stop_threshold:
                break
        return max_overlap

    # ------------------------------------------------------------- utilities
    def _shape_prior(self, images: List[Image.Image]) -> Optional[Dict[str, float]]:
        if not images:
            return None
        aspects = []
        fill_ratios = []
        for image in images:
            w, h = image.size
            if w <= 0 or h <= 0:
                continue
            aspects.append(w / h)
            fill = float(w * h)
            if fill > 0:
                fill_ratios.append(float(self._foreground_like_mask(image).mean()))
        areas = [float(image.width * image.height) for image in images if image.width > 0 and image.height > 0]
        if not aspects:
            return None
        area_median = float(np.median(areas)) if areas else 1.0
        area_ratios = [area / max(area_median, 1.0) for area in areas]
        return {
            "aspect_median": float(np.median(aspects)),
            "aspect_min": float(np.min(aspects)),
            "aspect_max": float(np.max(aspects)),
            "area_ratio_min": float(np.min(area_ratios)) if area_ratios else 1.0,
            "area_ratio_max": float(np.max(area_ratios)) if area_ratios else 1.0,
            "fill_median": float(np.median(fill_ratios)) if fill_ratios else 1.0,
        }

    def _foreground_like_mask(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGB"))
        if arr.ndim != 3 or arr.shape[2] < 3:
            return np.ones(arr.shape[:2], dtype=bool)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        near_white = (arr[..., 0] > 245) & (arr[..., 1] > 245) & (arr[..., 2] > 245)
        return ((sat > 25) | (val < 235)) & ~near_white

    def _shape_prior_penalty(
        self,
        crop: Image.Image,
        bbox: List[float],
        image_area: float,
        prior: Optional[Dict[str, float]],
    ) -> float:
        if not prior:
            return 0.0
        x1, y1, x2, y2 = bbox
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        aspect = w / h
        aspect_min = max(float(prior.get("aspect_min", aspect)), 1e-3)
        aspect_max = max(float(prior.get("aspect_max", aspect)), aspect_min)
        aspect_low = aspect_min / 1.35
        aspect_high = aspect_max * 1.35

        penalty = 0.0
        if aspect < aspect_low:
            penalty += min(0.16, 0.07 * np.log2(aspect_low / max(aspect, 1e-3)))
        elif aspect > aspect_high:
            penalty += min(0.16, 0.07 * np.log2(aspect / max(aspect_high, 1e-3)))

        fg_mask = self._foreground_like_mask(crop)
        fill_ratio = float(fg_mask.mean()) if fg_mask.size else 1.0
        prior_fill = float(prior.get("fill_median", 1.0))
        if prior_fill > 0:
            fill_delta = abs(np.log2(max(fill_ratio, 1e-3) / max(prior_fill, 1e-3)))
            if fill_delta > 0.65:
                penalty += min(0.10, 0.045 * (fill_delta - 0.65))

        area_ratio = (w * h) / max(image_area, 1.0)
        ref_area_min = float(prior.get("area_ratio_min", 1.0))
        ref_area_max = max(float(prior.get("area_ratio_max", 1.0)), ref_area_min)
        # Reference crops do not know the query image scale, so keep this as a
        # weak, class-specific prior and combine it with an absolute large-box
        # penalty for scene-level proposals.
        if ref_area_max > 0 and area_ratio > 0.12:
            excess = area_ratio / max(0.12 * ref_area_max, 1e-3) - 1.0
            if excess > 0:
                penalty += min(0.12, 0.035 * excess)
        if area_ratio > 0.24:
            penalty += min(0.16, 0.55 * (area_ratio - 0.24))
        return float(max(0.0, penalty))

    def _suppress_same_class_parts(
        self,
        detections: List[Detection],
        contain_threshold: float = 0.78,
        max_part_area_ratio: float = 0.72,
        max_whole_to_part_ratio: float = 4.5,
        max_center_distance_ratio: float = 0.26,
        part_score_margin: float = 0.30,
    ) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        drop = set()
        for i, part in enumerate(detections):
            if i in drop:
                continue
            part_area = self._box_area(part.bbox)
            if part_area <= 0:
                drop.add(i)
                continue
            for j, whole in enumerate(detections):
                if i == j or j in drop or part.class_name != whole.class_name:
                    continue
                whole_area = self._box_area(whole.bbox)
                if whole_area <= part_area:
                    continue
                area_ratio = part_area / max(whole_area, 1.0)
                if area_ratio > max_part_area_ratio:
                    continue
                if whole_area / max(part_area, 1.0) > max_whole_to_part_ratio:
                    continue
                if self._center_distance_ratio(part.bbox, whole.bbox) > max_center_distance_ratio:
                    continue
                inter = self._intersection_area(part.bbox, whole.bbox)
                if inter / max(part_area, 1.0) < contain_threshold:
                    continue
                # Prefer the larger same-class box unless the smaller contained
                # box wins by a clear margin. This targets head/body/label crops
                # that look highly similar but are not complete objects.
                if part.score <= whole.score + part_score_margin:
                    drop.add(i)
                    break
        if not drop:
            return detections
        kept = [det for idx, det in enumerate(detections) if idx not in drop]
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept

    def _suppress_multi_object_same_class_boxes(
        self,
        detections: List[Detection],
        contain_threshold: float = 0.65,
        max_child_area_ratio: float = 0.75,
        min_center_separation_ratio: float = 0.20,
    ) -> List[Detection]:
        if len(detections) <= 2:
            return detections
        drop = set()
        for i, large in enumerate(detections):
            large_area = self._box_area(large.bbox)
            if large_area <= 0:
                drop.add(i)
                continue
            children = []
            for j, child in enumerate(detections):
                if i == j or j in drop or large.class_name != child.class_name:
                    continue
                child_area = self._box_area(child.bbox)
                if child_area <= 0 or child_area >= large_area * max_child_area_ratio:
                    continue
                inter = self._intersection_area(child.bbox, large.bbox)
                if inter / max(child_area, 1.0) >= contain_threshold:
                    children.append((j, child))
            if len(children) < 2:
                continue
            for left_idx, left in children:
                for right_idx, right in children:
                    if left_idx >= right_idx:
                        continue
                    if self._center_separation_ratio(left.bbox, right.bbox, large.bbox) < min_center_separation_ratio:
                        continue
                    drop.add(i)
                    break
                if i in drop:
                    break
        if not drop:
            return detections
        kept = [det for idx, det in enumerate(detections) if idx not in drop]
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept

    def _box_area(self, box) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    def _center_distance_ratio(self, inner, outer) -> float:
        ix = (float(inner[0]) + float(inner[2])) * 0.5
        iy = (float(inner[1]) + float(inner[3])) * 0.5
        ox = (float(outer[0]) + float(outer[2])) * 0.5
        oy = (float(outer[1]) + float(outer[3])) * 0.5
        ow = max(float(outer[2]) - float(outer[0]), 1.0)
        oh = max(float(outer[3]) - float(outer[1]), 1.0)
        return float(np.hypot(ix - ox, iy - oy) / np.hypot(ow, oh))

    def _center_separation_ratio(self, a, b, reference) -> float:
        ax = (float(a[0]) + float(a[2])) * 0.5
        ay = (float(a[1]) + float(a[3])) * 0.5
        bx = (float(b[0]) + float(b[2])) * 0.5
        by = (float(b[1]) + float(b[3])) * 0.5
        rw = max(float(reference[2]) - float(reference[0]), 1.0)
        rh = max(float(reference[3]) - float(reference[1]), 1.0)
        return float(np.hypot(ax - bx, ay - by) / np.hypot(rw, rh))

    def _intersection_area(self, a, b) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

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

    def _suppress_contained_detections(
        self,
        detections: List[Detection],
        contain_threshold: float = 0.85,
        max_area_ratio: float = 0.35,
        min_large_score_ratio: float = 0.75,
    ) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        drop = set()
        for i, small in enumerate(detections):
            sx1, sy1, sx2, sy2 = small.bbox
            small_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
            if small_area <= 0:
                drop.add(i)
                continue
            for j, large in enumerate(detections):
                if i == j or i in drop or small.class_name != large.class_name:
                    continue
                lx1, ly1, lx2, ly2 = large.bbox
                large_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
                if large_area <= small_area:
                    continue
                if small_area / large_area > max_area_ratio:
                    continue
                if large.score < small.score * min_large_score_ratio:
                    continue
                ix1, iy1 = max(sx1, lx1), max(sy1, ly1)
                ix2, iy2 = min(sx2, lx2), min(sy2, ly2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter / small_area >= contain_threshold:
                    drop.add(i)
        if not drop:
            return detections
        kept = [det for idx, det in enumerate(detections) if idx not in drop]
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
