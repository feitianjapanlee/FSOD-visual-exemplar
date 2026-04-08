import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

from .graph_diffusion import GraphDiffusion, MaskedRoIPooling


@dataclass
class Detection:
    bbox: List[float]
    class_name: str
    score: float
    similarity: float


class FSODVFMDetector:
    def __init__(
        self,
        device: Optional[str] = None,
        max_proposals: int = 500,
        proposal_threshold: float = 0.01,
        match_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        graph_diffusion_steps: int = 30,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.max_proposals = max_proposals
        self.proposal_threshold = proposal_threshold
        self.match_threshold = match_threshold
        self.nms_threshold = nms_threshold
        self.graph_diffusion_steps = graph_diffusion_steps

        self._init_models()

    def _init_models(self):
        from transformers import (
            AutoImageProcessor,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            CLIPModel,
            CLIPProcessor,
        )
        from pathlib import Path

        checkpoint_dir = Path(__file__).parent.parent / "model_checkpoints"

        self.vision_model = self._load_vision_backbone(checkpoint_dir)
        self.vision_model.eval()
        self.vision_model.to(self.device)

        try:
            self.vision_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-large", trust_remote_code=True
            )
        except Exception:
            self.vision_processor = None

        self.vit_intermediate_size = 1024

        try:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model.eval()
            print("CLIP loaded successfully")
        except Exception as e:
            print(f"CLIP not available ({e})")
            self.clip_model = None
            self.clip_processor = None

        self.sam2_model = None
        self.mask_generator = None
        self.use_sam2 = False

        try:
            self.upn_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)
            self.upn_processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            )
            self.upn_model.eval()
        except Exception as e:
            print(f"GroundingDINO not available ({e})")
            self.upn_model = None
            self.upn_processor = None

        self.graph_diffusion = GraphDiffusion(num_steps=self.graph_diffusion_steps)
        self.roi_pooling = MaskedRoIPooling(output_size=7)

    def _load_vision_backbone(self, checkpoint_dir: Path):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "dinov3"))

        dinov3_path = checkpoint_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        if dinov3_path.exists():
            from dinov3.hub.backbones import dinov3_vitl16

            model = dinov3_vitl16(pretrained=False)
            state_dict = torch.load(dinov3_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("DINOv3-ViT-L/16 loaded from local checkpoint")
            return model
        else:
            print("DINOv3 checkpoint not found, using DINOv2-L")
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                "facebook/dinov2-large", trust_remote_code=True
            )
            return model

    @torch.no_grad()
    def detect_from_files(
        self,
        context_json_path: str,
        query_image_path: str,
        vis_path: Optional[str] = None,
    ) -> Dict:
        context_path = Path(context_json_path)
        context = json.loads(context_path.read_text(encoding="utf-8"))
        query_path = Path(query_image_path)

        class_db = self._build_class_database(context["context"], context_path.parent)
        query_image = Image.open(query_path).convert("RGB")

        proposals = self._generate_proposals(query_image)
        if not proposals:
            return {"image": str(query_path), "detections": []}

        img_area = query_image.width * query_image.height
        for prop in proposals:
            prop["image_area"] = img_area
            prop["image_width"] = query_image.width
            prop["image_height"] = query_image.height
            x1, y1, x2, y2 = [int(v) for v in prop["bbox"]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(query_image.width, x2)
            y2 = min(query_image.height, y2)
            prop["crop"] = query_image.crop((x1, y1, x2, y2))

        masks = self._extract_masks(query_image, proposals)

        query_features = self._extract_global_features(query_image)

        proposal_features = self._extract_roi_features(
            query_image, proposals, masks, query_features
        )

        class_scores = self._match_proposals_to_classes(proposal_features, class_db)

        refined_scores = self._apply_graph_diffusion(
            proposal_features, masks, class_scores
        )

        detections = self._build_detections(
            proposals, masks, refined_scores, class_scores, class_db
        )

        result = {
            "image": str(query_path),
            "detections": [
                {
                    "bbox": [round(float(v), 2) for v in det.bbox],
                    "class": det.class_name,
                    "score": round(float(det.score), 4),
                    "similarity": round(float(det.similarity), 4),
                }
                for det in detections
            ],
        }

        if vis_path:
            self._draw(query_image, detections, vis_path)

        return result

    def _build_class_database(self, context_items: List[Dict], base_dir: Path) -> Dict:
        db = {}
        for item in context_items:
            class_name = str(item.get("class_name", item["class"]))
            image_paths = [base_dir / p for p in item["refer_image"]]
            images = [Image.open(p).convert("RGB") for p in image_paths]

            support_masks = []
            support_features = []
            color_histograms = []
            clip_image_embeds = []

            for img in images:
                img_np = np.array(img)
                hist = self._compute_color_histogram(img)
                color_histograms.append(hist)

                if self.use_sam2 and self.mask_generator is not None:
                    try:
                        sam_masks = self.mask_generator.generate(img_np)
                        if sam_masks:
                            best_mask = max(sam_masks, key=lambda m: m["area"])
                            mask = best_mask["segmentation"].astype(np.float32)
                            support_masks.append(mask)
                    except Exception:
                        pass

                feat = self._extract_global_features(img)
                support_features.append(feat.squeeze(0))

                if self.clip_model is not None:
                    clip_emb = self._encode_clip_image(img)
                    clip_image_embeds.append(clip_emb)

            if support_features:
                mean_feat = torch.stack(support_features).mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=-1)

                avg_color_hist = np.mean(color_histograms, axis=0)
                avg_color_hist = avg_color_hist / (avg_color_hist.sum() + 1e-8)

                clip_embeds = (
                    torch.stack(clip_image_embeds) if clip_image_embeds else None
                )

                if support_masks:
                    first_shape = support_masks[0].shape
                    normalized_masks = []
                    for m in support_masks:
                        if m.shape != first_shape:
                            m = (
                                np.array(
                                    Image.fromarray((m * 255).astype(np.uint8)).resize(
                                        (first_shape[1], first_shape[0])
                                    )
                                ).astype(np.float32)
                                / 255.0
                            )
                        normalized_masks.append(m)
                    avg_mask = np.mean(normalized_masks, axis=0)
                    avg_mask = (avg_mask > 0.5).astype(np.float32)
                else:
                    avg_mask = None

                db[class_name] = {
                    "image_paths": [str(p) for p in image_paths],
                    "prototype": mean_feat,
                    "support_masks": support_masks,
                    "avg_mask": avg_mask,
                    "color_histograms": color_histograms,
                    "avg_color_hist": avg_color_hist,
                    "clip_image_embeds": clip_embeds,
                }
            else:
                db[class_name] = {
                    "image_paths": [str(p) for p in image_paths],
                    "prototype": None,
                    "support_masks": [],
                    "avg_mask": None,
                    "color_histograms": [],
                    "avg_color_hist": None,
                    "clip_image_embeds": None,
                }

        return db

    def _compute_color_histogram(
        self, image: Image.Image, bins: int = 16
    ) -> np.ndarray:
        img_np = np.array(image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _histogram_intersection(self, h1: np.ndarray, h2: np.ndarray) -> float:
        return float(np.minimum(h1, h2).sum())

    @torch.no_grad()
    def _encode_clip_image(self, image: Image.Image) -> torch.Tensor:
        if self.clip_model is None or self.clip_processor is None:
            return torch.zeros(512, device=self.device)
        inputs = self.clip_processor(images=[image], return_tensors="pt").to(
            self.device
        )
        feats = self.clip_model.get_image_features(**inputs)
        if hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1).squeeze(0)

    def _generate_proposals(self, image: Image.Image) -> List[Dict]:
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        proposals = []

        if self.upn_model is not None and self.upn_processor is not None:
            try:
                inputs = self.upn_processor(
                    images=image,
                    text="object . item . thing . product . toy .",
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.upn_model(**inputs)

                target_sizes = torch.tensor(
                    [[image.height, image.width]], device=self.device
                )
                results = self.upn_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.15,
                    text_threshold=0.15,
                    target_sizes=target_sizes,
                )[0]

                for box, score, label in zip(
                    results["boxes"], results["scores"], results["labels"]
                ):
                    if float(score) < self.proposal_threshold:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    proposals.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "score": float(score),
                        }
                    )
            except Exception as e:
                print(f"UPN proposal generation failed: {e}")

        if not proposals:
            if self.mask_generator is not None:
                try:
                    sam_masks = self.mask_generator.generate(img_np)
                    for mask_data in sam_masks[: self.max_proposals]:
                        bbox = mask_data["bbox"]
                        x1, y1, bw, bh = bbox
                        x2, y2 = x1 + bw, y1 + bh
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(x1 + 1, min(x2, w))
                        y2 = max(y1 + 1, min(y2, h))
                        score = mask_data.get("predicted_iou", 0.5)
                        proposals.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "score": float(score),
                                "mask": mask_data["segmentation"],
                            }
                        )
                except Exception as e:
                    print(f"SAM2 mask generation failed: {e}")

        if not proposals:
            proposals = self._generate_grid_proposals(h, w)

        proposals = sorted(proposals, key=lambda p: p["score"], reverse=True)
        return proposals[: self.max_proposals]

    def _generate_grid_proposals(
        self, h: int, w: int, min_size: int = 20
    ) -> List[Dict]:
        proposals = []
        scales = [0.1, 0.2, 0.3, 0.4, 0.5]
        aspect_ratios = [1.0, 0.7, 1.4, 0.5, 2.0]
        stride = 0.3

        for scale in scales:
            for ar in aspect_ratios:
                ph = int(h * scale)
                pw = int(ph * ar)
                ph = max(min_size, ph)
                pw = max(min_size, pw)

                for y in np.arange(0, h - ph, h * stride):
                    for x in np.arange(0, w - pw, w * stride):
                        proposals.append(
                            {
                                "bbox": [int(x), int(y), int(x + pw), int(y + ph)],
                                "score": 0.5,
                            }
                        )

        return proposals

    def _extract_masks(
        self, image: Image.Image, proposals: List[Dict]
    ) -> List[torch.Tensor]:
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        if self.use_sam2 and self.mask_generator is not None:
            try:
                all_masks = self.mask_generator.generate(img_np)
                proposal_masks = self._match_masks_to_proposals(all_masks, proposals)
                return proposal_masks
            except Exception as e:
                print(f"SAM2 mask generation failed: {e}")

        return [self._bbox_to_mask(p["bbox"], h, w) for p in proposals]

    def _match_masks_to_proposals(
        self, all_masks: List[Dict], proposals: List[Dict]
    ) -> List[torch.Tensor]:
        h, w = all_masks[0]["segmentation"].shape if all_masks else (0, 0)
        proposal_masks = []

        for prop in proposals:
            x1, y1, x2, y2 = [int(v) for v in prop["bbox"]]
            best_mask = None
            best_iou = 0.0

            for mask_data in all_masks:
                seg = mask_data["segmentation"]
                if seg.shape != (h, w):
                    continue
                mask_crop = seg[y1:y2, x1:x2]
                mask_area = mask_crop.sum()
                if mask_area < 10:
                    continue
                box_area = (x2 - x1) * (y2 - y1)
                iou = mask_area / box_area if box_area > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_mask = seg

            if best_mask is not None and best_iou > 0.1:
                proposal_masks.append(torch.from_numpy(best_mask).float())
            else:
                proposal_masks.append(self._bbox_to_mask(prop["bbox"], h, w))

        return proposal_masks

    def _bbox_to_mask(self, bbox: List[float], h: int, w: int) -> torch.Tensor:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        mask = torch.zeros(h, w, dtype=torch.float32)
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        mask[y1:y2, x1:x2] = 1.0
        return mask

    @torch.no_grad()
    def _extract_global_features(self, image: Image.Image) -> torch.Tensor:
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(518, 518),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std

        if hasattr(self.vision_model, "forward_features"):
            feat_dict = self.vision_model.forward_features(img_tensor)
            feat = feat_dict["x_norm_clstoken"]
        else:
            outputs = self.vision_model(img_tensor)
            feat = (
                outputs.pooler_output
                if hasattr(outputs, "pooler_output")
                else outputs.last_hidden_state[:, 0]
            )

        feat = feat.squeeze(0)
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def _extract_roi_features(
        self,
        image: Image.Image,
        proposals: List[Dict],
        masks: List[torch.Tensor],
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        img_h, img_w = image.height, image.width
        proc_h, proc_w = 518, 518

        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(proc_h, proc_w),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std

        x_scale = proc_w / img_w
        y_scale = proc_h / img_h
        patch_size = 16
        num_patches_h = proc_h // patch_size
        num_patches_w = proc_w // patch_size
        num_patches = num_patches_h * num_patches_w

        if hasattr(self.vision_model, "forward_features"):
            feat_dict = self.vision_model.forward_features(img_tensor)
            patch_tokens = feat_dict["x_prenorm"][0, 5:]
        else:
            outputs = self.vision_model(img_tensor, output_hidden_states=True)
            patch_tokens = outputs.last_hidden_state[0, 1:]

        if patch_tokens.shape[0] != num_patches:
            patch_tokens = patch_tokens[:num_patches]

        feat_map = (
            patch_tokens.reshape(num_patches_h, num_patches_w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        proposal_boxes = torch.tensor(
            [p["bbox"] for p in proposals], dtype=torch.float32, device=self.device
        )

        if len(masks) > 0:
            stacked_masks = torch.stack([m.to(self.device) for m in masks])
            stacked_masks = F.interpolate(
                stacked_masks.unsqueeze(1).float(),
                size=(num_patches_h, num_patches_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            stacked_masks = (stacked_masks > 0.5).float()
        else:
            stacked_masks = torch.ones(
                len(proposals),
                num_patches_h,
                num_patches_w,
                device=self.device,
                dtype=torch.float32,
            )

        roi_features = []
        for i, box in enumerate(proposal_boxes):
            x1, y1, x2, y2 = box.tolist()
            x1_f = x1 * x_scale / patch_size
            y1_f = y1 * y_scale / patch_size
            x2_f = x2 * x_scale / patch_size
            y2_f = y2 * y_scale / patch_size

            x1_f = max(0, min(x1_f, num_patches_w - 1))
            y1_f = max(0, min(y1_f, num_patches_h - 1))
            x2_f = max(x1_f + 1, min(x2_f, num_patches_w))
            y2_f = max(y1_f + 1, min(y2_f, num_patches_h))

            xi1, yi1, xi2, yi2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)

            feat_patch = feat_map[0, :, yi1:yi2, xi1:xi2]
            mask_patch = stacked_masks[i, yi1:yi2, xi1:xi2]

            if feat_patch.numel() == 0:
                roi_feat = torch.zeros(1024, device=self.device)
            else:
                if mask_patch.sum() > 0:
                    mask_up = (
                        F.interpolate(
                            mask_patch.unsqueeze(0).unsqueeze(0).float(),
                            size=feat_patch.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    weighted_feat = feat_patch * mask_up
                    roi_feat = weighted_feat.mean(dim=(1, 2))
                else:
                    roi_feat = feat_patch.mean(dim=(1, 2))

            roi_feat = F.normalize(roi_feat.unsqueeze(0), dim=-1).squeeze(0)
            roi_features.append(roi_feat)

        if not roi_features:
            return torch.empty(0, self.vit_intermediate_size, device=self.device)

        return torch.stack(roi_features)

    def _match_proposals_to_classes(
        self,
        proposal_features: torch.Tensor,
        class_db: Dict,
    ) -> Dict[str, torch.Tensor]:
        class_scores = {}

        for class_name, class_data in class_db.items():
            prototype = class_data["prototype"]
            if prototype is None:
                class_scores[class_name] = torch.zeros(len(proposal_features))
                continue

            prototype = prototype.to(proposal_features.device)
            similarities = torch.mm(proposal_features, prototype.unsqueeze(1)).squeeze(
                1
            )
            class_scores[class_name] = similarities.cpu()

        return class_scores

    def _apply_graph_diffusion(
        self,
        proposal_features: torch.Tensor,
        masks: List[torch.Tensor],
        class_scores: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if len(proposal_features) == 0:
            return class_scores

        device = proposal_features.device
        features = proposal_features.cpu()
        stacked_masks = (
            torch.stack([m for m in masks]) if masks else torch.zeros(0, 1, 1)
        )

        refined_scores = {}
        for class_name, scores in class_scores.items():
            initial_scores = scores.float().to(device)
            refined = self.graph_diffusion(
                features.to(device), stacked_masks.to(device), initial_scores
            )
            refined_scores[class_name] = refined.cpu()

        return refined_scores

    def _build_detections(
        self,
        proposals: List[Dict],
        masks: List[torch.Tensor],
        refined_scores: Dict[str, torch.Tensor],
        class_scores: Dict[str, torch.Tensor],
        class_db: Dict,
    ) -> List[Detection]:
        detections = []
        img_areas = []
        proposal_crops = []
        proposal_clip_embeds = []

        for i, prop in enumerate(proposals):
            bbox = prop["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(prop.get("image_width", 640), x2)
            y2 = min(prop.get("image_height", 480), y2)
            box_area = (x2 - x1) * (y2 - y1)
            img_area = prop.get("image_area", 640 * 480)
            area_ratio = box_area / img_area
            img_areas.append(area_ratio)

            crop = prop.get("crop")
            if crop is not None:
                crop_hist = self._compute_color_histogram(crop)
                clip_embed = self._encode_clip_image(crop) if self.clip_model else None
            else:
                crop_hist = None
                clip_embed = None
            proposal_crops.append(crop_hist)
            proposal_clip_embeds.append(clip_embed)

        for i, prop in enumerate(proposals):
            bbox = prop["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = img_areas[i]
            crop_hist = proposal_crops[i]
            crop_clip_embed = proposal_clip_embeds[i]

            max_area_ratio = 0.25
            min_area_ratio = 0.005
            if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
                continue

            best_class = None
            best_score = -1
            best_sim = 0
            best_color_sim = 0
            best_max_pair_sim = 0

            for class_name, class_data in class_db.items():
                ref_score = refined_scores.get(class_name)
                raw_score = class_scores.get(class_name)
                class_color_hist = class_data.get("avg_color_hist")
                clip_image_embeds = class_data.get("clip_image_embeds")

                if ref_score is None or raw_score is None:
                    continue
                if i >= len(ref_score) or i >= len(raw_score):
                    continue

                raw_sim = float(raw_score[i])
                ref_sim = float(ref_score[i])

                color_sim = 0.5
                if crop_hist is not None and class_color_hist is not None:
                    color_sim = self._histogram_intersection(
                        crop_hist, class_color_hist
                    )

                max_pair_sim = 0.0
                if clip_image_embeds is not None and crop_clip_embed is not None:
                    similarities = torch.matmul(
                        clip_image_embeds.to(crop_clip_embed.device),
                        crop_clip_embed.unsqueeze(1),
                    ).squeeze()
                    max_pair_sim = float(similarities.max().detach().cpu())

                proto_sim = raw_sim
                text_sim = ref_sim
                combined_similarity = (
                    0.45 * max_pair_sim
                    + 0.20 * proto_sim
                    + 0.10 * text_sim
                    + 0.25 * color_sim
                )

                size_penalty = 1.0 - (area_ratio / max_area_ratio) * 0.3
                combined_score = (
                    combined_similarity * size_penalty + 0.30 * prop["score"]
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_class = class_name
                    best_sim = combined_similarity
                    best_color_sim = color_sim
                    best_max_pair_sim = max_pair_sim

            if best_class is not None and best_score > self.match_threshold:
                if best_max_pair_sim >= 0.35 and best_color_sim >= 0.20:
                    detections.append(
                        Detection(
                            bbox=bbox,
                            class_name=best_class,
                            score=best_score,
                            similarity=best_sim,
                        )
                    )

        detections = self._per_class_nms(detections, self.nms_threshold)

        return detections

    def _per_class_nms(
        self, detections: List[Detection], iou_threshold: float
    ) -> List[Detection]:
        if not detections:
            return []

        kept = []
        classes = sorted(set(d.class_name for d in detections))
        for class_name in classes:
            cls_dets = [d for d in detections if d.class_name == class_name]
            boxes = torch.tensor([d.bbox for d in cls_dets], dtype=torch.float32)
            scores = torch.tensor([d.score for d in cls_dets], dtype=torch.float32)
            keep_idx = nms(boxes, scores, iou_threshold).tolist()
            kept.extend(cls_dets[i] for i in keep_idx)
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept

    def _draw(self, image: Image.Image, detections: List[Detection], vis_path: str):
        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = colors[idx % len(colors)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name[:20]} {det.score:.2f}"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        out = Path(vis_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
