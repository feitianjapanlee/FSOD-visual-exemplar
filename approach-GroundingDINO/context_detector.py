import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import CLIPModel, CLIPProcessor


@dataclass
class Detection:
    bbox: List[float]
    class_name: str
    score: float
    similarity: float
    proposal_score: float


class ContextConditionedDetector:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        self.detector_id = 'IDEA-Research/grounding-dino-tiny'
        self.clip_id = 'openai/clip-vit-base-patch32'

        self.det_processor = AutoProcessor.from_pretrained(self.detector_id)
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_id).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_id)
        self.clip_model = CLIPModel.from_pretrained(self.clip_id).to(self.device)
        self.det_model.eval()
        self.clip_model.eval()

    def detect_from_files(
        self,
        context_json_path: str,
        query_image_path: str,
        box_threshold: float = 0.2,
        text_threshold: float = 0.15,
        match_threshold: float = 0.22,
        nms_threshold: float = 0.45,
        max_box_area_ratio: float = 0.25,
        tiny_box_area_ratio: float = 0.015,
        tiny_box_min_proposal_score: float = 0.30,
        vis_path: Optional[str] = None,
    ) -> Dict:
        context_path = Path(context_json_path)
        context = json.loads(context_path.read_text(encoding='utf-8'))
        query_path = Path(query_image_path)

        class_db = self._build_class_database(context['context'], context_path.parent)
        query_image = Image.open(query_path).convert('RGB')

        proposal_prompt = self._build_generic_prompt(context['context'])
        proposals = self._propose_boxes(
            query_image,
            proposal_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections = self._classify_proposals(
            query_image=query_image,
            proposals=proposals,
            class_db=class_db,
            match_threshold=match_threshold,
            nms_threshold=nms_threshold,
            max_box_area_ratio=max_box_area_ratio,
            tiny_box_area_ratio=tiny_box_area_ratio,
            tiny_box_min_proposal_score=tiny_box_min_proposal_score,
        )

        result = {
            'image': str(query_path),
            'detections': [
                {
                    'bbox': [round(float(v), 2) for v in det.bbox],
                    'class': det.class_name,
                    'score': round(float(det.score), 4),
                    'similarity': round(float(det.similarity), 4),
                    'proposal_score': round(float(det.proposal_score), 4),
                }
                for det in detections
            ],
        }

        if vis_path:
            self._draw(query_image, detections, vis_path)
        return result

    def _build_class_database(self, context_items: List[Dict], base_dir: Path):
        db = {}
        for item in context_items:
            class_name = item.get('class_name', item['class'])
            image_paths = [base_dir / p for p in item['refer_image']]
            images = [Image.open(p).convert('RGB') for p in image_paths]
            image_embeds = self._encode_images(images)
            text_embed = self._encode_text([str(class_name)])[0]
            color_hists = [self._color_hist(img) for img in images]
            prototype = torch.nn.functional.normalize((image_embeds.mean(dim=0) + text_embed) / 2.0, dim=0)
            db[str(class_name)] = {
                'image_paths': [str(p) for p in image_paths],
                'prototype': prototype,
                'image_embeds': image_embeds,
                'text_embed': text_embed,
                'color_hists': color_hists,
            }
        return db

    def _build_generic_prompt(self, context_items: List[Dict]) -> str:
        class_names = [str(item.get('class_name', item['class'])) for item in context_items]
        generic_terms = ['object', 'item', 'product', 'vehicle', 'car', 'toy', 'decoration', 'thing']
        prompt_terms = generic_terms + class_names
        return ' . '.join(prompt_terms) + ' .'

    @torch.no_grad()
    def _propose_boxes(self, image: Image.Image, text_prompt: str, box_threshold: float, text_threshold: float):
        inputs = self.det_processor(images=image, text=text_prompt, return_tensors='pt').to(self.device)
        outputs = self.det_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold,
            target_sizes=target_sizes,
        )[0]

        proposals = []
        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            proposals.append({
                'bbox': box.detach().cpu(),
                'proposal_score': float(score.detach().cpu()),
                'label': label,
            })
        return proposals

    @torch.no_grad()
    def _classify_proposals(
        self,
        query_image: Image.Image,
        proposals: List[Dict],
        class_db: Dict,
        match_threshold: float,
        nms_threshold: float,
        max_box_area_ratio: float,
        tiny_box_area_ratio: float,
        tiny_box_min_proposal_score: float,
    ):
        if not proposals:
            return []

        crops = []
        valid_props = []
        image_area = float(query_image.width * query_image.height)
        for prop in proposals:
            x1, y1, x2, y2 = [int(round(v)) for v in prop['bbox'].tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(query_image.width, x2)
            y2 = min(query_image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
            # Scene-sized proposals are a recurrent false-positive mode for this toy benchmark.
            if area_ratio > max_box_area_ratio:
                continue
            proposal_score = float(prop['proposal_score'])
            if area_ratio < tiny_box_area_ratio and proposal_score < tiny_box_min_proposal_score:
                continue
            crop = query_image.crop((x1, y1, x2, y2))
            crops.append(crop)
            valid_props.append({
                **prop,
                'bbox': [x1, y1, x2, y2],
                'crop': crop,
                'area_ratio': area_ratio,
            })

        if not crops:
            return []

        crop_embeds = self._encode_images(crops)
        class_names = list(class_db.keys())

        detections = []
        for i, prop in enumerate(valid_props):
            crop_embed = crop_embeds[i]
            crop_hist = self._color_hist(prop['crop'])
            best = None

            for class_name in class_names:
                entry = class_db[class_name]
                pair_sims = torch.matmul(entry['image_embeds'], crop_embed)
                max_pair_sim = float(pair_sims.max().detach().cpu())
                proto_sim = float(torch.dot(crop_embed, entry['prototype']).detach().cpu())
                text_sim = float(torch.dot(crop_embed, entry['text_embed']).detach().cpu())
                color_sims = [self._hist_intersection(crop_hist, ref_hist) for ref_hist in entry['color_hists']]
                color_sim = max(color_sims) if color_sims else 0.0

                final_similarity = 0.45 * max_pair_sim + 0.20 * proto_sim + 0.10 * text_sim + 0.25 * color_sim
                proposal_score = float(prop['proposal_score'])
                final_score = 0.70 * final_similarity + 0.30 * proposal_score

                candidate = {
                    'class_name': class_name,
                    'similarity': final_similarity,
                    'proposal_score': proposal_score,
                    'score': final_score,
                    'color_sim': color_sim,
                    'max_pair_sim': max_pair_sim,
                }
                if best is None or candidate['score'] > best['score']:
                    best = candidate

            if best is None:
                continue

            # stricter filter to suppress color/material-near misses
            if best['similarity'] < match_threshold:
                continue
            if best['color_sim'] < 0.35:
                continue
            if best['max_pair_sim'] < 0.55:
                continue

            detections.append(Detection(
                bbox=prop['bbox'],
                class_name=best['class_name'],
                score=best['score'],
                similarity=best['similarity'],
                proposal_score=best['proposal_score'],
            ))

        if not detections:
            return []

        return self._per_class_nms(detections, nms_threshold)

    def _per_class_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
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

    def _normalize_clip_features(self, feats) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            tensor = feats
        elif hasattr(feats, 'pooler_output') and feats.pooler_output is not None:
            tensor = feats.pooler_output
        elif hasattr(feats, 'last_hidden_state') and feats.last_hidden_state is not None:
            tensor = feats.last_hidden_state[:, 0, :]
        else:
            raise TypeError(f'Unsupported CLIP feature output type: {type(feats)!r}')
        return torch.nn.functional.normalize(tensor, dim=-1)

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.clip_processor(images=images, return_tensors='pt').to(self.device)
        feats = self.clip_model.get_image_features(**inputs)
        return self._normalize_clip_features(feats)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.clip_processor(text=texts, return_tensors='pt', padding=True).to(self.device)
        feats = self.clip_model.get_text_features(**inputs)
        return self._normalize_clip_features(feats)

    def _color_hist(self, image: Image.Image, bins=(8, 8, 8)) -> np.ndarray:
        arr = np.array(image.convert('RGB'))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _hist_intersection(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.minimum(a, b).sum())

    def _draw(self, image: Image.Image, detections: List[Detection], vis_path: str):
        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.score:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = Path(vis_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
