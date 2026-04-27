#!/usr/bin/env python3
"""Visualize OVD_DINOv2 reference foreground crops.

This utility intentionally reuses OVDDINOv2Detector's foreground-crop methods
without constructing the full detector models. It loads only the SAM3 state
needed for exemplar-side crop inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from approach_OVD_DINOv2.detector import OVDDINOv2Detector


def load_exemplar(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "exemplar" in payload:
        return payload["exemplar"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected exemplar JSON structure in {path}")


def resolve_ref_path(base_dir: Path, rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute() and rel_path.exists():
        return rel_path
    candidate = base_dir / rel_path
    if candidate.exists():
        return candidate
    alt = base_dir.parent / rel_path
    if alt.exists():
        return alt
    return candidate


def make_cropper(args: argparse.Namespace):
    cropper = OVDDINOv2Detector.__new__(OVDDINOv2Detector)
    cropper.device = args.device
    cropper.enable_sam3 = not args.no_sam3
    cropper.sam3_repo_path = args.sam3_repo_path or os.environ.get(
        "SAM3_REPO_PATH", "/home/lee/workspace/keihin-prototype-sam"
    )
    cropper.sam3_checkpoint = Path(args.sam3_checkpoint)
    cropper.sam3_bpe_path = Path(args.sam3_bpe_path)
    cropper.sam3_resolution = args.sam3_resolution
    cropper.sam3_confidence_threshold = args.sam3_confidence_threshold
    cropper._sam3_model = None
    cropper._sam3_processor = None
    return cropper


def foreground_crop_with_label(
    cropper,
    image: Image.Image,
    category: Optional[str],
    class_name: Optional[str],
) -> tuple[Image.Image, str]:
    if cropper.enable_sam3:
        prompts = [p for p in (category, class_name) if p]
        prompts.append("object")
        for prompt in prompts:
            crop = cropper._sam3_foreground_crop(image, prompt)
            if crop is not None:
                return crop, f"SAM3: {prompt}"
    return cropper._saturation_foreground_crop(image), "saturation fallback"


def fit_image(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - fitted.width) // 2
    y = (box_h - fitted.height) // 2
    canvas.paste(fitted.convert("RGB"), (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize OVD_DINOv2 reference foreground crops")
    parser.add_argument("--exemplar", default="data/toy-91/exemplar.json")
    parser.add_argument("--output-dir", default="outputs/ovd_dinov2_foreground_check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-sam3", action="store_true")
    parser.add_argument("--sam3-repo-path", default=None)
    parser.add_argument(
        "--sam3-checkpoint",
        default="approach_OVD_DINOv2/model_checkpoints/sam3.pt",
    )
    parser.add_argument(
        "--sam3-bpe-path",
        default="approach_OVD_DINOv2/model_checkpoints/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.3)
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--tile-height", type=int, default=220)
    args = parser.parse_args()

    exemplar_path = Path(args.exemplar).expanduser().resolve()
    out_dir = Path(args.output_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cropper = make_cropper(args)
    items = load_exemplar(exemplar_path)

    rows = []
    manifest = []
    for item in items:
        class_name = str(item.get("class_name", item["class"]))
        category = item.get("category")
        for rel in item["refer_image"]:
            ref_path = resolve_ref_path(exemplar_path.parent, rel)
            image = Image.open(ref_path).convert("RGB")
            crop, method = foreground_crop_with_label(cropper, image, category, class_name)

            safe_stem = f"{item['class']}_{Path(rel).stem}".replace("/", "_")
            crop_path = crops_dir / f"{safe_stem}_crop.jpg"
            crop.save(crop_path, quality=95)

            rows.append((class_name, rel, method, image, crop))
            manifest.append(
                {
                    "class": item["class"],
                    "class_name": class_name,
                    "refer_image": rel,
                    "method": method,
                    "original_size": list(image.size),
                    "crop_size": list(crop.size),
                    "crop_path": str(crop_path),
                }
            )

    label_h = 62
    row_h = args.tile_height + label_h
    sheet_w = args.tile_width * 2
    sheet_h = max(1, row_h * len(rows))
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (class_name, rel, method, image, crop) in enumerate(rows):
        y = idx * row_h
        sheet.paste(fit_image(image, args.tile_width, args.tile_height), (0, y + label_h))
        sheet.paste(fit_image(crop, args.tile_width, args.tile_height), (args.tile_width, y + label_h))
        draw.text((8, y + 6), f"{class_name}", fill=(0, 0, 0))
        draw.text((8, y + 24), f"{rel}", fill=(60, 60, 60))
        draw.text((args.tile_width + 8, y + 6), "foreground crop", fill=(0, 0, 0))
        draw.text((args.tile_width + 8, y + 24), method, fill=(60, 60, 60))
        draw.line((0, y + row_h - 1, sheet_w, y + row_h - 1), fill=(220, 220, 220))

    sheet_path = out_dir / "foreground_contact_sheet.jpg"
    manifest_path = out_dir / "manifest.json"
    sheet.save(sheet_path, quality=95)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved contact sheet: {sheet_path}")
    print(f"Saved crops: {crops_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
