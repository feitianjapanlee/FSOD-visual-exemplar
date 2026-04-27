#!/usr/bin/env python3
"""Render *.gt.json annotations onto their corresponding images."""

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (0, 180, 90),
    (230, 80, 60),
    (60, 130, 230),
    (230, 160, 40),
    (160, 90, 220),
    (40, 180, 200),
    (220, 80, 160),
    (120, 170, 50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ground-truth boxes stored in *.gt.json files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/toy-91/query_images",
        help="Directory containing *.gt.json and image files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/toy91_gt_vis",
        help="Directory to write labeled JPG files and index.html.",
    )
    parser.add_argument(
        "--pattern",
        default="*.gt.json",
        help="Glob pattern for ground-truth JSON files inside --input-dir.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit the number of JSON files to render.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not generate index.html.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resolve_image_path(gt_path: Path, data: dict[str, Any]) -> Path:
    image_value = data.get("image")
    if image_value:
        image_path = Path(image_value)
        if image_path.exists():
            return image_path
        candidate = gt_path.parent / image_path.name
        if candidate.exists():
            return candidate

    stem = gt_path.name.removesuffix(".gt.json")
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = gt_path.with_name(stem + suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for {gt_path}")


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"bbox must have 4 values: {box}")
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    pad_x = 5
    pad_y = 3
    text_w, text_h = text_size(draw, text, font)
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    x, y = xy
    width, height = image_size

    if x + box_w > width:
        x = max(0, width - box_w)
    if y < 0:
        y = 0
    if y + box_h > height:
        y = max(0, height - box_h)

    draw.rectangle((x, y, x + box_w, y + box_h), fill=color)
    draw.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255), font=font)


def render_gt(gt_path: Path, output_dir: Path, class_colors: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    data = json.loads(gt_path.read_text())
    image_path = resolve_image_path(gt_path, data)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    font = load_font(max(13, round(min(width, height) / 45)))

    detections = data.get("detections", [])
    for idx, det in enumerate(detections, start=1):
        class_name = str(det.get("class", "object"))
        if class_name not in class_colors:
            class_colors[class_name] = PALETTE[len(class_colors) % len(PALETTE)]
        color = class_colors[class_name]
        x1, y1, x2, y2 = clamp_box(det["bbox"], width, height)

        for offset in range(line_width):
            draw.rectangle(
                (x1 - offset, y1 - offset, x2 + offset, y2 + offset),
                outline=color,
            )

        label = f"{idx}: {class_name}"
        draw_label(
            draw,
            (x1, y1 - (text_size(draw, label, font)[1] + 10)),
            label,
            font,
            color,
            image.size,
        )

    output_path = output_dir / f"{image_path.stem}_gt.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return {
        "gt_path": gt_path,
        "image_path": image_path,
        "output_path": output_path,
        "detections": len(detections),
    }


def write_index(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    index_path = output_dir / "index.html"
    total_boxes = sum(row["detections"] for row in rows)
    cards = []
    for row in rows:
        rel_img = row["output_path"].name
        title = html.escape(row["image_path"].name)
        gt_name = html.escape(row["gt_path"].name)
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(rel_img)}"><img src="{html.escape(rel_img)}" alt="{title}"></a>
              <div class="meta">
                <strong>{title}</strong>
                <span>{row["detections"]} boxes</span>
                <small>{gt_name}</small>
              </div>
            </article>
            """
        )

    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ground Truth Visualization</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fa;
      color: #17202a;
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid #d8dee8;
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 4px;
      font-weight: 650;
    }}
    header p {{
      margin: 0;
      color: #52616f;
      font-size: 14px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
      padding: 16px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #111820;
    }}
    .meta {{
      display: grid;
      gap: 4px;
      padding: 10px 12px 12px;
      font-size: 13px;
    }}
    .meta span, .meta small {{
      color: #52616f;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Ground Truth Visualization</h1>
    <p>{len(rows)} images, {total_boxes} boxes</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    gt_paths = sorted(input_dir.glob(args.pattern))
    if args.max_images is not None:
        gt_paths = gt_paths[: args.max_images]
    if not gt_paths:
        raise SystemExit(f"No files matched {input_dir / args.pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_colors: dict[str, tuple[int, int, int]] = {}
    rows = [render_gt(path, output_dir, class_colors) for path in gt_paths]

    index_path = None if args.no_index else write_index(output_dir, rows)
    print(f"Rendered {len(rows)} images to {output_dir}")
    if index_path:
        print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
