#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from fsod_vfm.detector import FSODVFMDetector


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def evaluate_predictions(predictions, ground_truths, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_class_stats = {}

    for gt_data in ground_truths:
        image_name = Path(gt_data["image"]).name
        pred_data = next(
            (p for p in predictions if Path(p["image"]).name == image_name), None
        )

        if pred_data is None:
            total_fn += len(gt_data["detections"])
            continue

        matched_gt = set()
        for pred in pred_data.get("detections", []):
            best_iou = 0
            best_gt_idx = -1
            pred_class = pred["class"]

            for gt_idx, gt_det in enumerate(gt_data["detections"]):
                if gt_idx in matched_gt:
                    continue
                if gt_det["class"] != pred_class:
                    continue
                iou = compute_iou(pred["bbox"], gt_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
                class_name = pred_class
                if class_name not in per_class_stats:
                    per_class_stats[class_name] = {"tp": 0, "fp": 0, "fn": 0}
                per_class_stats[class_name]["tp"] += 1
            else:
                total_fp += 1
                class_name = pred_class
                if class_name not in per_class_stats:
                    per_class_stats[class_name] = {"tp": 0, "fp": 0, "fn": 0}
                per_class_stats[class_name]["fp"] += 1

        for gt_idx, gt_det in enumerate(gt_data["detections"]):
            if gt_idx not in matched_gt:
                total_fn += 1
                class_name = gt_det["class"]
                if class_name not in per_class_stats:
                    per_class_stats[class_name] = {"tp": 0, "fp": 0, "fn": 0}
                per_class_stats[class_name]["fn"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    per_class_f1 = {}
    for class_name, stats in per_class_stats.items():
        p = (
            stats["tp"] / (stats["tp"] + stats["fp"])
            if (stats["tp"] + stats["fp"]) > 0
            else 0
        )
        r = (
            stats["tp"] / (stats["tp"] + stats["fn"])
            if (stats["tp"] + stats["fn"]) > 0
            else 0
        )
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class_f1[class_name] = {"precision": p, "recall": r, "f1": f}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_class": per_class_f1,
    }


def draw_comparison(query_image_path, gt_detections, pred_detections, output_path):
    img = cv2.imread(str(query_image_path))
    if img is None:
        img = np.array(Image.open(str(query_image_path)).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for det in gt_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"GT: {det['class'][:15]}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    for det in pred_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"PD: {det['class'][:15]} {det['score']:.2f}",
            (x1, max(15, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main():
    parser = argparse.ArgumentParser(description="FSOD-VFM Batch Benchmark")
    parser.add_argument("--sample-list", required=True, help="Path to sample list file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--context", required=True, help="Path to context JSON")
    parser.add_argument(
        "--data-root", default="data/toy-91", help="Data root directory"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold"
    )
    parser.add_argument("--visualize", action="store_true", help="Save visualizations")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Max images to process"
    )

    args = parser.parse_args()

    sample_list_path = Path(args.sample_list)
    sample_list = sample_list_path.read_text().strip().split("\n")

    if args.max_images:
        sample_list = sample_list[: args.max_images]

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_json = (
        data_root / args.context
        if not Path(args.context).is_absolute()
        else Path(args.context)
    )
    if not context_json.exists():
        for candidate in [
            Path(args.context),
            data_root / "context.json",
            Path("data/toy-91/context.json"),
        ]:
            if candidate.exists():
                context_json = candidate
                break

    print(f"Using context: {context_json}")

    detector = FSODVFMDetector(
        max_proposals=500,
        proposal_threshold=0.01,
        match_threshold=0.08,
        nms_threshold=0.45,
        graph_diffusion_steps=30,
    )

    predictions = []
    ground_truths = []
    all_times = []

    for idx, line in enumerate(sample_list):
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) >= 2:
            query_name = parts[0].strip()
            gt_name = (
                parts[1].strip()
                if len(parts) > 1
                else query_name.replace(".jpg", ".gt.json")
            )
        else:
            query_name = line.strip()
            gt_name = query_name.replace(".jpg", ".gt.json")

        query_path = data_root / "query_images" / query_name
        gt_path = data_root / "query_images" / gt_name

        if not query_path.exists():
            query_path = data_root / query_name
        if not gt_path.exists():
            gt_path = data_root / gt_name

        if not query_path.exists():
            print(f"Query not found: {query_path}")
            continue

        print(f"[{idx + 1}/{len(sample_list)}] Processing: {query_path.name}")

        start_time = time.time()
        try:
            result = detector.detect_from_files(
                context_json_path=str(context_json),
                query_image_path=str(query_path),
                vis_path=str(output_dir / f"{query_path.stem}_vis.jpg")
                if args.visualize
                else None,
            )
            elapsed = time.time() - start_time
            all_times.append(elapsed)
            predictions.append(result)
        except Exception as e:
            print(f"Error processing {query_path.name}: {e}")
            continue

        if gt_path.exists():
            try:
                gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                ground_truths.append(gt_data)
            except Exception as e:
                print(f"Error loading GT {gt_path}: {e}")

        if (idx + 1) % 10 == 0:
            avg_time = sum(all_times) / len(all_times) if all_times else 0
            print(f"  Average time: {avg_time:.3f}s/image")

    predictions_file = output_dir / "predictions.json"
    predictions_file.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))

    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory peak: {allocated:.2f} GB")

    if ground_truths:
        metrics = evaluate_predictions(predictions, ground_truths, args.iou_threshold)
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(
            f"TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}"
        )
        print("\nPer-class F1:")
        for class_name, stats in metrics["per_class"].items():
            print(
                f"  {class_name}: P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1']:.3f}"
            )

        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))
    else:
        print("\nNo ground truth files found, skipping evaluation")

    avg_time = sum(all_times) / len(all_times) if all_times else 0
    print(f"\nAverage inference time: {avg_time:.3f}s/image")
    print(f"Total images processed: {len(predictions)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
