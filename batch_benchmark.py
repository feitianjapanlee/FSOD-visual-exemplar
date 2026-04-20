#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image




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
    parser = argparse.ArgumentParser(description="FSOD Batch Benchmark")
    parser.add_argument("--approach", required=True, help="Approach name (GroundingDINO/FSODVFM)")
    parser.add_argument("--sample-list", default='data/toy-91/sample_list.txt', help="Path to sample list file (default: data/toy-91/sample_list.txt)")
    parser.add_argument("--output-dir", default='outputs/latest', help="Output directory (default: outputs/latest)")
    parser.add_argument("--exemplar", default='data/toy-91/exemplar.json', help="Path to exemplar JSON (default: data/toy-91/exemplar.json)")
    parser.add_argument("--data-root", default="data/toy-91", help="Data root directory (default: data/toy-91)")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument('--device', default='cuda', help="Device to run on (cuda/cpu/mps default: cuda)")
    parser.add_argument('--text-threshold', type=float, default=0.15, help="Text threshold (default: 0.15)")
    parser.add_argument('--match-threshold', type=float, default=0.22, help="Match threshold (default: 0.22)")
    parser.add_argument('--nms-threshold', type=float, default=0.45, help="NMS threshold (default: 0.45)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold (GroundingDINO only) (default: 0.5)")
    parser.add_argument('--box-threshold', type=float, default=0.20, help="Box threshold (GroundingDINO only) (default: 0.20)")
    parser.add_argument('--max-box-area-ratio', type=float, default=0.25, help="Max box area ratio (GroundingDINO only) (default: 0.25)")
    parser.add_argument('--tiny-box-area-ratio', type=float, default=0.015, help="Tiny box area ratio (GroundingDINO only) (default: 0.015)")
    parser.add_argument('--tiny-box-min-proposal-score', type=float, default=0.30, help="Tiny box min proposal score (GroundingDINO only) (default: 0.30)")
    parser.add_argument("--max-proposals", type=int, default=500, help="Max proposals (FSODVFM only) (default: 500)")
    parser.add_argument("--proposal-threshold", type=float, default=0.01, help="Proposal threshold (FSODVFM only) (default: 0.01)")
    parser.add_argument("--graph-steps", type=int, default=30, help="Graph diffusion steps (FSODVFM only) (default: 30)")
    args = parser.parse_args()

    approach = args.approach.lower()

    sample_list_path = Path(args.sample_list)
    sample_list = sample_list_path.read_text().strip().split("\n")

    if args.max_images:
        sample_list = sample_list[: args.max_images]

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exemplar_json = (
        data_root / args.exemplar
        if not Path(args.exemplar).is_absolute()
        else Path(args.exemplar)
    )
    if not exemplar_json.exists():
        for candidate in [
            Path(args.exemplar),
            data_root / "exemplar.json",
            Path("data/toy-91/exemplar.json"),
        ]:
            if candidate.exists():
                exemplar_json = candidate
                break

    print(f"Using exemplar: {exemplar_json}")

    if approach == "fsodvfm":    
        from approach_FSODVFM.fsod_vfm.detector import FSODVFMDetector

        detector = FSODVFMDetector(
            device=args.device,
            max_proposals=args.max_proposals,
            proposal_threshold=args.proposal_threshold,
            graph_diffusion_steps=args.graph_steps,
            match_threshold=args.match_threshold,
            nms_threshold=args.nms_threshold,
        )
    elif approach == "groundingdino":
        from approach_GroundingDINO.exemplar_detector import ExemplarConditionedDetector

        detector = ExemplarConditionedDetector(device=args.device)
    else:
        print(f"Unsupported approach: {approach}")
        return

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
            if approach == "fsodvfm":
                 result = detector.detect_from_files(
                    exemplar_json_path=str(exemplar_json),
                    query_image_path=str(query_path),
                    vis_path=str(output_dir / f"{query_path.stem}_vis.jpg") if args.visualize else None,
                )
            else: # approach == "groundingdino":
                result = detector.detect_from_files(
                    exemplar_json_path=str(exemplar_json),
                    query_image_path=str(query_path),
                    vis_path=str(output_dir / f"{query_path.stem}_vis.jpg") if args.visualize else None,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    match_threshold=args.match_threshold,
                    nms_threshold=args.nms_threshold,
                    max_box_area_ratio=args.max_box_area_ratio,
                    tiny_box_area_ratio=args.tiny_box_area_ratio,
                    tiny_box_min_proposal_score=args.tiny_box_min_proposal_score,
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

    avg_time = sum(all_times) / len(all_times) if all_times else 0

    gpu_memory_peak = None
    if torch.cuda.is_available():
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**3

    if ground_truths:
        metrics = evaluate_predictions(predictions, ground_truths, args.iou_threshold)
        metrics["approach"] = args.approach
        metrics["thresholds"] = {
            "text_threshold": args.text_threshold,
            "match_threshold": args.match_threshold,
            "nms_threshold": args.nms_threshold,
            "iou_threshold": args.iou_threshold,
            "box_threshold": args.box_threshold,
            "max_box_area_ratio": args.max_box_area_ratio,
            "tiny_box_area_ratio": args.tiny_box_area_ratio,
            "tiny_box_min_proposal_score": args.tiny_box_min_proposal_score,
            "max_proposals": args.max_proposals,
            "proposal_threshold": args.proposal_threshold,
            "graph_steps": args.graph_steps,
        }
        metrics["gpu_memory_peak_gb"] = gpu_memory_peak
        metrics["avg_inference_time_seconds"] = avg_time
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
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
        metrics = {"approach": args.approach, "avg_inference_time_seconds": avg_time}
        if gpu_memory_peak is not None:
            metrics["gpu_memory_peak_gb"] = gpu_memory_peak
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))

    print(f"\nAverage inference time: {avg_time:.3f}s/image")
    print(f"Total images processed: {len(predictions)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
