import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from context_detector import ContextConditionedDetector


def gpu_mem_mb():
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], text=True).strip().splitlines()
        vals = [int(x.strip()) for x in out if x.strip()]
        return max(vals) if vals else 0
    except Exception:
        return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-list', default='outputs/batch10/sample_list.txt')
    parser.add_argument('--output-dir', default='outputs/batch10')
    parser.add_argument('--context', default='data/toy-91/context.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--box-threshold', type=float, default=0.20)
    parser.add_argument('--text-threshold', type=float, default=0.15)
    parser.add_argument('--match-threshold', type=float, default=0.22)
    parser.add_argument('--nms-threshold', type=float, default=0.45)
    parser.add_argument('--max-box-area-ratio', type=float, default=0.25)
    parser.add_argument('--tiny-box-area-ratio', type=float, default=0.015)
    parser.add_argument('--tiny-box-min-proposal-score', type=float, default=0.30)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--skip-vis', action='store_true')
    return parser.parse_args()


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def load_ground_truth(query_path):
    gt_path = Path(query_path).with_suffix('.gt.json')
    if not gt_path.exists():
        return None, gt_path
    gt = json.loads(gt_path.read_text(encoding='utf-8'))
    gt_detections = [
        {
            'bbox': [float(v) for v in det['bbox']],
            'class': str(det['class']),
        }
        for det in gt.get('detections', [])
    ]
    return gt_detections, gt_path


def evaluate_image(pred_detections, gt_detections, iou_threshold):
    preds = [
        {
            'bbox': [float(v) for v in det['bbox']],
            'class': str(det['class']),
            'score': float(det.get('score', 0.0)),
        }
        for det in pred_detections
    ]
    gts = [
        {
            'bbox': [float(v) for v in det['bbox']],
            'class': str(det['class']),
        }
        for det in gt_detections
    ]

    matched_gt = set()
    matched_pairs = []
    tp = 0
    fp = 0

    for pred_idx, pred in sorted(enumerate(preds), key=lambda item: item[1]['score'], reverse=True):
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt or pred['class'] != gt['class']:
                continue
            iou = box_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is None:
            fp += 1
            continue

        matched_gt.add(best_gt_idx)
        tp += 1
        matched_pairs.append({
            'pred_index': pred_idx,
            'gt_index': best_gt_idx,
            'class': pred['class'],
            'iou': round(best_iou, 4),
        })

    fn = len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_iou = sum(pair['iou'] for pair in matched_pairs) / len(matched_pairs) if matched_pairs else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_matched_iou': avg_iou,
        'matches': matched_pairs,
        'unmatched_predictions': [preds[i] for i in range(len(preds)) if i not in {m['pred_index'] for m in matched_pairs}],
        'unmatched_ground_truth': [gts[i] for i in range(len(gts)) if i not in matched_gt],
    }


def main():
    args = parse_args()

    sample_list = [
        line.strip()
        for line in Path(args.sample_list).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not sample_list:
        raise ValueError(f'No samples found in {args.sample_list}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ContextConditionedDetector(device=args.device)

    # Warm up model kernels and caches without writing outputs.
    _ = detector.detect_from_files(
        context_json_path=args.context,
        query_image_path=sample_list[0],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        match_threshold=args.match_threshold,
        nms_threshold=args.nms_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
        tiny_box_area_ratio=args.tiny_box_area_ratio,
        tiny_box_min_proposal_score=args.tiny_box_min_proposal_score,
        vis_path=None,
    )

    times = []
    max_mem = gpu_mem_mb()
    summary = []
    total = len(sample_list)
    totals = defaultdict(float)
    class_totals = defaultdict(lambda: defaultdict(float))
    has_ground_truth = True

    for idx, q in enumerate(sample_list, 1):
        stem = Path(q).stem
        vis_path = None if args.skip_vis else str(output_dir / f'{stem}.vis.jpg')

        t0 = time.perf_counter()
        pred = detector.detect_from_files(
            context_json_path=args.context,
            query_image_path=q,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            match_threshold=args.match_threshold,
            nms_threshold=args.nms_threshold,
            max_box_area_ratio=args.max_box_area_ratio,
            tiny_box_area_ratio=args.tiny_box_area_ratio,
            tiny_box_min_proposal_score=args.tiny_box_min_proposal_score,
            vis_path=vis_path,
        )
        dt = time.perf_counter() - t0
        det_count = len(pred.get('detections', []))

        times.append(dt)
        max_mem = max(max_mem, gpu_mem_mb())

        out_json = output_dir / f'{stem}.pred.json'
        out_json.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding='utf-8')
        item_summary = {
            'query': q,
            'seconds': dt,
            'detections': det_count,
        }

        gt_detections, gt_path = load_ground_truth(q)
        if gt_detections is None:
            has_ground_truth = False
            item_summary['ground_truth'] = None
        else:
            eval_result = evaluate_image(pred.get('detections', []), gt_detections, args.iou_threshold)
            item_summary['ground_truth'] = str(gt_path)
            item_summary['gt_detections'] = len(gt_detections)
            item_summary['evaluation'] = {
                'tp': eval_result['tp'],
                'fp': eval_result['fp'],
                'fn': eval_result['fn'],
                'precision': round(eval_result['precision'], 4),
                'recall': round(eval_result['recall'], 4),
                'f1': round(eval_result['f1'], 4),
                'avg_matched_iou': round(eval_result['avg_matched_iou'], 4),
                'matches': eval_result['matches'],
                'unmatched_predictions': eval_result['unmatched_predictions'],
                'unmatched_ground_truth': eval_result['unmatched_ground_truth'],
            }
            totals['tp'] += eval_result['tp']
            totals['fp'] += eval_result['fp']
            totals['fn'] += eval_result['fn']
            totals['matched_iou_sum'] += eval_result['avg_matched_iou'] * eval_result['tp']
            totals['matched_count'] += eval_result['tp']
            for det in gt_detections:
                class_totals[det['class']]['gt'] += 1
            for det in pred.get('detections', []):
                class_totals[str(det['class'])]['pred'] += 1
            for match in eval_result['matches']:
                class_totals[match['class']]['tp'] += 1
            for det in eval_result['unmatched_predictions']:
                class_totals[det['class']]['fp'] += 1
            for det in eval_result['unmatched_ground_truth']:
                class_totals[det['class']]['fn'] += 1

        summary.append(item_summary)
        status = f'[{idx}/{total}] {q} sec={dt:.4f} det={det_count}'
        if gt_detections is not None:
            ev = item_summary['evaluation']
            status += f" tp={ev['tp']} fp={ev['fp']} fn={ev['fn']}"
        print(status)

    evaluation_summary = None
    if has_ground_truth:
        precision = totals['tp'] / (totals['tp'] + totals['fp']) if (totals['tp'] + totals['fp']) else 0.0
        recall = totals['tp'] / (totals['tp'] + totals['fn']) if (totals['tp'] + totals['fn']) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class = {}
        for class_name, values in sorted(class_totals.items()):
            tp = values['tp']
            fp = values['fp']
            fn = values['fn']
            cls_precision = tp / (tp + fp) if (tp + fp) else 0.0
            cls_recall = tp / (tp + fn) if (tp + fn) else 0.0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) else 0.0
            per_class[class_name] = {
                'gt': int(values['gt']),
                'pred': int(values['pred']),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': round(cls_precision, 4),
                'recall': round(cls_recall, 4),
                'f1': round(cls_f1, 4),
            }
        evaluation_summary = {
            'iou_threshold': args.iou_threshold,
            'tp': int(totals['tp']),
            'fp': int(totals['fp']),
            'fn': int(totals['fn']),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'avg_matched_iou': round(
                totals['matched_iou_sum'] / totals['matched_count'], 4
            ) if totals['matched_count'] else 0.0,
            'per_class': per_class,
        }

    result = {
        'count': len(times),
        'avg_seconds': sum(times) / len(times) if times else 0.0,
        'max_seconds': max(times) if times else 0.0,
        'min_seconds': min(times) if times else 0.0,
        'max_gpu_mem_mb': max_mem,
        'config': {
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
            'match_threshold': args.match_threshold,
            'nms_threshold': args.nms_threshold,
            'max_box_area_ratio': args.max_box_area_ratio,
            'tiny_box_area_ratio': args.tiny_box_area_ratio,
            'tiny_box_min_proposal_score': args.tiny_box_min_proposal_score,
            'iou_threshold': args.iou_threshold,
        },
        'evaluation': evaluation_summary,
        'items': summary,
    }
    (output_dir / 'benchmark_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('SUMMARY')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
