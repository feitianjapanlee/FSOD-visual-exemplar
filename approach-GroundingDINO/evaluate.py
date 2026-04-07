import argparse
import json
from collections import defaultdict


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_image(preds, gts, iou_thresh=0.5):
    by_class_gt = defaultdict(list)
    for gt in gts:
        by_class_gt[gt['class']].append({**gt, 'matched': False})

    preds = sorted(preds, key=lambda x: x.get('score', 0), reverse=True)
    tp, fp = 0, 0
    for pred in preds:
        cls = pred['class']
        best_iou = 0.0
        best_gt = None
        for gt in by_class_gt.get(cls, []):
            if gt['matched']:
                continue
            val = iou(pred['bbox'], gt['bbox'])
            if val > best_iou:
                best_iou = val
                best_gt = gt
        if best_gt is not None and best_iou >= iou_thresh:
            best_gt['matched'] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for cls_gts in by_class_gt.values() for gt in cls_gts if not gt['matched'])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Prediction JSON')
    parser.add_argument('--gt', required=True, help='GT JSON with detections field')
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()

    pred = json.load(open(args.pred, 'r', encoding='utf-8'))
    gt = json.load(open(args.gt, 'r', encoding='utf-8'))

    result = evaluate_image(pred.get('detections', []), gt.get('detections', []), iou_thresh=args.iou)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
