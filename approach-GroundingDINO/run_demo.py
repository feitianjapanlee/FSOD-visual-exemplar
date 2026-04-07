import argparse
import json
from pathlib import Path

from context_detector import ContextConditionedDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vis', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--box-threshold', type=float, default=0.20)
    parser.add_argument('--text-threshold', type=float, default=0.15)
    parser.add_argument('--match-threshold', type=float, default=0.22)
    parser.add_argument('--nms-threshold', type=float, default=0.45)
    args = parser.parse_args()

    detector = ContextConditionedDetector(device=args.device)
    results = detector.detect_from_files(
        context_json_path=args.context,
        query_image_path=args.query,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        match_threshold=args.match_threshold,
        nms_threshold=args.nms_threshold,
        vis_path=args.vis,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote predictions to {output_path}')


if __name__ == '__main__':
    main()
