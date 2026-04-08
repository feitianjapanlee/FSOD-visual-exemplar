#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from fsod_vfm.detector import FSODVFMDetector


def main():
    parser = argparse.ArgumentParser(description="FSOD-VFM Few-Shot Object Detection")
    parser.add_argument("--context", required=True, help="Path to context JSON file")
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument(
        "--output", default="outputs/predictions.json", help="Output JSON path"
    )
    parser.add_argument("--vis", default=None, help="Visualization image path")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--max-proposals", type=int, default=500, help="Max proposals")
    parser.add_argument(
        "--proposal-threshold", type=float, default=0.01, help="Proposal threshold"
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.3, help="Match threshold"
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.45, help="NMS threshold"
    )
    parser.add_argument(
        "--graph-steps", type=int, default=30, help="Graph diffusion steps"
    )

    args = parser.parse_args()

    detector = FSODVFMDetector(
        device=args.device,
        max_proposals=args.max_proposals,
        proposal_threshold=args.proposal_threshold,
        match_threshold=args.match_threshold,
        nms_threshold=args.nms_threshold,
        graph_diffusion_steps=args.graph_steps,
    )

    result = detector.detect_from_files(
        context_json_path=args.context,
        query_image_path=args.query,
        vis_path=args.vis,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Detected {len(result['detections'])} objects")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
