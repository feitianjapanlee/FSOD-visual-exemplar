import json
import os
import subprocess
import time
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


def main():
    sample_list = Path('outputs/batch10/sample_list.txt').read_text().strip().splitlines()
    detector = ContextConditionedDetector(device='cuda')

    # warmup
    _ = detector.detect_from_files(
        context_json_path='data/toy-91/context.json',
        query_image_path=sample_list[0],
        vis_path=None,
    )

    times = []
    max_mem = gpu_mem_mb()
    summary = []

    for idx, q in enumerate(sample_list, 1):
        t0 = time.perf_counter()
        pred = detector.detect_from_files(
            context_json_path='data/toy-91/context.json',
            query_image_path=q,
            vis_path=None,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        max_mem = max(max_mem, gpu_mem_mb())
        out_json = Path('outputs/batch10') / (Path(q).stem + '.pred.json')
        out_json.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append({
            'query': q,
            'seconds': dt,
            'detections': len(pred.get('detections', [])),
        })
        print(f'[{idx}/10] {q} sec={dt:.4f} det={len(pred.get("detections", []))}')

    result = {
        'count': len(times),
        'avg_seconds': sum(times) / len(times) if times else 0.0,
        'max_seconds': max(times) if times else 0.0,
        'min_seconds': min(times) if times else 0.0,
        'max_gpu_mem_mb': max_mem,
        'items': summary,
    }
    Path('outputs/batch10/benchmark_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('SUMMARY')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
