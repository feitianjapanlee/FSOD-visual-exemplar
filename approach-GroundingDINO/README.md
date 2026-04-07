# Few-shot Novel Object Detection Prototype

Context-image-conditioned object detection prototype.

## What this is
A validation pipeline for detecting object instances in a query image given:
- one or more classes
- one or more reference images per class
- no task-specific training on those classes

The implementation uses:
- GroundingDINO for generic region proposals
- CLIP for visual matching between proposals and reference images
- optional text scoring from class names

This is a practical prototype for evaluation and iteration, not a new foundation model.

## Why this design
Directly training a true generalized few-shot detector is expensive. For a usable baseline that works on novel classes without per-class training, a strong pattern is:
1. generate candidate boxes with an open-vocabulary detector
2. encode candidates and reference images into a shared embedding space
3. classify boxes by nearest reference-class prototype
4. apply NMS and confidence thresholds

This gives a zero additional training baseline and is easy to validate.

## Project layout
- `requirements.txt` Python dependencies
- `context_detector.py` main pipeline
- `demo_context.json` sample input format aligned with the bundled sample data
- `data/toy-91/context.json` validated sample context used for the included toy/object dataset
- `run_demo.py` CLI entrypoint
- `evaluate.py` simple evaluator for prediction JSON against GT JSON

## Input format
```json
{
  "context": [
    {
      "class": 1,
      "class_name": "Stuffed Bear Blue SB-1-B",
      "refer_image": ["refer_images/1-01.jpg"]
    },
    {
      "class": 2,
      "class_name": "Pocket Tomika P060",
      "refer_image": ["refer_images/2-01.jpg"]
    }
  ]
}
```

Notes:
- `refer_image` paths are resolved relative to the context JSON file location.
- `class_name` is optional in schema terms, but recommended because it improves text matching and makes outputs readable.
- The bundled sample dataset lives under `data/toy-91/`, so `data/toy-91/context.json` references `refer_images/...` relative to that directory.

## Output format
```json
{
  "image": "cars-on-road.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "Tesla Model X",
      "score": 0.83,
      "similarity": 0.79,
      "proposal_score": 0.87
    }
  ]
}
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Using the bundled sample dataset:
```bash
python run_demo.py \
  --context data/toy-91/context.json \
  --query data/toy-91/query_images/019e588b-a94a-4d91-a6e2-17d9fdd17c45.jpg \
  --output outputs/predictions.json \
  --vis outputs/vis.jpg
```

If you want to use `demo_context.json`, place it beside a matching `refer_images/` directory or update the paths to fit your dataset layout.

## Evaluation
`evaluate.py` compares prediction JSON against a GT JSON file using class-aware IoU matching.

Minimal GT format:
```json
{
  "image": "data/toy-91/query_images/019e588b-a94a-4d91-a6e2-17d9fdd17c45.jpg",
  "detections": [
    {
      "bbox": [100, 258, 261, 399],
      "class": "Stuffed Bear Blue SB-1-B"
    }
  ]
}
```

Example:
```bash
python evaluate.py \
  --pred outputs/smoke/pred.json \
  --gt sample_gt.json \
  --iou 0.5
```

## Notes
- GroundingDINO still needs a prompt to propose objects. This prototype uses a generic prompt like `object . vehicle . product . item .` to maximize recall.
- For some domains, proposal recall may be the limiting factor.
- On newer `transformers` versions, CLIP feature helpers may return model output objects rather than raw tensors; the current implementation handles that.
- If you want a stronger system later, likely upgrades are OWLv2 / GroundingDINO 1.5 / SAM2-assisted regioning / SigLIP2 embeddings.
