# Remote quickstart

cd ~/workspace/fewshot-context-detector
source .venv/bin/activate

# GPU availability check
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY

# Run example
python run_demo.py \
  --context demo_context.json \
  --query path/to/query.jpg \
  --output outputs/predictions.json \
  --vis outputs/vis.jpg
