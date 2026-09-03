#!/usr/bin/env bash
# One-shot setup for the shared 4x RTX 6000 Ada training box (Linux). Run from the repo root.
#   bash scripts/remote_setup.sh && uv run python scripts/smoke_test.py
#
# GPU POLICY: GPU 0 is FORBIDDEN. GPU 3 is the default for all carparts work. GPUs 1-2 only when idle (HANDOFF.md).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
case ",$CUDA_VISIBLE_DEVICES," in *,0,*) echo "GPU 0 is forbidden on this box (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 1;; esac
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (add 'export CUDA_VISIBLE_DEVICES=3' to your shell profile)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# Python 3.12 venv with CUDA 12.6 wheels (pyproject pins the PyTorch index). Add extras as needed.
uv sync --extra serve --extra kaggle --extra roboflow --extra dev

# Credentials: copy .env.example -> .env and fill KAGGLE_*/ROBOFLOW_API_KEY (or export them in the shell).
[ -f .env ] || cp .env.example .env

# Sanity: the policy in carparts/gpu_policy.py must see exactly the GPU(s) you intend to use.
uv run python - <<'EOF'
import carparts, torch
print("torch", torch.__version__, "| visible GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF

# Data: pulls carparts-seg (public), Kaggle (anonymous OK) and Roboflow (key) and builds all recipes.
uv run python scripts/prepare_data.py --all

cat <<'EOF'

Next (all on GPU 3 — see HANDOFF.md):
  uv run python scripts/smoke_test.py
  uv run python scripts/train.py    --recipe exterior_seg   --config rfdetr_48gb --name exterior_seg_xl
  uv run python scripts/train.py    --recipe engine_bay_det --config rfdetr_48gb --model rfdetr-large --name engine_bay_l
  uv run python scripts/evaluate.py --run exterior_seg_xl
  uv run python scripts/export.py   --run exterior_seg_xl --format onnx
  uv run python app/server.py       --run exterior_seg_xl --run engine_bay_l --host 0.0.0.0 --port 8000
EOF
