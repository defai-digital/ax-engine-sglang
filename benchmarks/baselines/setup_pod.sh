#!/usr/bin/env bash
# One-time setup on a RunPod H200/B200 pod (Ubuntu + CUDA image). Run from the repo root.
# Puts the venv and HF cache on /workspace (the persistent network volume).
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf}
VENV=${VENV:-/workspace/venv-phase0}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}

nvidia-smi
command -v uv >/dev/null || pip install -q uv
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install "vllm>=0.19.0" --torch-backend=auto
uv pip install -r benchmarks/baselines/requirements.txt

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, torch.version.cuda)"

# ~72 GB download. Set HF_TOKEN if you hit rate limits.
hf download "$MODEL" --exclude "*.pth"

# Warm the eval datasets so runs do not depend on the network mid-benchmark.
python - <<'EOF'
from datasets import load_dataset
load_dataset("openai/openai_humaneval", split="test")
load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
load_dataset("openai/gsm8k", "main", split="test")
print("datasets cached")
EOF

cat <<EOF

Setup done. In each new shell:
  export HF_HOME=$HF_HOME && source $VENV/bin/activate
Quick smoke test:
  MODES="ar mtp2" LIMIT=20 CONCURRENCY="1 8" SAMPLED=0 benchmarks/baselines/run_all.sh
Full sweep:
  benchmarks/baselines/run_all.sh
EOF
