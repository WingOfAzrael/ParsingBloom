#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  ParsingForge • Local launcher
#
#   ./deploy/local_deploy.sh                  → run once on CPU
#   ./deploy/local_deploy.sh --gpus           → run once on GPU
#   ./deploy/local_deploy.sh --schedule daily → daily scheduler (CPU)
#   ./deploy/local_deploy.sh --schedule hourly --gpus
#                                             → hourly scheduler (GPU)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Settings you rarely change ──────────────────────────────
HF_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
HF_MODEL_DIR="models/Llama-3.2-3B-Instruct"
CONDA_ENV=""            # leave blank unless you rely on conda

# ─── CLI flags ───────────────────────────────────────────────
SCHEDULE_MODE=""        # empty → run once
USE_GPU=0
EXTRA_PY_ARGS=()

usage () {
  cat <<EOF
Usage: $0 [--gpus] [--schedule hourly|daily|cron] [EXTRA_ARGS...]

  --gpus            Use the GPU (sets USE_GPU=1 and leaves CUDA_VISIBLE_DEVICES as-is)
  --schedule MODE   Start the APScheduler loop (hourly, daily or cron per config)
  -h, --help        Show this help and exit

Any EXTRA_ARGS are forwarded verbatim to pipeline.run_pipeline.
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)     USE_GPU=1; shift;;
    --schedule) SCHEDULE_MODE="${2:-}"; shift 2;;
    -h|--help)  usage;;
    *)          EXTRA_PY_ARGS+=("$1"); shift;;
  esac
done

# ─── Activate conda if you use one (optional) ────────────────
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ─── Poetry bootstrap --------------------------------------------------------
echo "Ensuring Poetry is installed …"
command -v poetry >/dev/null 2>&1 || \
  curl -sSL https://install.python-poetry.org | python3 -

echo "Installing project dependencies …"
poetry install --no-interaction

# ─── Download / link the HF model (cache-aware)  --------------
if [[ -d "$HF_MODEL_DIR" ]]; then
  echo "Model already present – skipping download."
else
  CACHE_DIR="${HF_HOME:-$HOME/.cache}/huggingface/hub"
  if ls "$CACHE_DIR"/models--${HF_MODEL_ID//\//--}*/ > /dev/null 2>&1; then
    echo "🔗  Found model in HF cache – symlinking."
    mkdir -p "$(dirname "$HF_MODEL_DIR")"
    ln -s "$CACHE_DIR"/models--${HF_MODEL_ID//\//--}* "$HF_MODEL_DIR"
  else
    echo "Downloading model …"
    poetry run huggingface-cli download "$HF_MODEL_ID" \
      --local-dir "$HF_MODEL_DIR" --local-dir-use-symlinks False
  fi
fi

# ─── One-off HF token into keyring if missing -----------------
python - <<'PY'
import keyring, os, getpass, sys
svc="email_agent"
if keyring.get_password(svc,"hf_api_token") is None:
    tok=os.getenv("HF_API_TOKEN") or getpass.getpass("Enter your HF token: ")
    keyring.set_password(svc,"hf_api_token",tok)
PY

# ─── CPU/GPU override logic -----------------------------------
# If the code honours PARSINGFORGE_DEVICE env it will win; otherwise
# the user still needs to set parser.device/device_map in config.yaml.
if [[ "$USE_GPU" -eq 1 ]]; then
  export PARSINGFORGE_DEVICE="cuda"          # pipeline can read this if implemented
  echo "⚡  Using GPU (CUDA visible: ${CUDA_VISIBLE_DEVICES:-ALL})."
else
  export CUDA_VISIBLE_DEVICES=""
  export PARSINGFORGE_DEVICE="cpu"
  echo "🔸  Forcing CPU (CUDA disabled)."
fi

# ─── Decide which Python entry point to call ------------------
if [[ -n "$SCHEDULE_MODE" ]]; then
  echo "⏲️  Starting scheduler ($SCHEDULE_MODE mode) …"
  poetry run python -m pipeline.scheduler "${EXTRA_PY_ARGS[@]}"
else
  echo "🚀  Running pipeline once …"
  poetry run python -m pipeline.run_pipeline "${EXTRA_PY_ARGS[@]}"
fi