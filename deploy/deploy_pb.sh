#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  ParsingBloom • Unified launcher
#
#  ./deploy/deploy_pb.sh                   → single run (default)
#  ./deploy/deploy_pb.sh --runs 5          → 5-rep determinism test
#  ./deploy/deploy_pb.sh --gpus             Use GPU
#  ./deploy/deploy_pb.sh --out-dir DIR      Override output dir
#  ./deploy/deploy_pb.sh --schedule hourly  Daemon mode (runs=1 only)
#  -h, --help                                Show this help and exit
# ─────────────────────────────────────────────────────────────

HF_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
HF_MODEL_DIR="models/Llama-3.2-3B-Instruct"
CONDA_ENV=""

# CLI defaults
SCHEDULE_MODE=""
USE_GPU=0
RUNS=1
USER_OUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [--gpus] [--runs N] [--out-dir DIR] [--schedule hourly|daily|cron] [EXTRA_ARGS...]

  --gpus             Use GPU (sets PARSINGBLOOM_DEVICE=cuda)
  --runs N           Number of replicates (default: 1)
  --out-dir DIR      Base output directory (default: data/ if --runs=1; data/pipeline_executes/ if >1)
  --schedule MODE    Daemon mode (only valid when --runs=1): hourly, daily, or cron per config
  -h, --help         Show this help and exit

Any EXTRA_ARGS are forwarded verbatim to the pipeline_execute script.
EOF
  exit 0
}

# ─── Parse flags ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) USE_GPU=1; EXTRA_ARGS+=("--gpus"); shift ;;
    --runs)      RUNS="${2:-}"; shift 2 ;;
    --out-dir)   USER_OUT_DIR="${2:-}"; shift 2 ;;
    --schedule)  SCHEDULE_MODE="${2:-}"; shift 2 ;;
    -h|--help)   usage ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done


#Determinism test directory is made use of when argument --runs is greater than 1. This choice of runs is made to ensure that the test is run multiple times to check for self-consistency in the results.
# ─── Decide out‐dir ─────────────────────────────────────────────────────────
if [[ -n "$USER_OUT_DIR" ]]; then
  OUT_DIR="$USER_OUT_DIR"
else
  if (( RUNS <= 1 )); then
    OUT_DIR="data"
  else
    OUT_DIR="data/determinism_tests"
  fi
fi

# ─── Optionally activate Conda ─────────────────────────────────────────────
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# ─── Ensure Poetry & deps ──────────────────────────────────────────────────
echo "Ensuring Poetry is installed …"
command -v poetry >/dev/null 2>&1 || \
  curl -sSL https://install.python-poetry.org | python3 -
echo "Installing project dependencies …"
poetry install --no-interaction

# ─── Download / link HF model ───────────────────────────────────────────────
if [[ -d "$HF_MODEL_DIR" ]]; then
  echo "Model already present – skipping download."
else
  CACHE_DIR="${HF_HOME:-$HOME/.cache}/huggingface/hub"
  if ls "$CACHE_DIR"/models--${HF_MODEL_ID//\//--}*/ &>/dev/null; then
    echo "🔗  Found model in HF cache – symlinking."
    mkdir -p "$(dirname "$HF_MODEL_DIR")"
    ln -s "$CACHE_DIR"/models--${HF_MODEL_ID//\//--}* "$HF_MODEL_DIR"
  else
    echo "Downloading model …"
    poetry run huggingface-cli download "$HF_MODEL_ID" \
      --local-dir "$HF_MODEL_DIR" --local-dir-use-symlinks False
  fi
fi

# ─── One-off HF token into keyring ──────────────────────────────────────────
python - <<'PY'
import keyring, os, getpass
svc="email_agent"
if keyring.get_password(svc,"hf_api_token") is None:
    tok=os.getenv("HF_API_TOKEN") or getpass.getpass("Enter your HF token: ")
    keyring.set_password(svc,"hf_api_token",tok)
PY

# ─── CPU/GPU override ──────────────────────────────────────────────────────
if (( USE_GPU )); then
  export PARSINGBLOOM_DEVICE="cuda"
  echo "⚡  Using GPU (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-ALL})"
else
  export CUDA_VISIBLE_DEVICES=""
  export PARSINGBLOOM_DEVICE="cpu"
  echo "🔸  Forcing CPU (CUDA disabled)"
fi

# ─── Gmail OAuth pre-flight ─────────────────────────────────────────────────
echo "Ensuring Gmail authentication…"
poetry run python - <<'PYCODE'
import os, importlib
from core.config_loader import load_config
from google.auth.exceptions import RefreshError

cfg = load_config()
def_name = cfg.default_connector
conn_cfg = cfg.connectors[def_name]
mod = importlib.import_module(f"connectors.{def_name}")
Connector = getattr(mod, f"{def_name.capitalize()}Connector")
token_file = conn_cfg.token_path

try:
    Connector(conn_cfg)
except RefreshError:
    if token_file and os.path.exists(token_file):
        os.remove(token_file)
    Connector(conn_cfg)
PYCODE

# ─── Define core run function ───────────────────────────────────────────────
run_once() {
  echo "🚀  Running pipeline_execute.py (runs=$RUNS → out=$OUT_DIR)…"
  poetry run python src/pipeline/pipeline_execute.py \
    --runs "$RUNS" --out-dir "$OUT_DIR" "${EXTRA_ARGS[@]}"

  if (( RUNS > 1 )); then
    echo "📊  Generating plots…"
    poetry run python src/pipeline.determinism_plot.py --dir "$OUT_DIR"
  fi
}

# ─── Dispatch: schedule vs one‐off ─────────────────────────────────────────
if [[ -n "$SCHEDULE_MODE" ]]; then
  if (( RUNS > 1 )); then
    echo "⚠️  Scheduling mode only supports --runs 1. To batch tests, invoke without --schedule."
    exit 1
  fi

  echo "⏲️  Starting scheduler ($SCHEDULE_MODE mode)…"
  # legacy scheduler still calls run_pipeline (identical to your old daemon)
  poetry run python -m src/pipeline.scheduler
else
  run_once
fi