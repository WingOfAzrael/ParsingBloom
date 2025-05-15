#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# ParsingBloom • Docker deployment helper
#
#  ./deploy_pb_docker.sh                   → run once, CPU, foreground
#  ./deploy_pb_docker.sh --gpus            → run once, GPU, foreground
#  ./deploy_pb_docker.sh --runs 5           → 5-rep determinism test
#  ./deploy_pb_docker.sh --out-dir DIR      → custom output dir
#  ./deploy_pb_docker.sh --schedule daily -d → daily scheduler, CPU, detached
#  ./deploy_pb_docker.sh --schedule hourly --gpus -d
#                                          → hourly scheduler, GPU, detached
# ────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_CPU="parsingbloom:latest"
IMAGE_GPU="parsingbloom-gpu:latest"
DOCKERFILE_CPU="Dockerfile"
DOCKERFILE_GPU="Dockerfile.gpu"
CONTAINER_BASE="parsingbloom"

# ─── CLI defaults ──────────────────────────────────────────────────────
USE_GPU=0
RUNS=1
USER_OUT_DIR=""
SCHEDULE_MODE=""
DETACH=""
RM_FLAG="--rm"     # remove container after single-run exit
GPU_FLAGS=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --gpus                         Use GPU image & flags
  --runs N                       Number of replicates (default: 1)
  --out-dir DIR                  Override base output directory
  --schedule [hourly|daily|cron] Run APScheduler loop instead (runs=1 only)
  -d, --detach                   Start container in background
  -h, --help                     Show this help and exit

If --schedule is omitted, the pipeline runs once and the container is
auto-removed when it finishes (unless -d is also given).
EOF
  exit 1
}

# ─── Parse flags ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      USE_GPU=1
      IMAGE="$IMAGE_GPU"
      DOCKERFILE="$DOCKERFILE_GPU"
      GPU_FLAGS="--gpus all"
      shift
      ;;
    --runs)
      RUNS="${2:-}"; shift 2
      ;;
    --out-dir)
      USER_OUT_DIR="${2:-}"; shift 2
      ;;
    --schedule)
      SCHEDULE_MODE="${2:-}"; shift 2
      # scheduling mode only ever makes sense for single runs
      ;;
    -d|--detach)
      DETACH="-d"
      RM_FLAG=""     # keep container alive for scheduler
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ─── Compute default OUT_DIR if not overridden ─────────────────────────
if [[ -n "$USER_OUT_DIR" ]]; then
  OUT_DIR="$USER_OUT_DIR"
else
  if (( RUNS <= 1 )); then
    OUT_DIR="data"
  else
    OUT_DIR="data/determinism_tests"
  fi
fi

# ─── Build the Docker image ───────────────────────────────────────────
echo "Building image ($IMAGE) from $DOCKERFILE …"
docker build -t "$IMAGE" -f "$DOCKERFILE" .

# ─── Choose internal command & container name ─────────────────────────
if [[ -n "$SCHEDULE_MODE" ]]; then
  # scheduler loop: only valid when RUNS=1
  if (( RUNS > 1 )); then
    echo "⚠️  Scheduling only supports single runs. Drop --runs for schedule mode."
    exit 1
  fi
  CMD="python -m pipeline.scheduler"
  CONTAINER_NAME="${CONTAINER_BASE}-sched"
  # keep container alive => already unset RM_FLAG
else
  CMD="python -m src/pipeline.pipeline_execute --runs $RUNS --out-dir $OUT_DIR ${EXTRA_ARGS[*]}"
  CONTAINER_NAME="$CONTAINER_BASE"
fi

# ─── Run the container ────────────────────────────────────────────────
echo "Starting container '$CONTAINER_NAME' …"
# propagate PARSINGBLOOM_DEVICE for CPU/GPU selection
PARSINGBLOOM_DEVICE=$( ((USE_GPU)) && echo "cuda" || echo "cpu" )

docker run $DETACH $RM_FLAG --name "$CONTAINER_NAME" \
  $GPU_FLAGS \
  -e HF_API_TOKEN="${HF_API_TOKEN:?please export HF_API_TOKEN}" \
  -e PARSINGBLOOM_DEVICE="$PARSINGBLOOM_DEVICE" \
  -v "$PWD/config":/app/config \
  -v "$PWD/models":/app/models \
  "$IMAGE" bash -c "$CMD"

if [[ -z "$DETACH" ]]; then
  echo "Finished."
else
  echo "Container '$CONTAINER_NAME' is now live."
  echo "Stop with: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
fi
