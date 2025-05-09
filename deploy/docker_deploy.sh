#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# parsingforge • Deployment helper
#
#   ./docker_deploy.sh                 → run once, CPU, foreground
#   ./docker_deploy.sh --gpus          → run once, GPU, foreground
#   ./docker_deploy.sh --schedule daily -d
#                                      → daily scheduler, CPU, detached
#   ./docker_deploy.sh --schedule hourly --gpus -d
#                                      → hourly scheduler, GPU, detached
# ────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_CPU="parsingforge:latest"
IMAGE_GPU="parsingforge-gpu:latest"
DOCKERFILE_CPU="Dockerfile"
DOCKERFILE_GPU="Dockerfile.gpu"
CONTAINER_NAME="parsingforge"

DETACH=""          # will hold "-d" if requested
GPU_FLAGS=""       # "--gpus all" if requested
IMAGE="$IMAGE_CPU"
DOCKERFILE="$DOCKERFILE_CPU"
SCHEDULE_MODE=""   # empty → run once
RM_FLAG="--rm"     # auto-remove container when it exits (run-once only)

usage () {
  cat <<EOF
Usage: $0 [OPTIONS]

Options
  --schedule [hourly|daily|cron]  Run the APScheduler loop instead of a single scrape
  -d, --detach                    Start container in the background
  --gpus                          Build and run the GPU image (adds --gpus all)
  -h, --help                      Show this message

If --schedule is omitted, the pipeline runs once and the container is removed
automatically when it finishes.
EOF
  exit 1
}

# ─── Parse CLI args ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedule)
      [[ $# -lt 2 ]] && { echo "Missing value for --schedule"; usage; }
      SCHEDULE_MODE="$2"
      shift 2
      ;;
    -d|--detach)
      DETACH="-d"
      shift
      ;;
    --gpus)
      IMAGE="$IMAGE_GPU"
      DOCKERFILE="$DOCKERFILE_GPU"
      GPU_FLAGS="--gpus all"
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

# ─── Build image ────────────────────────────────────────────────
echo "Building Docker image ($IMAGE) from $DOCKERFILE …"
docker build -t "$IMAGE" -f "$DOCKERFILE" .

# ─── Decide command to run inside container ─────────────────────
if [[ -n "$SCHEDULE_MODE" ]]; then
  CMD="python -m pipeline.scheduler"
  CONTAINER_NAME="${CONTAINER_NAME}-sched"
  RM_FLAG=""             # keep the container; it runs indefinitely
else
  CMD="python -m pipeline.run_pipeline"
fi

# ─── Launch container ───────────────────────────────────────────
echo "Starting container '$CONTAINER_NAME' …"
docker run $DETACH $RM_FLAG --name "$CONTAINER_NAME" \
  $GPU_FLAGS \
  -e HF_API_TOKEN="${HF_API_TOKEN:?please export HF_API_TOKEN}" \
  -v "$PWD/config":/app/config \
  -v "$PWD/models":/app/models \
  "$IMAGE" bash -c "$CMD"

if [[ -z "$DETACH" ]]; then
  echo "Finished."
else
  echo "Container '$CONTAINER_NAME' is now live."
  echo "Stop with: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
fi