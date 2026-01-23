#!/bin/bash
# Script to run YOLO Detection Service in Docker on old-nano

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root so Docker build context is correct
cd "$PROJECT_ROOT"

# Default values
MODEL_PATH="/home/ordun/Documents/snowcrash/yolov8n.pt"
DEVICE=0
HOST="0.0.0.0"
PORT=8080
CONFIDENCE=0.25

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model PATH] [--device NUM] [--host HOST] [--port PORT] [--confidence FLOAT]"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

# Get absolute path and directory
MODEL_PATH=$(realpath "$MODEL_PATH")
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_FILE=$(basename "$MODEL_PATH")

echo "Building YOLO Detection Service Docker image..."
echo "Project root: $PROJECT_ROOT"
docker build -f tools/Dockerfile.yolo_service -t yolo-detection-service:latest .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

echo "Starting YOLO Detection Service container..."
docker run -d \
    --name yolo-detection-service \
    --runtime nvidia \
    --network host \
    --device /dev/video${DEVICE}:/dev/video${DEVICE} \
    -v ${MODEL_DIR}:/models:ro \
    -e MODEL_PATH=/models/${MODEL_FILE} \
    -e CAMERA_DEVICE=${DEVICE} \
    -e HOST=${HOST} \
    -e PORT=${PORT} \
    -e CONFIDENCE=${CONFIDENCE} \
    yolo-detection-service:latest \
    python3 /app/tools/yolo_detection_service.py \
        --model /models/${MODEL_FILE} \
        --device ${DEVICE} \
        --host ${HOST} \
        --port ${PORT} \
        --confidence ${CONFIDENCE}

echo ""
echo "Service started!"
echo "View logs: docker logs -f yolo-detection-service"
echo "Stop service: docker stop yolo-detection-service"
echo "Remove container: docker rm yolo-detection-service"

