#!/bin/bash
# Script to run Jetson Inference Service in Docker on old-nano

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root so Docker build context is correct
cd "$PROJECT_ROOT"

# Default values
CAMERA_DEVICE=0
HOST="0.0.0.0"
PORT=9000
JETSON_INFERENCE_PATH="/usr/local/bin"  # In Docker, jetson-inference is in /usr/local/bin

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            CAMERA_DEVICE="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device NUM] [--host HOST] [--port PORT]"
            exit 1
            ;;
    esac
done

echo "Building Jetson Inference Service Docker image..."
echo "Project root: $PROJECT_ROOT"
docker build -f tools/Dockerfile.jetson_inference -t jetson-inference-service:latest .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

echo "Starting Jetson Inference Service container..."
docker run -d \
    --name jetson-inference-service \
    --runtime nvidia \
    --network host \
    --device /dev/video${CAMERA_DEVICE}:/dev/video${CAMERA_DEVICE} \
    -e CAMERA_DEVICE=${CAMERA_DEVICE} \
    -e HOST=${HOST} \
    -e PORT=${PORT} \
    -e JETSON_INFERENCE_PATH=/usr/local/bin \
    jetson-inference-service:latest \
    python3 /app/jetson_inference_service.py \
        --device ${CAMERA_DEVICE} \
        --host ${HOST} \
        --port ${PORT} \
        --jetson-inference-path /usr/local/bin

echo ""
echo "Service started!"
echo "View logs: docker logs -f jetson-inference-service"
echo "Stop service: docker stop jetson-inference-service"
echo "Remove container: docker rm jetson-inference-service"

