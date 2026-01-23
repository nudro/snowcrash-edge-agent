#!/bin/bash
# Script to run Jetson Inference service on old-nano

# Default values
CAMERA_DEVICE=0
HOST="0.0.0.0"
PORT=8081  # Changed from 8080 to avoid conflict with other services
JETSON_INFERENCE_PATH="~/Documents/jetson-inference"

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
        --jetson-inference-path)
            JETSON_INFERENCE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device NUM] [--host HOST] [--port PORT] [--jetson-inference-path PATH]"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting Jetson Inference Detection Service..."
echo "  Camera: /dev/video${CAMERA_DEVICE}"
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  Jetson Inference: ${JETSON_INFERENCE_PATH}"
echo ""

python3 tools/jetson_inference_service.py \
    --device ${CAMERA_DEVICE} \
    --host ${HOST} \
    --port ${PORT} \
    --jetson-inference-path "${JETSON_INFERENCE_PATH}"

