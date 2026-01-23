# Running YOLO Detection Service in Docker (old-nano)

## Overview

Due to Python/JetPack compatibility issues on old-nano (Jetson Nano with Python 3.6), the YOLO detection service should be run inside a Docker container using the official Ultralytics image.

## Prerequisites

- Docker installed on old-nano
- NVIDIA Container Toolkit configured for Jetson
- Camera connected at `/dev/video0`

## Quick Start

### 1. Build Docker Image

On old-nano:
```bash
cd ~/Documents/snowcrash
docker build -f tools/Dockerfile.yolo_service -t yolo-detection-service:latest .
```

### 2. Run Service (using script)

```bash
./tools/run_yolo_service_docker.sh
```

Or manually:
```bash
docker run -d \
    --name yolo-detection-service \
    --runtime nvidia \
    --network host \
    --device /dev/video0:/dev/video0 \
    -v ~/Documents/snowcrash:/models:ro \
    yolo-detection-service:latest \
    python3 /app/tools/yolo_detection_service.py \
        --model /models/yolov8n.pt \
        --device 0 \
        --host 0.0.0.0 \
        --port 8080
```

### 3. Check Status

```bash
# View logs
docker logs -f yolo-detection-service

# Check if running
docker ps | grep yolo-detection-service
```

### 4. Stop Service

```bash
docker stop yolo-detection-service
docker rm yolo-detection-service
```

## Manual Docker Run (with custom options)

```bash
docker run -d \
    --name yolo-detection-service \
    --runtime nvidia \
    --network host \
    --device /dev/video0:/dev/video0 \
    -v ~/Documents/snowcrash:/models:ro \
    yolo-detection-service:latest \
    python3 /app/tools/yolo_detection_service.py \
        --model /models/yolov8n.pt \
        --device 0 \
        --host 0.0.0.0 \
        --port 8080 \
        --confidence 0.25
```

## Troubleshooting

### Container fails to start
```bash
# Check logs
docker logs yolo-detection-service

# Check if camera device is accessible
ls -l /dev/video0
```

### Camera access denied
```bash
# Make sure camera device is passed correctly
docker run ... --device /dev/video0:/dev/video0 ...
```

### Model not found
```bash
# Verify model path is mounted correctly
docker exec yolo-detection-service ls -l /models/yolov8n.pt
```

## Benefits of Docker Approach

1. **Python compatibility** - Uses Python 3.8+ from Ultralytics image
2. **Pre-installed dependencies** - Ultralytics, OpenCV, etc. already included
3. **Isolation** - Doesn't affect system Python installation
4. **Easy updates** - Rebuild image when needed

