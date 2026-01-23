# Docker LLM Integration - Implementation Summary

## Overview

Snowcrash now uses Docker containers running `llama-server` by default, with automatic fallback to `llama-cpp-python` if Docker is unavailable. This provides better memory management and isolation.

## What Changed

### New Files Created

1. **`agent/docker_llm.py`**
   - `DockerLLMAdapter` class - LangChain-compatible wrapper for llama-server HTTP API
   - Connects to `llama-server` running in Docker containers
   - Implements same interface as `LlamaCpp` (drop-in replacement)

2. **`agent/container_manager.py`**
   - `ContainerManager` class - Manages Docker container lifecycle
   - Starts/stops containers for each model
   - Port assignment: llama=8080, phi-3=8081, gemma=8082
   - Health checks to ensure server is ready

### Modified Files

1. **`agent/simple_agent.py`**
   - Added `use_docker` parameter (default: `True`)
   - Added `docker_port` and `container_manager` parameters
   - Tries Docker first, falls back to llama-cpp-python automatically
   - No changes needed to existing code that uses `SimpleAgent`

2. **`main.py`**
   - Creates `ContainerManager` before initializing agent
   - Passes container manager to `SimpleAgent`
   - Registers cleanup handlers to stop containers on exit
   - Handles SIGINT/SIGTERM for graceful shutdown

3. **`requirements.txt`**
   - Added `httpx>=0.24.0` for HTTP client (used by DockerLLMAdapter)

## How It Works

### Startup Flow

1. `main.py` creates `ContainerManager`
2. `SimpleAgent` tries to start Docker container for selected model
3. Container runs `llama-server` with conservative memory settings (`-ngl 8 -c 512`)
4. `DockerLLMAdapter` connects to HTTP API (`http://localhost:PORT/v1/chat/completions`)
5. If Docker fails, automatically falls back to `llama-cpp-python`

### Memory Settings

**Docker containers use conservative settings to leave room for YOLO:**
- `--n-gpu-layers 8` (instead of 15)
- `--ctx-size 512` (instead of 1024)
- `--batch-size 128` (instead of 256)
- `--memory="2g"` (Docker memory limit)

This leaves ~7GB free for YOLO and system processes.

### Port Assignment

- **llama**: Port 8080
- **phi-3**: Port 8081
- **gemma**: Port 8082

## Usage

### Default (Docker Mode)

```bash
# Just run as before - Docker is now default
python3 main.py --model llama --gui-viewer chatgui
```

The system will:
1. Start Docker container automatically
2. Use llama-server HTTP API
3. Fall back to llama-cpp-python if Docker fails

### Force llama-cpp-python (Disable Docker)

To use llama-cpp-python instead of Docker, modify `main.py`:

```python
agent = SimpleAgent(
    model_type=args.model,
    use_docker=False,  # Disable Docker
    ...
)
```

### Manual Container Management

```python
from agent.container_manager import ContainerManager

# Create manager
manager = ContainerManager()

# Start container for llama
port = manager.start_model_container("llama")
# Returns: 8080

# Check if running
if manager.is_container_running("llama"):
    print("Container is running")

# Stop container
manager.stop_model_container("llama")

# Stop all containers
manager.stop_all_containers()
```

## Container Details

### Container Names
- `snowcrash-llama-server`
- `snowcrash-phi3-server`
- `snowcrash-gemma-server`

### Model Paths (Inside Container)
- llama: `/models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- phi-3: `/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf`
- gemma: `/models/gemma2b/gemma-3n-E2B-it-Q4_K_M.gguf`

### Docker Command (Generated Automatically)

```bash
docker run -d \
  --name snowcrash-llama-server \
  --runtime nvidia \
  --network host \
  --shm-size 2g \
  --memory 2g \
  -v /home/ordun/Documents/snowcrash/models:/models:ro \
  --rm \
  <image-tag> \
  llama-server \
  -m /models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --port 8080 \
  --n-gpu-layers 8 \
  --ctx-size 512 \
  --batch-size 128
```

## Error Handling

### Docker Not Available
- Automatically falls back to llama-cpp-python
- Prints warning message
- Continues normally

### Container Start Fails
- Falls back to llama-cpp-python
- Prints error message
- Continues normally

### HTTP API Errors
- Retries handled by httpx client
- Timeout: 60 seconds
- Clear error messages

## Cleanup

Containers are automatically stopped when:
- `main.py` exits normally
- SIGINT (Ctrl+C) is received
- SIGTERM is received
- Python process terminates

## Benefits

1. **Better Memory Management** - Containers handle fragmentation
2. **Isolation** - LLM crashes don't kill main process
3. **Production Ready** - Uses wrapper script approach
4. **Backward Compatible** - Falls back to llama-cpp-python
5. **YOLO Compatible** - Conservative memory settings leave room for YOLO

## Testing

After transferring to Orin:

```bash
# Test Docker mode
python3 main.py --model llama --gui-viewer chatgui

# Check containers
docker ps

# Check logs
docker logs snowcrash-llama-server

# Test HTTP API directly
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}]}'
```

## Troubleshooting

### Container Won't Start
- Check Docker is installed: `docker --version`
- Check models directory exists: `ls /home/ordun/Documents/snowcrash/models`
- Check autotag works: `autotag llama_cpp`
- Check logs: `docker logs snowcrash-llama-server`

### HTTP API Not Responding
- Wait a few seconds after container starts
- Check port is correct: `netstat -tuln | grep 8080`
- Check container is running: `docker ps`

### Memory Issues
- Containers use `-ngl 8` by default (conservative)
- Can reduce further: modify `container_manager.py`
- Check memory: `tegrastats`

## Files Summary

**New:**
- `agent/docker_llm.py` - HTTP API adapter
- `agent/container_manager.py` - Container management
- `DOCKER_INTEGRATION.md` - This file

**Modified:**
- `agent/simple_agent.py` - Docker support added
- `main.py` - Container startup/cleanup
- `requirements.txt` - Added httpx

**No Changes Needed:**
- `tools/tracking_viewer_chatgui.py` - Works automatically
- `tools/tracking_viewer_audiogui.py` - Works automatically
- All other files - Backward compatible

