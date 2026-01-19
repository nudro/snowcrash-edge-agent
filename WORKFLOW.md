# Snowcrash Workflow Guide

This document outlines the step-by-step workflow for setting up and deploying Snowcrash on the Old Nano.

## Setup on DGX Spark

### Step 1: Initial Setup

```bash
cd /home/nudro/Documents/snowcrash

# Run setup script (creates venv, installs dependencies)
./scripts/setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download Models

Download all three SLM models (Phi-3 Mini, Llama 3.2, Gemma 2B):

```bash
python scripts/download_models.py
```

This will download GGUF quantized models into:
- `models/phi3-mini/`
- `models/llama3.2/`
- `models/gemma2b/`

### Step 3: Measure VRAM Requirements

Check if models fit within Old Nano constraints (8GB RAM + 1978 MB SWAP):

```bash
python scripts/measure_vram.py
```

This will:
- Calculate file sizes for each model
- Estimate VRAM usage during inference
- Verify if models fit in RAM or require SWAP
- Show SWAP usage if needed

### Step 4: Package for Transfer

Create compressed package for efficient transfer:

```bash
python scripts/package_for_transfer.py
```

This creates a timestamped tar.gz file in `package/` directory.

## Transfer to Old Nano

### Step 5: Transfer Package via SCP

```bash
# Transfer package to Old Nano
scp package/snowcrash-YYYYMMDD-HHMMSS.tar.gz old-nano:~/snowcrash/

# SSH into Old Nano
ssh old-nano
```

### Step 6: Setup on Old Nano

On Old Nano:

```bash
cd ~/snowcrash

# Extract package
tar -xzf snowcrash-YYYYMMDD-HHMMSS.tar.gz

# Install dependencies (may need to adapt for ARM/Ubuntu 18.04)
pip3 install -r requirements.txt

# If llama-cpp-python needs compilation for ARM:
pip3 install llama-cpp-python --no-cache-dir

# Install YOLO model (will auto-download on first use)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Running Snowcrash

### Step 7: Start MCP Server

On Old Nano:

```bash
cd ~/snowcrash
python3 mcp_server/server.py
```

The MCP server will:
- Load models from `models/` directory
- Provide YOLO object detection tool
- Expose agentic capabilities via MCP protocol

## Development Workflow

### Testing Models Locally (DGX Spark)

Test models before transferring:

```bash
# Test Phi-3 Mini
python3 -c "from llama_cpp import Llama; llm = Llama(model_path='models/phi3-mini/Phi-3-mini-4k-instruct-q4_K_M.gguf'); print(llm('Hello, world!'))"
```

### Adding New Tools

1. Create tool in `tools/` directory
2. Register in `mcp_server/server.py`
3. Test locally
4. Re-package and transfer

## Packaging Strategy

The packaging script (`package_for_transfer.py`):
- ✅ Includes models (GGUF files)
- ✅ Includes MCP server code
- ✅ Includes tools (YOLO, etc.)
- ✅ Includes scripts and requirements
- ❌ Excludes `.git`, `__pycache__`, `.pyc` files
- ❌ Excludes venv and test files

**Efficiency:**
- Models are already compressed (GGUF format)
- tar.gz provides additional compression (~10-30% reduction)
- Total package size should be reasonable for SCP transfer

## Troubleshooting

### Model Download Issues
- Check internet connection
- Verify HuggingFace URLs are accessible
- Models are large (1-5 GB each), allow time for download

### VRAM Measurement
- Estimates are conservative
- Actual usage depends on context window, batch size
- Monitor with `tegrastats` during inference

### Transfer Issues
- Large packages may timeout with SCP
- Use `scp -C` for compression during transfer
- Or transfer models separately if needed

### Old Nano Setup
- Ubuntu 18.04 may need older package versions
- Compile llama-cpp-python from source if wheels unavailable
- May need to adjust CUDA paths for older JetPack

## Next Steps

After initial setup:
1. Test each model individually
2. Benchmark performance (tokens/sec)
3. Add more tools to MCP server
4. Integrate with LangChain agent
5. Test in disconnected mode

