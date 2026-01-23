#!/bin/bash
# Installation script for Snowcrash on Orin Nano
# Installs dependencies and downloads models
# Supports both Docker mode (default) and llama-cpp-python fallback

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Snowcrash Installation for Orin Nano"
echo "=========================================="
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  Python version: $(python3 --version)"
echo ""

# Check Docker (optional but recommended)
echo "[2/7] Checking Docker..."
DOCKER_MODE_AVAILABLE=0
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "  ✓ Docker found: $DOCKER_VERSION"
    
    # Check for jetson-containers autotag
    if command -v autotag &> /dev/null; then
        echo "  ✓ jetson-containers autotag found"
        DOCKER_MODE_AVAILABLE=1
    else
        echo "  [WARNING] autotag not found - Docker mode may not work"
        echo "  Install jetson-containers: https://github.com/dusty-nv/jetson-containers"
        DOCKER_MODE_AVAILABLE=0
    fi
else
    echo "  [WARNING] Docker not found - will use llama-cpp-python fallback"
    echo "  Docker mode provides better memory management (recommended)"
    DOCKER_MODE_AVAILABLE=0
fi
echo ""

# Upgrade pip
echo "[3/7] Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"
echo ""

# Install Python dependencies
echo "[4/7] Installing Python dependencies..."
echo "  This may take several minutes..."
echo "  Installing: httpx (for Docker LLM), llama-cpp-python (fallback), and others..."
python3 -m pip install -r requirements.txt
echo "  ✓ Dependencies installed"
echo ""

# Make scripts executable
echo "[5/7] Making scripts executable..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x mcp_server/server.py 2>/dev/null || true
echo "  ✓ Scripts are executable"
echo ""

# Verify new Docker integration files
echo "[6/7] Verifying Docker integration files..."
if [ -f "agent/docker_llm.py" ] && [ -f "agent/container_manager.py" ]; then
    echo "  ✓ Docker LLM adapter found"
    echo "  ✓ Container manager found"
    if [ "$DOCKER_MODE_AVAILABLE" -eq 1 ]; then
        echo "  ✓ Docker mode will be used by default"
    else
        echo "  ✓ Will fallback to llama-cpp-python (Docker not available)"
    fi
else
    echo "  [WARNING] Docker integration files not found"
    echo "  Docker mode will not be available"
fi
echo ""

# Download YOLO model
echo "[7/7] Downloading YOLO model..."
YOLO_MODEL_PATH="$HOME/Documents/snowcrash/models/yolo26n-seg.pt"
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "  Downloading yolo26n-seg.pt to $YOLO_MODEL_PATH..."
    # Create models directory if it doesn't exist
    mkdir -p "$(dirname "$YOLO_MODEL_PATH")"
    # Download to temp location first, then move to final location
    python3 -c "from ultralytics import YOLO; YOLO('yolo26n-seg.pt')" 2>/dev/null || {
        echo "  [WARNING] Failed to auto-download YOLO model"
        echo "  You can download it manually later"
    }
    # Move downloaded file to correct location if it exists in current directory
    if [ -f "yolo26n-seg.pt" ]; then
        mv "yolo26n-seg.pt" "$YOLO_MODEL_PATH"
        echo "  ✓ YOLO model downloaded to $YOLO_MODEL_PATH"
    elif [ -f "$YOLO_MODEL_PATH" ]; then
        echo "  ✓ YOLO model already exists at $YOLO_MODEL_PATH"
    fi
else
    echo "  ✓ YOLO model already exists at $YOLO_MODEL_PATH"
fi
echo ""

# Model download prompt
echo ""
echo "=========================================="
echo "Model Downloads"
echo "=========================================="
echo ""
echo "Do you already have models downloaded? (y/n)"
echo "  If yes, all model downloads will be skipped."
echo "  Models should be in: ~/Documents/snowcrash/models/"
read -p "Skip model downloads? [y/N]: " skip_models

if [[ "$skip_models" =~ ^[Yy]$ ]]; then
    echo ""
    echo "  ✓ Skipping all model downloads"
    echo "  Assuming models are already in ~/Documents/snowcrash/models/"
else
    echo ""
    echo "Would you like to download SLM models now? (y/n)"
    echo "  - phi3-mini (~2.3 GB)"
    echo "  - llama3.2 (~2.0 GB)"
    echo "  - gemma2b (~1.4 GB)"
    read -p "Download SLM models? [y/N]: " download_slm

    if [[ "$download_slm" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading SLM models..."
        python3 scripts/download_models.py
        echo "  ✓ SLM models downloaded"
    else
        echo "  Skipping SLM model download"
        echo "  You can download them later with: python3 scripts/download_models.py"
    fi

    echo ""
    echo "Would you like to download Whisper STT models? (y/n)"
    echo "  - Recommended: 'base' model (~150 MB)"
    read -p "Download Whisper models? [y/N]: " download_whisper

    if [[ "$download_whisper" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading Whisper model (base)..."
        python3 scripts/download_whisper.py --model-size base --device cpu --compute-type int8
        echo "  ✓ Whisper model downloaded"
    else
        echo "  Skipping Whisper model download"
        echo "  You can download it later with: python3 scripts/download_whisper.py"
    fi
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Configuration Summary:"
if [ "$DOCKER_MODE_AVAILABLE" -eq 1 ]; then
    echo "  ✓ Docker mode: ENABLED (default)"
    echo "    - Uses llama-server in Docker containers"
    echo "    - Better memory management"
    echo "    - Automatic fallback to llama-cpp-python if Docker fails"
else
    echo "  ✓ Docker mode: DISABLED (Docker not available)"
    echo "    - Will use llama-cpp-python directly"
    echo "    - Install Docker + jetson-containers for better memory management"
fi
echo ""
echo "Next steps:"
echo "  1. Test the installation:"
echo "     python3 main.py --model llama --help"
echo ""
echo "  2. Run with GUI viewer (Docker mode):"
echo "     python3 main.py --model llama --gui-viewer chatgui"
echo ""
echo "  3. Run with different models:"
echo "     python3 main.py --model phi-3 --gui-viewer chatgui"
echo "     python3 main.py --model gemma --gui-viewer chatgui"
echo ""
echo "  4. Check Docker containers (if Docker mode enabled):"
echo "     docker ps"
echo "     docker logs snowcrash-llama-server"
echo ""
echo "  5. Run with STT:"
echo "     python3 main.py --model llama --gui-viewer audiogui"
echo ""
echo "Note: Models should be in ~/Documents/snowcrash/models/"
echo "  - llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
echo "  - phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"
echo "  - gemma2b/gemma-3n-E2B-it-Q4_K_M.gguf"
echo ""

