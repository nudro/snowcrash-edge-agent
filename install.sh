#!/bin/bash
# Installation script for Snowcrash on Orin Nano
# Installs dependencies and downloads models

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Snowcrash Installation for Orin Nano"
echo "=========================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  Python version: $(python3 --version)"
echo ""

# Upgrade pip
echo "[2/6] Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"
echo ""

# Install Python dependencies
echo "[3/6] Installing Python dependencies..."
echo "  This may take several minutes..."
python3 -m pip install -r requirements.txt
echo "  ✓ Dependencies installed"
echo ""

# Make scripts executable
echo "[4/6] Making scripts executable..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x mcp_server/server.py 2>/dev/null || true
echo "  ✓ Scripts are executable"
echo ""

# Download YOLO model
echo "[5/6] Downloading YOLO model..."
if [ ! -f "yolov8n.pt" ]; then
    echo "  Downloading yolov8n.pt..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || {
        echo "  [WARNING] Failed to auto-download YOLO model"
        echo "  You can download it manually later"
    }
    if [ -f "yolov8n.pt" ]; then
        echo "  ✓ YOLO model downloaded"
    fi
else
    echo "  ✓ YOLO model already exists"
fi
echo ""

# Model download prompt
echo "[6/6] Model downloads"
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

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test the installation:"
echo "     python3 main.py --model phi3-mini --help"
echo ""
echo "  2. Run with GUI viewer:"
echo "     python3 main.py --model phi3-mini"
echo ""
echo "  3. Run with chat GUI:"
echo "     python3 main.py --model phi3-mini --gui-viewer chatgui"
echo ""
echo "  4. Run with STT:"
echo "     python3 main.py --model phi3-mini --stt"
echo ""

