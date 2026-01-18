#!/bin/bash
# Setup script for Snowcrash on DGX Spark

set -e

echo "Setting up Snowcrash environment..."

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Create virtual environment (optional, but recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.py
chmod +x scripts/*.sh
chmod +x mcp_server/server.py

echo "[OK] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download models: python scripts/download_models.py"
echo "  2. Measure VRAM: python scripts/measure_vram.py"
echo "  3. Package for transfer: python scripts/package_for_transfer.py"

