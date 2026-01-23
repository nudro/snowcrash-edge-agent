#!/bin/bash
# Delete VILA-2.7b model files from Jetson
# This frees disk space (not RAM - model only uses RAM when loaded)
#
# Note: VILA models are stored in jetson-containers Docker volume.
# This is separate from the Snowcrash project directory.
# Default path is the jetson-containers data directory.
# Override with: bash delete_vila_model.sh /custom/path/to/jetson-containers/data

set -e

# Default to jetson-containers data directory
# Common locations:
# - /home/ordun/Documents/Lora-GenAI/Documents/Lorawan/jetson-containers/data (if using that setup)
# - /home/ordun/Documents/jetson-containers/data
# - /data (if mounted separately)
# Override by passing custom path as first argument
DATA_DIR="${1:-/home/ordun/Documents/Lora-GenAI/Documents/Lorawan/jetson-containers/data}"

echo "=== VILA Model Deletion Script ==="
echo ""
echo "This will delete VILA-2.7b model files to free disk space."
echo "Model location: $DATA_DIR/models/"
echo ""

# Check if directories exist
VILA_HF_PATH="$DATA_DIR/models/huggingface/models--Efficient-Large-Model--VILA-2.7b"
VILA_MLC_PATH="$DATA_DIR/models/mlc/dist/VILA-2.7b"

if [ ! -d "$VILA_HF_PATH" ] && [ ! -d "$VILA_MLC_PATH" ]; then
    echo "⚠️  VILA model files not found at expected locations."
    echo "   Searched:"
    echo "   - $VILA_HF_PATH"
    echo "   - $VILA_MLC_PATH"
    echo ""
    echo "Checking for any VILA-related files..."
    find "$DATA_DIR/models" -name "*VILA*" -type d 2>/dev/null || echo "   No VILA files found."
    exit 1
fi

# Show disk usage before deletion
echo "Disk usage before deletion:"
if [ -d "$VILA_HF_PATH" ]; then
    echo "  HuggingFace cache:"
    du -sh "$VILA_HF_PATH" 2>/dev/null || echo "    (cannot read)"
fi
if [ -d "$VILA_MLC_PATH" ]; then
    echo "  MLC cache:"
    du -sh "$VILA_MLC_PATH" 2>/dev/null || echo "    (cannot read)"
fi
echo ""

# Confirm deletion
read -p "Delete VILA model files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Delete HuggingFace cache
if [ -d "$VILA_HF_PATH" ]; then
    echo "Deleting HuggingFace cache..."
    rm -rf "$VILA_HF_PATH"
    echo "✅ Deleted: $VILA_HF_PATH"
else
    echo "ℹ️  HuggingFace cache not found (may already be deleted)"
fi

# Delete MLC cache
if [ -d "$VILA_MLC_PATH" ]; then
    echo "Deleting MLC cache..."
    rm -rf "$VILA_MLC_PATH"
    echo "✅ Deleted: $VILA_MLC_PATH"
else
    echo "ℹ️  MLC cache not found (may already be deleted)"
fi

echo ""
echo "=== Deletion Complete ==="
echo ""
echo "To verify, run:"
echo "  find $DATA_DIR/models -name '*VILA*' -type d"
echo ""
echo "Note: This freed DISK space. To free RAM, stop any running processes."

