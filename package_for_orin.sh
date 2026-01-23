#!/bin/bash
# Package Snowcrash for transfer to Jetson Orin via SCP
# Excludes models (will be downloaded on target) and unnecessary files
#
# Usage: bash package_for_orin.sh
# Output: snowcrash-orin-YYYYMMDD-HHMMSS.tar.gz

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"
cd "$PROJECT_ROOT"

PACKAGE_NAME="snowcrash-orin-$(date +%Y%m%d-%H%M%S).tar.gz"
TEMP_PACKAGE_PATH=$(mktemp --suffix=.tar.gz)
PACKAGE_PATH="$PROJECT_ROOT/$PACKAGE_NAME"

echo "=========================================="
echo "Packaging Snowcrash for Jetson Orin"
echo "=========================================="
echo "Source: $PROJECT_ROOT"
echo "Package: $PACKAGE_NAME"
echo ""

# Create comprehensive exclude file for tar
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" << 'EOF'
# Python cache and compiled files
__pycache__/
*.pyc
*.pyo
*.pyd
*.py[cod]
*$py.class

# Virtual environments
venv/
.venv/
env/
.ENV/
ENV/

# Version control
.git/
.gitignore
.gitattributes

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# Build and distribution
build/
dist/
*.egg-info/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Logs and temporary files
*.log
*.tmp
*.cache
*.bak

# Model files (exclude all model formats)
models/*/*.gguf
models/*/*.nemo
models/*/*.pt
models/*/*.engine
models/*/*.onnx
models/*/*.trt
*.pt
*.engine
*.onnx
*.trt

# Package files
package/
*.tar.gz
*.zip

# Stray files
=0.3.0

# Archive directory (old code)
archive/
EOF

echo "Excluding:"
echo "  ✓ Python cache (__pycache__/, *.pyc)"
echo "  ✓ Virtual environments (venv/, .venv/)"
echo "  ✓ Version control (.git/)"
echo "  ✓ IDE files (.vscode/, .idea/)"
echo "  ✓ Model files (*.gguf, *.nemo, *.pt, *.engine)"
echo "  ✓ Build artifacts (build/, dist/, *.egg-info/)"
echo "  ✓ Package files (*.tar.gz)"
echo "  ✓ Archive directory"
echo ""

# Verify key files exist before packaging
echo "Verifying key files..."
MISSING_FILES=()

KEY_FILES=(
    "main.py"
    "install.sh"
    "requirements.txt"
    "README.md"
    "agent/simple_agent.py"
    "agent/docker_llm.py"
    "agent/container_manager.py"
    "agent/prompt_templates.py"
    "agent/query_router.py"
    "tools/yolo_utils.py"
    "tools/yolo_detection.py"
    "tools/tracking_viewer_chatgui.py"
    "tools/tracking_web_viewer.py"
    "scripts/convert_yolo_to_tensorrt.py"
    "scripts/llama_wrapper.sh"
)

for file in "${KEY_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "[WARNING] Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$EXCLUDE_FILE" "$TEMP_PACKAGE_PATH"
        exit 1
    fi
fi

# Create the package
echo "Creating compressed package..."
echo "This may take a moment..."

# Run tar command and capture output
TAR_OUTPUT=$(tar -czf "$TEMP_PACKAGE_PATH" \
    --exclude-from="$EXCLUDE_FILE" \
    --exclude=".git" \
    --exclude="venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="*.tar.gz" \
    --exclude="*.pt" \
    --exclude="*.engine" \
    -C "$PROJECT_ROOT" \
    . 2>&1)

TAR_EXIT_CODE=$?

# Filter out harmless warnings but keep errors
FILTERED_OUTPUT=$(echo "$TAR_OUTPUT" | grep -v "Removing leading" | grep -v "^$" || true)

# Check if tar succeeded
if [ $TAR_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Failed to create package (exit code: $TAR_EXIT_CODE)"
    echo ""
    echo "Tar output:"
    echo "$FILTERED_OUTPUT"
    echo ""
    rm -f "$EXCLUDE_FILE" "$TEMP_PACKAGE_PATH"
    exit 1
fi

# Show warnings if any (but don't fail)
if [ -n "$FILTERED_OUTPUT" ]; then
    echo "[WARNING] Tar warnings:"
    echo "$FILTERED_OUTPUT"
    echo ""
fi

# Move temp package to final location
mv "$TEMP_PACKAGE_PATH" "$PACKAGE_PATH"

# Clean up
rm -f "$EXCLUDE_FILE"

# Get package size
PACKAGE_SIZE=$(du -h "$PACKAGE_PATH" | cut -f1)

# List what's included (sample)
echo ""
echo "Package contents (sample):"
tar -tzf "$PACKAGE_PATH" | head -20
echo "  ... (and more)"
echo ""

echo "=========================================="
echo "Package created successfully!"
echo "=========================================="
echo "Package: $PACKAGE_PATH"
echo "Size: $PACKAGE_SIZE"
echo ""

# Count files in package
FILE_COUNT=$(tar -tzf "$PACKAGE_PATH" | wc -l)
echo "Files included: $FILE_COUNT"
echo ""

echo "=========================================="
echo "Transfer Instructions"
echo "=========================================="
echo ""
echo "1. Create snowcrash directory on Orin:"
echo "   ssh orin-nano 'mkdir -p ~/Documents/snowcrash'"
echo ""
echo "2. Transfer package:"
echo "   scp $PACKAGE_PATH orin-nano:~/Documents/snowcrash/"
echo ""
echo "3. On Orin Nano, extract and install:"
echo "   ssh orin-nano"
echo "   cd ~/Documents/snowcrash"
echo "   tar -xzf $PACKAGE_NAME"
echo "   cd snowcrash-orin3"
echo "   bash install.sh"
echo ""
echo "=========================================="
echo "What's Included"
echo "=========================================="
echo "✓ All Python source code (agent/, tools/, scripts/, mcp_server/)"
echo "✓ Configuration files (requirements.txt, install.sh)"
echo "✓ Documentation (README.md, docs/)"
echo "✓ Shell scripts (package_for_orin.sh, scripts/*.sh)"
echo "✓ Test files (scripts/test_*.py)"
echo ""
echo "What's Excluded"
echo "=========================================="
echo "✗ Model files (*.gguf, *.nemo, *.pt, *.engine)"
echo "✗ Virtual environment (venv/)"
echo "✗ Python cache (__pycache__/, *.pyc)"
echo "✗ Build artifacts"
echo ""
echo "Note: Models should be downloaded separately on the Orin"
echo "      using install.sh or manually."
echo ""
