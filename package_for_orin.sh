#!/bin/bash
# Package Snowcrash for transfer to Orin Nano via SCP
# Excludes models (will be downloaded on target) and unnecessary files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"
cd "$PROJECT_ROOT"

PACKAGE_NAME="snowcrash-orin-$(date +%Y%m%d-%H%M%S).tar.gz"
PACKAGE_PATH="$PROJECT_ROOT/$PACKAGE_NAME"

echo "=========================================="
echo "Packaging Snowcrash for Orin Nano"
echo "=========================================="
echo "Source: $PROJECT_ROOT"
echo "Package: $PACKAGE_NAME"
echo ""

# Create temporary exclude file for tar
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
venv/
.venv/
env/
.ENV/
*.log
*.tmp
*.cache
.DS_Store
Thumbs.db
*.swp
*.swo
*~
.pytest_cache/
.mypy_cache/
build/
dist/
*.egg-info/
.vscode/
.idea/
# Exclude models (will be downloaded on target)
models/*.gguf
models/*/*.gguf
*.pt
# Exclude package files themselves
package/
*.tar.gz
EOF

echo "Excluding:"
echo "  - __pycache__/, *.pyc"
echo "  - .git/, venv/"
echo "  - models/*.gguf, *.pt (will be downloaded on target)"
echo "  - IDE files, temp files"
echo ""

# Create the package
echo "Creating compressed package..."
tar -czf "$PACKAGE_PATH" \
    --exclude-from="$EXCLUDE_FILE" \
    --exclude="$PACKAGE_NAME" \
    --exclude="*/__pycache__" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    --exclude="*.pyd" \
    agent/ \
    docs/ \
    mcp_server/ \
    scripts/ \
    tests/ \
    tools/ \
    main.py \
    README.md \
    requirements.txt \
    WORKFLOW.md \
    install.sh \
    2>/dev/null || {
    echo "[ERROR] Failed to create package"
    rm -f "$EXCLUDE_FILE"
    exit 1
}

# Clean up
rm -f "$EXCLUDE_FILE"

# Get package size
PACKAGE_SIZE=$(du -h "$PACKAGE_PATH" | cut -f1)

echo ""
echo "=========================================="
echo "Package created successfully!"
echo "=========================================="
echo "Package: $PACKAGE_PATH"
echo "Size: $PACKAGE_SIZE"
echo ""
echo "To transfer to Orin Nano:"
echo "  # Create snowcrash directory on Orin Nano"
echo "  ssh orin-nano 'mkdir -p ~/snowcrash'"
echo "  # Transfer package"
echo "  scp $PACKAGE_PATH orin-nano:~/snowcrash/"
echo ""
echo "On Orin Nano, extract and install:"
echo "  ssh orin-nano"
echo "  cd ~/snowcrash"
echo "  tar -xzf $PACKAGE_NAME"
echo "  bash install.sh"
echo ""

