#!/bin/bash
# llama_wrapper.sh - Production-ready wrapper for llama-cli with automatic fallback
# Handles memory fragmentation issues on Jetson Orin
#
# Usage:
#   ./llama_wrapper.sh [model_path] [prompt] [gpu_layers] [context_size] [threads]
#
# Examples:
#   ./llama_wrapper.sh
#   ./llama_wrapper.sh /models/llama3.2/model.gguf "What is AI?"
#   ./llama_wrapper.sh /models/llama3.2/model.gguf "Hello" 8 512 4

# Don't use set -e here - we want to catch errors and fallback

# Show usage if help requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [model_path] [prompt] [gpu_layers] [context_size] [threads]"
    echo ""
    echo "Arguments (all optional):"
    echo "  model_path    - Path to GGUF model (default: /models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf)"
    echo "  prompt        - Text prompt (default: 'Hello, world!')"
    echo "  gpu_layers    - Number of GPU layers (default: 8)"
    echo "  context_size  - Context window size (default: 512)"
    echo "  threads       - CPU threads for fallback (default: 4)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 /models/llama3.2/model.gguf 'What is AI?'"
    echo "  $0 /models/llama3.2/model.gguf 'Hello' 8 512 4"
    exit 0
fi

# Default values
MODEL_PATH="${1:-/models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"
PROMPT="${2:-Hello, world!}"
GPU_LAYERS="${3:-8}"
CONTEXT_SIZE="${4:-512}"
THREADS="${5:-4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[llama-wrapper] Starting inference...${NC}"
echo "  Model: $MODEL_PATH"
echo "  Prompt: $PROMPT"
echo ""

# Clear caches to reduce fragmentation (non-blocking, may require sudo)
if command -v sudo &> /dev/null; then
    echo -e "${YELLOW}[llama-wrapper] Clearing caches...${NC}"
    sudo sync 2>/dev/null || true
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    echo ""
fi

# Try GPU mode first
echo -e "${GREEN}[llama-wrapper] Attempting GPU inference (${GPU_LAYERS} layers)...${NC}"
if llama-cli -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -ngl "$GPU_LAYERS" \
    -c "$CONTEXT_SIZE" 2>&1; then
    echo -e "\n${GREEN}[llama-wrapper] GPU inference completed successfully${NC}"
    exit 0
fi

# If we get here, GPU mode failed
GPU_EXIT_CODE=$?
echo -e "\n${YELLOW}[llama-wrapper] GPU allocation failed (exit code: $GPU_EXIT_CODE)${NC}"
echo -e "${YELLOW}[llama-wrapper] Falling back to CPU mode...${NC}"
echo ""

# Fallback to CPU-only mode
if llama-cli -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -ngl 0 \
    -c "$CONTEXT_SIZE" \
    --threads "$THREADS" 2>&1; then
    echo -e "\n${GREEN}[llama-wrapper] CPU inference completed successfully${NC}"
    exit 0
fi

# Both modes failed
CPU_EXIT_CODE=$?
echo -e "\n${RED}[llama-wrapper] Both GPU and CPU modes failed${NC}"
echo -e "${RED}[llama-wrapper] GPU exit code: $GPU_EXIT_CODE${NC}"
echo -e "${RED}[llama-wrapper] CPU exit code: $CPU_EXIT_CODE${NC}"
exit 1

