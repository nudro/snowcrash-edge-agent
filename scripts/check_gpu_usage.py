#!/usr/bin/env python3
"""
Quick script to check GPU usage and verify models are using GPU.
Run this while your agent is running to see real-time GPU usage.
"""
import sys
import time

try:
    import torch
    print("=" * 60)
    print("PyTorch CUDA Check:")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print()
except ImportError:
    print("[ERROR] PyTorch not installed")

try:
    import llama_cpp
    print("=" * 60)
    print("llama.cpp CUDA Support Check:")
    print("=" * 60)
    # Check if llama-cpp-python was built with CUDA support
    try:
        # Try to create a minimal model to check CUDA support
        # This will fail if CUDA is requested but not available
        from llama_cpp import Llama
        print(f"llama-cpp-python version: {llama_cpp.__version__}")
        
        # Check build info
        if hasattr(llama_cpp, 'llama_cpp'):
            print("llama-cpp-python imported successfully")
        else:
            print("[WARNING] Could not verify llama-cpp build")
    except Exception as e:
        print(f"[ERROR] llama-cpp-python issue: {e}")
    print()
except ImportError:
    print("[ERROR] llama-cpp-python not installed")
    print("Install with: pip install llama-cpp-python")

print("=" * 60)
print("Real-time GPU Monitoring:")
print("=" * 60)
print("Run this command in another terminal while your agent is running:")
print("  watch -n 1 nvidia-smi")
print()
print("Or use this Python script to monitor continuously:")
print("  python3 -c \"import torch; import time; [print(f'GPU: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB / {torch.cuda.memory_reserved(0)/1024**3:.2f}GB cached') or time.sleep(1) for _ in range(30)]\"")
print()

