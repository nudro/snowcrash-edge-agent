#!/usr/bin/env python3
"""
Measure model VRAM requirements and validate against Old Nano constraints.
Old Nano: 8GB unified memory + 1978 MB SWAP
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Old Nano constraints
OLD_NANO_RAM = 8 * 1024  # MB
OLD_NANO_SWAP = 1978  # MB
OLD_NANO_TOTAL = OLD_NANO_RAM + OLD_NANO_SWAP  # MB

def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    if not filepath.exists():
        return 0.0
    return filepath.stat().st_size / (1024 * 1024)

def estimate_vram_usage(model_size_mb: float) -> float:
    """
    Estimate VRAM usage from model file size.
    GGUF models typically use:
    - Q4_K_M: ~1.5-2x model file size during inference (KV cache + overhead)
    - Q5_K_M: ~2-2.5x model file size
    - Context window affects KV cache size
    
    Conservative estimate: 2x model size for safety margin
    """
    # Base: model loaded in memory
    base_memory = model_size_mb
    
    # KV cache and overhead (conservative: +50-100% of model size)
    overhead_factor = 1.8  # Conservative estimate
    
    estimated_vram = model_size_mb * overhead_factor
    
    # Add small base overhead for system
    system_overhead = 500  # MB for OS, CUDA runtime, etc.
    
    return estimated_vram + system_overhead

def check_model(model_name: str, model_dir: Path):
    """Check a single model's memory requirements."""
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        print(f"  {model_name}: No GGUF files found")
        return
    
    for model_file in gguf_files:
        file_size_mb = get_file_size_mb(model_file)
        estimated_vram_mb = estimate_vram_usage(file_size_mb)
        
        fits_ram = estimated_vram_mb <= OLD_NANO_RAM
        fits_total = estimated_vram_mb <= OLD_NANO_TOTAL
        needs_swap = estimated_vram_mb > OLD_NANO_RAM
        
        status_ram = "[OK]" if fits_ram else "[WARNING]"
        status_total = "[OK]" if fits_total else "[FAIL]"
        
        print(f"\n  Model: {model_file.name}")
        print(f"    File size: {file_size_mb:.1f} MB")
        print(f"    Estimated VRAM: {estimated_vram_mb:.1f} MB")
        print(f"    Fits in 8GB RAM: {status_ram} {'Yes' if fits_ram else 'No'}")
        print(f"    Fits with SWAP: {status_total} {'Yes' if fits_total else 'No'}")
        
        if needs_swap and fits_total:
            swap_usage = estimated_vram_mb - OLD_NANO_RAM
            print(f"    SWAP usage: {swap_usage:.1f} MB / {OLD_NANO_SWAP} MB")

def main():
    """Measure all models."""
    print("Measuring model VRAM requirements for Old Nano")
    print(f"Old Nano: {OLD_NANO_RAM} MB RAM + {OLD_NANO_SWAP} MB SWAP = {OLD_NANO_TOTAL} MB total\n")
    
    models_to_check = {
        "Phi-3 Mini": MODELS_DIR / "phi3-mini",
        "Llama 3.2": MODELS_DIR / "llama3.2",
        "Gemma 2B": MODELS_DIR / "gemma2b"
    }
    
    for name, model_dir in models_to_check.items():
        print(f"\n{name}:")
        check_model(name, model_dir)
    
    print("\n" + "="*50)
    print("Note: Actual VRAM usage may vary based on:")
    print("  - Context window size")
    print("  - Batch size")
    print("  - CUDA/GPU memory management")
    print("  - System overhead")

if __name__ == "__main__":
    main()

