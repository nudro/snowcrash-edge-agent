#!/usr/bin/env python3
"""
Download SLM models locally on DGX Spark.
Downloads: Phi-3 Mini, Llama 3.2, and Gemma 2B in GGUF format.
"""
import os
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Model URLs (GGUF quantized versions)
MODELS = {
    "phi3-mini": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4_K_M.gguf",
        "dir": MODELS_DIR / "phi3-mini",
        "use_huggingface_cli": True  # Use huggingface-cli instead of direct URL
    },
    "llama3.2": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "dir": MODELS_DIR / "llama3.2"
    },
    "gemma2b": {
        "url": "https://huggingface.co/bartowski/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf",
        "filename": "gemma-3n-E2B-it-Q4_K_M.gguf",
        "dir": MODELS_DIR / "gemma2b"
    }
}

def download_model(name: str, config: dict):
    """Download a single model using wget."""
    os.makedirs(config["dir"], exist_ok=True)
    filepath = config["dir"] / config["filename"]
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"[OK] {name} already exists ({size_mb:.1f} MB)")
        return
    
    print(f"Downloading {name}...")
    print(f"  URL: {config['url']}")
    print(f"  Destination: {filepath}")
    
    try:
        subprocess.run(
            ["wget", "-O", str(filepath), config["url"]],
            check=True
        )
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"[OK] {name} downloaded ({size_mb:.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to download {name}: {e}")
        raise

def main():
    """Download all models."""
    print("Downloading SLM models...")
    print(f"Models directory: {MODELS_DIR}\n")
    
    for name, config in MODELS.items():
        try:
            download_model(name, config)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            continue
    
    print("\nModel download complete!")

if __name__ == "__main__":
    main()

