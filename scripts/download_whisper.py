#!/usr/bin/env python3
"""
Download Whisper model for faster-whisper (INT8 CPU compatible).
Downloads model to local cache for offline use on Jetson devices.

Model sizes:
- tiny:   ~39M parameters,  ~75 MB
- base:   ~74M parameters,  ~150 MB (recommended for Jetson)
- small:  ~244M parameters, ~500 MB
- medium: ~769M parameters, ~1.5 GB
- large:  ~1550M parameters, ~3 GB
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("[ERROR] faster-whisper not installed. Install with:")
    print("  pip install faster-whisper>=0.3.0")
    sys.exit(1)

def download_whisper_model(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """
    Download Whisper model for offline use.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large). Default: base
        device: Device to use (cpu or cuda). Default: cpu
        compute_type: Compute type (int8, float16, float32). Default: int8
    """
    print(f"Downloading Whisper model: {model_size}")
    print(f"  Device: {device}")
    print(f"  Compute type: {compute_type}")
    print(f"  (Model will be cached for offline use)\n")
    
    try:
        # Initialize model - this will download if not cached
        # faster-whisper automatically caches models in ~/.cache/huggingface/hub/
        print(f"Initializing WhisperModel({model_size}, device={device}, compute_type={compute_type})...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[OK] Model {model_size} downloaded and ready for use")
        print(f"\nNote: Model is cached at ~/.cache/huggingface/hub/")
        print(f"      For offline deployment, copy this cache directory to Jetson device.")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to download model: {e}")
        return False

def main():
    """Download Whisper model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Whisper model for faster-whisper")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (cpu or cuda). Default: cpu"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type. Default: int8 (CPU compatible)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Whisper Model Downloader for Snowcrash")
    print("=" * 60)
    print(f"\nRecommended for Jetson devices:")
    print(f"  - Old Nano (Maxwell): cpu + int8")
    print(f"  - Orin Nano (Ampere): cpu + int8 (or cuda + int8_float16 for faster)")
    print(f"\nModel size '{args.model_size}' is recommended for edge devices.\n")
    
    success = download_whisper_model(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nTo use this model in Snowcrash:")
        print(f"  from faster_whisper import WhisperModel")
        print(f"  model = WhisperModel('{args.model_size}', device='{args.device}', compute_type='{args.compute_type}')")
    else:
        print("\n[ERROR] Model download failed. Check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

