#!/usr/bin/env python3
"""
Convert YOLO26 PyTorch models to TensorRT (.engine) format for Jetson Orin.

TensorRT provides 2-5x faster inference on Jetson devices.
This script must be run ON the Jetson Orin (hardware-specific conversion).

Usage:
    python scripts/convert_yolo_to_tensorrt.py [model_path] [--precision fp16|int8] [--imgsz 1280]
    
Example:
    python scripts/convert_yolo_to_tensorrt.py /home/ordun/Documents/snowcrash/models/yolo26n-seg.pt --precision fp16 --imgsz 1280
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def convert_to_tensorrt(
    model_path: str,
    precision: str = "fp16",
    imgsz: int = 1280,
    output_path: Optional[str] = None
) -> str:
    """
    Convert YOLO model to TensorRT format.
    
    Args:
        model_path: Path to .pt PyTorch model
        precision: TensorRT precision ('fp16' or 'int8')
        imgsz: Input image size (must match your usage)
        output_path: Optional output path (default: same as input with .engine extension)
    
    Returns:
        Path to converted .engine file
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not model_path.suffix == ".pt":
        raise ValueError(f"Expected .pt file, got: {model_path.suffix}")
    
    # Determine output path
    if output_path is None:
        output_path = model_path.with_suffix(".engine")
    else:
        output_path = Path(output_path)
    
    print(f"[CONVERT] Loading PyTorch model: {model_path}")
    print(f"[CONVERT] Output TensorRT engine: {output_path}")
    print(f"[CONVERT] Precision: {precision}")
    print(f"[CONVERT] Input size: {imgsz}x{imgsz}")
    print("")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Export to TensorRT
    print("[CONVERT] Starting TensorRT conversion...")
    print("[CONVERT] This may take 10-30 minutes (first time conversion)...")
    print("")
    
    try:
        # Export with specified precision
        if precision == "fp16":
            model.export(
                format='engine',
                imgsz=imgsz,
                half=True,  # FP16
                simplify=True,
                workspace=4  # 4GB workspace
            )
        elif precision == "int8":
            model.export(
                format='engine',
                imgsz=imgsz,
                int8=True,  # INT8 quantization
                simplify=True,
                workspace=2  # Reduced to 2GB for INT8 (requires calibration data)
            )
        else:
            raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'int8'")
        
        # Find the exported file (Ultralytics creates it in the same directory)
        exported_file = model_path.with_suffix(".engine")
        
        if exported_file.exists():
            print(f"[SUCCESS] TensorRT engine created: {exported_file}")
            print(f"[INFO] File size: {exported_file.stat().st_size / (1024*1024):.2f} MB")
            return str(exported_file)
        else:
            raise FileNotFoundError(f"Expected engine file not found: {exported_file}")
            
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO26 PyTorch models to TensorRT format"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to YOLO .pt model file"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "int8"],
        default="fp16",
        help="TensorRT precision (fp16 recommended, int8 for maximum speed with slight accuracy loss)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size (must match your usage, default: 1280)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .engine file (default: same as input with .engine extension)"
    )
    
    args = parser.parse_args()
    
    try:
        engine_path = convert_to_tensorrt(
            model_path=args.model_path,
            precision=args.precision,
            imgsz=args.imgsz,
            output_path=args.output
        )
        
        print("")
        print("=" * 60)
        print("[SUCCESS] Conversion complete!")
        print("=" * 60)
        print(f"TensorRT engine: {engine_path}")
        print("")
        print("Next steps:")
        print("1. Update your code to use .engine file (or auto-detect)")
        print("2. Benchmark performance (should see 2-5x speedup)")
        print("3. Test accuracy (FP16: minimal loss, INT8: slight loss)")
        print("")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

