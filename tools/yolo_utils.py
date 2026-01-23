#!/usr/bin/env python3
"""
YOLO utility functions for TensorRT and PyTorch model loading.

Provides auto-detection of TensorRT (.engine) vs PyTorch (.pt) formats.
"""
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def load_yolo_model(
    model_path: str,
    device: Optional[str] = None,
    verbose: bool = True
):
    """
    Load YOLO model with auto-detection of TensorRT (.engine) or PyTorch (.pt) format.
    
    TensorRT engines provide 2-5x faster inference on Jetson devices.
    This function checks for .engine file first, falls back to .pt if not found.
    
    Args:
        model_path: Path to model file (.pt or .engine)
        device: Device to use ('cuda', 'cpu', etc.) - only for PyTorch models
        verbose: Print which format is being loaded
    
    Returns:
        YOLO model instance
    
    Example:
        # Auto-detects TensorRT if available, otherwise PyTorch
        model = load_yolo_model("/path/to/yolo26n-seg.pt")
    """
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
    
    model_path_obj = Path(model_path)
    
    # Check for TensorRT engine first (faster)
    engine_path = model_path_obj.with_suffix(".engine")
    if engine_path.exists():
        if verbose:
            print(f"[YOLO] Loading TensorRT engine: {engine_path}")
        return YOLO(str(engine_path))
    elif model_path_obj.exists():
        if verbose:
            print(f"[YOLO] Loading PyTorch model: {model_path_obj}")
        if device:
            return YOLO(str(model_path_obj), device=device)
        else:
            return YOLO(str(model_path_obj))
    else:
        raise FileNotFoundError(f"Model not found: {model_path_obj} (also checked: {engine_path})")

