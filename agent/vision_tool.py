#!/usr/bin/env python3
"""
Vision Tool for LLM Tool Binding.
Allows LLM to "see" the current frame by calling YOLOE-26.
REQUIRES YOLOE-26 model (yoloe-26*-seg.pt or yoloe-26*-seg-pf.pt).
"""
import sys
import json
import cv2
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from langchain.tools import tool
from langchain_core.tools import ToolException

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Models directory path (relative to project root)
MODELS_DIR = PROJECT_ROOT / "models"

try:
    from ultralytics import YOLO
    from tools.yolo_utils import load_yolo_model
except ImportError:
    YOLO = None


def create_vision_tool(frame: Optional[Any] = None):
    """
    Create a vision tool that uses the current frame from web viewer.
    
    Args:
        frame: OpenCV frame (numpy array) from web viewer, or None
        
    Returns:
        LangChain tool function
    """
    @tool
    def detect_objects_in_frame(
        confidence_threshold: float = 0.15
    ) -> str:
        """
        Detect objects in the current video frame using YOLOE-26.
        
        Use this tool when you need to "see" what's in the current frame,
        detect hazards, identify objects, or analyze the scene.
        
        This tool runs YOLOE-26 on the current frame and returns detected objects
        with their classes, confidence scores, and bounding box coordinates.
        
        Args:
            confidence_threshold: Detection confidence threshold (0.0-1.0), default 0.15
            
        Returns:
            JSON string with detection results:
            {
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.85,
                        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
                    },
                    ...
                ],
                "count": 3,
                "hazards": ["person", "car"]  // if hazards detected
            }
            
        Example use cases:
            - "Are there any hazards in the frame?" -> Call this tool to see the scene
            - "What objects are visible?" -> Call this tool to detect objects
            - "Is the path clear?" -> Call this tool to check for obstacles
        """
        if frame is None:
            raise ToolException("No frame available. Web viewer must be running.")
        
        try:
            import os
            from pathlib import Path
            
            # FORCE YOLOE-26 usage - no fallback to regular YOLO26
            # Check if YOLOE-26 model is available (with text prompt support)
            # Prefer non-prompt-free versions for text prompting
            yoloe_models_with_prompts = [
                str(MODELS_DIR / "yoloe-26n-seg.pt"),
                str(MODELS_DIR / "yoloe-26s-seg.pt"),
                str(MODELS_DIR / "yoloe-26m-seg.pt"),
                str(MODELS_DIR / "yoloe-26l-seg.pt"),
            ]
            
            # Fallback to prompt-free versions if text-prompt versions not available
            yoloe_models_prompt_free = [
                str(MODELS_DIR / "yoloe-26n-seg-pf.pt"),
                str(MODELS_DIR / "yoloe-26s-seg-pf.pt"),
                str(MODELS_DIR / "yoloe-26m-seg-pf.pt"),
            ]
            
            model_path = None
            use_yoloe = False
            use_text_prompts = False
            
            # Try text-prompt versions first
            for yoloe_path in yoloe_models_with_prompts:
                if os.path.exists(yoloe_path):
                    model_path = yoloe_path
                    use_yoloe = True
                    use_text_prompts = True
                    break
            
            # Fallback to prompt-free if text-prompt versions not found
            if not use_yoloe:
                for yoloe_path in yoloe_models_prompt_free:
                    if os.path.exists(yoloe_path):
                        model_path = yoloe_path
                        use_yoloe = True
                        break
            
            # REQUIRE YOLOE - raise error if not found
            if not use_yoloe or model_path is None:
                available_models = yoloe_models_with_prompts + yoloe_models_prompt_free
                raise ToolException(
                    f"YOLOE-26 model not found. Required one of: {available_models}. "
                    "Please download a YOLOE-26 model to use the vision tool."
                )
            
            # Load model
            model = load_yolo_model(model_path, verbose=False)
            
            # Check if model is TensorRT engine (requires specific imgsz)
            engine_path = Path(model_path).with_suffix('.engine')
            is_tensorrt_engine = engine_path.exists()
            if is_tensorrt_engine:
                imgsz_for_detection = 640  # TensorRT engines are compiled for specific size (640x640)
            else:
                imgsz_for_detection = 640  # PyTorch models - use smaller size to save memory
            
            # If using YOLOE with text prompts, set hazard-related classes
            if use_yoloe and use_text_prompts:
                # Define hazard classes for text prompt detection
                hazard_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle", 
                                "obstacle", "barrier", "construction", "debris"]
                try:
                    # Set text prompts for hazard detection
                    model.set_classes(hazard_classes, model.get_text_pe(hazard_classes))
                except Exception as e:
                    # If set_classes fails, continue without text prompts
                    use_text_prompts = False
            
            # Run detection on frame
            results = model(frame, conf=confidence_threshold, verbose=False, imgsz=imgsz_for_detection)
            
            # Parse results
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detections.append({
                            "class": result.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": {
                                "x1": float(box.xyxy[0][0]),
                                "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]),
                                "y2": float(box.xyxy[0][3])
                            }
                        })
            
            # Analyze for hazards (potential obstacles)
            hazards = []
            hazard_classes_list = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
            detected_hazard_classes = [d["class"] for d in detections if d["class"] in hazard_classes_list]
            if detected_hazard_classes:
                hazards.extend(detected_hazard_classes)
            
            result = {
                "detections": detections,
                "count": len(detections),
                "hazards": hazards if hazards else []
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            raise ToolException(f"Error detecting objects: {str(e)}")
    
    return detect_objects_in_frame

