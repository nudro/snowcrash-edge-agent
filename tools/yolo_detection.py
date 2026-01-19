#!/usr/bin/env python3
"""
YOLO Object Detection Tool for MCP Server.
Supports both image files and webcam capture via GStreamer.
"""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from mcp.types import Tool

class YOLODetectionTool:
    """YOLO object detection tool for MCP server."""
    
    def __init__(self, model_path: str = "yolo26n-seg.pt"):
        """Initialize YOLO model."""
        self.model_path = model_path
        self.model = None
        
    def _load_model(self):
        """Lazy load YOLO model."""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        
        if self.model is None:
            self.model = YOLO(self.model_path)
    
    def get_tool_schema(self) -> Tool:
        """Get tool schema for MCP."""
        return Tool(
            name="yolo_object_detection",
            description="Perform object detection on an image or webcam using YOLO. Detects common objects like people, cars, animals, etc. Can use image file or webcam capture.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file OR 'webcam' or 'camera' to capture from webcam"
                    },
                    "camera_device": {
                        "type": "integer",
                        "description": "Camera device index (default 0). Only used when image_path is 'webcam' or 'camera'",
                        "default": 0
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Confidence threshold (0.0-1.0), default 0.25",
                        "default": 0.25
                    },
                    "use_gstreamer": {
                        "type": "boolean",
                        "description": "Use GStreamer for webcam capture (recommended for Jetson, default True)",
                        "default": True
                    }
                },
                "required": ["image_path"]
            }
        )
    
    def _capture_from_webcam_gstreamer(self, device: int = 0, output_path: Optional[str] = None) -> str:
        """
        Capture a frame from webcam using GStreamer.
        Compatible with Jetson devices.
        
        Uses OpenCV with GStreamer backend for Jetson compatibility,
        falls back to standard OpenCV if GStreamer backend fails.
        
        Args:
            device: Camera device index (default 0)
            output_path: Path to save captured frame (optional, uses temp file if not provided)
            
        Returns:
            Path to saved image file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Try GStreamer pipeline first (better for Jetson)
        # GStreamer pipeline format: v4l2src ! ... ! appsink
        gstreamer_pipeline = (
            f"v4l2src device=/dev/video{device} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink"
        )
        
        try:
            # Try OpenCV with GStreamer backend (works well on Jetson)
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    cv2.imwrite(output_path, frame)
                    return output_path
        except Exception:
            pass
        
        # Fallback to standard OpenCV (works everywhere)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device /dev/video{device}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture frame from device /dev/video{device}")
            
            cv2.imwrite(output_path, frame)
            return output_path
        finally:
            cap.release()
    
    def _capture_from_webcam_opencv(self, device: int = 0, output_path: Optional[str] = None) -> str:
        """
        Capture a frame from webcam using OpenCV (fallback).
        
        Args:
            device: Camera device index (default 0)
            output_path: Path to save captured frame (optional, uses temp file if not provided)
            
        Returns:
            Path to saved image file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture frame from device {device}")
            
            cv2.imwrite(output_path, frame)
            return output_path
        finally:
            cap.release()
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute object detection."""
        try:
            self._load_model()
            
            image_path = arguments.get("image_path")
            confidence = arguments.get("confidence_threshold", 0.25)
            camera_device = arguments.get("camera_device", 0)
            use_gstreamer = arguments.get("use_gstreamer", True)
            
            # Check if webcam capture is requested
            is_webcam = image_path.lower() in ["webcam", "camera", "cam"]
            
            if is_webcam:
                # Capture from webcam
                try:
                    if use_gstreamer:
                        # Use GStreamer (recommended for Jetson)
                        captured_image = self._capture_from_webcam_gstreamer(camera_device)
                        source_info = f"webcam (device /dev/video{camera_device} via GStreamer)"
                    else:
                        # Fallback to OpenCV
                        captured_image = self._capture_from_webcam_opencv(camera_device)
                        source_info = f"webcam (device {camera_device} via OpenCV)"
                    
                    # Use captured image for detection
                    detection_image_path = captured_image
                except Exception as e:
                    return {
                        "error": f"Webcam capture failed: {str(e)}",
                        "success": False
                    }
            else:
                # Validate image path
                img_path = Path(image_path)
                if not img_path.exists():
                    return {
                        "error": f"Image not found: {image_path}",
                        "success": False
                    }
                detection_image_path = str(img_path)
                source_info = str(image_path)
            
            # Run detection
            results = self.model(str(detection_image_path), conf=confidence)
            
            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        "class": self.model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": {
                            "x1": float(box.xyxy[0][0]),
                            "y1": float(box.xyxy[0][1]),
                            "x2": float(box.xyxy[0][2]),
                            "y2": float(box.xyxy[0][3])
                        }
                    })
            
            result = {
                "success": True,
                "source": source_info,
                "detections": detections,
                "count": len(detections)
            }
            
            # Clean up temporary captured image if webcam was used
            if is_webcam and detection_image_path.startswith(tempfile.gettempdir()):
                try:
                    Path(detection_image_path).unlink()
                except:
                    pass
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

