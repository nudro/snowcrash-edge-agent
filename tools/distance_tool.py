#!/usr/bin/env python3
"""
Distance Estimation Tool for MCP Server.
Estimates distance from camera to detected objects using bounding box height and known object sizes.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from mcp.types import Tool


# Known object heights in meters (for distance estimation)
DEFAULT_OBJECT_HEIGHTS = {
    "person": 1.7,  # Average person height
    "bicycle": 1.2,
    "car": 1.5,  # Average car height
    "motorcycle": 1.0,
    "airplane": 4.0,
    "bus": 3.0,
    "train": 4.0,
    "truck": 3.0,
    "boat": 2.0,
    "traffic light": 3.0,
    "fire hydrant": 1.0,
    "stop sign": 2.5,
    "parking meter": 1.0,
    "bench": 0.5,
    "bird": 0.3,
    "cat": 0.3,
    "dog": 0.5,
    "horse": 1.5,
    "sheep": 1.0,
    "cow": 1.5,
    "elephant": 3.0,
    "bear": 1.5,
    "zebra": 1.5,
    "giraffe": 5.0,
    "backpack": 0.4,
    "umbrella": 1.5,
    "handbag": 0.3,
    "tie": 0.2,
    "suitcase": 0.6,
    "frisbee": 0.3,
    "skis": 1.5,
    "snowboard": 1.5,
    "sports ball": 0.25,
    "kite": 0.5,
    "baseball bat": 1.0,
    "baseball glove": 0.3,
    "skateboard": 0.1,
    "surfboard": 1.5,
    "tennis racket": 0.7,
    "bottle": 0.3,
    "wine glass": 0.2,
    "cup": 0.15,
    "fork": 0.2,
    "knife": 0.2,
    "spoon": 0.2,
    "bowl": 0.15,
    "banana": 0.2,
    "apple": 0.1,
    "sandwich": 0.1,
    "orange": 0.1,
    "broccoli": 0.2,
    "carrot": 0.2,
    "hot dog": 0.15,
    "pizza": 0.3,
    "donut": 0.1,
    "cake": 0.2,
    "chair": 1.0,
    "couch": 0.9,
    "potted plant": 0.5,
    "bed": 0.5,
    "dining table": 0.8,
    "toilet": 0.5,
    "tv": 0.6,
    "laptop": 0.03,
    "mouse": 0.03,
    "remote": 0.15,
    "keyboard": 0.03,
    "cell phone": 0.15,
    "microwave": 0.3,
    "oven": 0.6,
    "toaster": 0.2,
    "sink": 0.5,
    "refrigerator": 1.8,
    "book": 0.03,
    "clock": 0.3,
    "vase": 0.3,
    "scissors": 0.15,
    "teddy bear": 0.3,
    "hair drier": 0.3,
    "toothbrush": 0.2,
}


class DistanceTool:
    """Distance estimation tool for MCP server."""
    
    def __init__(self, model_path: str = "/home/ordun/Documents/snowcrash/models/yolo26n-seg.pt", focal_length_mm: float = 3.6, sensor_height_mm: float = 4.69):
        """
        Initialize distance estimation tool.
        
        Args:
            model_path: Path to YOLO model
            focal_length_mm: Camera focal length in mm (default for common webcam)
            sensor_height_mm: Camera sensor height in mm (default for common webcam)
        """
        self.model_path = model_path
        self.model = None
        self.focal_length_mm = focal_length_mm
        self.sensor_height_mm = sensor_height_mm
        
        # Default image dimensions (will be updated from actual captures)
        self.image_height_px = 480
    
    def _load_model(self):
        """Lazy load YOLO model."""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        
        if self.model is None:
            self.model = YOLO(self.model_path)
    
    def get_tool_schema(self) -> Tool:
        """Get tool schema for MCP."""
        return Tool(
            name="estimate_object_distances",
            description="Estimate distance from camera to detected objects. Uses bounding box height and known object sizes. Can use image file or webcam capture.",
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
                    },
                    "focal_length_mm": {
                        "type": "number",
                        "description": "Camera focal length in mm (optional, uses default if not provided)",
                        "default": 3.6
                    }
                },
                "required": ["image_path"]
            }
        )
    
    def _capture_from_webcam(self, device: int = 0, use_gstreamer: bool = True) -> cv2.Mat:
        """Capture a frame from webcam."""
        if use_gstreamer:
            # Use higher resolution for better small object detection
            gstreamer_pipeline = (
                f"v4l2src device=/dev/video{device} ! "
                "image/jpeg,width=1280,height=720,framerate=30/1 ! "
                "jpegdec ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink"
            )
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("GStreamer failed, falling back to standard OpenCV...")
                cap = cv2.VideoCapture(device)
        else:
            cap = cv2.VideoCapture(device)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device /dev/video{device}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        
        return frame
    
    def _estimate_distance(self, bbox: List[float], class_name: str, image_height: int) -> Optional[float]:
        """
        Estimate distance to object using pinhole camera model.
        
        Formula: distance = (known_height * focal_length) / (detected_height_px * pixel_size)
        Where pixel_size = sensor_height_mm / image_height_px
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: Object class name
            image_height: Image height in pixels
            
        Returns:
            Estimated distance in meters, or None if object height unknown
        """
        # Get bounding box height in pixels
        _, y1, _, y2 = bbox
        bbox_height_px = abs(y2 - y1)
        
        # Get known object height
        known_height_m = DEFAULT_OBJECT_HEIGHTS.get(class_name)
        if known_height_m is None:
            return None
        
        # Calculate pixel size
        pixel_size_mm = self.sensor_height_mm / image_height
        
        # Calculate distance using pinhole camera model
        # distance = (known_height * focal_length) / (detected_height_px * pixel_size)
        distance_m = (known_height_m * self.focal_length_mm) / (bbox_height_px * pixel_size_mm)
        
        return distance_m
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute distance estimation tool.
        
        Args:
            arguments: Tool arguments (can include 'frame' for direct frame input)
            
        Returns:
            Distance estimation results
        """
        self._load_model()
        
        image_path = arguments.get("image_path")
        camera_device = arguments.get("camera_device", 0)
        confidence_threshold = arguments.get("confidence_threshold", 0.25)
        use_gstreamer = arguments.get("use_gstreamer", True)
        frame = arguments.get("frame")  # Allow passing frame directly
        
        # Override focal length if provided
        if "focal_length_mm" in arguments:
            self.focal_length_mm = float(arguments["focal_length_mm"])
        
        # If frame provided directly, use it (e.g., from web viewer)
        if frame is not None:
            self.image_height_px = frame.shape[0]
            image_for_detection = frame
            source_info = "web viewer frame"
        # Capture or load image
        elif image_path:
            is_webcam = image_path.lower() in ["webcam", "camera", "cam", "me", "myself"]
            
            if is_webcam:
                frame = self._capture_from_webcam(camera_device, use_gstreamer)
                self.image_height_px = frame.shape[0]
                image_for_detection = frame
                source_info = f"webcam (device /dev/video{camera_device})"
            else:
                frame = cv2.imread(image_path)
                if frame is None:
                    return {"error": f"Cannot read image: {image_path}"}
                self.image_height_px = frame.shape[0]
                image_for_detection = frame
                source_info = image_path
        else:
            return {"error": "Either 'frame' or 'image_path' must be provided"}
        
        # Run detection
        # Use higher resolution for better small object detection
        results = self.model(
            image_for_detection, 
            conf=confidence_threshold, 
            verbose=False,
            imgsz=1280  # Higher resolution for better small object detection
        )
        
        # Process detections and estimate distances
        detections_with_distances = []
        nearest_object = None
        min_distance = float('inf')
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = self.model.names[cls_id]
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Estimate distance
                distance_m = self._estimate_distance(bbox, cls_name, self.image_height_px)
                
                detection = {
                    "class": cls_name,
                    "bounding_box": bbox,
                    "confidence": round(conf, 2),
                    "estimated_height_meters": DEFAULT_OBJECT_HEIGHTS.get(cls_name)
                }
                
                if distance_m is not None:
                    distance_ft = distance_m * 3.28084  # Convert to feet
                    detection["distance_meters"] = round(distance_m, 2)
                    detection["distance_feet"] = round(distance_ft, 1)
                    
                    # Track nearest object
                    if distance_m < min_distance:
                        min_distance = distance_m
                        nearest_object = {
                            "class": cls_name,
                            "distance_meters": round(distance_m, 2),
                            "bounding_box": bbox
                        }
                else:
                    detection["distance_meters"] = None
                    detection["distance_feet"] = None
                
                detections_with_distances.append(detection)
        
        return {
            "detections_with_distances": detections_with_distances,
            "nearest_object": nearest_object,
            "source": source_info,
            "timestamp": datetime.now().isoformat()
        }

