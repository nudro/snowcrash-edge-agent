#!/usr/bin/env python3
"""
Geographic Estimation Tool for MCP Server.
Estimates geographic location (lat/lon) of detected objects by fusing YOLO detections
with GPS coordinates and distance estimates.

Uses Shapely (memory-efficient) for geometric calculations instead of GeoPandas.
"""
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from shapely.geometry import Point
    from shapely.ops import transform
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Point = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    pyproj = None

from mcp.types import Tool

# Use distance estimation constants
from tools.distance_tool import DEFAULT_OBJECT_HEIGHTS, DistanceTool


class GeographicTool:
    """
    Geographic estimation tool using Shapely for memory efficiency.
    Fuses YOLO detections with GPS coordinates to estimate object locations.
    """
    
    def __init__(self):
        """Initialize geographic tool."""
        self.model = None
        self.model_path = "yolov8n.pt"
        self.distance_tool = DistanceTool()  # Reuse distance estimation
        
        # Camera calibration parameters (defaults - should be calibrated per device)
        # Focal length in pixels (assuming 640x480 webcam)
        self.focal_length_px = 480.0  # Approximate for webcam
        self.image_width_px = 640
        self.image_height_px = 480
        self.principal_point_x = 320.0  # Center of image
        self.principal_point_y = 240.0
        
        # Camera pose (will be updated with GPS data)
        self.camera_lat = None
        self.camera_lon = None
        self.camera_alt = 0.0  # Meters above sea level
        self.camera_yaw = 0.0  # Degrees: 0 = North, 90 = East
        self.camera_pitch = 0.0  # Degrees: 0 = horizontal, negative = down
        self.camera_roll = 0.0  # Degrees
    
    def _load_model(self):
        """Lazy load YOLO model."""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        if self.model is None:
            self.model = YOLO(self.model_path)
    
    def get_tool_schema(self) -> Tool:
        """Get tool schema for MCP."""
        return Tool(
            name="estimate_object_geography",
            description="Estimate geographic location (latitude/longitude) of detected objects by fusing YOLO detections with GPS coordinates and distance estimates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file or 'webcam' for camera capture"
                    },
                    "camera_device": {
                        "type": "integer",
                        "description": "Camera device index (default: 0)",
                        "default": 0
                    },
                    "camera_latitude": {
                        "type": "number",
                        "description": "Camera GPS latitude (degrees). Required if GPS data not available from device.",
                        "required": False
                    },
                    "camera_longitude": {
                        "type": "number",
                        "description": "Camera GPS longitude (degrees). Required if GPS data not available from device.",
                        "required": False
                    },
                    "camera_altitude": {
                        "type": "number",
                        "description": "Camera altitude in meters (default: 0.0)",
                        "default": 0.0
                    },
                    "camera_yaw": {
                        "type": "number",
                        "description": "Camera yaw angle in degrees (0=North, 90=East, default: 0.0)",
                        "default": 0.0
                    },
                    "camera_pitch": {
                        "type": "number",
                        "description": "Camera pitch angle in degrees (0=horizontal, default: 0.0)",
                        "default": 0.0
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "YOLO confidence threshold (default: 0.25)",
                        "default": 0.25
                    },
                    "frame": {
                        "type": "object",
                        "description": "Optional: pre-captured frame (numpy array) to avoid re-opening camera"
                    }
                }
            }
        )
    
    def _get_device_gps(self) -> Optional[Tuple[float, float]]:
        """
        Attempt to get GPS coordinates from device.
        Jetson devices typically don't have built-in GPS - would need external GPS module.
        
        Returns:
            (lat, lon) tuple if GPS available, None otherwise
        """
        # TODO: Implement GPS module detection for Jetson devices
        # Common GPS modules: USB GPS dongles, UART GPS modules (e.g., NEO-M9N)
        # For now, return None (user must provide GPS coordinates)
        
        # Example: If GPS module attached via /dev/ttyUSB0 or similar:
        # try:
        #     import serial
        #     gps_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        #     # Parse NMEA sentences to get lat/lon
        #     ...
        # except:
        #     return None
        
        return None
    
    def _pixel_to_geo(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_m: float,
        camera_lat: float,
        camera_lon: float,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates + distance to geographic coordinates (lat/lon).
        
        Uses pinhole camera model and geographic projection.
        
        Args:
            pixel_x: X coordinate in image (pixels)
            pixel_y: Y coordinate in image (pixels)
            distance_m: Distance from camera to object (meters)
            camera_lat: Camera GPS latitude (degrees)
            camera_lon: Camera GPS longitude (degrees)
            camera_yaw: Camera yaw angle (degrees, 0=North)
            camera_pitch: Camera pitch angle (degrees, 0=horizontal)
            
        Returns:
            (lat, lon) tuple of object location
        """
        # Convert pixel to camera coordinates (pinhole model)
        # Camera coordinate system: Z forward, X right, Y down
        u = pixel_x - self.principal_point_x
        v = pixel_y - self.principal_point_y
        
        # Convert to camera frame (meters)
        # Using pinhole model: x_cam = (u * z) / f, y_cam = (v * z) / f
        z_cam = distance_m  # Distance along camera optical axis
        x_cam = (u * z_cam) / self.focal_length_px
        y_cam = (v * z_cam) / self.focal_length_px
        
        # Apply camera pitch (rotation around X axis)
        pitch_rad = math.radians(camera_pitch)
        y_cam_rotated = y_cam * math.cos(pitch_rad) - z_cam * math.sin(pitch_rad)
        z_cam_rotated = y_cam * math.sin(pitch_rad) + z_cam * math.cos(pitch_rad)
        x_cam_rotated = x_cam
        
        # Apply camera yaw (rotation around Y axis, clockwise from North)
        yaw_rad = math.radians(camera_yaw)
        x_cam_final = x_cam_rotated * math.cos(yaw_rad) + z_cam_rotated * math.sin(yaw_rad)
        z_cam_final = -x_cam_rotated * math.sin(yaw_rad) + z_cam_rotated * math.cos(yaw_rad)
        
        # Convert camera coordinates to local ENU (East-North-Up) frame
        # Camera frame: X right, Y down, Z forward
        # ENU frame: East = X, North = Y, Up = -Z
        east_offset_m = x_cam_final
        north_offset_m = z_cam_final
        up_offset_m = -y_cam_rotated  # Note: camera Y down, ENU up positive
        
        # Convert ENU offset to geographic coordinates
        # Approximate: 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(camera_lat))
        
        object_lat = camera_lat + (north_offset_m / meters_per_degree_lat)
        object_lon = camera_lon + (east_offset_m / meters_per_degree_lon)
        
        return (object_lat, object_lon)
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate geographic locations of detected objects.
        
        Args:
            arguments: Tool arguments including image_path, GPS coordinates, etc.
            
        Returns:
            Dictionary with detections and their geographic coordinates
        """
        if not SHAPELY_AVAILABLE:
            return {
                "error": "Shapely not installed. Install with: pip install shapely",
                "success": False
            }
        
        try:
            self._load_model()
            
            image_path = arguments.get("image_path")
            camera_device = arguments.get("camera_device", 0)
            confidence_threshold = arguments.get("confidence_threshold", 0.25)
            
            # Get GPS coordinates (from arguments or device)
            camera_lat = arguments.get("camera_latitude")
            camera_lon = arguments.get("camera_longitude")
            
            # If not provided, try to get from device
            if camera_lat is None or camera_lon is None:
                device_gps = self._get_device_gps()
                if device_gps:
                    camera_lat, camera_lon = device_gps
                else:
                    return {
                        "error": "GPS coordinates required. Provide 'camera_latitude' and 'camera_longitude', or ensure GPS module is attached.",
                        "success": False
                    }
            
            # Update camera pose
            self.camera_lat = camera_lat
            self.camera_lon = camera_lon
            self.camera_alt = arguments.get("camera_altitude", 0.0)
            self.camera_yaw = arguments.get("camera_yaw", 0.0)
            self.camera_pitch = arguments.get("camera_pitch", 0.0)
            
            # Get frame or capture from camera/image
            frame = arguments.get("frame")
            
            if frame is not None:
                # Use provided frame (e.g., from web viewer)
                image_for_detection = frame
                source_info = "web viewer frame"
                self.image_height_px = frame.shape[0]
                self.image_width_px = frame.shape[1]
                self.principal_point_x = self.image_width_px / 2.0
                self.principal_point_y = self.image_height_px / 2.0
            elif image_path and image_path.lower() in ["webcam", "camera", "cam"]:
                # Capture from webcam (reuse distance_tool's method)
                from tools.yolo_detection import YOLODetectionTool
                yolo_tool = YOLODetectionTool()
                captured_path = await yolo_tool._capture_from_webcam_gstreamer(camera_device) if hasattr(yolo_tool, '_capture_from_webcam_gstreamer') else None
                if not captured_path:
                    return {"error": "Failed to capture from webcam", "success": False}
                import cv2
                image_for_detection = cv2.imread(captured_path)
                source_info = f"webcam (device {camera_device})"
            else:
                # Load from image file
                from pathlib import Path
                import cv2
                if not Path(image_path).exists():
                    return {"error": f"Image not found: {image_path}", "success": False}
                image_for_detection = cv2.imread(image_path)
                source_info = str(image_path)
            
            if image_for_detection is None:
                return {"error": "Failed to load image", "success": False}
            
            # Update image dimensions
            self.image_height_px = image_for_detection.shape[0]
            self.image_width_px = image_for_detection.shape[1]
            self.principal_point_x = self.image_width_px / 2.0
            self.principal_point_y = self.image_height_px / 2.0
            
            # Run YOLO detection
            results = self.model(image_for_detection, conf=confidence_threshold, verbose=False)
            
            # Process detections and estimate geographic locations
            detections_with_geo = []
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    cls_name = self.model.names[cls_id]
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    # Calculate bounding box center (pixel coordinates)
                    center_x = (bbox[0] + bbox[2]) / 2.0
                    center_y = (bbox[1] + bbox[3]) / 2.0
                    
                    # Estimate distance using distance tool
                    distance_m = self.distance_tool._estimate_distance(
                        bbox, cls_name, self.image_height_px
                    )
                    
                    detection = {
                        "class": cls_name,
                        "bounding_box": bbox,
                        "center_pixel": {"x": round(center_x, 1), "y": round(center_y, 1)},
                        "confidence": round(conf, 2),
                        "distance_meters": round(distance_m, 2) if distance_m else None
                    }
                    
                    # If distance estimated, calculate geographic coordinates
                    if distance_m is not None:
                        object_lat, object_lon = self._pixel_to_geo(
                            center_x,
                            center_y,
                            distance_m,
                            camera_lat,
                            camera_lon,
                            self.camera_yaw,
                            self.camera_pitch
                        )
                        
                        detection["latitude"] = round(object_lat, 6)
                        detection["longitude"] = round(object_lon, 6)
                        
                        # Create Shapely Point for potential geometric operations
                        if Point is not None:
                            detection["geometry"] = {
                                "type": "Point",
                                "coordinates": [object_lon, object_lat]  # GeoJSON format (lon, lat)
                            }
                    else:
                        detection["latitude"] = None
                        detection["longitude"] = None
                    
                    detections_with_geo.append(detection)
            
            return {
                "success": True,
                "camera_location": {
                    "latitude": camera_lat,
                    "longitude": camera_lon,
                    "altitude_meters": self.camera_alt
                },
                "source": source_info,
                "detections_with_geography": detections_with_geo,
                "count": len(detections_with_geo)
            }
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False
            }

