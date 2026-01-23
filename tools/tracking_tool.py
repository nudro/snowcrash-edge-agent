#!/usr/bin/env python3
"""
Object Tracking Tool for MCP Server.
Tracks objects across frames with persistent IDs using YOLO's built-in tracker (BoT-SORT/ByteTrack).
"""
import json
import tempfile
import time
import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

try:
    from ultralytics import YOLO
    try:
        from ultralytics.solutions.speed_estimation import SpeedEstimator
        SPEED_ESTIMATOR_AVAILABLE = True
    except ImportError:
        SPEED_ESTIMATOR_AVAILABLE = False
        SpeedEstimator = None
except ImportError:
    YOLO = None
    SPEED_ESTIMATOR_AVAILABLE = False
    SpeedEstimator = None

from mcp.types import Tool


class TrackingTool:
    """Object tracking tool for MCP server using YOLO's built-in tracker."""
    
    def __init__(self, model_path: str = "/home/ordun/Documents/snowcrash/models/yolo26n-seg.pt"):
        """Initialize tracking tool."""
        self.model_path = model_path
        self.model = None
        self.tracker = None
        self.frame_count = 0
        # Store track history: track_id -> list of (frame, timestamp, bbox, class, conf)
        self.track_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # Track metadata: track_id -> {first_seen, last_seen, class, max_confidence}
        self.track_metadata: Dict[int, Dict[str, Any]] = {}
        
    def _load_model(self):
        """Lazy load YOLO model."""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        
        if self.model is None:
            self.model = YOLO(self.model_path)
    
    def get_tool_schema(self) -> Tool:
        """Get tool schema for MCP."""
        return Tool(
            name="track_objects",
            description="Track objects across frames with persistent IDs. Uses YOLO's built-in tracker (BoT-SORT/ByteTrack). Can use image file or webcam capture.",
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
                    "track_history_frames": {
                        "type": "integer",
                        "description": "Number of frames to keep in track history (default 30, 0 to disable)",
                        "default": 30
                    },
                    "reset_tracks": {
                        "type": "boolean",
                        "description": "Reset track history before processing (default False)",
                        "default": False
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Duration to track in seconds (0-60, default 0 for single frame). Only used with webcam.",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 60
                    }
                },
                "required": ["image_path"]
            }
        )
    
    def _open_webcam(self, device: int = 0, use_gstreamer: bool = True) -> cv2.VideoCapture:
        """Open webcam video capture."""
        if use_gstreamer:
            gstreamer_pipeline = (
                f"v4l2src device=/dev/video{device} ! "
                "image/jpeg,width=640,height=480,framerate=30/1 ! "
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
        
        return cap
    
    def _capture_from_webcam(self, device: int = 0, use_gstreamer: bool = True) -> cv2.Mat:
        """Capture a single frame from webcam."""
        cap = self._open_webcam(device, use_gstreamer)
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam")
            return frame
        finally:
            cap.release()
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tracking tool.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tracking results
        """
        self._load_model()
        
        image_path = arguments.get("image_path")
        camera_device = arguments.get("camera_device", 0)
        confidence_threshold = arguments.get("confidence_threshold", 0.25)
        use_gstreamer = arguments.get("use_gstreamer", True)
        track_history_frames = arguments.get("track_history_frames", 30)
        reset_tracks = arguments.get("reset_tracks", False)
        duration_seconds = arguments.get("duration_seconds", 0)
        
        # Clamp duration to 0-60 seconds
        duration_seconds = max(0, min(60, float(duration_seconds)))
        
        # Reset tracks if requested
        if reset_tracks:
            self.track_history.clear()
            self.track_metadata.clear()
            self.frame_count = 0
        
        # Capture or load image
        is_webcam = image_path.lower() in ["webcam", "camera", "cam"]
        
        # If webcam and duration > 0, run continuous tracking
        if is_webcam and duration_seconds > 0:
            return await self._track_continuous(
                camera_device, use_gstreamer, confidence_threshold,
                track_history_frames, duration_seconds
            )
        
        # Single frame processing (existing logic)
        if is_webcam:
            frame = self._capture_from_webcam(camera_device, use_gstreamer)
            source_info = f"webcam (device /dev/video{camera_device})"
        else:
            frame = cv2.imread(image_path)
            if frame is None:
                return {"error": f"Cannot read image: {image_path}"}
            source_info = image_path
        
        self.frame_count += 1
        current_time = datetime.now()
        timestamp_str = current_time.isoformat()
        
        # Run tracking (YOLO with tracker mode)
        # Use 'track' mode which uses BoT-SORT or ByteTrack
        results = self.model.track(
            frame,
            conf=confidence_threshold,
            persist=True,  # Persist tracks across frames
            verbose=False,
            imgsz=1280  # Higher resolution for better small object detection
        )
        
        # Process tracked objects
        active_tracks = []
        active_track_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                track_id_int = int(track_id)
                active_track_ids.add(track_id_int)
                
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = self.model.names[cls_id]
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Calculate center position
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Update track metadata
                if track_id_int not in self.track_metadata:
                    self.track_metadata[track_id_int] = {
                        "first_seen": timestamp_str,
                        "class": cls_name,
                        "max_confidence": conf,
                        "age_frames": 0
                    }
                else:
                    self.track_metadata[track_id_int]["max_confidence"] = max(
                        self.track_metadata[track_id_int]["max_confidence"],
                        conf
                    )
                
                # Calculate velocity (if we have previous position)
                velocity = {"vx": 0.0, "vy": 0.0}
                if track_id_int in self.track_history and len(self.track_history[track_id_int]) > 0:
                    prev_data = self.track_history[track_id_int][-1]
                    prev_bbox = prev_data["bounding_box"]
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    # Assume ~30 FPS for velocity calculation
                    velocity = {
                        "vx": round((center_x - prev_center_x) / 30.0, 1),
                        "vy": round((center_y - prev_center_y) / 30.0, 1)
                    }
                
                # Update track history
                track_data = {
                    "frame": self.frame_count,
                    "timestamp": timestamp_str,
                    "bounding_box": bbox,
                    "class": cls_name,
                    "confidence": conf
                }
                
                if track_history_frames > 0:
                    self.track_history[track_id_int].append(track_data)
                    # Limit history size
                    if len(self.track_history[track_id_int]) > track_history_frames:
                        self.track_history[track_id_int] = self.track_history[track_id_int][-track_history_frames:]
                else:
                    # Just keep last one if history disabled
                    self.track_history[track_id_int] = [track_data]
                
                # Update metadata
                metadata = self.track_metadata[track_id_int]
                metadata["age_frames"] = len(self.track_history[track_id_int])
                metadata["last_updated"] = timestamp_str
                metadata["class"] = cls_name  # Update class (in case it changes)
                
                # Build track info
                track_info = {
                    "track_id": track_id_int,
                    "class": cls_name,
                    "bounding_box": bbox,
                    "confidence": round(conf, 2),
                    "age_frames": metadata["age_frames"],
                    "position": {"x": round(center_x, 1), "y": round(center_y, 1)},
                    "velocity": velocity,
                    "trajectory_length": len(self.track_history[track_id_int]),
                    "first_seen": metadata["first_seen"],
                    "last_updated": timestamp_str
                }
                
                active_tracks.append(track_info)
        
        # Remove lost tracks from metadata (tracks not seen in this frame)
        lost_tracks = []
        for track_id in list(self.track_metadata.keys()):
            if track_id not in active_track_ids:
                # Track is lost (not in current frame)
                lost_tracks.append(track_id)
        
        # Clean up old lost tracks (optional - keep metadata for a bit)
        
        return {
            "active_tracks": active_tracks,
            "total_active_tracks": len(active_tracks),
            "lost_tracks": len(lost_tracks),
            "frame_number": self.frame_count,
            "timestamp": timestamp_str,
            "source": source_info
        }
    
    async def _track_continuous(
        self, 
        camera_device: int, 
        use_gstreamer: bool,
        confidence_threshold: float,
        track_history_frames: int,
        duration_seconds: float
    ) -> Dict[str, Any]:
        """Track objects continuously for a specified duration."""
        cap = None
        try:
            cap = self._open_webcam(camera_device, use_gstreamer)
            source_info = f"webcam (device /dev/video{camera_device})"
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            all_tracks_seen = {}  # track_id -> metadata
            
            print(f"[TRACKING] Starting continuous tracking for {duration_seconds} seconds...")
            
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                current_time = datetime.now()
                timestamp_str = current_time.isoformat()
                
                # Run tracking
                results = self.model.track(
                    frame,
                    conf=confidence_threshold,
                    persist=True,
                    verbose=False,
                    imgsz=1280  # Higher resolution for better small object detection
                )
                
                # Process tracked objects
                active_track_ids = set()
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                        track_id_int = int(track_id)
                        active_track_ids.add(track_id_int)
                        
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.model.names[cls_id]
                        bbox = box.xyxy[0].tolist()
                        
                        # Calculate center position
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Update or create track metadata
                        if track_id_int not in all_tracks_seen:
                            all_tracks_seen[track_id_int] = {
                                "track_id": track_id_int,
                                "class": cls_name,
                                "first_seen": timestamp_str,
                                "last_seen": timestamp_str,
                                "first_bbox": bbox,
                                "first_position": {"x": center_x, "y": center_y},
                                "last_bbox": bbox,
                                "last_position": {"x": center_x, "y": center_y},
                                "max_confidence": conf,
                                "min_confidence": conf,
                                "frames_seen": 0,
                                "total_confidence": 0.0,
                                "first_frame_time": time.time(),
                                "last_frame_time": time.time()
                            }
                        
                        # Update track info
                        track_info = all_tracks_seen[track_id_int]
                        track_info["last_seen"] = timestamp_str
                        track_info["last_bbox"] = bbox
                        track_info["last_position"] = {"x": center_x, "y": center_y}
                        track_info["last_frame_time"] = time.time()
                        track_info["max_confidence"] = max(track_info["max_confidence"], conf)
                        track_info["min_confidence"] = min(track_info["min_confidence"], conf)
                        track_info["frames_seen"] += 1
                        track_info["total_confidence"] += conf
                        # Update class (use most recent)
                        track_info["class"] = cls_name
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
            
            elapsed_time = time.time() - start_time
            print(f"[TRACKING] Completed tracking. Processed {self.frame_count} frames in {elapsed_time:.1f} seconds.")
            
            # Build final results with aggregated track data
            active_tracks = []
            for track_id, track_info in all_tracks_seen.items():
                avg_confidence = track_info["total_confidence"] / track_info["frames_seen"] if track_info["frames_seen"] > 0 else 0.0
                
                # Try using SpeedEstimator if available, otherwise fall back to manual calculation
                first_pos = track_info.get("first_position", {"x": 0, "y": 0})
                last_pos = track_info.get("last_position", {"x": 0, "y": 0})
                first_time = track_info.get("first_frame_time", start_time)
                last_time = track_info.get("last_frame_time", start_time)
                
                # Calculate time delta (avoid division by zero)
                time_delta = max(0.001, last_time - first_time)  # At least 1ms
                
                # Try SpeedEstimator approach first
                velocity = None
                distance_moved = None
                average_speed_pixels_per_second = None
                
                if SPEED_ESTIMATOR_AVAILABLE and SpeedEstimator is not None:
                    try:
                        # Calculate pixel displacement
                        dx = last_pos["x"] - first_pos["x"]
                        dy = last_pos["y"] - first_pos["y"]
                        pixel_distance = ((dx ** 2) + (dy ** 2)) ** 0.5
                        
                        # Calculate frames per second (approximate)
                        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 30.0
                        
                        # Calculate time delta in frames
                        frame_delta = track_info["frames_seen"] if track_info["frames_seen"] > 0 else 1
                        dt_seconds = frame_delta / fps if fps > 0 else time_delta
                        
                        # Use SpeedEstimator-style calculation (pixel-based for now)
                        # Note: SpeedEstimator requires meter_per_pixel for real-world speeds
                        # Here we use pixel-based calculation similar to SpeedEstimator's approach
                        if dt_seconds > 0:
                            velocity = {
                                "vx": round(dx / dt_seconds, 2),
                                "vy": round(dy / dt_seconds, 2)
                            }
                            distance_moved = round(pixel_distance, 1)
                            average_speed_pixels_per_second = round(pixel_distance / dt_seconds, 2)
                    except Exception as e:
                        # Fall back to manual calculation if SpeedEstimator approach fails
                        print(f"[TRACKING] SpeedEstimator calculation failed for track {track_id}: {e}. Using fallback.")
                
                # Fallback to manual calculation if SpeedEstimator not used or failed
                if velocity is None:
                    # FALLBACK: Manual calculation (old code preserved)
                    velocity = {
                        "vx": round((last_pos["x"] - first_pos["x"]) / time_delta, 2),
                        "vy": round((last_pos["y"] - first_pos["y"]) / time_delta, 2)
                    }
                    
                    # Total distance moved (in pixels)
                    distance_moved = ((last_pos["x"] - first_pos["x"]) ** 2 + (last_pos["y"] - first_pos["y"]) ** 2) ** 0.5
                    distance_moved = round(distance_moved, 1)
                    
                    average_speed_pixels_per_second = round(distance_moved / time_delta, 2) if time_delta > 0 else 0.0
                
                active_tracks.append({
                    "track_id": track_id,
                    "class": track_info["class"],
                    "frames_seen": track_info["frames_seen"],
                    "avg_confidence": round(avg_confidence, 2),
                    "max_confidence": round(track_info["max_confidence"], 2),
                    "min_confidence": round(track_info["min_confidence"], 2),
                    "first_seen": track_info["first_seen"],
                    "last_seen": track_info["last_seen"],
                    "duration_tracked_seconds": round(elapsed_time, 2),
                    "first_position": {
                        "x": round(first_pos["x"], 1),
                        "y": round(first_pos["y"], 1)
                    },
                    "last_position": {
                        "x": round(last_pos["x"], 1),
                        "y": round(last_pos["y"], 1)
                    },
                    "first_bounding_box": track_info.get("first_bbox", []),
                    "last_bounding_box": track_info.get("last_bbox", []),
                    "velocity": velocity,
                    "distance_moved_pixels": distance_moved,
                    "average_speed_pixels_per_second": average_speed_pixels_per_second
                })
            
            return {
                "active_tracks": active_tracks,
                "total_active_tracks": len(active_tracks),
                "duration_seconds": round(elapsed_time, 2),
                "frames_processed": self.frame_count,
                "timestamp": datetime.now().isoformat(),
                "source": source_info
            }
        finally:
            if cap is not None:
                cap.release()

