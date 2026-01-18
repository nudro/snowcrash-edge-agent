#!/usr/bin/env python3
"""
LangChain tool wrappers for MCP tools.
Converts MCP tools to LangChain-compatible tools.
"""
import json
import asyncio
from typing import Any, Dict, Optional
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import ToolException

from tools.yolo_detection import YOLODetectionTool
from tools.statistics_tool import StatisticsTool
from tools.distance_tool import DistanceTool
from tools.tracking_tool import TrackingTool
from tools.geographic_tool import GeographicTool

# Initialize MCP tools
_yolo_tool = YOLODetectionTool()
_statistics_tool = StatisticsTool()
_distance_tool = DistanceTool()
_tracking_tool = TrackingTool()
_geographic_tool = GeographicTool()

@tool
async def yolo_object_detection(
    image_path: str,
    confidence_threshold: float = 0.25
) -> str:
    """
    Perform object detection on an image using YOLO.
    
    Use this tool when the user asks to detect objects in an image, 
    identify what's in a picture, analyze an image, or find objects in a video.
    
    Args:
        image_path: Path to the image file to analyze
        confidence_threshold: Confidence threshold (0.0-1.0), default 0.25
        
    Returns:
        JSON string with detection results including detected objects, 
        their classes, confidence scores, and bounding boxes.
        
    Example:
        "Detect objects in /path/to/image.jpg"
    """
    try:
        result = await _yolo_tool.execute({
            "image_path": image_path,
            "confidence_threshold": confidence_threshold
        })
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            raise ToolException(f"YOLO detection failed: {error_msg}")
        
        # Update statistics tool with detections
        if "detections" in result:
            _statistics_tool.update(result["detections"])
        
        # Format result for readable output
        detections = result.get("detections", [])
        source = result.get("source", image_path)
        
        if not detections:
            return f"Detection completed on {source}. Result: No objects detected."
        
        # Create human-readable output
        output_lines = [f"Detection completed on {source}. Found {result.get('count', 0)} object(s):"]
        for det in detections:
            cls_name = det.get("class", "unknown")
            conf = det.get("confidence", 0.0)
            output_lines.append(f"- {cls_name} (confidence: {conf:.1%})")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        raise ToolException(f"Error running YOLO detection: {str(e)}")


# Synchronous wrapper for compatibility
def yolo_object_detection_sync(image_path: str, confidence_threshold: float = 0.25) -> str:
    """Synchronous wrapper for yolo_object_detection."""
    return asyncio.run(yolo_object_detection.ainvoke({
        "image_path": image_path,
        "confidence_threshold": confidence_threshold
    }))


@tool
async def get_detection_statistics(reset: bool = False) -> str:
    """
    Get aggregated statistics about object detections over time.
    Returns counts, average confidence, and most common objects.
    
    Args:
        reset: Reset statistics after returning (default False)
    """
    result = await _statistics_tool.execute({"reset": reset})
    return json.dumps(result, indent=2)


@tool
async def estimate_object_distances(
    image_path: str,
    camera_device: int = 0,
    confidence_threshold: float = 0.25
) -> str:
    """
    Estimate distance from camera to detected objects.
    Uses bounding box height and known object sizes.
    
    Args:
        image_path: Path to image file OR 'webcam' or 'camera' to capture from webcam
        camera_device: Camera device index (default 0). Only used when image_path is 'webcam' or 'camera'
        confidence_threshold: Confidence threshold (0.0-1.0), default 0.25
    """
    result = await _distance_tool.execute({
        "image_path": image_path,
        "camera_device": camera_device,
        "confidence_threshold": confidence_threshold
    })
    return json.dumps(result, indent=2)


@tool
async def track_objects(
    image_path: str,
    camera_device: int = 0,
    confidence_threshold: float = 0.25,
    track_history_frames: int = 30,
    duration_seconds: float = 0
) -> str:
    """
    Track objects across frames with persistent IDs using YOLO's built-in tracker.
    
    Args:
        image_path: Path to image file OR 'webcam' or 'camera' to capture from webcam
        camera_device: Camera device index (default 0). Only used when image_path is 'webcam' or 'camera'
        confidence_threshold: Confidence threshold (0.0-1.0), default 0.25
        track_history_frames: Number of frames to keep in track history (default 30, 0 to disable)
        duration_seconds: Duration to track in seconds (0-60, default 0 for single frame). Only used with webcam.
    """
    result = await _tracking_tool.execute({
        "image_path": image_path,
        "camera_device": camera_device,
        "confidence_threshold": confidence_threshold,
        "track_history_frames": track_history_frames,
        "duration_seconds": duration_seconds
    })
    return json.dumps(result, indent=2)


@tool
async def estimate_object_geography(
    image_path: str,
    camera_latitude: float,
    camera_longitude: float,
    camera_device: int = 0,
    camera_altitude: float = 0.0,
    camera_yaw: float = 0.0,
    camera_pitch: float = 0.0,
    confidence_threshold: float = 0.25
) -> str:
    """
    Estimate geographic location (latitude/longitude) of detected objects by fusing YOLO detections with GPS coordinates.
    Uses Shapely (memory-efficient) for geographic calculations.
    
    Args:
        image_path: Path to image file OR 'webcam' or 'camera' to capture from webcam
        camera_latitude: Camera GPS latitude (degrees, required)
        camera_longitude: Camera GPS longitude (degrees, required)
        camera_device: Camera device index (default 0). Only used when image_path is 'webcam' or 'camera'
        camera_altitude: Camera altitude in meters (default 0.0)
        camera_yaw: Camera yaw angle in degrees (0=North, 90=East, default 0.0)
        camera_pitch: Camera pitch angle in degrees (0=horizontal, default 0.0)
        confidence_threshold: Confidence threshold (0.0-1.0), default 0.25
    """
    result = await _geographic_tool.execute({
        "image_path": image_path,
        "camera_device": camera_device,
        "camera_latitude": camera_latitude,
        "camera_longitude": camera_longitude,
        "camera_altitude": camera_altitude,
        "camera_yaw": camera_yaw,
        "camera_pitch": camera_pitch,
        "confidence_threshold": confidence_threshold
    })
    return json.dumps(result, indent=2)


def get_langchain_tools() -> list:
    """Get all LangChain tools."""
    return [
        yolo_object_detection,
        get_detection_statistics,
        estimate_object_distances,
        track_objects,
        estimate_object_geography
    ]

