#!/usr/bin/env python3
"""
Remote YOLO Detection Tool

Allows the orin-nano agent to query old-nano for YOLO object detection.
When a person is detected, old-nano sends a desktop notification to orin-nano.
"""
import json
import asyncio
from typing import Optional, Dict, Any
import httpx
from langchain.tools import tool
from langchain_core.tools import ToolException


async def query_remote_yolo(
    host: str = "old-nano",
    port: int = 8080,
    source: str = "camera",
    send_notification: bool = True,
    confidence_threshold: float = 0.25,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Query remote device (old-nano) for YOLO object detection.
    
    Args:
        host: Hostname or IP of old-nano
        port: HTTP port of YOLO service
        source: "camera" for webcam or path to image file
        send_notification: If True, old-nano will send desktop notification when person detected
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with detection results
    """
    url = f"http://{host}:{port}/detect"
    
    payload = {
        "source": source,
        "send_notification": send_notification,
        "confidence_threshold": confidence_threshold
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as e:
        raise ToolException(
            f"Cannot connect to {host}:{port}. "
            f"Ensure old-nano YOLO service is running. Error: {e}"
        )
    except httpx.TimeoutException:
        raise ToolException(
            f"Request to {host}:{port} timed out after {timeout}s. "
            f"Old-nano may be busy or unreachable."
        )
    except httpx.HTTPStatusError as e:
        raise ToolException(
            f"HTTP error from {host}:{port}: {e.response.status_code}. "
            f"Response: {e.response.text}"
        )
    except Exception as e:
        raise ToolException(f"Error querying remote YOLO: {str(e)}")


@tool
async def remote_yolo_detection(
    host: str = "old-nano",
    send_notification: bool = True,
    confidence_threshold: float = 0.25,
    source: str = "camera"
) -> str:
    """
    Query remote device (old-nano) for YOLO object detection.
    If a person is detected, old-nano will send a desktop notification ("ALERT: PERSON") to orin-nano.
    
    Use this tool when you need to check what old-nano sees, or when asked to query the remote device.
    This is useful for distributed detection across multiple devices.
    
    Args:
        host: Hostname or IP of old-nano (default: "old-nano")
        send_notification: If True, old-nano sends desktop notification when person detected (default: True)
        confidence_threshold: Detection confidence threshold 0.0-1.0 (default: 0.25)
        source: "camera" for webcam or path to image file (default: "camera")
        
    Returns:
        Human-readable string with detection results
        
    Example:
        "Query old-nano for person detection" -> Calls this tool
        "Check what old-nano sees" -> Calls this tool
        "If you see a person object on old-nano, make an alert" -> Calls this tool
    """
    try:
        result = await query_remote_yolo(
            host=host,
            source=source,
            send_notification=send_notification,
            confidence_threshold=confidence_threshold
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            raise ToolException(f"Remote YOLO detection failed: {error_msg}")
        
        detections = result.get("detections", [])
        source_info = result.get("source", source)
        
        # Check for person detection and notification status
        persons = [d for d in detections if d.get("class", "").lower() == "person"]
        notification_sent = result.get("notification_sent", False)
        
        # Build response
        output_lines = [
            f"Remote detection from {host} completed on {source_info}."
        ]
        
        if not detections:
            output_lines.append("Result: No objects detected.")
        else:
            output_lines.append(f"Found {result.get('count', 0)} object(s):")
            for det in detections:
                cls_name = det.get("class", "unknown")
                conf = det.get("confidence", 0.0)
                output_lines.append(f"- {cls_name} (confidence: {conf:.1%})")
        
        # Add notification status
        if persons and send_notification:
            if notification_sent:
                output_lines.append(f"\n✓ Alert sent: Person detected on {host}!")
            else:
                output_lines.append(f"\n⚠ Person detected but notification may have failed.")
        
        return "\n".join(output_lines)
        
    except ToolException:
        # Re-raise ToolException as-is
        raise
    except Exception as e:
        raise ToolException(f"Error running remote YOLO detection: {str(e)}")


# Synchronous wrapper for compatibility
def remote_yolo_detection_sync(
    host: str = "old-nano",
    send_notification: bool = True,
    confidence_threshold: float = 0.25,
    source: str = "camera"
) -> str:
    """Synchronous wrapper for remote_yolo_detection."""
    return asyncio.run(remote_yolo_detection.ainvoke({
        "host": host,
        "send_notification": send_notification,
        "confidence_threshold": confidence_threshold,
        "source": source
    }))

