"""
Color detection utility for edge devices.
Detects dominant colors of objects using OpenCV - no internet required.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# Color name mapping (RGB to color names)
COLOR_NAMES = {
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (128, 128, 128): "gray",
    (192, 192, 192): "silver",
    (255, 0, 0): "red",
    (255, 165, 0): "orange",
    (255, 255, 0): "yellow",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (75, 0, 130): "indigo",
    (238, 130, 238): "violet",
    (255, 20, 147): "pink",
    (165, 42, 42): "brown",
    (255, 192, 203): "pink",
    (0, 191, 255): "sky blue",
    (139, 0, 139): "dark magenta",
    (255, 140, 0): "dark orange",
    (50, 50, 50): "dark gray",
    (128, 0, 128): "purple",
    (255, 215, 0): "gold",
    (255, 228, 225): "misty rose",
}


def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Map RGB values to color name using nearest color match.
    
    Args:
        rgb: RGB tuple (R, G, B)
    
    Returns:
        Color name string
    """
    r, g, b = rgb
    
    # Find nearest color in COLOR_NAMES using Euclidean distance
    min_distance = float('inf')
    nearest_color = "unknown"
    
    for color_rgb, color_name in COLOR_NAMES.items():
        distance = np.sqrt(
            (r - color_rgb[0]) ** 2 + 
            (g - color_rgb[1]) ** 2 + 
            (b - color_rgb[2]) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            nearest_color = color_name
    
    return nearest_color


def _get_dominant_color_region(region: np.ndarray) -> Tuple[int, int, int]:
    """
    Get dominant color from an image region using k-means clustering.
    
    Args:
        region: Image region (numpy array, BGR format from OpenCV)
    
    Returns:
        RGB tuple (R, G, B) of dominant color
    """
    # Reshape to list of pixels
    pixels = region.reshape(-1, 3)
    
    # Convert BGR to RGB
    pixels_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    pixels_rgb = pixels_rgb.reshape(-1, 3)
    
    # Use k-means to find dominant color (k=1 means just the mean)
    # For efficiency on edge devices, we can use k=1 or calculate mean directly
    # k=1: just get the average color
    if len(pixels_rgb) == 0:
        return (0, 0, 0)
    
    # Calculate mean RGB values
    mean_rgb = pixels_rgb.mean(axis=0).astype(int)
    
    # Alternatively, use k-means for more accurate dominant color (k=1-3)
    # For edge devices, mean is faster and usually sufficient
    try:
        # Use median for more robust color (less affected by outliers)
        dominant_rgb = np.median(pixels_rgb, axis=0).astype(int)
    except:
        dominant_rgb = mean_rgb
    
    return tuple(dominant_rgb.tolist())


def detect_object_colors(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    object_class: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect colors of objects in a frame using bounding boxes.
    
    Args:
        frame: OpenCV image (BGR format)
        detections: List of detection dicts with 'bbox' keys containing x1, y1, x2, y2
        object_class: Optional filter for specific object class (e.g., "car")
    
    Returns:
        List of detections with added 'color_rgb' and 'color_name' fields
    """
    detections_with_colors = []
    
    for detection in detections:
        # Filter by class if specified
        if object_class and detection.get("class") != object_class:
            continue
        
        bbox = detection.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Extract region of interest (ROI)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
        
        # Get dominant color from ROI
        try:
            color_rgb = _get_dominant_color_region(roi)
            color_name = _rgb_to_color_name(color_rgb)
            
            # Add color info to detection
            detection_with_color = detection.copy()
            detection_with_color["color_rgb"] = color_rgb
            detection_with_color["color_name"] = color_name
            
            detections_with_colors.append(detection_with_color)
        except Exception as e:
            # If color detection fails, keep original detection without color
            detections_with_colors.append(detection)
    
    return detections_with_colors


def detect_colors_from_yolo_results_with_masks(
    frame: np.ndarray,
    yolo_results,
    object_class: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect colors from YOLO segmentation results using masks (more accurate).
    
    Uses segmentation masks to extract only object pixels, avoiding background
    contamination that causes inaccurate color detection.
    
    Args:
        frame: OpenCV image (BGR format)
        yolo_results: YOLO model results object (from segmentation model like yolo26n-seg.pt)
        object_class: Optional filter for specific object class
    
    Returns:
        List of detections with color information
    """
    detections = []
    
    result = yolo_results[0]
    
    # Check if we have boxes (detections)
    if result.boxes is None:
        return []
    
    # Check if we have masks (segmentation)
    has_masks = result.masks is not None
    
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls)
        cls_name = result.names[cls_id]
        
        # Filter by class if specified
        if object_class and cls_name != object_class:
            continue
        
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        # Use segmentation mask if available, otherwise fall back to bounding box
        if has_masks and i < len(result.masks.data):
            try:
                # Get mask for this detection
                mask = result.masks.data[i].cpu().numpy()
                
                # Resize mask to frame size if needed
                h, w = frame.shape[:2]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Create binary mask (threshold at 0.5)
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Get pixels where mask is 1
                mask_indices = np.where(mask_binary == 1)
                if len(mask_indices[0]) > 0:
                    # Extract BGR values from masked region (frame is BGR)
                    masked_pixels_bgr = frame[mask_indices[0], mask_indices[1]]  # Shape: (N, 3) where N = number of masked pixels
                    
                    # Convert BGR to RGB by reversing the channel order
                    # BGR format: [B, G, R] -> RGB format: [R, G, B]
                    masked_pixels_rgb = masked_pixels_bgr[:, [2, 1, 0]]  # Reverse channels: BGR -> RGB
                    
                    # Get dominant color from masked pixels only
                    try:
                        # Use median for more robust color (less affected by outliers)
                        dominant_rgb = np.median(masked_pixels_rgb, axis=0).astype(int)
                        color_name = _rgb_to_color_name(tuple(dominant_rgb.tolist()))
                    except:
                        # Fallback to mean
                        dominant_rgb = masked_pixels_rgb.mean(axis=0).astype(int)
                        color_name = _rgb_to_color_name(tuple(dominant_rgb.tolist()))
                else:
                    # Fallback to bounding box if mask is empty
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        dominant_rgb = _get_dominant_color_region(roi)
                        color_name = _rgb_to_color_name(dominant_rgb)
                    else:
                        dominant_rgb = (128, 128, 128)
                        color_name = "gray"
            except Exception as e:
                # Fallback to bounding box method if mask processing fails
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    dominant_rgb = _get_dominant_color_region(roi)
                    color_name = _rgb_to_color_name(dominant_rgb)
                else:
                    dominant_rgb = (128, 128, 128)
                    color_name = "gray"
        else:
            # Fallback to bounding box method if no masks
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                dominant_rgb = _get_dominant_color_region(roi)
                color_name = _rgb_to_color_name(dominant_rgb)
            else:
                dominant_rgb = (128, 128, 128)
                color_name = "gray"
        
        detection = {
            "class": cls_name,
            "confidence": float(box.conf),
            "bbox": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3])
            },
            "color_rgb": tuple(dominant_rgb.tolist()),
            "color_name": color_name
        }
        
        detections.append(detection)
    
    return detections


def detect_colors_from_yolo_results(
    frame: np.ndarray,
    yolo_results,
    object_class: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect colors from YOLO results directly.
    
    This function now tries to use masks if available (from segmentation models),
    otherwise falls back to bounding box method.
    
    Args:
        frame: OpenCV image (BGR format)
        yolo_results: YOLO model results object
        object_class: Optional filter for specific object class
    
    Returns:
        List of detections with color information
    """
    # Try mask-based detection first (more accurate with segmentation models)
    try:
        return detect_colors_from_yolo_results_with_masks(frame, yolo_results, object_class)
    except Exception:
        # Fallback to original bounding box method
        detections = []
        
        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes:
                cls_id = int(box.cls)
                cls_name = yolo_results[0].names[cls_id]
                
                # Filter by class if specified
                if object_class and cls_name != object_class:
                    continue
                
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                detection = {
                    "class": cls_name,
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3])
                    }
                }
                
                detections.append(detection)
        
        # Detect colors for all detections using bounding box method
        return detect_object_colors(frame, detections, object_class)

