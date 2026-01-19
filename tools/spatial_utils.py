#!/usr/bin/env python3
"""
Spatial relationship utilities for object detection.
Computes spatial relationships between objects (left/right/above/below/near).
"""
from typing import List, Dict, Any, Optional, Tuple
import math


def compute_object_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute center point of a bounding box.
    
    Args:
        bbox: Dict with 'x1', 'y1', 'x2', 'y2' keys
    
    Returns:
        (center_x, center_y) tuple
    """
    return (
        (bbox["x1"] + bbox["x2"]) / 2.0,
        (bbox["y1"] + bbox["y2"]) / 2.0
    )


def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        point1: (x, y) tuple
        point2: (x, y) tuple
    
    Returns:
        Distance in pixels
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_cardinal_direction(point1: Tuple[float, float], point2: Tuple[float, float]) -> str:
    """
    Compute cardinal direction from point1 to point2.
    
    In image coordinates:
    - X increases to the right (East)
    - Y increases downward (South)
    - Therefore: -X = West, +X = East, -Y = North, +Y = South
    
    Args:
        point1: (x, y) tuple of reference point (e.g., ID5 car)
        point2: (x, y) tuple of target point (e.g., ID1 car)
    
    Returns:
        Cardinal direction string (e.g., "northwest", "southeast", "north", etc.)
    """
    dx = point2[0] - point1[0]  # Positive = target is to the right (East) of reference
    dy = point2[1] - point1[1]  # Positive = target is below (South) of reference
    
    # Handle edge case: same position
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return "at the same position"
    
    # Calculate angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
    # In image coordinates: angle = atan2(dx, -dy) * 180 / pi
    # We negate dy because in images Y increases downward (South), but in standard coordinates Y increases upward (North)
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    
    # Determine cardinal direction based on angle
    # North: 0° (±22.5° = 337.5°-22.5°)
    # Northeast: 45° (22.5°-67.5°)
    # East: 90° (67.5°-112.5°)
    # Southeast: 135° (112.5°-157.5°)
    # South: 180° (157.5°-202.5°)
    # Southwest: 225° (202.5°-247.5°)
    # West: 270° (247.5°-292.5°)
    # Northwest: 315° (292.5°-337.5°)
    
    if angle_deg >= 337.5 or angle_deg < 22.5:
        return "north"
    elif 22.5 <= angle_deg < 67.5:
        return "northeast"
    elif 67.5 <= angle_deg < 112.5:
        return "east"
    elif 112.5 <= angle_deg < 157.5:
        return "southeast"
    elif 157.5 <= angle_deg < 202.5:
        return "south"
    elif 202.5 <= angle_deg < 247.5:
        return "southwest"
    elif 247.5 <= angle_deg < 292.5:
        return "west"
    else:  # 292.5 <= angle_deg < 337.5
        return "northwest"


def compute_spatial_relationships(
    detections: List[Dict[str, Any]],
    near_threshold: float = 200.0
) -> Dict[str, Dict[str, Any]]:
    """
    Compute spatial relationships between detected objects.
    
    Args:
        detections: List of detection dicts with 'bbox' keys containing x1, y1, x2, y2
        near_threshold: Distance threshold in pixels for "near" relationship (default: 200)
    
    Returns:
        Dict mapping "class1_index_to_class2_index" to relationship dict containing:
        - left_of: bool (A is left of B if A.x_center < B.x_center)
        - right_of: bool
        - above: bool (A is above B if A.y_center < B.y_center)
        - below: bool
        - near: bool (distance < threshold)
        - distance: float (distance in pixels)
    """
    relationships = {}
    
    for i, det1 in enumerate(detections):
        bbox1 = det1.get("bbox", {})
        center1 = compute_object_center(bbox1)
        class1 = det1.get("class", "unknown")
        
        for j, det2 in enumerate(detections):
            if i == j:
                continue
                
            bbox2 = det2.get("bbox", {})
            center2 = compute_object_center(bbox2)
            class2 = det2.get("class", "unknown")
            
            # Compute relationships
            distance = compute_distance(center1, center2)
            
            # Compute cardinal direction from obj2 to obj1 (direction obj1 is relative to obj2)
            cardinal_dir = compute_cardinal_direction(center2, center1)
            
            key = f"{class1}_{i}_to_{class2}_{j}"
            relationships[key] = {
                "left_of": center1[0] < center2[0],
                "right_of": center1[0] > center2[0],
                "above": center1[1] < center2[1],
                "below": center1[1] > center2[1],
                "near": distance < near_threshold,
                "distance": round(distance, 2),
                "cardinal_direction": cardinal_dir  # Direction from obj2 to obj1
            }
    
    return relationships


def find_objects_by_spatial_query(
    detections: List[Dict[str, Any]],
    relationships: Dict[str, Dict[str, Any]],
    target_class: str,
    spatial_relation: str,
    reference_class: str
) -> List[Dict[str, Any]]:
    """
    Find objects matching spatial criteria.
    
    Example: "car next to fire hydrant" -> find cars where near(fire hydrant) = True
    
    Args:
        detections: List of detection dicts
        relationships: Spatial relationships dict from compute_spatial_relationships()
        target_class: Class to find (e.g., "car")
        spatial_relation: Relationship type ("left_of", "right_of", "above", "below", "near")
        reference_class: Reference object class (e.g., "fire hydrant")
    
    Returns:
        List of matching detection dicts
    """
    matching_detections = []
    
    # Find reference objects
    reference_indices = [
        i for i, det in enumerate(detections)
        if det.get("class", "").lower() == reference_class.lower()
    ]
    
    if not reference_indices:
        return matching_detections
    
    # Find target objects that have the specified relationship to any reference object
    for i, det in enumerate(detections):
        if det.get("class", "").lower() != target_class.lower():
            continue
        
        # Check if this target object has the relationship to any reference object
        for ref_idx in reference_indices:
            # Look up relationship
            ref_det = detections[ref_idx]
            ref_class = ref_det.get("class", "unknown")
            target_class_name = det.get("class", "unknown")
            
            # Try both directions (A to B and B to A)
            key1 = f"{target_class_name}_{i}_to_{ref_class}_{ref_idx}"
            key2 = f"{ref_class}_{ref_idx}_to_{target_class_name}_{i}"
            
            rel = relationships.get(key1) or relationships.get(key2)
            
            if rel and rel.get(spatial_relation, False):
                matching_detections.append(det)
                break  # Found a match, no need to check other references
    
    return matching_detections


def format_spatial_context(detections: List[Dict[str, Any]], relationships: Dict[str, Dict[str, Any]]) -> str:
    """
    Format detections and relationships as a structured context string for LLM reasoning.
    
    Args:
        detections: List of detection dicts
        relationships: Spatial relationships dict
    
    Returns:
        Formatted string describing objects and their spatial relationships
    """
    lines = ["Detected objects:"]
    
    for i, det in enumerate(detections):
        bbox = det.get("bbox", {})
        center = compute_object_center(bbox)
        class_name = det.get("class", "unknown")
        confidence = det.get("confidence", 0.0)
        color = det.get("color_name", None)
        
        info = f"  {i}. {class_name} (confidence: {confidence:.1%}, center: ({center[0]:.1f}, {center[1]:.1f}))"
        if color:
            info += f", color: {color}"
        lines.append(info)
    
    # Add key spatial relationships (only "near" and directional for clarity)
    lines.append("\nKey spatial relationships:")
    for key, rel in relationships.items():
        if rel.get("near", False) or any([rel.get("left_of"), rel.get("right_of"), rel.get("above"), rel.get("below")]):
            parts = key.split("_to_")
            if len(parts) == 2:
                obj1 = parts[0].rsplit("_", 1)[0]  # Remove index
                obj2 = parts[1].rsplit("_", 1)[0]
                
                rel_desc = []
                if rel.get("left_of"):
                    rel_desc.append("left of")
                if rel.get("right_of"):
                    rel_desc.append("right of")
                if rel.get("above"):
                    rel_desc.append("above")
                if rel.get("below"):
                    rel_desc.append("below")
                if rel.get("near"):
                    rel_desc.append(f"near (distance: {rel.get('distance', 0):.1f}px)")
                
                if rel_desc:
                    lines.append(f"  - {obj1} is {' and '.join(rel_desc)} {obj2}")
    
    return "\n".join(lines)

