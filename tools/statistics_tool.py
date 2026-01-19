#!/usr/bin/env python3
"""
Statistics Aggregation Tool for MCP Server.
Tracks detection statistics over time (counts, average confidence, most common objects).
"""
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from mcp.types import Tool


class StatisticsTool:
    """Statistics aggregation tool for tracking detection statistics."""
    
    def __init__(self):
        """Initialize statistics tracking."""
        self.stats = defaultdict(lambda: {
            "count": 0,
            "total_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 1.0
        })
        self.total_detections = 0
    
    def get_tool_schema(self) -> Tool:
        """Get tool schema for MCP."""
        return Tool(
            name="get_detection_statistics",
            description="Get aggregated statistics about object detections over time. Returns counts, average confidence, and most common objects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "reset": {
                        "type": "boolean",
                        "description": "Reset statistics after returning (default False)",
                        "default": False
                    }
                }
            }
        )
    
    def update(self, detections: List[Dict[str, Any]]):
        """
        Update statistics with new detections.
        
        Args:
            detections: List of detection dicts with 'class' and 'confidence' keys
        """
        for detection in detections:
            cls_name = detection.get("class", "unknown")
            confidence = float(detection.get("confidence", 0.0))
            
            self.stats[cls_name]["count"] += 1
            self.stats[cls_name]["total_confidence"] += confidence
            self.stats[cls_name]["max_confidence"] = max(
                self.stats[cls_name]["max_confidence"],
                confidence
            )
            self.stats[cls_name]["min_confidence"] = min(
                self.stats[cls_name]["min_confidence"],
                confidence
            )
            self.total_detections += 1
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute statistics aggregation tool.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Statistics dictionary
        """
        reset = arguments.get("reset", False)
        
        # Calculate statistics
        detection_count_by_class = {}
        average_confidence_by_class = {}
        
        for cls_name, data in self.stats.items():
            count = data["count"]
            detection_count_by_class[cls_name] = count
            
            if count > 0:
                avg_conf = data["total_confidence"] / count
                average_confidence_by_class[cls_name] = round(avg_conf, 2)
        
        # Find most common object
        most_common_object = None
        max_count = 0
        for cls_name, count in detection_count_by_class.items():
            if count > max_count:
                max_count = count
                most_common_object = cls_name
        
        result = {
            "total_detections": self.total_detections,
            "detection_count_by_class": detection_count_by_class,
            "average_confidence_by_class": average_confidence_by_class,
            "most_common_object": most_common_object,
            "detection_timestamp": datetime.now().isoformat()
        }
        
        # Reset if requested
        if reset:
            self.stats.clear()
            self.total_detections = 0
        
        return result

