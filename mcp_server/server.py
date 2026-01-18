#!/usr/bin/env python3
"""
MCP Server for Snowcrash.
Provides agentic SLM capabilities with local tools (YOLO object detection, etc.)
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import tools
from tools.yolo_detection import YOLODetectionTool
from tools.statistics_tool import StatisticsTool
from tools.distance_tool import DistanceTool
from tools.tracking_tool import TrackingTool
from tools.geographic_tool import GeographicTool

# Initialize MCP server
app = Server("snowcrash-mcp")

# Register tools
yolo_tool = YOLODetectionTool()
statistics_tool = StatisticsTool()
distance_tool = DistanceTool()
tracking_tool = TrackingTool()
geographic_tool = GeographicTool()

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    tools = [
        yolo_tool.get_tool_schema(),
        statistics_tool.get_tool_schema(),
        distance_tool.get_tool_schema(),
        tracking_tool.get_tool_schema(),
        geographic_tool.get_tool_schema()
    ]
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tool."""
    if name == "yolo_object_detection":
        result = await yolo_tool.execute(arguments)
        # Update statistics from detections
        if result.get("success") and "detections" in result:
            statistics_tool.update(result["detections"])
        return [TextContent(type="text", text=json.dumps(result))]
    
    elif name == "get_detection_statistics":
        result = await statistics_tool.execute(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    
    elif name == "estimate_object_distances":
        result = await distance_tool.execute(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    
    elif name == "track_objects":
        result = await tracking_tool.execute(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    
    elif name == "estimate_object_geography":
        result = await geographic_tool.execute(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

