#!/usr/bin/env python3
"""
Test MCP tools directly without full MCP server.
Useful for quick testing of tool functionality.
"""
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

async def test_yolo_tool():
    """Test YOLO detection tool with a sample image."""
    from tools.yolo_detection import YOLODetectionTool
    
    print("Testing YOLO Object Detection Tool")
    print("=" * 50)
    
    tool = YOLODetectionTool()
    
    # Test tool schema
    schema = tool.get_tool_schema()
    print(f"\nTool: {schema.name}")
    print(f"Description: {schema.description}")
    print(f"Schema: {json.dumps(schema.inputSchema, indent=2)}")
    
    # Test with a sample image (if available)
    # You can use any image file for testing
    test_image = Path(__file__).parent.parent / "test_image.jpg"
    
    if test_image.exists():
        print(f"\nTesting with image: {test_image}")
        try:
            result = await tool.execute({
                "image_path": str(test_image),
                "confidence_threshold": 0.25
            })
            print(f"\nResult:")
            print(json.dumps(result, indent=2))
            return result.get("success", False)
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"\n[WARNING] No test image found at {test_image}")
        print("  Tool schema is valid. To test execution, place an image at:")
        print(f"  {test_image}")
        print("\n[OK] Tool registration successful")
        return True

async def main():
    """Run tool tests."""
    import asyncio
    
    print("MCP Tools Test")
    print("=" * 50)
    
    results = await asyncio.gather(
        test_yolo_tool()
    )
    
    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All tool tests passed!")
        return 0
    else:
        print("[WARNING] Some tool tests had issues")
        return 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))

