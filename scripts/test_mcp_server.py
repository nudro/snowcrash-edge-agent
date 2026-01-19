#!/usr/bin/env python3
"""
Simple test script to verify MCP server works.
Tests tool registration and basic functionality.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        from tools.yolo_detection import YOLODetectionTool
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_tool_registration():
    """Test tool registration."""
    print("\nTesting tool registration...")
    try:
        from tools.yolo_detection import YOLODetectionTool
        tool = YOLODetectionTool()
        schema = tool.get_tool_schema()
        print(f"[OK] Tool registered: {schema.name}")
        print(f"  Description: {schema.description}")
        return True
    except Exception as e:
        print(f"[FAIL] Tool registration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_creation():
    """Test MCP server creation."""
    print("\nTesting server creation...")
    try:
        from mcp.server import Server
        app = Server("snowcrash-mcp-test")
        print(f"[OK] Server created: {app.name}")
        return True
    except Exception as e:
        print(f"[FAIL] Server creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing MCP Server Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_tool_registration,
        test_server_creation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All tests passed!")
        print("\nMCP server is ready. You can now run:")
        print("  python3 mcp_server/server.py")
        return 0
    else:
        print("[FAIL] Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

