#!/usr/bin/env python3
"""
Test webcam capture with YOLO detection.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.yolo_detection import YOLODetectionTool


async def test_webcam_capture():
    """Test webcam capture with YOLO."""
    print("Testing Webcam Capture with YOLO")
    print("=" * 50)
    print()
    
    tool = YOLODetectionTool()
    
    print("[CAMERA] Capturing frame from webcam...")
    print("   Device: /dev/video0")
    print("   Using: GStreamer (primary), OpenCV (fallback)")
    print()
    
    try:
        # Test webcam capture
        result = await tool.execute({
            "image_path": "webcam",
            "camera_device": 0,
            "use_gstreamer": True,
            "confidence_threshold": 0.25
        })
        
        if result.get("success"):
            print("[OK] Webcam capture successful!")
            print()
            print(f"Source: {result.get('source')}")
            print(f"Objects detected: {result.get('count', 0)}")
            print()
            
            detections = result.get("detections", [])
            if detections:
                print("Detected objects:")
                for i, det in enumerate(detections, 1):
                    cls = det.get("class", "unknown")
                    conf = det.get("confidence", 0.0)
                    print(f"  {i}. {cls} (confidence: {conf:.2%})")
            else:
                print("No objects detected in the captured frame.")
            
            print()
            print("[OK] Webcam test completed successfully!")
            return True
        else:
            error = result.get("error", "Unknown error")
            print(f"[FAIL] Webcam capture failed: {error}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run webcam test."""
    success = await test_webcam_capture()
    
    print()
    print("=" * 50)
    if success:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

