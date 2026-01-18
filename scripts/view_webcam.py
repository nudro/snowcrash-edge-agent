#!/usr/bin/env python3
"""
View live webcam feed in a window.
Press 'q' to quit, 's' to save a frame, 'd' to detect objects with YOLO.
"""
import sys
import cv2
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing YOLO for object detection
try:
    from tools.yolo_detection import YOLODetectionTool
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available. Object detection disabled.")


def view_webcam_with_yolo(device=0, use_gstreamer=True):
    """View webcam with optional YOLO object detection overlay."""
    
    # Try GStreamer pipeline first (better for Jetson)
    if use_gstreamer:
        gstreamer_pipeline = (
            f"v4l2src device=/dev/video{device} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink"
        )
        
        print(f"Attempting to open webcam with GStreamer...")
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("GStreamer failed, falling back to standard OpenCV...")
            cap = cv2.VideoCapture(device)
    else:
        cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        print(f"[FAIL] Cannot open camera device /dev/video{device}")
        return 1
    
    print(f"[OK] Webcam opened: /dev/video{device}")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    if YOLO_AVAILABLE:
        print("  'd' - Run YOLO object detection on current frame")
    print()
    
    # Initialize YOLO if available
    yolo_tool = None
    if YOLO_AVAILABLE:
        try:
            yolo_tool = YOLODetectionTool()
            print("YOLO loaded. Press 'd' to detect objects.")
        except Exception as e:
            print(f"[WARNING] Could not load YOLO: {e}")
            yolo_tool = None
    
    frame_count = 0
    saved_count = 0
    
    print("Displaying webcam feed... (Press 'q' to quit)")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Display frame
            display_frame = frame.copy()
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show webcam feed
            cv2.imshow("Webcam Feed - Press 'q' to quit, 'd' for detection", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save frame
                saved_count += 1
                filename = f"webcam_frame_{saved_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Frame saved: {filename}")
            elif key == ord('d') and yolo_tool:
                # Run YOLO detection
                print("Running YOLO detection...")
                
                # Save frame temporarily for YOLO
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, frame)
                
                try:
                    # Run detection
                    result = asyncio.run(yolo_tool.execute({
                        "image_path": tmp_path,
                        "confidence_threshold": 0.25
                    }))
                    
                    if result.get("success"):
                        detections = result.get("detections", [])
                        print(f"\n[OK] Detected {len(detections)} object(s):")
                        
                        # Draw detections on frame
                        for det in detections:
                            bbox = det.get("bbox", {})
                            x1 = int(bbox.get("x1", 0))
                            y1 = int(bbox.get("y1", 0))
                            x2 = int(bbox.get("x2", 0))
                            y2 = int(bbox.get("y2", 0))
                            cls = det.get("class", "unknown")
                            conf = det.get("confidence", 0.0)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{cls}: {conf:.2%}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            print(f"  - {cls} ({conf:.2%})")
                        
                        # Show frame with detections
                        cv2.imshow("Webcam Feed - Press 'q' to quit, 'd' for detection", frame)
                        print()
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"[FAIL] Detection failed: {error}")
                    
                    # Clean up temp file
                    Path(tmp_path).unlink()
                    
                except Exception as e:
                    print(f"[FAIL] Error during detection: {e}")
                    # Clean up temp file on error
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[OK] Closed webcam. Processed {frame_count} frames.")
        if saved_count > 0:
            print(f"  Saved {saved_count} frame(s).")
    
    return 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View live webcam feed")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--no-gstreamer",
        action="store_true",
        help="Disable GStreamer, use standard OpenCV"
    )
    
    args = parser.parse_args()
    
    return view_webcam_with_yolo(
        device=args.device,
        use_gstreamer=not args.no_gstreamer
    )


if __name__ == "__main__":
    sys.exit(main())

