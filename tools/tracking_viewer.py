#!/usr/bin/env python3
"""
GUI viewer for object detection and tracking.
Displays live video with bounding boxes, track IDs, and trajectories.
"""
import cv2
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def run_tracking_viewer(
    model_path: str = "/home/ordun/Documents/snowcrash/models/yolo26n-seg.pt",
    device: int = 0,
    confidence_threshold: float = 0.25,
    use_gstreamer: bool = True,
    duration_seconds: float = 0,  # 0 = run until 'q' pressed
    show_trajectories: bool = True,
    trajectory_length: int = 30
):
    """
    Display live tracking viewer with bounding boxes, track IDs, and trajectories.
    
    Args:
        model_path: Path to YOLO model file
        device: Camera device index
        confidence_threshold: Detection confidence threshold
        use_gstreamer: Use GStreamer backend (recommended for Jetson)
        duration_seconds: Duration to run (0 = until 'q' pressed)
        show_trajectories: Show trajectory trails for tracked objects
        trajectory_length: Number of previous positions to show in trajectory
    """
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
    
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Setup webcam capture
    if use_gstreamer:
        gstreamer_pipeline = (
            f"v4l2src device=/dev/video{device} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink"
        )
        
        print(f"Opening webcam with GStreamer: /dev/video{device}")
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("GStreamer failed, falling back to standard OpenCV...")
            cap = cv2.VideoCapture(device)
    else:
        cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device /dev/video{device}")
    
    print(f"[OK] Webcam opened: /dev/video{device}")
    print("Displaying tracking viewer. Press 'q' to quit.")
    print()
    
    frame_count = 0
    start_time = time.time()
    elapsed = 0.0
    
    # Store trajectories: track_id -> list of (x, y) positions
    trajectories: Dict[int, List[tuple]] = defaultdict(list)
    
    try:
        end_time = start_time + duration_seconds if duration_seconds > 0 else None
        
        while True:
            # Check duration if specified
            if end_time and time.time() >= end_time:
                break
            
            ret, frame = cap.read()
            
            if not ret:
                print("[FAIL] Failed to read frame from webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Run tracking (YOLO with tracker mode)
            results = model.track(
                frame,
                conf=confidence_threshold,
                persist=True,
                verbose=False
            )
            
            # Draw detections and tracking info
            annotated_frame = frame.copy()
            
            # Draw trajectory trails (before bounding boxes)
            if show_trajectories:
                for track_id, trajectory in trajectories.items():
                    if len(trajectory) > 1:
                        # Draw trajectory as a line
                        points = trajectory[-trajectory_length:]  # Last N points
                        for i in range(1, len(points)):
                            cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)
            
            # Draw bounding boxes and track IDs
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                    track_id_int = int(track_id)
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    cls_name = model.names[cls_id]
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Calculate center for trajectory
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Update trajectory
                    if show_trajectories:
                        trajectories[track_id_int].append((center_x, center_y))
                        # Limit trajectory length
                        if len(trajectories[track_id_int]) > trajectory_length:
                            trajectories[track_id_int] = trajectories[track_id_int][-trajectory_length:]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with track ID and class
                    label = f"ID:{track_id_int} {cls_name} {conf:.0%}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    label_y = max(y1 - 10, label_size[1])
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1, label_y - label_size[1] - 5),
                        (x1 + label_size[0], label_y + 5),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
                    
                    # Draw track ID circle at center
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Add info text overlay
            display_text = f"Time: {elapsed:.1f}s | Frame: {frame_count} | Tracks: {len(trajectories)}"
            cv2.putText(annotated_frame, display_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Object Tracking Viewer - Press 'q' to quit", annotated_frame)
            
            # Handle quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            # Clean up old trajectories (remove tracks not seen in recent frames)
            # This is handled by limiting trajectory length per track
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[OK] Closed webcam. Processed {frame_count} frames in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GUI viewer for object detection and tracking")
    parser.add_argument("--model", type=str, default="/home/ordun/Documents/snowcrash/models/yolo26n-seg.pt", help="YOLO model path")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-gstreamer", action="store_true", help="Disable GStreamer")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = until 'q' pressed)")
    parser.add_argument("--no-trajectories", action="store_true", help="Disable trajectory visualization")
    
    args = parser.parse_args()
    
    run_tracking_viewer(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_gstreamer=not args.no_gstreamer,
        duration_seconds=args.duration,
        show_trajectories=not args.no_trajectories
    )

