#!/usr/bin/env python3
"""
Live YOLO object detection with webcam feed display.
Shows video window with bounding boxes and labels.
Supports timestamp-based detection logging.
"""
import cv2
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def run_live_detection(
    model_path: str = "yolov8n.pt",
    device: int = 0,
    confidence_threshold: float = 0.25,
    use_gstreamer: bool = True,
    log_timestamps: bool = False,
    log_frames: bool = False,
    log_both: bool = False,
    log_file: Optional[str] = None
):
    """
    Run live object detection with webcam feed.
    
    Args:
        model_path: Path to YOLO model file
        device: Camera device index
        confidence_threshold: Detection confidence threshold
        use_gstreamer: Use GStreamer backend (recommended for Jetson)
        log_timestamps: Log detections with wall-clock timestamps
        log_frames: Log detections with frame numbers
        log_both: Log both timestamps and frame numbers
        log_file: Optional CSV file path for logging detections
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
    print("Displaying live detection. Press 'q' to quit.")
    
    # Determine logging mode
    if log_both:
        log_mode = "both"
    elif log_timestamps:
        log_mode = "timestamps"
    elif log_frames:
        log_mode = "frames"
    else:
        log_mode = "none"
    
    if log_mode != "none":
        print(f"[OK] Logging mode: {log_mode}")
        if log_file:
            print(f"  Logging to: {log_file}")
    print()
    
    frame_count = 0
    start_time = time.time()
    elapsed = 0.0
    log_fp = None
    
    # Setup CSV log file with appropriate columns
    if log_file:
        log_fp = open(log_file, 'w')
        if log_mode == "both":
            log_fp.write("timestamp,wall_clock_time,elapsed_seconds,frame_number,object_class,confidence\n")
        elif log_mode == "timestamps":
            log_fp.write("timestamp,wall_clock_time,elapsed_seconds,object_class,confidence\n")
        elif log_mode == "frames":
            log_fp.write("timestamp,frame_number,object_class,confidence\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[FAIL] Failed to read frame from webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            # Get wall-clock time (device's current time)
            wall_clock = datetime.now()
            wall_clock_str = wall_clock.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            timestamp_iso = wall_clock.isoformat()
            
            # Run YOLO detection
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Add timestamp and frame counter to display
            if log_mode == "both":
                display_text = f"Time: {wall_clock_str} | Frame: {frame_count} | Elapsed: {elapsed:.1f}s"
            elif log_mode == "timestamps":
                display_text = f"Time: {wall_clock_str} | Elapsed: {elapsed:.1f}s"
            elif log_mode == "frames":
                display_text = f"Frame: {frame_count} | Elapsed: {elapsed:.1f}s"
            else:
                display_text = f"Time: {elapsed:.1f}s | Frame: {frame_count}"
            cv2.putText(annotated_frame, display_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Count detections for this frame
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(annotated_frame, f"Objects: {num_detections}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Live YOLO Detection - Press 'q' to quit", annotated_frame)
            
            # Log detections based on mode
            if num_detections > 0:
                detections_this_frame = []
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    cls_name = model.names[cls_id]
                    detections_this_frame.append((cls_name, conf))
                    
                    # Log to CSV file based on mode
                    if log_fp:
                        if log_mode == "both":
                            log_fp.write(f"{timestamp_iso},{wall_clock_str},{elapsed:.3f},{frame_count},{cls_name},{conf:.4f}\n")
                        elif log_mode == "timestamps":
                            log_fp.write(f"{timestamp_iso},{wall_clock_str},{elapsed:.3f},{cls_name},{conf:.4f}\n")
                        elif log_mode == "frames":
                            log_fp.write(f"{timestamp_iso},{frame_count},{cls_name},{conf:.4f}\n")
                        log_fp.flush()
                
                # Print to console based on mode
                if log_mode == "both":
                    print(f"[{wall_clock_str}] Frame {frame_count} ({elapsed:.3f}s) - Detected {num_detections} object(s):")
                    for cls_name, conf in detections_this_frame:
                        print(f"  - {cls_name}: {conf:.2%}")
                elif log_mode == "timestamps":
                    print(f"[{wall_clock_str}] ({elapsed:.3f}s) - Detected {num_detections} object(s):")
                    for cls_name, conf in detections_this_frame:
                        print(f"  - {cls_name}: {conf:.2%}")
                elif log_mode == "frames":
                    print(f"Frame {frame_count} ({elapsed:.1f}s) - Detected {num_detections} object(s):")
                    for cls_name, conf in detections_this_frame:
                        print(f"  - {cls_name}: {conf:.2%}")
                elif frame_count % 30 == 0:  # Print every 30 frames if no logging mode
                    print(f"Frame {frame_count} ({elapsed:.1f}s): Detected {num_detections} object(s)")
                    for cls_name, conf in detections_this_frame:
                        print(f"  - {cls_name}: {conf:.2%}")
            
            # Handle quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_fp:
            log_fp.close()
            print(f"[OK] Detection log saved to: {log_file}")
        print(f"[OK] Closed webcam. Processed {frame_count} frames in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live YOLO object detection with webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-gstreamer", action="store_true", help="Disable GStreamer")
    parser.add_argument("--timestamps", action="store_true", help="Log detections with wall-clock timestamps")
    parser.add_argument("--frames", action="store_true", help="Log detections with frame numbers")
    parser.add_argument("--both", action="store_true", help="Log detections with both timestamps and frame numbers")
    parser.add_argument("--log-file", type=str, help="File to log detections (CSV format)")
    
    args = parser.parse_args()
    
    # Determine logging mode
    if args.both:
        log_both = True
        log_timestamps = False
        log_frames = False
    elif args.timestamps:
        log_timestamps = True
        log_frames = False
        log_both = False
    elif args.frames:
        log_frames = True
        log_timestamps = False
        log_both = False
    elif args.log_file:
        # If log file specified but no mode, default to timestamps
        log_timestamps = True
        log_frames = False
        log_both = False
    else:
        log_timestamps = False
        log_frames = False
        log_both = False
    
    run_live_detection(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_gstreamer=not args.no_gstreamer,
        log_timestamps=log_timestamps,
        log_frames=log_frames,
        log_both=log_both,
        log_file=args.log_file
    )

