#!/usr/bin/env python3
"""
YOLO Detection Service for old-nano

Lightweight HTTP server that runs YOLOv8 detection and can perform
continuous person detection with alert notifications.
"""
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from flask import Flask, jsonify, request
except ImportError as e:
    print(f"[ERROR] Flask not installed: {e}")
    print("[ERROR] Install with: pip3 install flask")
    sys.exit(1)

# Don't import cv2 directly - let YOLO handle camera capture
# This avoids OpenCV "Illegal instruction" issues on Jetson Nano

# Add project root to path (handle both direct run and Docker)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Also add tools directory
TOOLS_DIR = Path(__file__).parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from tools.desktop_notification import send_person_alert
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    print("[WARNING] desktop_notification not available. Alerts will be disabled.")


class YOLODetectionService:
    """HTTP service for YOLO object detection."""
    
    def __init__(
        self,
        model_path: str = "/home/ordun/Documents/snowcrash/yolov8n.pt",
        camera_device: int = 0,
        host: str = "0.0.0.0",
        port: int = 8080,
        confidence_threshold: float = 0.25
    ):
        # Flag to track if person was detected (for notification polling)
        self.person_detected = False
        self.person_detected_lock = threading.Lock()
        """Initialize YOLO detection service."""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.camera_device = camera_device
        self.host = host
        self.port = port
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ YOLO model loaded")
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Continuous detection state
        self.continuous_detection_running = False
        self.continuous_detection_thread = None
        self.continuous_detection_lock = threading.Lock()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({"status": "ok", "service": "yolo_detection"})
        
        @self.app.route('/detect', methods=['POST'])
        def detect():
            """Run single detection."""
            try:
                data = request.get_json() or {}
                source = data.get('source', 'camera')
                send_notification = data.get('send_notification', False)
                confidence = data.get('confidence_threshold', self.confidence_threshold)
                
                # Use YOLO's built-in source handling (camera or image file)
                # This avoids OpenCV import issues
                if source == 'camera':
                    yolo_source = f"/dev/video{self.camera_device}"
                else:
                    # Image file path
                    img_path = Path(source)
                    if not img_path.exists():
                        return jsonify({"success": False, "error": f"Image not found: {source}"}), 400
                    yolo_source = str(img_path)
                
                # Run detection using YOLO's predict (handles camera/image internally)
                results = self.model.predict(
                    source=yolo_source,
                    conf=confidence,
                    verbose=False,
                    stream=False,
                    save=False
                )
                
                # Parse results
                detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            detections.append({
                                "class": self.model.names[int(box.cls)],
                                "confidence": float(box.conf),
                                "bbox": {
                                    "x1": float(box.xyxy[0][0]),
                                    "y1": float(box.xyxy[0][1]),
                                    "x2": float(box.xyxy[0][2]),
                                    "y2": float(box.xyxy[0][3])
                                }
                            })
                
                # Check for person
                persons = [d for d in detections if d["class"].lower() == "person"]
                notification_sent = False
                
                if persons and send_notification and NOTIFICATION_AVAILABLE:
                    # Send notification for first person detected
                    person = persons[0]
                    notification_sent = send_person_alert(
                        hostname="orin-nano",
                        confidence=person["confidence"]
                    )
                
                return jsonify({
                    "success": True,
                    "source": f"camera /dev/video{self.camera_device}" if source == 'camera' else source,
                    "detections": detections,
                    "count": len(detections),
                    "persons_detected": len(persons),
                    "notification_sent": notification_sent
                })
                
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/watch_person', methods=['POST'])
        def watch_person():
            """Start continuous person detection until person found or stopped."""
            try:
                with self.continuous_detection_lock:
                    if self.continuous_detection_running:
                        return jsonify({
                            "success": False,
                            "error": "Continuous detection already running"
                        }), 400
                    
                    data = request.get_json() or {}
                    confidence = data.get('confidence_threshold', self.confidence_threshold)
                    
                    # Start continuous detection in background thread
                    self.continuous_detection_running = True
                    self.continuous_detection_thread = threading.Thread(
                        target=self._continuous_person_detection,
                        args=(confidence,),
                        daemon=True
                    )
                    self.continuous_detection_thread.start()
                    
                    return jsonify({
                        "success": True,
                        "message": "Continuous person detection started"
                    })
                    
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/stop_watch', methods=['POST'])
        def stop_watch():
            """Stop continuous person detection."""
            try:
                with self.continuous_detection_lock:
                    if not self.continuous_detection_running:
                        return jsonify({
                            "success": False,
                            "error": "No continuous detection running"
                        }), 400
                    
                    self.continuous_detection_running = False
                    
                    # Wait for thread to finish
                    if self.continuous_detection_thread:
                        self.continuous_detection_thread.join(timeout=2.0)
                    
                    return jsonify({
                        "success": True,
                        "message": "Continuous person detection stopped"
                    })
                    
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/watch_status', methods=['GET'])
        def watch_status():
            """Get status of continuous detection."""
            with self.continuous_detection_lock:
                with self.person_detected_lock:
                    status = {
                        "running": self.continuous_detection_running,
                        "person_detected": self.person_detected
                    }
                    # Reset flag after reading
                    if self.person_detected:
                        self.person_detected = False
                    return jsonify(status)
        
        @self.app.route('/video_feed')
        def video_feed():
            """MJPEG video stream from old-nano camera."""
            try:
                from flask import Response
                return Response(
                    self._generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] Video feed error: {e}")
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
    
    def _capture_frame(self):
        """Capture a single frame using YOLO's built-in camera interface.
        This avoids direct OpenCV import issues on Jetson Nano.
        """
        # Use YOLO's predict with camera source - it handles capture internally
        # Capture single frame by using imgsz=1 and max_det=0 to skip detection
        results = self.model.predict(
            source=f"/dev/video{self.camera_device}",
            imgsz=640,
            conf=0.25,
            verbose=False,
            stream=False,
            save=False
        )
        
        # Get frame from results (YOLO captures it internally)
        # Note: This requires YOLO to actually process, but avoids cv2 import
        for result in results:
            # YOLO stores the original image in result.orig_img
            return result.orig_img
        
        raise RuntimeError(f"Failed to capture frame from /dev/video{self.camera_device}")
    
    def _generate_video_stream(self):
        """Generate MJPEG video stream from camera for GUI display."""
        import sys
        import traceback
        
        camera_path = f"/dev/video{self.camera_device}"
        print(f"[VIDEO] Starting video stream from {camera_path}")
        
        try:
            # Try using device index instead of path (YOLO might prefer this)
            # First try with device index
            try:
                source = int(self.camera_device)
                print(f"[VIDEO] Trying device index: {source}")
            except:
                source = camera_path
                print(f"[VIDEO] Using device path: {source}")
            
            # Use YOLO's stream mode to capture and process frames
            frame_count = 0
            for results in self.model.predict(
                source=source,
                conf=self.confidence_threshold,
                verbose=False,
                stream=True,
                save=False
            ):
                frame_count += 1
                if frame_count == 1:
                    print(f"[VIDEO] First frame received successfully!")
                
                # Get annotated frame from YOLO results
                try:
                    annotated_frame = results.plot()  # YOLO's built-in annotation
                except Exception as e:
                    print(f"[ERROR] Failed to plot frame: {e}")
                    continue
                
                # Check for person in current frame (for notification)
                if results.boxes is not None:
                    for box in results.boxes:
                        class_name = self.model.names[int(box.cls)]
                        if class_name.lower() == "person":
                            print(f"[VIDEO] Person detected in frame!")
                
                # Encode frame as JPEG
                try:
                    import cv2
                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        print(f"[ERROR] Failed to encode frame as JPEG")
                except ImportError:
                    print(f"[ERROR] OpenCV (cv2) not available for encoding")
                    break
                except Exception as e:
                    print(f"[ERROR] JPEG encoding error: {e}")
                    continue
                
                time.sleep(0.033)  # ~30 FPS
                
        except FileNotFoundError as e:
            print(f"[ERROR] Camera device not found: {camera_path}")
            print(f"[ERROR] {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] Video stream error: {e}")
            traceback.print_exc()
            import sys
            sys.stdout.flush()
    
    def _continuous_person_detection(self, confidence: float):
        """Continuous person detection loop using YOLO's built-in streaming.
        Avoids direct OpenCV usage.
        """
        print("[WATCH] Starting continuous person detection...")
        
        frame_count = 0
        
        try:
            # Use YOLO's stream mode for continuous camera capture
            # YOLO handles camera opening internally
            for results in self.model.predict(
                source=f"/dev/video{self.camera_device}",
                conf=confidence,
                verbose=False,
                stream=True,  # Stream mode for continuous processing
                save=False
            ):
                if not self.continuous_detection_running:
                    break
                
                # Check for person in results
                if results.boxes is not None:
                    for box in results.boxes:
                        class_name = self.model.names[int(box.cls)]
                        if class_name.lower() == "person":
                                    conf = float(box.conf)
                                    print(f"[WATCH] Person detected! Confidence: {conf:.2%}")
                                    
                                    # Set flag for notification polling
                                    with self.person_detected_lock:
                                        self.person_detected = True
                                    
                                    # Send notification
                                    if NOTIFICATION_AVAILABLE:
                                        send_person_alert(
                                            hostname="orin-nano",
                                            confidence=conf
                                        )
                                    
                                    # Stop detection after first person found
                                    with self.continuous_detection_lock:
                                        self.continuous_detection_running = False
                                    print("[WATCH] Person detection stopped (person found)")
                                    return
                
                # Small delay to prevent CPU overload
                frame_count += 1
                if frame_count % 10 == 0:  # Reduce processing frequency
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"[ERROR] Continuous detection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.continuous_detection_lock:
                self.continuous_detection_running = False
            print("[WATCH] Continuous person detection stopped")
    
    def run(self):
        """Start the Flask server."""
        print(f"[OK] Starting YOLO detection service on {self.host}:{self.port}")
        print(f"     Camera: /dev/video{self.camera_device}")
        print(f"     Model: {self.model_path}")
        print(f"     Press Ctrl+C to stop")
        print()
        
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="YOLO Detection Service for old-nano")
    parser.add_argument("--model", type=str, 
                       default=os.environ.get("MODEL_PATH", "/home/ordun/Documents/snowcrash/yolov8n.pt"),
                       help="Path to YOLO model")
    parser.add_argument("--device", type=int, 
                       default=int(os.environ.get("CAMERA_DEVICE", "0")), 
                       help="Camera device index")
    parser.add_argument("--host", type=str, 
                       default=os.environ.get("HOST", "0.0.0.0"), 
                       help="Host to bind to")
    parser.add_argument("--port", type=int, 
                       default=int(os.environ.get("PORT", "8080")), 
                       help="Port to bind to")
    parser.add_argument("--confidence", type=float, 
                       default=float(os.environ.get("CONFIDENCE", "0.25")), 
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    service = YOLODetectionService(
        model_path=args.model,
        camera_device=args.device,
        host=args.host,
        port=args.port,
        confidence_threshold=args.confidence
    )
    
    try:
        service.run()
    except KeyboardInterrupt:
        print("\n[OK] Shutting down service...")
        with service.continuous_detection_lock:
            service.continuous_detection_running = False

