#!/usr/bin/env python3
"""
Web-based GUI for object detection and tracking.
Serves HTML interface with live video stream and track data display.
Memory-efficient: uses MJPEG streaming, no frame storage.
"""
import cv2
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict
import socket

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from flask import Flask, Response, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    request = None


# HTML template for the tracking viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Tracking Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            overflow: hidden;
        }
        .header {
            background: #2a2a2a;
            padding: 10px 20px;
            border-bottom: 2px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 18px;
            color: #4CAF50;
        }
        .header-info {
            font-size: 12px;
            color: #aaa;
        }
        .container {
            display: flex;
            height: calc(100vh - 50px);
        }
        .video-panel {
            flex: 1;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .video-panel img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .tracks-panel {
            width: 350px;
            background: #2a2a2a;
            border-left: 2px solid #444;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }
        .tracks-panel h2 {
            font-size: 16px;
            margin-bottom: 10px;
            color: #4CAF50;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }
        .tracks-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .track-item {
            background: #333;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .track-item strong {
            color: #4CAF50;
        }
        .track-item .track-id {
            font-size: 14px;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 5px;
        }
        .track-item .field {
            margin: 3px 0;
            color: #ccc;
        }
        .no-tracks {
            color: #888;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        ::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Object Tracking Viewer</h1>
        <div class="header-info">
            <div>Device: <strong id="device-ip">Loading...</strong></div>
            <div id="current-time">Loading...</div>
        </div>
    </div>
    <div class="container">
        <div class="video-panel">
            <img id="video-stream" src="/video_feed" alt="Video Stream">
        </div>
        <div class="tracks-panel">
            <h2>Active Tracks</h2>
            <div class="tracks-container" id="tracks-container">
                <div class="no-tracks">No tracks detected yet...</div>
            </div>
        </div>
    </div>

    <script>
        // Update time every second
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toLocaleString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Update device IP
        fetch('/device_info')
            .then(r => r.json())
            .then(data => {
                document.getElementById('device-ip').textContent = data.ip;
            });

        // Update tracks every 500ms
        function updateTracks() {
            fetch('/tracks')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('tracks-container');
                    if (data.tracks.length === 0) {
                        container.innerHTML = '<div class="no-tracks">No tracks detected yet...</div>';
                        return;
                    }
                    
                    const htmlParts = data.tracks.map(track => {
                        const vx = (track.velocity && track.velocity.vx != null) ? track.velocity.vx.toFixed(2) : '0.00';
                        const vy = (track.velocity && track.velocity.vy != null) ? track.velocity.vy.toFixed(2) : '0.00';
                        const dist = (track.distance_moved_pixels != null) ? track.distance_moved_pixels.toFixed(1) : '0.0';
                        const speed = (track.average_speed_pixels_per_second != null) ? track.average_speed_pixels_per_second.toFixed(2) : '0.00';
                        let firstSeen = 'N/A';
                        let lastSeen = 'N/A';
                        try {
                            if (track.first_seen) {
                                firstSeen = String(new Date(track.first_seen).toLocaleTimeString() || 'N/A');
                            }
                            if (track.last_seen) {
                                lastSeen = String(new Date(track.last_seen).toLocaleTimeString() || 'N/A');
                            }
                        } catch (e) {
                            // Date parsing failed, use ISO string or fallback
                            firstSeen = track.first_seen ? String(track.first_seen).substring(11, 19) : 'N/A';
                            lastSeen = track.last_seen ? String(track.last_seen).substring(11, 19) : 'N/A';
                        }
                        const trackId = String(track.track_id || '?');
                        const trackClass = String(track.class || 'unknown');
                        
                        return '<div class="track-item">' +
                            '<div class="track-id">Track ID: ' + trackId + '</div>' +
                            '<div class="field"><strong>Class:</strong> ' + trackClass + '</div>' +
                            '<div class="field"><strong>Velocity:</strong> vx=' + vx + ', vy=' + vy + ' px/s</div>' +
                            '<div class="field"><strong>Distance Moved:</strong> ' + dist + ' px</div>' +
                            '<div class="field"><strong>Avg Speed:</strong> ' + speed + ' px/s</div>' +
                            '<div class="field"><strong>First Seen:</strong> ' + firstSeen + '</div>' +
                            '<div class="field"><strong>Last Seen:</strong> ' + lastSeen + '</div>' +
                            '</div>';
                    });
                    container.innerHTML = htmlParts.join('');
                })
                .catch(err => console.error('Error fetching tracks:', err));
        }
        setInterval(updateTracks, 500);
        updateTracks();
    </script>
</body>
</html>
"""


def get_local_ip():
    """Get the local IP address of the device."""
    try:
        # Connect to a remote address (doesn't actually connect)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


class TrackingWebViewer:
    """Web-based tracking viewer with video stream and track data."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: int = 0,
        confidence_threshold: float = 0.25,
        use_gstreamer: bool = True,
        port: int = 8080
    ):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not installed. Install with: pip install flask")
        
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_gstreamer = use_gstreamer
        self.port = port
        
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Track data storage (shared between threads)
        self.tracks_data = {}
        self.tracks_lock = threading.Lock()
        
        # Historical track metadata (for duration calculations)
        # Store all track metadata including inactive tracks
        self.track_metadata_history: Dict[int, Dict] = {}
        self.track_metadata_lock = threading.Lock()
        
        # Video capture (will be set in thread)
        self.cap = None
        self.frame = None  # Annotated frame for display
        self.raw_frame = None  # Raw frame without annotations (for detection tools)
        self.frame_lock = threading.Lock()
        
        # Flask app (with suppressed logging)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)
        self._setup_routes()
        
        # Device info
        self.device_ip = get_local_ip()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/video_feed')
        def video_feed():
            """MJPEG video stream endpoint."""
            return Response(
                self._generate_video_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        
        @self.app.route('/tracks')
        def get_tracks():
            """Get current track data as JSON."""
            with self.tracks_lock:
                tracks = list(self.tracks_data.values())
            return jsonify({"tracks": tracks})
        
        @self.app.route('/device_info')
        def device_info():
            """Get device information."""
            return jsonify({
                "ip": self.device_ip,
                "device": f"/dev/video{self.device}"
            })
        
    def _generate_video_frames(self):
        """Generate MJPEG frames from video capture."""
        while True:
            frame_to_send = None
            
            # Get current frame (with lock)
            with self.frame_lock:
                if self.frame is not None:
                    frame_to_send = self.frame.copy()
            
            # Encode and send frame if available
            if frame_to_send is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to prevent CPU overload (~30 FPS)
            time.sleep(0.033)
    
    def _open_webcam(self):
        """Open webcam with GStreamer or OpenCV."""
        if self.use_gstreamer:
            gstreamer_pipeline = (
                f"v4l2src device=/dev/video{self.device} ! "
                "image/jpeg,width=640,height=480,framerate=30/1 ! "
                "jpegdec ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink"
            )
            
            print(f"Opening webcam with GStreamer: /dev/video{self.device}")
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("GStreamer failed, falling back to standard OpenCV...")
                cap = cv2.VideoCapture(self.device)
        else:
            cap = cv2.VideoCapture(self.device)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device /dev/video{self.device}")
        
        return cap
    
    def get_track_duration(self, track_id: Optional[int] = None, object_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calculate how long a track (by ID or object class) has been in the frame.
        
        Args:
            track_id: Specific track ID to query (optional)
            object_class: Object class name (e.g., "car", "person") to find all instances (optional)
            
        Returns:
            List of dictionaries containing track info with duration in minutes.
            Each dict contains: track_id, class, duration_minutes, first_seen, last_seen, is_active
            
        Note:
            If both track_id and object_class are provided, track_id takes precedence.
            If neither is provided, returns all tracks.
        """
        current_time = time.time()
        results = []
        
        with self.track_metadata_lock:
            # If track_id is specified, find that specific track
            if track_id is not None:
                if track_id in self.track_metadata_history:
                    track_info = self.track_metadata_history[track_id]
                    first_time = track_info.get("first_frame_time", current_time)
                    last_time = track_info.get("last_frame_time", current_time)
                    
                    # If track is currently active, use current time for last_seen
                    is_active = track_id in self.tracks_data
                    if is_active:
                        last_time = current_time
                    
                    duration_seconds = last_time - first_time
                    duration_minutes = duration_seconds / 60.0
                    
                    results.append({
                        "track_id": track_id,
                        "class": track_info.get("class", "unknown"),
                        "duration_minutes": round(duration_minutes, 2),
                        "duration_seconds": round(duration_seconds, 1),
                        "first_seen": track_info.get("first_seen", "N/A"),
                        "last_seen": track_info.get("last_seen", "N/A"),
                        "is_active": is_active
                    })
            # If object_class is specified, find all tracks of that class
            elif object_class is not None:
                object_class_lower = object_class.lower()
                for track_id_val, track_info in self.track_metadata_history.items():
                    track_class = track_info.get("class", "").lower()
                    if track_class == object_class_lower:
                        first_time = track_info.get("first_frame_time", current_time)
                        last_time = track_info.get("last_frame_time", current_time)
                        
                        # If track is currently active, use current time
                        is_active = track_id_val in self.tracks_data
                        if is_active:
                            last_time = current_time
                        
                        duration_seconds = last_time - first_time
                        duration_minutes = duration_seconds / 60.0
                        
                        results.append({
                            "track_id": track_id_val,
                            "class": track_info.get("class", "unknown"),
                            "duration_minutes": round(duration_minutes, 2),
                            "duration_seconds": round(duration_seconds, 1),
                            "first_seen": track_info.get("first_seen", "N/A"),
                            "last_seen": track_info.get("last_seen", "N/A"),
                            "is_active": is_active
                        })
            # If neither specified, return all tracks
            else:
                for track_id_val, track_info in self.track_metadata_history.items():
                    first_time = track_info.get("first_frame_time", current_time)
                    last_time = track_info.get("last_frame_time", current_time)
                    
                    is_active = track_id_val in self.tracks_data
                    if is_active:
                        last_time = current_time
                    
                    duration_seconds = last_time - first_time
                    duration_minutes = duration_seconds / 60.0
                    
                    results.append({
                        "track_id": track_id_val,
                        "class": track_info.get("class", "unknown"),
                        "duration_minutes": round(duration_minutes, 2),
                        "duration_seconds": round(duration_seconds, 1),
                        "first_seen": track_info.get("first_seen", "N/A"),
                        "last_seen": track_info.get("last_seen", "N/A"),
                        "is_active": is_active
                    })
        
        # Sort by duration (longest first)
        results.sort(key=lambda x: x["duration_minutes"], reverse=True)
        return results
    
    def _tracking_loop(self, duration_seconds: float = 0):
        """Main tracking loop (runs in separate thread)."""
        cap = None
        try:
            cap = self._open_webcam()
            print(f"[OK] Webcam opened: /dev/video{self.device}")
            
            trajectories: Dict[int, List[tuple]] = defaultdict(list)
            track_metadata: Dict[int, Dict] = {}
            
            start_time = time.time()
            end_time = start_time + duration_seconds if duration_seconds > 0 else None
            frame_count = 0
            
            while True:
                # Check duration
                if end_time and time.time() >= end_time:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp_str = datetime.now().isoformat()
                
                # Run tracking
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    persist=True,
                    verbose=False
                )
                
                # Store raw frame BEFORE annotations (for detection tools)
                raw_frame_for_detection = frame.copy()
                
                # Draw annotations
                annotated_frame = frame.copy()
                
                # Draw trajectories
                for track_id, trajectory in trajectories.items():
                    if len(trajectory) > 1:
                        points = trajectory[-30:]  # Last 30 points
                        for i in range(1, len(points)):
                            cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)
                
                # Process tracked objects
                active_track_ids = set()
                all_track_ids_processed_this_frame = set()  # Track ALL IDs processed (both real and temp)
                # Collect detections for statistics update
                frame_detections = []
                
                # Process boxes - handle both tracked (with IDs) and non-tracked (detections only)
                if results[0].boxes is not None:
                    # Check if we have track IDs
                    has_track_ids = results[0].boxes.id is not None
                    
                    # Iterate through boxes
                    boxes_list = list(results[0].boxes)
                    track_ids_list = list(results[0].boxes.id) if has_track_ids else [None] * len(boxes_list)
                    
                    for idx, box in enumerate(boxes_list):
                        track_id = track_ids_list[idx] if has_track_ids else None
                        
                        # Assign a temporary track ID if none exists yet (for first frame detections)
                        # Real tracking IDs will be assigned by YOLO after a few frames
                        if track_id is not None:
                            track_id_int = int(track_id)
                            active_track_ids.add(track_id_int)
                            all_track_ids_processed_this_frame.add(track_id_int)
                        else:
                            # Use box index as temporary ID until tracking assigns one
                            # Use negative IDs temporarily to distinguish from real track IDs
                            track_id_int = -(idx + 1)  # -1, -2, -3, etc.
                            all_track_ids_processed_this_frame.add(track_id_int)
                        
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.model.names[cls_id]
                        bbox = box.xyxy[0].tolist()
                        
                        # Collect detection for statistics
                        frame_detections.append({
                            "class": cls_name,
                            "confidence": conf
                        })
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Initialize trajectory if not exists
                        if track_id_int not in trajectories:
                            trajectories[track_id_int] = []
                        
                        # Update trajectory
                        trajectories[track_id_int].append((center_x, center_y))
                        if len(trajectories[track_id_int]) > 30:
                            trajectories[track_id_int] = trajectories[track_id_int][-30:]
                        
                        # Update track metadata
                        if track_id_int not in track_metadata:
                            track_metadata[track_id_int] = {
                                "track_id": track_id_int,
                                "class": cls_name,
                                "first_seen": timestamp_str,
                                "last_seen": timestamp_str,
                                "first_position": {"x": center_x, "y": center_y},
                                "last_position": {"x": center_x, "y": center_y},
                                "first_frame_time": time.time(),
                                "last_frame_time": time.time(),
                                "frames_seen": 0
                            }
                        
                        track_info = track_metadata[track_id_int]
                        track_info["last_seen"] = timestamp_str
                        track_info["last_position"] = {"x": center_x, "y": center_y}
                        track_info["last_frame_time"] = time.time()
                        track_info["frames_seen"] += 1
                        track_info["class"] = cls_name
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"ID:{track_id_int} {cls_name} {conf:.0%}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        label_y = max(y1 - 10, label_size[1])
                        cv2.rectangle(
                            annotated_frame,
                            (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0], label_y + 5),
                            (0, 255, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # Clean up inactive tracks from trajectories (keep only active ones)
                trajectories = {tid: traj for tid, traj in trajectories.items() if tid in all_track_ids_processed_this_frame}
                
                # Update shared track data with velocity calculations
                current_tracks = {}
                for track_id, track_info in track_metadata.items():
                    # Include all tracks that were active this frame (real IDs or temporary)
                    if track_id in all_track_ids_processed_this_frame:
                        first_pos = track_info["first_position"]
                        last_pos = track_info["last_position"]
                        first_time = track_info["first_frame_time"]
                        last_time = track_info["last_frame_time"]
                        
                        time_delta = max(0.001, last_time - first_time)
                        distance_moved = ((last_pos["x"] - first_pos["x"]) ** 2 + 
                                         (last_pos["y"] - first_pos["y"]) ** 2) ** 0.5
                        
                        current_tracks[track_id] = {
                            "track_id": track_id,
                            "class": track_info["class"],
                            "velocity": {
                                "vx": round((last_pos["x"] - first_pos["x"]) / time_delta, 2),
                                "vy": round((last_pos["y"] - first_pos["y"]) / time_delta, 2)
                            },
                            "distance_moved_pixels": round(distance_moved, 1),
                            "average_speed_pixels_per_second": round(distance_moved / time_delta, 2) if time_delta > 0 else 0.0,
                            "first_seen": track_info["first_seen"],
                            "last_seen": track_info["last_seen"]
                        }
                
                # Update shared tracks data
                with self.tracks_lock:
                    self.tracks_data = current_tracks
                
                # Update historical track metadata (preserve all tracks including inactive)
                with self.track_metadata_lock:
                    # Update or add track metadata to history
                    for track_id, track_info in track_metadata.items():
                        if track_id in all_track_ids_processed_this_frame:
                            # Track is currently active - update with latest info
                            self.track_metadata_history[track_id] = track_info.copy()
                        else:
                            # Track is no longer active but preserve last known state
                            if track_id not in self.track_metadata_history:
                                self.track_metadata_history[track_id] = track_info.copy()
                            else:
                                # Update last_seen and last_frame_time for inactive tracks
                                self.track_metadata_history[track_id]["last_seen"] = track_info["last_seen"]
                                self.track_metadata_history[track_id]["last_frame_time"] = track_info["last_frame_time"]
                
                # Update shared frames (both raw and annotated for different use cases)
                with self.frame_lock:
                    self.frame = annotated_frame  # For display in GUI
                    self.raw_frame = raw_frame_for_detection  # For detection/distance tools (no annotations)
                
                # Update statistics tool with detections from this frame
                if frame_detections:
                    try:
                        # Import statistics tool from langchain_tools (shared instance)
                        from agent.langchain_tools import _statistics_tool
                        _statistics_tool.update(frame_detections)
                    except (ImportError, AttributeError):
                        # Statistics tool not available - silently skip (optional feature)
                        pass
                
                time.sleep(0.001)  # Small delay
        
        finally:
            if cap:
                cap.release()
            print(f"[OK] Tracking loop ended. Processed {frame_count} frames.")
    
    def run(self, duration_seconds: float = 0):
        """Start the web server and tracking loop."""
        # Start tracking loop in separate thread
        tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(duration_seconds,),
            daemon=True
        )
        tracking_thread.start()
        
        # Start Flask server
        print(f"\n[OK] Starting web server on http://{self.device_ip}:{self.port}")
        print(f"     Open your browser to: http://{self.device_ip}:{self.port}")
        print(f"     Tracking for {duration_seconds} seconds (0 = until stopped)" if duration_seconds > 0 else "     Press Ctrl+C to stop")
        print()
        
        # Suppress Flask/Werkzeug logging (only log errors)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Run Flask (blocking, but with suppressed logging)
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based object tracking viewer")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-gstreamer", action="store_true", help="Disable GStreamer")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = until stopped)")
    
    args = parser.parse_args()
    
    viewer = TrackingWebViewer(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_gstreamer=not args.no_gstreamer,
        port=args.port
    )
    
    viewer.run(duration_seconds=args.duration)

