#!/usr/bin/env python3
"""
Jetson Inference Detection Service for old-nano

Simple service that streams camera video via HTTP using GStreamer.
"""
import subprocess
import sys
import time
import threading
from pathlib import Path
from flask import Flask, Response, jsonify

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.desktop_notification import send_person_alert
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False

class JetsonInferenceService:
    """Simple service that streams camera video via HTTP."""
    
    def __init__(self, camera_device: int = 0, host: str = "0.0.0.0", port: int = 9000):
        self.camera_device = camera_device
        self.host = host
        self.port = port
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Continuous detection state
        self.continuous_detection_running = False
        self.continuous_detection_thread = None
        self.continuous_detection_lock = threading.Lock()
        self.person_detected = False
        self.person_detected_lock = threading.Lock()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "ok", "service": "jetson_inference"})
        
        @self.app.route('/video_feed')
        def video_feed():
            """MJPEG video stream from camera."""
            print(f"[VIDEO] /video_feed accessed")
            return Response(
                self._generate_video_stream(),
                mimetype='multipart/x-mixed-replace; boundary=--frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        
        @self.app.route('/watch_person', methods=['POST'])
        def watch_person():
            """Start continuous person detection."""
            with self.continuous_detection_lock:
                if self.continuous_detection_running:
                    return jsonify({"success": False, "error": "Already running"}), 400
                
                self.continuous_detection_running = True
                self.continuous_detection_thread = threading.Thread(
                    target=self._continuous_person_detection,
                    daemon=True
                )
                self.continuous_detection_thread.start()
                return jsonify({"success": True, "message": "Person detection started"})
        
        @self.app.route('/stop_watch', methods=['POST'])
        def stop_watch():
            """Stop continuous person detection."""
            with self.continuous_detection_lock:
                self.continuous_detection_running = False
                if self.continuous_detection_thread:
                    self.continuous_detection_thread.join(timeout=2.0)
                return jsonify({"success": True, "message": "Person detection stopped"})
        
        @self.app.route('/watch_status', methods=['GET'])
        def watch_status():
            """Get status of continuous detection."""
            with self.continuous_detection_lock:
                with self.person_detected_lock:
                    status = {
                        "running": self.continuous_detection_running,
                        "person_detected": self.person_detected
                    }
                    if self.person_detected:
                        self.person_detected = False
                    return jsonify(status)
    
    def _generate_video_stream(self):
        """Generate MJPEG video stream with detectnet bounding boxes."""
        # Use detectnet to process video and overlay bounding boxes
        # detectnet.py can output annotated frames that we can stream
        yield from self._generate_video_stream_with_detectnet_overlay()
    
    def _generate_video_stream_with_detectnet_overlay(self):
        """Generate video stream with detectnet bounding boxes overlaid."""
        import tempfile
        import os
        
        detectnet_path = Path.home() / "jetson-inference" / "build" / "aarch64" / "bin" / "detectnet.py"
        if not detectnet_path.exists():
            import shutil
            detectnet_cmd = shutil.which("detectnet.py")
            if detectnet_cmd:
                detectnet_path = Path(detectnet_cmd)
            else:
                print("[VIDEO] detectnet.py not found, using raw stream")
                yield from self._generate_video_stream_raw()
                return
        
        # Use detectnet with output to a file pattern, then read and stream frames
        # detectnet.py will write annotated frames with bounding boxes
        temp_dir = tempfile.mkdtemp(prefix='detectnet_stream_')
        output_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
        
        try:
            print(f"[VIDEO] Starting detectnet with overlay to {temp_dir}")
            
            # Start detectnet process that outputs annotated frames
            # Use file pattern so detectnet writes sequential frames
            import os as os_module
            env = os_module.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            cmd = [
                "python3", "-u", str(detectnet_path),
                f"/dev/video{self.camera_device}",
                f"file://{output_pattern}",
                "--headless",
                "--width=640",
                "--height=480"
            ]
            
            print(f"[VIDEO] Running: {' '.join(cmd)}")
            detectnet_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )
            
            # Wait for first frame
            time.sleep(3)
            
            if detectnet_process.poll() is not None:
                stderr = detectnet_process.stderr.read() if detectnet_process.stderr else ""
                print(f"[ERROR] detectnet exited: {stderr[:500]}")
                yield from self._generate_video_stream_raw()
                return
            
            print("[VIDEO] detectnet started, streaming annotated frames...")
            
            # Read frames sequentially and stream them
            frame_num = 1
            last_frame_time = time.time()
            
            while True:
                frame_file = os.path.join(temp_dir, f"frame_{frame_num:06d}.jpg")
                
                if os.path.exists(frame_file):
                    try:
                        with open(frame_file, 'rb') as f:
                            frame_data = f.read()
                        
                        if len(frame_data) > 100:  # Valid JPEG
                            # Stream as MJPEG
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   frame_data + b'\r\n')
                            
                            if frame_num == 1:
                                print(f"[VIDEO] First annotated frame received!")
                            
                            frame_num += 1
                            last_frame_time = time.time()
                            
                            # Clean up old frames (keep last 2)
                            if frame_num > 3:
                                old_frame = os.path.join(temp_dir, f"frame_{frame_num-3:06d}.jpg")
                                try:
                                    os.unlink(old_frame)
                                except:
                                    pass
                            
                            # Rate limit to ~30 FPS
                            time.sleep(0.033)
                        else:
                            time.sleep(0.01)
                    except Exception as e:
                        print(f"[ERROR] Error reading frame: {e}")
                        time.sleep(0.1)
                else:
                    # Frame not ready, wait a bit
                    time.sleep(0.01)
                    
                    # Check if process died
                    if detectnet_process.poll() is not None:
                        stderr = detectnet_process.stderr.read() if detectnet_process.stderr else ""
                        print(f"[ERROR] detectnet died: {stderr[:500]}")
                        yield from self._generate_video_stream_raw()
                        return
                    
                    # If no frames for 5 seconds, something wrong
                    if time.time() - last_frame_time > 5.0 and frame_num == 1:
                        print("[WARNING] No frames from detectnet, checking...")
                        if detectnet_process.poll() is not None:
                            stderr = detectnet_process.stderr.read() if detectnet_process.stderr else ""
                            print(f"[ERROR] detectnet died: {stderr[:500]}")
                            yield from self._generate_video_stream_raw()
                            return
                        last_frame_time = time.time()
                        
        except Exception as e:
            print(f"[ERROR] detectnet overlay stream failed: {e}")
            import traceback
            traceback.print_exc()
            yield from self._generate_video_stream_raw()
        finally:
            # Cleanup
            try:
                if 'detectnet_process' in locals() and detectnet_process.poll() is None:
                    detectnet_process.terminate()
                    detectnet_process.wait(timeout=2)
            except:
                pass
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except:
                pass
    
    def _generate_video_stream_raw(self):
        """Generate raw MJPEG video stream using GStreamer (no overlays)."""
        print(f"[VIDEO] Starting raw stream from /dev/video{self.camera_device}")
        
        pipelines = [
            # Pipeline 1: Jetson hardware-accelerated with 640x480 resolution
            [
                "gst-launch-1.0", "-q",
                "v4l2src", f"device=/dev/video{self.camera_device}",
                "!", "video/x-raw,width=640,height=480,framerate=30/1",
                "!", "nvvidconv",
                "!", "video/x-raw(memory:NVMM),format=I420,width=640,height=480",
                "!", "nvvidconv", "!", "video/x-raw,format=I420,width=640,height=480",
                "!", "jpegenc", "quality=85",
                "!", "multipartmux", "boundary=--frame",
                "!", "fdsink", "fd=1"
            ],
            # Pipeline 2: Simple fallback with 640x480 resolution
            [
                "gst-launch-1.0", "-q",
                "v4l2src", f"device=/dev/video{self.camera_device}",
                "!", "video/x-raw,width=640,height=480,framerate=30/1",
                "!", "videoconvert",
                "!", "jpegenc", "quality=85",
                "!", "multipartmux", "boundary=--frame",
                "!", "fdsink", "fd=1"
            ]
        ]
        
        process = None
        for i, pipeline in enumerate(pipelines, 1):
            print(f"[VIDEO] Trying pipeline {i}...")
            try:
                process = subprocess.Popen(
                    pipeline,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                time.sleep(0.5)
                
                if process.poll() is None:
                    print(f"[VIDEO] Pipeline {i} started!")
                    break
                else:
                    stderr = process.stderr.read().decode('utf-8', errors='ignore')
                    print(f"[VIDEO] Pipeline {i} failed: {stderr[:200]}")
                    process = None
            except Exception as e:
                print(f"[VIDEO] Pipeline {i} exception: {e}")
                if process:
                    process.terminate()
                process = None
        
        if process is None:
            error_msg = "All GStreamer pipelines failed"
            print(f"[ERROR] {error_msg}")
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + 
                   error_msg.encode() + b'\r\n\r\n')
            return
        
        print("[VIDEO] Streaming...")
        try:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    if process.poll() is not None:
                        stderr = process.stderr.read().decode('utf-8', errors='ignore')
                        print(f"[ERROR] GStreamer exited: {stderr[:300]}")
                        break
                    time.sleep(0.01)
                    continue
                yield chunk
        except Exception as e:
            print(f"[ERROR] Stream error: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except:
                    process.kill()
    
    def _generate_video_stream_with_detectnet_direct(self):
        """Generate video stream directly from detectnet with bounding boxes."""
        # Use detectnet.py to process video and output annotated frames
        # detectnet.py can output to stdout as MJPEG using GStreamer pipeline
        
        detectnet_path = Path.home() / "jetson-inference" / "build" / "aarch64" / "bin" / "detectnet.py"
        if not detectnet_path.exists():
            import shutil
            detectnet_cmd = shutil.which("detectnet.py")
            if detectnet_cmd:
                detectnet_path = Path(detectnet_cmd)
            else:
                print("[ERROR] detectnet.py not found")
                yield from self._generate_video_stream_raw()
                return
        
        print("[VIDEO] Starting detectnet with direct MJPEG output...")
        
        # Use detectnet.py with output to stdout as MJPEG
        # detectnet processes video and overlays bounding boxes automatically
        # We'll pipe its output directly to the HTTP stream
        try:
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # detectnet.py with output to stdout as MJPEG
            # Format: detectnet.py input output [options]
            # For stdout, we can use a GStreamer pipeline or detectnet's built-in output
            # Actually, detectnet might not support stdout directly, so we'll use a different approach
            
            # Better: Use detectnet with a GStreamer pipeline that outputs MJPEG
            # OR: Use detectnet's RTSP output and capture it
            # OR: Use detectnet with appsink and convert to MJPEG
            
            # Simplest working approach: Use detectnet's display output capability
            # But we need to capture it. Let's use a GStreamer pipeline that includes detectnet
            
            # Actually, the best approach is to use detectnet.py with a GStreamer appsink
            # that outputs MJPEG. But detectnet might not support this directly.
            
            # Fallback: Use detectnet in a subprocess and capture its annotated output
            # via a named pipe or shared memory, but that's complex.
            
            # For now, let's use detectnet's built-in capability to output annotated video
            # and capture it using GStreamer's appsrc or similar
            
            # Actually, the simplest: Use detectnet.py with output to a file pattern,
            # then use GStreamer to read and stream those files as they're created
            
            # But that's what we tried before and it didn't work well.
            
            # Better approach: Use detectnet's overlay on the raw stream
            # Run detectnet separately to get detections, then overlay on raw frames
            # But we don't have OpenCV on old-nano.
            
            # Best solution: Use detectnet.py's built-in streaming if it has it,
            # OR use the raw stream and send detections separately via JSON
            
            # For now, let's just use the raw stream - it works reliably
            # The detections are already being sent via the watch_person endpoint
            print("[VIDEO] Using raw stream (detections sent separately via watch_person)")
            yield from self._generate_video_stream_raw()
            
        except Exception as e:
            print(f"[ERROR] detectnet direct stream setup failed: {e}")
            import traceback
            traceback.print_exc()
            yield from self._generate_video_stream_raw()
    
    def _continuous_person_detection(self):
        """Continuous person detection using detectnet.py."""
        print("[WATCH] Starting person detection...")
        
        detectnet_path = Path.home() / "jetson-inference" / "build" / "aarch64" / "bin" / "detectnet.py"
        
        if not detectnet_path.exists():
            import shutil
            detectnet_cmd = shutil.which("detectnet.py")
            if detectnet_cmd:
                detectnet_path = Path(detectnet_cmd)
            else:
                print(f"[ERROR] detectnet.py not found at {detectnet_path}")
                with self.continuous_detection_lock:
                    self.continuous_detection_running = False
                return
        
        print(f"[WATCH] Using detectnet: {detectnet_path}")
        
        try:
            # detectnet.py arguments: input, output (optional), --network, --threshold, etc.
            # Use --network=ssd-mobilenet-v2 or --network=pednet for person detection
            # Default detectnet uses COCO classes including "person"
            cmd = [
                "python3", "-u", str(detectnet_path),  # -u flag for unbuffered output
                f"/dev/video{self.camera_device}", 
                "--headless",
                "--width=640",
                "--height=480"
            ]
            print(f"[WATCH] Running command: {' '.join(cmd)}")
            
            # Set environment for unbuffered output
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,  # Unbuffered
                env=env
            )
            
            print("[WATCH] detectnet process started, monitoring output...")
            line_count = 0
            last_output_time = time.time()
            
            # Read output line by line with better buffering handling
            while True:
                if not self.continuous_detection_running:
                    print("[WATCH] Detection stopped by user")
                    process.terminate()
                    break
                
                # Check if process has exited
                return_code = process.poll()
                if return_code is not None:
                    # Process exited, read any remaining output
                    try:
                        remaining = process.stdout.read()
                        if remaining:
                            for rem_line in remaining.splitlines():
                                rem_line = rem_line.strip()
                                if rem_line:
                                    line_count += 1
                                    print(f"[WATCH] detectnet output (line {line_count}): {rem_line[:200]}")
                    except:
                        pass
                    break
                
                # Try to read a line
                try:
                    line = process.stdout.readline()
                    if not line:
                        # No output yet, check if process is still alive
                        if process.poll() is not None:
                            break
                        # Wait a bit before checking again
                        time.sleep(0.1)
                        # If no output for 10 seconds, something might be wrong
                        if time.time() - last_output_time > 10.0:
                            print("[WARNING] No output from detectnet for 10 seconds, checking process...")
                            if process.poll() is not None:
                                break
                            last_output_time = time.time()  # Reset timer
                        continue
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    last_output_time = time.time()
                    
                    # Print every 30th line for debugging (reduce spam)
                    if line_count % 30 == 0:
                        print(f"[WATCH] detectnet output (line {line_count}): {line[:100]}")
                    
                    # Parse detectnet output - it outputs detection info in various formats
                    # Format examples:
                    # - "detected 1 objects (class #0 'person'  confidence=0.85)"
                    # - "person: 85%"
                    # - "class: person, confidence: 0.85"
                    line_lower = line.lower()
                    
                    # Check for person detection in various formats
                    person_detected = False
                    
                    # Pattern 1: "class #0 'person'" or "class 0 'person'"
                    if ("class" in line_lower and "person" in line_lower) or \
                       ("'person'" in line_lower) or \
                       ('"person"' in line_lower):
                        person_detected = True
                    
                    # Pattern 2: "person:" or "person " followed by confidence/percentage
                    if "person:" in line_lower or (line_lower.startswith("person ") and ("%" in line or "confidence" in line_lower)):
                        person_detected = True
                    
                    # Pattern 3: "detected" + "person" in same line
                    if "detected" in line_lower and "person" in line_lower:
                        person_detected = True
                    
                    # Pattern 4: Check for class ID 0 (person in COCO dataset)
                    # detectnet might output: "class 0" or "class #0"
                    if ("class" in line_lower and (" 0 " in line_lower or " #0" in line_lower or "=0" in line_lower)) and \
                       ("confidence" in line_lower or "%" in line or "detected" in line_lower):
                        # This might be person (class 0 in COCO), but verify
                        # If we see confidence > threshold, assume person
                        import re
                        conf_match = re.search(r'confidence[=:]?\s*([0-9.]+)', line_lower)
                        if conf_match:
                            conf_val = float(conf_match.group(1))
                            if conf_val > 0.3:  # Reasonable threshold
                                person_detected = True
                                print(f"[WATCH] Detected class 0 with confidence {conf_val}, assuming person")
                    
                    if person_detected:
                        print(f"[WATCH] Person detection found! Line: {line}")
                        print("[WATCH] Person detected!")
                        with self.person_detected_lock:
                            self.person_detected = True
                        
                        if NOTIFICATION_AVAILABLE:
                            print("[WATCH] Sending alert to orin-nano...")
                            try:
                                success = send_person_alert(hostname="orin-nano", confidence=0.8)
                                if success:
                                    print("[WATCH] Alert sent successfully!")
                                else:
                                    print("[WATCH] Alert failed - check SSH connection")
                            except Exception as e:
                                print(f"[WATCH] Alert error: {e}")
                        else:
                            print("[WATCH] Notification not available (desktop_notification module not found)")
                        
                        with self.continuous_detection_lock:
                            self.continuous_detection_running = False
                        process.terminate()
                        print("[WATCH] Person detection stopped (person found)")
                        return
                        
                except Exception as e:
                    print(f"[WATCH] Error reading output: {e}")
                    time.sleep(0.1)
                    continue
                    
            # Check if process exited unexpectedly
            return_code = process.poll()
            if return_code is not None:
                print(f"[WARNING] detectnet process exited with code {return_code}")
                if return_code != 0:
                    # Try to read stderr for error details
                    try:
                        stderr_output = process.stderr.read() if process.stderr else ""
                        if stderr_output:
                            print(f"[ERROR] detectnet stderr: {stderr_output[:500]}")
                    except:
                        pass
                    print("[ERROR] detectnet may have crashed. Check camera permissions and detectnet installation.")
                    print("[ERROR] Common issues:")
                    print("[ERROR]   - Camera device not accessible: ls -l /dev/video0")
                    print("[ERROR]   - detectnet.py not found or not executable")
                    print("[ERROR]   - Missing dependencies or models")
                    print("[ERROR]   - GStreamer pipeline errors")
            
            process.wait()
        except FileNotFoundError:
            print(f"[ERROR] detectnet.py not found. Is jetson-inference installed?")
            print(f"[ERROR] Expected at: {detectnet_path}")
        except Exception as e:
            print(f"[ERROR] Detection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.continuous_detection_lock:
                self.continuous_detection_running = False
            print("[WATCH] Detection stopped")
    
    def run(self):
        """Start the Flask server."""
        print(f"[OK] Starting service on {self.host}:{self.port}")
        print(f"     Camera: /dev/video{self.camera_device}")
        print(f"     Press Ctrl+C to stop\n")
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson Inference Detection Service")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    
    args = parser.parse_args()
    
    service = JetsonInferenceService(
        camera_device=args.device,
        host=args.host,
        port=args.port
    )
    
    try:
        service.run()
    except KeyboardInterrupt:
        print("\n[OK] Shutting down...")
        with service.continuous_detection_lock:
            service.continuous_detection_running = False
