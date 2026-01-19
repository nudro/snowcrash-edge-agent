#!/usr/bin/env python3
"""
Web-based GUI for object detection and tracking with audio input visualization.
Serves HTML interface with live video stream, track data display, and audio transcription panel.
Memory-efficient: uses MJPEG streaming, no frame storage.
Audio input uses Parakeet STT for real-time transcription.
"""
import cv2
import sys
import time
import json
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict
from queue import Queue, Empty
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


# HTML template with audio visualization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Snowcrash - Audio Tracking Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }
        .header {
            background: #161b22;
            padding: 12px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
        }
        .header h1 {
            font-size: 16px;
            color: #58a6ff;
            font-weight: 600;
        }
        .header-info {
            font-size: 11px;
            color: #8b949e;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .mic-indicator-small {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #8b949e;
        }
        .mic-icon-small {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #30363d;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            transition: all 0.3s;
        }
        .mic-icon-small.listening {
            background: #238636;
            animation: pulse-small 1.5s ease-in-out infinite;
        }
        .mic-icon-small.processing {
            background: #f0883e;
            animation: pulse-small 1s ease-in-out infinite;
        }
        @keyframes pulse-small {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .container {
            display: flex;
            height: calc(100vh - 45px);
        }
        .video-panel {
            flex: 1;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            border-right: 1px solid #30363d;
        }
        .video-panel img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .sidebar {
            width: 400px;
            display: flex;
            flex-direction: column;
            background: #161b22;
            border-left: 1px solid #30363d;
        }
        .tracks-panel {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            border-bottom: 1px solid #30363d;
            max-height: 40%;
        }
        .tracks-panel h2 {
            font-size: 13px;
            margin-bottom: 10px;
            color: #58a6ff;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        .tracks-container {
            overflow-y: auto;
        }
        .track-item {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 8px;
            font-size: 11px;
        }
        .track-item .track-id {
            font-size: 12px;
            font-weight: 600;
            color: #f0883e;
            margin-bottom: 4px;
        }
        .track-item .field {
            margin: 2px 0;
            color: #8b949e;
        }
        .track-item strong {
            color: #c9d1d9;
        }
        .no-tracks {
            color: #6e7681;
            font-style: italic;
            text-align: center;
            padding: 20px;
            font-size: 11px;
        }
        .terminal-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 60%;
            background: #0d1117;
        }
        .terminal-panel h2 {
            font-size: 13px;
            margin: 12px;
            color: #58a6ff;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        .terminal-display {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            background: #0a0e14;
            border: 1px solid #30363d;
            border-radius: 4px;
            margin: 0 12px 12px 12px;
            font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            line-height: 1.6;
            color: #c9d1d9;
        }
        .terminal-line {
            margin: 2px 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .terminal-prompt {
            color: #58a6ff;
        }
        .terminal-output {
            color: #c9d1d9;
        }
        .terminal-error {
            color: #f85149;
        }
        .terminal-transcription {
            color: #238636;
            font-weight: 500;
        }
        .terminal-response {
            color: #79c0ff;
        }
        .quick-prompts {
            padding: 8px 12px;
            border-top: 1px solid #30363d;
            background: #0d1117;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 100px;
            overflow-y: auto;
        }
        .quick-prompt-btn {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 4px;
            padding: 4px 8px;
            color: #9ca3af;
            font-size: 10px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .quick-prompt-btn:hover {
            background: #374151;
            border-color: #58a6ff;
            color: #58a6ff;
        }
        .quick-prompt-btn:active {
            background: #1e40af;
        }
        .loading-indicator {
            color: #6e7681;
            font-size: 11px;
            font-style: italic;
            padding: 8px 12px;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0d1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Snowcrash - Audio Tracking Viewer</h1>
        <div class="header-info">
            <div>Device: <strong id="device-ip">Loading...</strong></div>
            <div id="current-time">Loading...</div>
            <div class="mic-indicator-small" id="mic-indicator-small">
                <div class="mic-icon-small" id="mic-icon-small">🎤</div>
                <span id="mic-status-small">Audio</span>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="video-panel">
            <img id="video-stream" src="/video_feed" alt="Video Stream">
        </div>
        <div class="sidebar">
            <div class="tracks-panel">
                <h2>Active Tracks</h2>
                <div class="tracks-container" id="tracks-container">
                    <div class="no-tracks">No tracks detected yet...</div>
                </div>
            </div>
            <div class="terminal-panel">
                <h2>Terminal Output</h2>
                <div class="terminal-display" id="terminal-display">
                    <div class="terminal-line terminal-output">Snowcrash Audio Agent initialized. Listening for voice commands...</div>
                    <div class="terminal-line terminal-prompt">$</div>
                </div>
                <div class="quick-prompts">
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Status Check')">STATUS CHECK</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Target Count')">TARGET COUNT</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Distance Report')">DISTANCE REPORT</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Position Update')">POSITION UPDATE</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Color Intel')">COLOR INTEL</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Track IDs')">TRACK IDs</button>
                    <button class="quick-prompt-btn" onclick="triggerQuickPrompt('Environment Scan')">ENVIRONMENT SCAN</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update time every second
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Update device IP
        fetch('/device_info')
            .then(r => r.json())
            .then(data => {
                document.getElementById('device-ip').textContent = data.ip;
            });

        // Terminal output functions
        function addTerminalLine(text, className = 'terminal-output') {
            const terminal = document.getElementById('terminal-display');
            // Remove existing prompt
            const existingPrompt = terminal.querySelector('.terminal-prompt');
            if (existingPrompt) {
                existingPrompt.remove();
            }
            
            const line = document.createElement('div');
            line.className = 'terminal-line ' + className;
            line.textContent = text;
            terminal.appendChild(line);
            
            // Add new prompt
            const prompt = document.createElement('div');
            prompt.className = 'terminal-line terminal-prompt';
            prompt.textContent = '$';
            terminal.appendChild(prompt);
            
            terminal.scrollTop = terminal.scrollHeight;
        }

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
                            firstSeen = track.first_seen ? String(track.first_seen).substring(11, 19) : 'N/A';
                            lastSeen = track.last_seen ? String(track.last_seen).substring(11, 19) : 'N/A';
                        }
                        const trackId = String(track.track_id || '?');
                        const trackClass = String(track.class || 'unknown');
                        
                        return '<div class="track-item">' +
                            '<div class="track-id">Track ID: ' + trackId + '</div>' +
                            '<div class="field"><strong>Class:</strong> ' + trackClass + '</div>' +
                            '<div class="field"><strong>Velocity:</strong> vx=' + vx + ', vy=' + vy + ' px/s</div>' +
                            '<div class="field"><strong>Distance:</strong> ' + dist + ' px</div>' +
                            '<div class="field"><strong>Speed:</strong> ' + speed + ' px/s</div>' +
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
        
        // Poll for STT transcriptions and audio status
        let currentTranscriptionId = null;
        let lastTranscription = '';
        
        function updateMicStatus(status) {
            const micIcon = document.getElementById('mic-icon-small');
            const micStatus = document.getElementById('mic-status-small');
            
            micIcon.classList.remove('listening', 'processing');
            
            if (status === 'listening') {
                micIcon.classList.add('listening');
                micStatus.textContent = 'Listening';
            } else if (status === 'processing') {
                micIcon.classList.add('processing');
                micStatus.textContent = 'Processing';
            } else {
                micStatus.textContent = 'Ready';
            }
        }
        
        function pollSTTTranscription() {
            fetch('/stt_transcription')
                .then(r => r.json())
                .then(data => {
                    if (data.enabled) {
                        if (data.is_listening) {
                            updateMicStatus('listening');
                        } else if (data.is_processing) {
                            updateMicStatus('processing');
                        } else {
                            updateMicStatus('ready');
                        }
                        
                        // Handle completed transcription (skip partial - only show final)
                        if (data.has_transcription && data.transcription) {
                            // Add final transcription only if new
                            if (data.transcription !== lastTranscription || !currentTranscriptionId || currentTranscriptionId !== data.transcription_id) {
                                addTerminalLine('You: ' + data.transcription, 'terminal-transcription');
                                lastTranscription = data.transcription;
                                currentTranscriptionId = data.transcription_id;
                                
                                // Send to agent
                                sendToAgent(data.transcription);
                            }
                        }
                    } else {
                        updateMicStatus('ready');
                    }
                })
                .catch(err => console.error('Error polling STT:', err));
        }
        
        function sendToAgent(message) {
            // Send to server with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
                signal: controller.signal
            })
            .then(r => {
                clearTimeout(timeoutId);
                if (!r.ok) {
                    throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                }
                return r.json();
            })
            .then(data => {
                // Add assistant response (terminal-style)
                if (data.response) {
                    addTerminalLine(data.response, 'terminal-response');
                } else if (data.error) {
                    addTerminalLine('Error: ' + data.error, 'terminal-error');
                } else {
                    addTerminalLine('No response received.', 'terminal-error');
                }
            })
            .catch(err => {
                clearTimeout(timeoutId);
                let errorMsg = 'Error: ';
                if (err.name === 'AbortError') {
                    errorMsg += 'Request timed out (60s)';
                } else {
                    errorMsg += err.message;
                }
                addTerminalLine(errorMsg, 'terminal-error');
            });
        }
        
        // Poll every 300ms for better responsiveness
        setInterval(pollSTTTranscription, 300);
        pollSTTTranscription();
        
        // Quick prompt functionality (triggers via voice simulation)
        function triggerQuickPrompt(promptType) {
            const promptMap = {
                'Status Check': "What's the current situation?",
                'Target Count': "How many objects detected?",
                'Distance Report': "Distances to all targets",
                'Position Update': "Positions of all objects",
                'Color Intel': "Color identification of targets",
                'Track IDs': "List all active track IDs",
                'Environment Scan': "Based on the objects in the video, what kind of environment is this?"
            };
            
            const prompt = promptMap[promptType] || promptType;
            sendToAgent(prompt);
        }
    </script>
</body>
</html>
"""


def get_local_ip():
    """Get the local IP address of the device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


class TrackingViewerAudioGUI:
    """Web-based tracking viewer with audio input visualization and transcription."""
    
    def __init__(
        self,
        model_path: str = "yolo26n-seg.pt",
        device: int = 0,
        confidence_threshold: float = 0.25,
        use_gstreamer: bool = True,
        port: int = 8080,
        agent = None,  # SimpleAgent instance (shared LLM - no extra memory)
        stt_model = None,  # ParakeetSTT instance (REQUIRED)
        stt_card: int = 1,  # ALSA card number for USB microphone
        stt_chunk_duration: float = 3.0  # Audio chunk duration for STT
    ):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not installed. Install with: pip install flask")
        if stt_model is None:
            raise RuntimeError("STT model is REQUIRED for AudioGUI. Provide ParakeetSTT instance.")
        
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_gstreamer = use_gstreamer
        self.port = port
        self.agent = agent  # SimpleAgent instance (reuses LLM - no extra memory)
        self.stt_model = stt_model  # ParakeetSTT instance (REQUIRED)
        self.stt_card = stt_card
        self.stt_chunk_duration = stt_chunk_duration
        
        # STT transcription queue (thread-safe, memory-efficient)
        self.stt_transcription_queue = Queue(maxsize=10)
        self.stt_listening = False
        self.stt_processing = False
        self.current_transcription_id = 0
        self.partial_transcription = ""
        
        # Track data storage (shared between threads)
        self.tracks_data = {}
        self.tracks_lock = threading.Lock()
        
        # Historical track metadata
        self.track_metadata_history: Dict[int, Dict] = {}
        self.track_metadata_lock = threading.Lock()
        
        # Video capture (will be set in thread)
        self.cap = None
        self.frame = None
        self.raw_frame = None
        self.frame_lock = threading.Lock()
        
        # Flask app
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)
        self._setup_routes()
        
        # Device info
        self.device_ip = get_local_ip()
        
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # STT thread
        self.stt_thread = None
        self.stt_running = False
    
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
        
        @self.app.route('/stt_transcription')
        def get_stt_transcription():
            """Get latest STT transcription and status."""
            try:
                transcription = None
                has_transcription = False
                
                try:
                    transcription = self.stt_transcription_queue.get_nowait()
                    has_transcription = True
                    self.current_transcription_id += 1
                except Empty:
                    pass
                
                return jsonify({
                    "enabled": True,
                    "is_listening": self.stt_listening,
                    "is_processing": self.stt_processing,
                    "has_transcription": has_transcription,
                    "transcription": transcription if has_transcription else None,
                    "transcription_id": self.current_transcription_id if has_transcription else None,
                    "partial_transcription": self.partial_transcription
                })
            except Exception as e:
                return jsonify({
                    "enabled": False,
                    "error": str(e)
                })
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Handle chat messages using SimpleAgent."""
            if not self.agent:
                return jsonify({"error": "Agent not available. Chat requires SimpleAgent instance."}), 500
            
            try:
                data = request.get_json()
                message = data.get('message', '').strip()
                
                if not message:
                    return jsonify({"error": "Empty message"}), 400
                
                # Use agent to generate response (minimal logging)
                if hasattr(self.agent, 'run_sync'):
                    response = self.agent.run_sync(message)
                elif hasattr(self.agent, 'run'):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(self.agent.run(message))
                    finally:
                        loop.close()
                else:
                    return jsonify({"error": "Agent does not support run_sync or run methods"}), 500
                
                # Ensure response is a string
                if response is None:
                    response = "No response generated."
                elif not isinstance(response, str):
                    response = str(response)
                
                return jsonify({"response": response})
                
            except Exception as e:
                print(f"[ERROR] Chat error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
    
    def _generate_video_frames(self):
        """Generate MJPEG frames from video capture."""
        while True:
            frame_to_send = None
            
            with self.frame_lock:
                if self.frame is not None:
                    frame_to_send = self.frame.copy()
            
            if frame_to_send is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
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
    
    def _stt_listening_loop(self):
        """Background thread for continuous STT listening."""
        self.stt_running = True
        
        try:
            if hasattr(self.stt_model, 'listen_continuous'):
                while self.stt_running:
                    try:
                        self.stt_listening = True
                        self.partial_transcription = "Listening..."
                        
                        transcription = self.stt_model.listen_continuous(
                            card=self.stt_card,
                            chunk_duration=self.stt_chunk_duration,
                            sample_rate=16000,
                            max_silence_chunks=3,  # Reduced from 5 for faster initial response
                            silence_chunks_after_speech=1,  # Reduced from 2 to 1 for faster response (reduces lag)
                            timestamps=False
                        )
                        
                        if transcription and transcription.strip():
                            self.stt_processing = True
                            self.partial_transcription = transcription.strip()
                            
                            # Add to queue
                            try:
                                self.stt_transcription_queue.put_nowait(transcription.strip())
                            except:
                                # Queue full - remove oldest
                                try:
                                    self.stt_transcription_queue.get_nowait()
                                    self.stt_transcription_queue.put_nowait(transcription.strip())
                                except:
                                    pass
                            
                            # Reset for next round
                            time.sleep(0.5)  # Brief pause before next listen
                            self.partial_transcription = ""
                        else:
                            self.partial_transcription = ""
                        
                        self.stt_listening = False
                        self.stt_processing = False
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        self.stt_listening = False
                        self.stt_processing = False
                        self.partial_transcription = ""
                        time.sleep(1)
        finally:
            self.stt_running = False
            self.stt_listening = False
            self.stt_processing = False
    
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
                
                raw_frame_for_detection = frame.copy()
                annotated_frame = frame.copy()
                
                # Draw trajectories
                for track_id, trajectory in trajectories.items():
                    if len(trajectory) > 1:
                        points = trajectory[-30:]
                        for i in range(1, len(points)):
                            cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)
                
                # Process tracked objects
                active_track_ids = set()
                all_track_ids_processed_this_frame = set()
                frame_detections = []
                
                if results[0].boxes is not None:
                    has_track_ids = results[0].boxes.id is not None
                    boxes_list = list(results[0].boxes)
                    track_ids_list = list(results[0].boxes.id) if has_track_ids else [None] * len(boxes_list)
                    
                    for idx, box in enumerate(boxes_list):
                        track_id = track_ids_list[idx] if has_track_ids else None
                        
                        if track_id is not None:
                            track_id_int = int(track_id)
                            active_track_ids.add(track_id_int)
                            all_track_ids_processed_this_frame.add(track_id_int)
                        else:
                            track_id_int = -(idx + 1)
                            all_track_ids_processed_this_frame.add(track_id_int)
                        
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.model.names[cls_id]
                        bbox = box.xyxy[0].tolist()
                        
                        frame_detections.append({"class": cls_name, "confidence": conf})
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        if track_id_int not in trajectories:
                            trajectories[track_id_int] = []
                        trajectories[track_id_int].append((center_x, center_y))
                        if len(trajectories[track_id_int]) > 30:
                            trajectories[track_id_int] = trajectories[track_id_int][-30:]
                        
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
                
                trajectories = {tid: traj for tid, traj in trajectories.items() if tid in all_track_ids_processed_this_frame}
                
                # Update shared track data
                current_tracks = {}
                for track_id, track_info in track_metadata.items():
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
                
                with self.tracks_lock:
                    self.tracks_data = current_tracks
                
                with self.track_metadata_lock:
                    for track_id, track_info in track_metadata.items():
                        if track_id in all_track_ids_processed_this_frame:
                            self.track_metadata_history[track_id] = track_info.copy()
                        else:
                            if track_id not in self.track_metadata_history:
                                self.track_metadata_history[track_id] = track_info.copy()
                            else:
                                self.track_metadata_history[track_id]["last_seen"] = track_info["last_seen"]
                                self.track_metadata_history[track_id]["last_frame_time"] = track_info["last_frame_time"]
                
                with self.frame_lock:
                    self.frame = annotated_frame
                    self.raw_frame = raw_frame_for_detection
                
                if frame_detections:
                    try:
                        from agent.langchain_tools import _statistics_tool
                        _statistics_tool.update(frame_detections)
                    except:
                        pass
                
                time.sleep(0.001)
        
        finally:
            if cap:
                cap.release()
            print(f"[OK] Tracking loop ended. Processed {frame_count} frames.")
    
    def run(self, duration_seconds: float = 0):
        """Start the web server and tracking loop."""
        tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(duration_seconds,),
            daemon=True
        )
        tracking_thread.start()
        
        # Start STT listening thread (always enabled for AudioGUI)
        self.stt_thread = threading.Thread(
            target=self._stt_listening_loop,
            daemon=True
        )
        self.stt_thread.start()
        print(f"[STT] Audio input enabled (Card: {self.stt_card})")
        
        print(f"\n[OK] Starting AudioGUI web server on http://{self.device_ip}:{self.port}")
        print(f"     Open your browser to: http://{self.device_ip}:{self.port}")
        print(f"     Audio input: Enabled (Card {self.stt_card})")
        print(f"     Chat integrated with SimpleAgent (shares LLM - no extra memory)")
        print(f"     Press Ctrl+C to stop")
        print()
        
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based object tracking viewer with audio input visualization")
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt", help="YOLO model path")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-gstreamer", action="store_true", help="Disable GStreamer")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = until stopped)")
    parser.add_argument("--stt-model-path", type=str, default=None, help="Path to Parakeet .nemo model (required)")
    parser.add_argument("--stt-card", type=int, default=1, help="ALSA card number for USB microphone (default: 1)")
    parser.add_argument("--stt-chunk-duration", type=float, default=3.0, help="Audio chunk duration for STT in seconds (default: 3.0)")
    
    args = parser.parse_args()
    
    # Initialize STT model (REQUIRED for AudioGUI)
    try:
        from tools.parakeet_stt import create_parakeet_stt
        print("[STT] Initializing Parakeet STT for audio input...")
        stt_model = create_parakeet_stt(model_path=args.stt_model_path)
        print("[OK] Parakeet STT initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize STT: {e}")
        print("[ERROR] AudioGUI requires STT. Exiting.")
        sys.exit(1)
    
    viewer = TrackingViewerAudioGUI(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_gstreamer=not args.no_gstreamer,
        port=args.port,
        agent=None,  # Will be set by main.py when integrated
        stt_model=stt_model,
        stt_card=args.stt_card,
        stt_chunk_duration=args.stt_chunk_duration
    )
    
    viewer.run(duration_seconds=args.duration)

