#!/usr/bin/env python3
"""
Web-based GUI for object detection and tracking with Jan.ai-style chat integration.
Serves HTML interface with live video stream, track data display, and chat panel.
Memory-efficient: uses MJPEG streaming, no frame storage.
Chat uses the same SimpleAgent LLM instance (shared memory).
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


# HTML template with chat integration (Jan.ai-style)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Snowcrash - Tracking Viewer with Chat</title>
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
            padding: 10px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 70px;
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
            flex-direction: column;
            gap: 4px;
            align-items: flex-end;
            justify-content: center;
        }
        .memory-bar-container {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 220px;
        }
        .memory-label {
            font-size: 10px;
            color: #8b949e;
            white-space: nowrap;
        }
        .memory-bar-wrapper {
            flex: 1;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            height: 16px;
            position: relative;
            overflow: hidden;
        }
        .memory-bar {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease, background-color 0.3s ease;
            background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        }
        .memory-bar.warning {
            background: linear-gradient(90deg, #f85149 0%, #ff6b6b 100%);
        }
        .memory-bar.moderate {
            background: linear-gradient(90deg, #d29922 0%, #f1e05a 100%);
        }
        .memory-text {
            font-size: 9px;
            color: #c9d1d9;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
            white-space: nowrap;
        }
        .container {
            display: flex;
            height: calc(100vh - 70px);
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
            max-height: 50%;
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
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 50%;
            background: #0d1117;
        }
        .chat-panel h2 {
            font-size: 13px;
            margin: 12px;
            color: #58a6ff;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .chat-message {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .message-header {
            font-size: 11px;
            font-weight: 600;
            color: #58a6ff;
        }
        .message-content {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.5;
            color: #c9d1d9;
        }
        .message-user .message-content {
            background: #21262d;
            border: 1px solid #30363d;
        }
        .message-assistant .message-content {
            background: #161b22;
            border: 1px solid #30363d;
        }
        .quick-prompts {
            padding: 8px 12px;
            border-top: 1px solid #30363d;
            background: #0d1117;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 120px;
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
        .chat-input-container {
            padding: 12px;
            border-top: 1px solid #30363d;
            background: #161b22;
        }
        .chat-input-form {
            display: flex;
            gap: 8px;
        }
        .chat-input {
            flex: 1;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            color: #c9d1d9;
            font-size: 13px;
            font-family: inherit;
        }
        .chat-input:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .chat-send {
            background: #238636;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            color: #fff;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }
        .chat-send:hover {
            background: #2ea043;
        }
        .chat-send:disabled {
            background: #30363d;
            color: #6e7681;
            cursor: not-allowed;
        }
        .chat-loading {
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
        <h1>Snowcrash - Object Tracking & Chat</h1>
        <div class="header-info">
            <div>Device: <strong id="device-ip">Loading...</strong></div>
            <div id="current-time">Loading...</div>
            <div class="memory-bar-container">
                <span class="memory-label">GPU:</span>
                <div class="memory-bar-wrapper">
                    <div class="memory-bar" id="gpu-memory-bar" style="width: 0%"></div>
                    <div class="memory-text" id="gpu-memory-text">0%</div>
                </div>
            </div>
            <div class="memory-bar-container">
                <span class="memory-label">RAM:</span>
                <div class="memory-bar-wrapper">
                    <div class="memory-bar" id="ram-memory-bar" style="width: 0%"></div>
                    <div class="memory-text" id="ram-memory-text">0%</div>
                </div>
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
            <div class="chat-panel">
                <h2>Chat</h2>
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-message message-assistant">
                        <div class="message-header">Assistant</div>
                        <div class="message-content">Ask the Snowcrash Agent</div>
                    </div>
                </div>
                <div class="quick-prompts">
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Status Check')">STATUS CHECK</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Target Count')">TARGET COUNT</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Distance Report')">DISTANCE REPORT</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Position Update')">POSITION UPDATE</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Color Intel')">COLOR INTEL</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Track IDs')">TRACK IDs</button>
                    <button class="quick-prompt-btn" onclick="sendQuickPrompt('Environment Scan')">ENVIRONMENT SCAN</button>
                </div>
                <div class="chat-input-container">
                    <form class="chat-input-form" id="chat-form" onsubmit="sendMessage(event)">
                        <input type="text" class="chat-input" id="chat-input" placeholder="Type your message..." autocomplete="off">
                        <button type="submit" class="chat-send" id="chat-send">Send</button>
                    </form>
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

        // Update memory usage every 2 seconds
        function updateMemoryUsage() {
            fetch('/memory_usage')
                .then(r => {
                    if (!r.ok) {
                        throw new Error(`HTTP error! status: ${r.status}`);
                    }
                    return r.json();
                })
                .then(data => {
                    // Update GPU memory
                    if (data.gpu) {
                        const gpuPercent = Math.round(data.gpu.percent);
                        const gpuBar = document.getElementById('gpu-memory-bar');
                        const gpuText = document.getElementById('gpu-memory-text');
                        gpuBar.style.width = gpuPercent + '%';
                        gpuText.textContent = `${gpuPercent}% (${data.gpu.used_gb.toFixed(1)}/${data.gpu.total_gb.toFixed(1)}GB)`;
                        
                        // Change color based on usage
                        gpuBar.className = 'memory-bar';
                        if (gpuPercent >= 85) {
                            gpuBar.classList.add('warning');
                        } else if (gpuPercent >= 70) {
                            gpuBar.classList.add('moderate');
                        }
                    } else {
                        document.getElementById('gpu-memory-text').textContent = 'N/A';
                    }
                    
                    // Update RAM memory
                    if (data.ram) {
                        const ramPercent = Math.round(data.ram.percent);
                        const ramBar = document.getElementById('ram-memory-bar');
                        const ramText = document.getElementById('ram-memory-text');
                        ramBar.style.width = ramPercent + '%';
                        ramText.textContent = `${ramPercent}% (${data.ram.used_gb.toFixed(1)}/${data.ram.total_gb.toFixed(1)}GB)`;
                        
                        // Change color based on usage
                        ramBar.className = 'memory-bar';
                        if (ramPercent >= 85) {
                            ramBar.classList.add('warning');
                        } else if (ramPercent >= 70) {
                            ramBar.classList.add('moderate');
                        }
                    }
                })
                .catch(err => console.error('Error fetching memory usage:', err));
        }
        setInterval(updateMemoryUsage, 2000); // Update every 2 seconds
        updateMemoryUsage(); // Initial update

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
        
        // Poll for STT transcriptions every 500ms (memory-efficient polling)
        let sttPolling = false;
        function pollSTTTranscription() {
            if (!sttPolling) return; // Only poll if STT is enabled
            
            fetch('/stt_transcription')
                .then(r => r.json())
                .then(data => {
                    if (data.enabled && data.has_transcription && data.transcription) {
                        // Auto-fill chat input with transcription and send
                        const input = document.getElementById('chat-input');
                        input.value = data.transcription;
                        
                        // Trigger send automatically
                        const event = new Event('submit', { bubbles: true, cancelable: true });
                        document.getElementById('chat-form').dispatchEvent(event);
                    }
                })
                .catch(err => console.error('Error polling STT:', err));
        }
        
        // Check if STT is enabled on page load
        fetch('/stt_transcription')
            .then(r => r.json())
            .then(data => {
                if (data.enabled) {
                    sttPolling = true;
                    setInterval(pollSTTTranscription, 500); // Poll every 500ms
                    console.log('[STT] Audio input enabled - transcriptions will auto-fill chat');
                }
            });

        // Quick prompt functionality
        function sendQuickPrompt(promptType) {
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
            const input = document.getElementById('chat-input');
            input.value = prompt;
            
            // Trigger send
            const event = new Event('submit', { bubbles: true, cancelable: true });
            document.getElementById('chat-form').dispatchEvent(event);
        }
        
        // Chat functionality
        function addMessage(role, content) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message message-' + role;
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.textContent = role === 'user' ? 'You' : 'Assistant';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(header);
            messageDiv.appendChild(contentDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function sendMessage(event) {
            event.preventDefault();
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Disable send button
            const sendBtn = document.getElementById('chat-send');
            sendBtn.disabled = true;
            
            // Show loading
            const messagesContainer = document.getElementById('chat-messages');
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'chat-loading';
            loadingDiv.id = 'chat-loading';
            loadingDiv.textContent = 'Thinking...';
            messagesContainer.appendChild(loadingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            // Send to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(r => r.json())
            .then(data => {
                // Remove loading
                const loading = document.getElementById('chat-loading');
                if (loading) loading.remove();
                
                // Add assistant response
                if (data.response) {
                    addMessage('assistant', data.response);
                } else if (data.error) {
                    addMessage('assistant', 'Error: ' + data.error);
                }
                
                sendBtn.disabled = false;
                input.focus();
            })
            .catch(err => {
                const loading = document.getElementById('chat-loading');
                if (loading) loading.remove();
                addMessage('assistant', 'Error: Failed to send message. ' + err.message);
                sendBtn.disabled = false;
            });
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


class TrackingViewerJanChat:
    """Web-based tracking viewer with Jan.ai-style chat integration."""
    
    def __init__(
        self,
        model_path: str = "yolo26n-seg.pt",
        device: int = 0,
        confidence_threshold: float = 0.25,
        use_gstreamer: bool = True,
        port: int = 8080,
        agent = None,  # SimpleAgent instance (shared LLM - no extra memory)
        stt_model = None,  # ParakeetSTT or WhisperModel instance (optional)
        stt_enabled: bool = False,  # Enable STT audio input
        stt_card: int = 1,  # ALSA card number for USB microphone
        stt_chunk_duration: float = 3.0  # Audio chunk duration for STT
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
        self.agent = agent  # SimpleAgent instance (reuses LLM - no extra memory)
        self.stt_model = stt_model  # ParakeetSTT or WhisperModel instance
        self.stt_enabled = stt_enabled
        self.stt_card = stt_card
        self.stt_chunk_duration = stt_chunk_duration
        
        # STT transcription queue (thread-safe, memory-efficient)
        # Stores transcriptions from background STT thread
        self.stt_transcription_queue = Queue(maxsize=10)  # Max 10 transcriptions to prevent memory buildup
        self.stt_thread = None
        self.stt_running = False
        
        print(f"Loading YOLO model: {model_path}")
        
        # Clear GPU memory cache before loading model
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                # Clear Python garbage collector
                gc.collect()
                # Clear GPU cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Check available memory
                props = torch.cuda.get_device_properties(0)
                total_mem = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                available = total_mem - cached
                print(f"[DEBUG] GPU memory - Total: {total_mem:.2f} GB, Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB, Available: {available:.2f} GB")
                
                # If less than 1GB available, try to free more
                if available < 1.0:
                    print(f"[WARNING] Low GPU memory ({available:.2f} GB). Attempting aggressive cleanup...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Load model - YOLO will automatically use GPU if available
        self.model = YOLO(model_path)
        
        # Verify model type
        if hasattr(self.model, 'task'):
            print(f"[DEBUG] Model task: {self.model.task}")
        if 'seg' in model_path.lower():
            print(f"[DEBUG] Segmentation model detected: {model_path}")
        else:
            print(f"[WARNING] Model path does not contain 'seg' - may not support segmentation: {model_path}")
        
        # Track data storage (shared between threads)
        self.tracks_data = {}
        self.tracks_lock = threading.Lock()
        
        # Historical track metadata (for duration calculations)
        self.track_metadata_history: Dict[int, Dict] = {}
        self.track_metadata_lock = threading.Lock()
        
        # Video capture (will be set in thread)
        self.cap = None
        self.frame = None  # Annotated frame for display
        self.raw_frame = None  # Raw frame without annotations
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
        
        @self.app.route('/memory_usage')
        def memory_usage():
            """Get system memory usage (RAM and GPU VRAM)."""
            memory_data = {
                "ram": None,
                "gpu": None
            }
            
            # Get RAM usage
            try:
                import psutil
                ram = psutil.virtual_memory()
                memory_data["ram"] = {
                    "total_gb": ram.total / (1024**3),
                    "used_gb": ram.used / (1024**3),
                    "available_gb": ram.available / (1024**3),
                    "percent": ram.percent
                }
            except ImportError:
                # Fallback: try reading /proc/meminfo
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                        lines = meminfo.split('\n')
                        mem_total = int([l for l in lines if 'MemTotal:' in l][0].split()[1]) * 1024
                        mem_available = int([l for l in lines if 'MemAvailable:' in l][0].split()[1]) * 1024
                        mem_used = mem_total - mem_available
                        memory_data["ram"] = {
                            "total_gb": mem_total / (1024**3),
                            "used_gb": mem_used / (1024**3),
                            "available_gb": mem_available / (1024**3),
                            "percent": (mem_used / mem_total) * 100
                        }
                except:
                    pass
            
            # Get GPU VRAM usage
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    
                    # Use reserved memory as the actual usage (PyTorch caching)
                    memory_data["gpu"] = {
                        "total_gb": total / (1024**3),
                        "allocated_gb": allocated / (1024**3),
                        "reserved_gb": reserved / (1024**3),
                        "used_gb": reserved / (1024**3),  # Show reserved as "used"
                        "percent": (reserved / total) * 100 if total > 0 else 0
                    }
            except ImportError:
                pass
            except Exception as e:
                # GPU check failed, leave as None
                pass
            
            return jsonify(memory_data)
        
        @self.app.route('/stt_transcription')
        def get_stt_transcription():
            """Get latest STT transcription (polling endpoint)."""
            if not self.stt_enabled:
                return jsonify({"transcription": None, "enabled": False})
            
            try:
                # Non-blocking get from queue
                transcription = self.stt_transcription_queue.get_nowait()
                return jsonify({
                    "transcription": transcription,
                    "enabled": True,
                    "has_transcription": True
                })
            except Empty:
                # No new transcription
                return jsonify({
                    "transcription": None,
                    "enabled": True,
                    "has_transcription": False
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
                
                # Use agent to generate response (reuses LLM - no extra memory)
                if hasattr(self.agent, 'run_sync'):
                    response = self.agent.run_sync(message)
                elif hasattr(self.agent, 'run'):
                    # Async version - run in sync mode
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(self.agent.run(message))
                    finally:
                        loop.close()
                else:
                    return jsonify({"error": "Agent does not support run_sync or run methods"}), 500
                
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
    
    def get_track_duration(self, track_id: Optional[int] = None, object_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """Calculate how long a track has been in the frame (same as TrackingWebViewer)."""
        current_time = time.time()
        results = []
        
        with self.track_metadata_lock:
            if track_id is not None:
                if track_id in self.track_metadata_history:
                    track_info = self.track_metadata_history[track_id]
                    first_time = track_info.get("first_frame_time", current_time)
                    last_time = track_info.get("last_frame_time", current_time)
                    is_active = track_id in self.tracks_data
                    if is_active:
                        last_time = current_time
                    
                    duration_seconds = last_time - first_time
                    results.append({
                        "track_id": track_id,
                        "class": track_info.get("class", "unknown"),
                        "duration_minutes": round(duration_seconds / 60.0, 2),
                        "duration_seconds": round(duration_seconds, 1),
                        "first_seen": track_info.get("first_seen", "N/A"),
                        "last_seen": track_info.get("last_seen", "N/A"),
                        "is_active": is_active
                    })
            elif object_class is not None:
                object_class_lower = object_class.lower()
                for track_id_val, track_info in self.track_metadata_history.items():
                    track_class = track_info.get("class", "").lower()
                    if track_class == object_class_lower:
                        first_time = track_info.get("first_frame_time", current_time)
                        last_time = track_info.get("last_frame_time", current_time)
                        is_active = track_id_val in self.tracks_data
                        if is_active:
                            last_time = current_time
                        
                        duration_seconds = last_time - first_time
                        results.append({
                            "track_id": track_id_val,
                            "class": track_info.get("class", "unknown"),
                            "duration_minutes": round(duration_seconds / 60.0, 2),
                            "duration_seconds": round(duration_seconds, 1),
                            "first_seen": track_info.get("first_seen", "N/A"),
                            "last_seen": track_info.get("last_seen", "N/A"),
                            "is_active": is_active
                        })
            else:
                for track_id_val, track_info in self.track_metadata_history.items():
                    first_time = track_info.get("first_frame_time", current_time)
                    last_time = track_info.get("last_frame_time", current_time)
                    is_active = track_id_val in self.tracks_data
                    if is_active:
                        last_time = current_time
                    
                    duration_seconds = last_time - first_time
                    results.append({
                        "track_id": track_id_val,
                        "class": track_info.get("class", "unknown"),
                        "duration_minutes": round(duration_seconds / 60.0, 2),
                        "duration_seconds": round(duration_seconds, 1),
                        "first_seen": track_info.get("first_seen", "N/A"),
                        "last_seen": track_info.get("last_seen", "N/A"),
                        "is_active": is_active
                    })
        
        results.sort(key=lambda x: x["duration_minutes"], reverse=True)
        return results
    
    def _tracking_loop(self, duration_seconds: float = 0):
        """Main tracking loop (runs in separate thread) - same logic as TrackingWebViewer."""
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
                
                # Run tracking (bounding boxes only, no masks)
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
                
                # Update shared track data with velocity calculations
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
    
    def _stt_listening_loop(self):
        """Background thread for continuous STT listening (memory-efficient)."""
        if not self.stt_model or not self.stt_enabled:
            return
        
        print(f"[STT] Starting continuous audio listening (Card: {self.stt_card})...")
        self.stt_running = True
        
        try:
            # Use Parakeet's continuous listening if available
            if hasattr(self.stt_model, 'listen_continuous'):
                # Parakeet STT
                while self.stt_running:
                    try:
                        transcription = self.stt_model.listen_continuous(
                            card=self.stt_card,
                            chunk_duration=self.stt_chunk_duration,
                            sample_rate=16000,
                            max_silence_chunks=5,
                            silence_chunks_after_speech=2,  # Wait for 2 silent chunks after speech ends
                            timestamps=False
                        )
                        
                        if transcription and transcription.strip():
                            # Add to queue (will overwrite old if queue full - prevents memory buildup)
                            try:
                                self.stt_transcription_queue.put_nowait(transcription.strip())
                            except:
                                # Queue full - remove oldest and add new
                                try:
                                    self.stt_transcription_queue.get_nowait()
                                    self.stt_transcription_queue.put_nowait(transcription.strip())
                                except:
                                    pass
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"[STT] Error in listening loop: {e}")
                        time.sleep(1)  # Wait before retrying
            else:
                # Whisper fallback - use simple recording loop
                while self.stt_running:
                    try:
                        # For Whisper, we'd need a different approach
                        # For now, just log that STT model doesn't support continuous listening
                        print("[STT] STT model doesn't support continuous listening")
                        break
                    except Exception as e:
                        print(f"[STT] Error: {e}")
                        time.sleep(1)
        finally:
            self.stt_running = False
            print("[STT] STT listening loop stopped")
    
    def run(self, duration_seconds: float = 0):
        """Start the web server and tracking loop."""
        tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(duration_seconds,),
            daemon=True
        )
        tracking_thread.start()
        
        # Start STT listening thread if enabled
        if self.stt_enabled and self.stt_model:
            self.stt_thread = threading.Thread(
                target=self._stt_listening_loop,
                daemon=True
            )
            self.stt_thread.start()
            print(f"[STT] Audio input enabled (Card: {self.stt_card})")
        
        print(f"\n[OK] Starting web server on http://{self.device_ip}:{self.port}")
        print(f"     Open your browser to: http://{self.device_ip}:{self.port}")
        print(f"     Chat integrated with SimpleAgent (shares LLM - no extra memory)")
        if self.stt_enabled:
            print(f"     Audio input: Enabled (Card {self.stt_card})")
        print(f"     Press Ctrl+C to stop")
        print()
        
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based object tracking viewer with Jan.ai-style chat")
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt", help="YOLO model path")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-gstreamer", action="store_true", help="Disable GStreamer")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = until stopped)")
    parser.add_argument("--stt-audio", action="store_true", help="Enable STT audio input (requires --stt-model-path)")
    parser.add_argument("--stt-model-path", type=str, default=None, help="Path to Parakeet .nemo model (required for --stt-audio)")
    parser.add_argument("--stt-card", type=int, default=1, help="ALSA card number for USB microphone (default: 1)")
    parser.add_argument("--stt-chunk-duration", type=float, default=3.0, help="Audio chunk duration for STT in seconds (default: 3.0)")
    
    args = parser.parse_args()
    
    # Initialize STT model if audio input is enabled
    stt_model = None
    if args.stt_audio:
        try:
            from tools.parakeet_stt import create_parakeet_stt
            print("[STT] Initializing Parakeet STT for audio input...")
            stt_model = create_parakeet_stt(model_path=args.stt_model_path)
            print("[OK] Parakeet STT initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize STT: {e}")
            print("[ERROR] STT audio input disabled")
            args.stt_audio = False
    
    viewer = TrackingViewerJanChat(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_gstreamer=not args.no_gstreamer,
        port=args.port,
        agent=None,  # Will be set by main.py when integrated
        stt_model=stt_model,
        stt_enabled=args.stt_audio,
        stt_card=args.stt_card,
        stt_chunk_duration=args.stt_chunk_duration
    )
    
    viewer.run(duration_seconds=args.duration)

