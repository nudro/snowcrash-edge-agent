# Distributed Multi-Device Setup Plan

## Overview

This plan outlines how to connect **orin-nano** (primary agent device) and **old-nano** (secondary detection device) via an air-gapped WiFi network (GL-SFT1200-0cc-5G, network name 'opal') to enable distributed object detection with audio feedback.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Air-Gapped Network                      │
│                  GL-SFT1200-0cc-5G ('opal')                 │
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐       │
│  │   orin-nano      │          │   old-nano       │       │
│  │  (Primary Agent) │◄────────►│ (YOLO Service)   │       │
│  │                  │   HTTP   │                  │       │
│  │ • Agent System   │          │ • YOLOv8         │       │
│  │ • Camera Viewer  │          │ • Audio Output   │       │
│  │ • Remote Tool    │          │ • HTTP Server    │       │
│  └──────────────────┘          └──────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components to Build

### 1. YOLO Detection Service on old-nano (`tools/yolo_detection_service.py`)

**Purpose**: Lightweight HTTP server on old-nano that runs YOLOv8 detection and triggers SSH desktop notifications.

**Features**:
- HTTP endpoint to receive detection requests
- Captures frame from camera (/dev/video0 - USB camera)
- Runs YOLOv8 detection
- Checks for "person" class
- Sends SSH notification to orin-nano desktop ("ALERT: PERSON") when person detected
- Returns detection results as JSON

**API Endpoints**:
- `POST /detect` - Run detection with optional SSH notification on person detection
  ```json
  {
    "source": "camera" | "image_path",
    "send_notification": true,
    "confidence_threshold": 0.25
  }
  ```
- `GET /health` - Health check

**SSH Notification**:
- Uses SSH to send desktop notification to orin-nano
- Calls `notify-send` on orin-nano via SSH: "ALERT: PERSON"
- Only sends notification when person detected (avoid spam)
- Requires SSH key-based authentication (set up in Phase 1)

### 2. Remote YOLO Tool on orin-nano (`agent/remote_yolo_tool.py`)

**Purpose**: LangChain tool that allows orin-nano agent to query old-nano's YOLO service.

**Features**:
- HTTP client to communicate with old-nano service
- Auto-discovery of old-nano IP on Opal network (or configurable hostname/IP)
- Retry logic for network reliability
- Integration with existing agent tool system

**Tool Signature**:
```python
@tool
async def remote_yolo_detection(
    device: str = "old-nano",
    trigger_audio: bool = True,
    confidence_threshold: float = 0.25,
    source: str = "camera"
) -> str:
    """
    Query remote device (old-nano) for YOLO object detection.
    If person is detected, old-nano will make a whistle sound.
    
    Args:
        device: Device identifier or IP/hostname (default: "old-nano")
        trigger_audio: If True, play whistle when person detected (default: True)
        confidence_threshold: Detection confidence (0.0-1.0, default: 0.25)
        source: "camera" for webcam or path to image file
    """
```

### 3. Network Configuration (`scripts/configure_opal_network.py`)

**Purpose**: Script to help configure both devices for Opal network connection.

**Features**:
- Network discovery helper (find devices on Opal network)
- IP address configuration validation
- Test connectivity between devices

**Usage**:
```bash
# On orin-nano
python scripts/configure_opal_network.py --find-devices

# On old-nano
python scripts/configure_opal_network.py --server-mode
```

### 4. Desktop Notification Utility (`tools/desktop_notification.py`)

**Purpose**: Send desktop notifications from old-nano to orin-nano via SSH.

**Implementation**:
- Uses SSH to execute `notify-send` command on orin-nano
- Requires SSH key-based authentication (passwordless)
- Works with X11/Wayland desktop environments
- Shows "ALERT: PERSON" notification on orin-nano desktop

**Requirements on orin-nano**:
- Desktop environment (X11 or Wayland)
- `notify-send` installed: `sudo apt-get install libnotify-bin`
- SSH server running: `sudo systemctl start ssh`

## Implementation Steps

### Phase 1: Network Setup (Before Connecting to Opal)

1. **Prepare both devices**:
   ```bash
   # On both orin-nano and old-nano
   # Ensure Python dependencies are installed
   pip3 install requests flask
   
   # For audio on old-nano (if using sounddevice)
   pip3 install sounddevice numpy
   # OR use system tools (aplay, speaker-test)
   sudo apt-get install alsa-utils
   ```

2. **Test local connectivity** (before moving to Opal network):
   - Connect both devices to same WiFi temporarily
   - Note their IP addresses
   - Test ping between devices

### Phase 2: Build old-nano Service

1. **Create YOLO detection service**:
   - File: `tools/yolo_detection_service.py`
   - HTTP server (Flask or FastAPI)
   - YOLOv8 integration
   - Audio output integration

2. **Create audio whistle utility**:
   - File: `tools/audio_whistle.py`
   - Generate/play whistle sound

3. **Create startup script**:
   - File: `scripts/start_yolo_service_old_nano.sh`
   - Start service on old-nano
   - Bind to all interfaces (0.0.0.0) for network access

### Phase 3: Build orin-nano Remote Tool

1. **Create remote tool**:
   - File: `agent/remote_yolo_tool.py`
   - HTTP client
   - Integration with LangChain tool system

2. **Register tool in agent system**:
   - Update `agent/langchain_tools.py` to include remote tool
   - Add to tool registry

3. **Update agent query keywords**:
   - Update `agent/query_keywords.py` if needed for remote queries

### Phase 4: Network Discovery & Configuration

1. **Create network configuration script**:
   - File: `scripts/configure_opal_network.py`
   - Device discovery
   - IP validation

2. **Documentation**:
   - Network setup instructions
   - Troubleshooting guide

### Phase 5: Integration & Testing

1. **Connect to Opal network**:
   - Both devices connect to 'opal' WiFi
   - Note IP addresses assigned

2. **Start services**:
   ```bash
   # On old-nano
   python3 tools/yolo_detection_service.py --host 0.0.0.0 --port 8080
   
   # On orin-nano
   python3 main.py --model phi-3 --gui-viewer chatgui
   ```

3. **Test from orin-nano**:
   - Use agent query: "If you see a person object on old-nano, make a whistle"
   - Verify audio output on old-nano
   - Verify camera stays open on orin-nano

## File Structure

```
snowcrash-orin3/
├── tools/
│   ├── yolo_detection_service.py  # NEW: HTTP service for old-nano
│   └── audio_whistle.py           # NEW: Audio generation/playback
├── agent/
│   ├── remote_yolo_tool.py        # NEW: Remote tool for agent
│   └── langchain_tools.py         # MODIFY: Register remote tool
├── scripts/
│   ├── configure_opal_network.py  # NEW: Network config helper
│   └── start_yolo_service_old_nano.sh  # NEW: Startup script
└── docs/
    └── DISTRIBUTED_SETUP_PLAN.md  # This file
```

## Network Requirements

### Opal Network Configuration

- **SSID**: opal
- **Security**: WPA2/WPA3 (as configured on GL-SFT1200)
- **IP Range**: Typically 192.168.8.x (check router settings)
- **Port**: 8080 for YOLO service (or configurable)

### Firewall Considerations

On old-nano, ensure firewall allows incoming connections:
```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp
# OR disable firewall temporarily for testing
sudo ufw disable  # (not recommended for production)
```

## Audio Requirements on old-nano

### Options for Audio Output

1. **3.5mm Audio Jack** (if available):
   ```bash
   # Test speaker
   speaker-test -t sine -f 800 -l 1
   ```

2. **USB Audio Device** (if available):
   ```bash
   # List audio devices
   aplay -l
   ```

3. **System Beep** (fallback):
   ```bash
   # Enable system beep
   sudo modprobe pcspkr
   # Play beep
   beep -f 800 -l 500
   ```

4. **Programmatic Generation** (recommended):
   - Use `sounddevice` or `pyaudio` + `numpy`
   - No hardware dependencies
   - More control over tone

## Testing Strategy

### 1. Local Testing (Same Network)
- Both devices on same WiFi (not Opal yet)
- Test HTTP communication
- Verify YOLO detection works
- Verify audio output works

### 2. Opal Network Testing
- Connect to Opal network
- Verify device discovery
- Test full workflow:
  - orin-nano agent queries old-nano
  - old-nano detects person
  - old-nano plays whistle
  - orin-nano receives detection results

### 3. Edge Cases
- Network disconnection handling
- Service restart behavior
- Multiple simultaneous requests
- Camera unavailable on old-nano

## Security Considerations

- **Air-gapped network**: Opal network should be isolated (no internet)
- **No authentication**: HTTP service is unauthenticated (acceptable for air-gapped)
- **Network isolation**: Devices only accessible within Opal network

## Future Enhancements

- Service discovery (mDNS/zeroconf) for automatic device finding
- WebSocket for real-time detection streaming
- Multi-device support (query multiple old-nano devices)
- Authentication/encryption for production use
- Health monitoring and auto-restart

## Questions to Resolve

1. **Camera on old-nano**: Does old-nano have a camera? If not, only image file detection possible.
2. **Audio device**: What audio output method is available on old-nano?
3. **Network IP assignment**: Static IPs or DHCP on Opal network?
4. **Service persistence**: Should service auto-start on boot?

## Next Steps

1. Review and approve this plan
2. Confirm camera availability on old-nano
3. Confirm audio output method
4. Begin Phase 1 implementation (network setup)
5. Build components incrementally
6. Test each phase before proceeding

