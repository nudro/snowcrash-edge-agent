# Snowcrash Distributed System Demo Guide

This guide walks you through starting the distributed system with old-nano (detection device) and orin-nano (primary agent with GUI).

## Prerequisites

- **old-nano**: Jetson device with USB camera connected at `/dev/video0`
- **orin-nano**: Primary Jetson device running the main agent
- Both devices on the same network (or air-gapped Opal network)
- SSH key-based authentication set up between old-nano and orin-nano
- Files transferred to both devices (see file transfer section below)

## Network Setup

### Current Setup (Local Network)
- Devices connected to local network
- Use `hostname -I` to find IP addresses

### Future Setup (Opal Air-Gapped Network)
- Connect both devices to GL-SFT1200-0cc-5G travel router (network: 'opal')
- Configure static IPs (to be added)
- Ensure devices can ping each other

## Step-by-Step Startup

### Step 1: Start Service on old-nano

**On old-nano (SSH or direct terminal):**

```bash
ssh old-nano
cd ~/Documents/snowcrash
python3 tools/jetson_inference_service.py --device 0 --host 0.0.0.0 --port 9000
```

**Expected output:**
```
[OK] Starting service on 0.0.0.0:9000
     Camera: /dev/video0
     Press Ctrl+C to stop

 * Running on http://10.163.1.173:9000/ (Press CTRL+C to quit)
```

**Keep this terminal open** - the service must stay running.

**Note:** Replace `10.163.1.173` with old-nano's actual IP address.

### Step 2: Verify old-nano Service (Optional)

**From your local machine or orin-nano, test the service:**

```bash
# Test health endpoint
curl http://10.163.1.173:9000/health

# Expected response:
# {"service":"jetson_inference","status":"ok"}

# Test video feed (will show binary data - that's OK)
curl http://10.163.1.173:9000/video_feed --max-time 3
```

**Check old-nano service logs** - you should see:
```
[VIDEO] /video_feed accessed
[VIDEO] Starting stream from /dev/video0
[VIDEO] Trying pipeline 1...
[VIDEO] Pipeline 1 started!
[VIDEO] Streaming...
```

### Step 3: Start GUI on orin-nano

**In a NEW terminal (keep old-nano service running):**

**⚠️ IMPORTANT: You MUST include `--old-nano-host` with the IP address!**

```bash
ssh orin-nano
cd ~/Documents/snowcrash
python3 main.py --model llama --gui-viewer chatgui --old-nano-host 10.163.1.173
```

**Replace `10.163.1.173` with old-nano's actual IP address.**

**If you forget `--old-nano-host`, the GUI will try to connect to hostname "old-nano" which won't work unless you have it configured in `/etc/hosts`. Always use the IP address!**

**Expected output:**
```
[MAIN] Starting Snowcrash...
[ContainerManager] Starting containers...
...
[GUI] Starting web interface on http://10.163.1.174:8083/
```

### Step 4: Open GUI in Browser

The GUI should automatically open in your browser. If not, navigate to:
```
http://10.163.1.174:8083/
```

**Replace `10.163.1.174` with orin-nano's actual IP address.**

### Step 5: Verify GUI Features

You should see:
- ✅ Two video feeds side-by-side:
  - **Orin-Nano Video**: Local camera feed from orin-nano
  - **Old-Nano Video**: Remote camera feed from old-nano
- ✅ **"PERSON WATCH"** button (to start continuous person detection)
- ✅ **"STOP WATCH"** button (to stop detection)
- ✅ **MEM** memory bar (RAM usage)
- ✅ Chat interface on the right

## Using Person Detection

### Start Person Watch

1. Click **"PERSON WATCH"** button in the GUI
2. old-nano will start running `detectnet.py` continuously
3. When a person is detected:
   - ✅ Red notification banner appears at top of GUI: "🚨 ALERT: PERSON DETECTED ON OLD-NANO! 🚨"
   - ✅ Alert file written to `~/Desktop/person_alerts.txt` on orin-nano
   - ✅ Detection automatically stops

### Stop Person Watch

- Click **"STOP WATCH"** button to manually stop detection

## Troubleshooting

### old-nano Service Won't Start

**Check camera:**
```bash
ls -l /dev/video0
# Should show: crw-rw---- 1 root video 81, 0 ...
```

**Check if camera is in use:**
```bash
lsof /dev/video0
# Should show nothing (or list what's using it)
```

**Check dependencies:**
```bash
python3 -c "import flask; print('Flask OK')"
# Install if missing: pip3 install flask
```

**Test detectnet.py directly:**
```bash
cd ~/jetson-inference/build/aarch64/bin
./detectnet.py /dev/video0
# Should show detection window - press Ctrl+C to stop
```

### GUI Can't Connect to old-nano

**⚠️ Most Common Issue: Missing `--old-nano-host` argument**

If you see: `Failed to connect to old-nano at old-nano:9000`

**Solution:** You forgot to include `--old-nano-host` when starting main.py!

**Fix:**
1. Stop main.py (Ctrl+C)
2. Restart with the IP address:
   ```bash
   python3 main.py --model llama --gui-viewer chatgui --old-nano-host 10.163.1.173
   ```
3. Replace `10.163.1.173` with old-nano's actual IP (get it with `hostname -I` on old-nano)

**Check network connectivity:**
```bash
# From orin-nano, ping old-nano
ping 10.163.1.173

# Test service directly
curl http://10.163.1.173:9000/health
# Should return: {"service":"jetson_inference","status":"ok"}
```

**Check firewall:**
```bash
# On old-nano
sudo ufw status
# If enabled, allow port 9000:
sudo ufw allow 9000
```

**Verify old-nano service is running:**
```bash
# On old-nano
ps aux | grep jetson_inference_service
# Should show the Python process
```

**Verify you're using the IP address, not hostname:**
- ✅ Correct: `--old-nano-host 10.163.1.173`
- ❌ Wrong: `--old-nano-host old-nano` (unless you have it in `/etc/hosts`)

### Video Feed Not Showing

**Check old-nano service logs** for `[VIDEO]` messages:
- Should see: `[VIDEO] Pipeline 1 started!`
- Should see: `[VIDEO] Streaming...`

**Check browser console (F12)** for errors:
- Look for connection errors to `http://10.163.1.173:9000/video_feed`

**Test video feed directly:**
```bash
curl http://10.163.1.173:9000/video_feed --max-time 3
# Should output binary MJPEG data
```

### Desktop Notification Not Working

**Check SSH connection from old-nano to orin-nano:**
```bash
# On old-nano
ssh orin-nano "echo 'SSH connection works'"
# Should work without password (SSH keys set up)
```

**Test notification manually:**
```bash
# On old-nano
python3 tools/desktop_notification.py --host orin-nano --title "TEST" --message "Test alert"
# Check orin-nano: cat ~/Desktop/person_alerts.txt
```

**Check alert file permissions:**
```bash
# On orin-nano
ls -l ~/Desktop/person_alerts.txt
# File should be created when alert is sent
```

### CUDA Out of Memory Error (GPU Memory Exhausted)

**If you see:**
```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
NvMapMemAllocInternalTagged: error 12
```

**This means:** GPU memory is exhausted. The YOLO segmentation model (yolo26n-seg.pt) requires significant GPU memory, especially with large image sizes.

**Solutions:**

**Option 1: Reduce image size (recommended)**
- The error shows `imgsz=1280` - this is very large
- Edit `tools/tracking_viewer_chatgui.py` and reduce `imgsz` parameter
- Try `imgsz=640` or `imgsz=480` instead of `1280`

**Option 2: Use detection model instead of segmentation**
- Segmentation models use more memory than detection models
- Switch from `yolo26n-seg.pt` to `yolov8n.pt` (detection only, no segmentation)
- Update model path in `main.py` or command line

**Option 3: Force CPU fallback**
- The code should automatically fallback to CPU on OOM
- Check if CPU fallback is working in the logs
- If not, you may need to explicitly set device to CPU

**Option 4: Free GPU memory**
- Stop other GPU processes (llama-server, etc.)
- Restart the service to clear GPU cache
- Check GPU memory: `nvidia-smi`

**Option 5: Reduce batch size or frame processing rate**
- Process fewer frames per second
- Skip frames if needed

**Quick fix - reduce image size:**
```bash
# Find where imgsz=1280 is set in tracking_viewer_chatgui.py
grep -n "imgsz=1280" tools/tracking_viewer_chatgui.py
# Change to imgsz=640 or imgsz=480
```

## File Transfer Commands

### Transfer Files to old-nano

```bash
# From your local machine
scp tools/jetson_inference_service.py tools/desktop_notification.py old-nano:~/Documents/snowcrash/tools/
```

### Transfer Files to orin-nano

```bash
# From your local machine
scp tools/tracking_viewer_chatgui.py orin-nano:~/Documents/snowcrash/tools/
scp main.py orin-nano:~/Documents/snowcrash/
```

## Service Architecture

```
┌─────────────────┐         HTTP (port 9000)        ┌─────────────────┐
│   old-nano      │◄─────────────────────────────────┤   orin-nano     │
│                 │                                  │                 │
│ - Camera        │                                  │ - Main Agent    │
│ - detectnet.py  │                                  │ - GUI (port 8083)│
│ - Flask Service │                                  │ - YOLO Tracking │
│   (port 9000)   │                                  │ - LLM Agent     │
│                 │                                  │                 │
│ Features:       │                                  │ Features:       │
│ - Video stream  │                                  │ - Dual video    │
│ - Person watch  │                                  │ - Person alerts │
│ - Alerts via    │                                  │ - Chat interface│
│   SSH           │                                  │                 │
└─────────────────┘                                  └─────────────────┘
```

## Future: Opal Network Setup

**To be added:**
- Static IP configuration for both devices on Opal network
- Network discovery/configuration scripts
- VPN setup instructions
- Air-gapped network testing procedures

## Quick Reference

### Find IP Addresses

```bash
# On old-nano
hostname -I
# Example output: 10.163.1.173

# On orin-nano
hostname -I
# Example output: 10.163.1.174
```

### Service URLs

- **old-nano service**: `http://10.163.1.173:9000/`
  - Health: `http://10.163.1.173:9000/health`
  - Video: `http://10.163.1.173:9000/video_feed`
  - Watch status: `http://10.163.1.173:9000/watch_status`

- **orin-nano GUI**: `http://10.163.1.174:8083/`

### Stop Services

**Stop old-nano service:**
- Press `Ctrl+C` in the old-nano service terminal

**Stop orin-nano GUI:**
- Press `Ctrl+C` in the orin-nano terminal running `main.py`

## Notes

- **⚠️ CRITICAL: Always include `--old-nano-host <IP>` when starting main.py!**
  - Use the IP address (e.g., `10.163.1.173`), not the hostname
  - Get old-nano's IP with: `hostname -I` on old-nano
- Keep old-nano service running while using the GUI
- The GUI will automatically reconnect to old-nano video feed
- Person detection runs continuously until person found or manually stopped
- Alerts are written to `~/Desktop/person_alerts.txt` on orin-nano
- Red notification banner appears in GUI when person detected
- If GUI shows "Failed to connect to old-nano at old-nano:9000", you forgot the `--old-nano-host` argument

