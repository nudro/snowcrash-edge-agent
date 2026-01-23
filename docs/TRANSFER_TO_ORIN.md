# Transfer Files to Orin Nano

This guide shows what to SCP to the Orin Nano after updating the code.

## Option 1: Quick Transfer (Updated Files Only)

If you just updated code and want to transfer only the changed files:

```bash
# From your development machine
# Replace 'orin-nano' with your Orin's hostname or IP

# Updated agent files
scp agent/simple_agent.py orin-nano:~/Documents/snowcrash/agent/
scp agent/vision_tool.py orin-nano:~/Documents/snowcrash/agent/

# TensorRT conversion script (if not already there)
scp scripts/convert_yolo_to_tensorrt.py orin-nano:~/Documents/snowcrash/scripts/

# Documentation
scp docs/TENSORRT_CONVERSION.md orin-nano:~/Documents/snowcrash/docs/
```

**Note:** The YOLOE model (`yoloe-26n-seg.pt`) should already be on the Orin in `~/Documents/snowcrash-orin3/models/`. If not, transfer it separately:

```bash
# Transfer the model file (11MB)
scp models/yoloe-26n-seg.pt orin-nano:~/Documents/snowcrash/models/
```

## Option 2: Full Package Transfer (Recommended for First-Time Setup)

Use the packaging script to create a complete package (excludes models):

```bash
# On your development machine
cd /home/nudro/Documents/snowcrash-orin3
bash package_for_orin.sh
```

This creates `snowcrash-orin-YYYYMMDD-HHMMSS.tar.gz` in the project root.

Then transfer:

```bash
# Create directory on Orin (if first time)
ssh orin-nano 'mkdir -p ~/Documents/snowcrash'

# Transfer the package
scp snowcrash-orin-*.tar.gz orin-nano:~/Documents/snowcrash/

# On Orin, extract
ssh orin-nano
cd ~/Documents/snowcrash
tar -xzf snowcrash-orin-*.tar.gz

# Install dependencies (if needed)
bash install.sh
```

## What Gets Transferred

### Option 1 (Quick Transfer) Includes:
- ✅ Updated `agent/simple_agent.py` (uses new model paths)
- ✅ Updated `agent/vision_tool.py` (uses new model paths)
- ✅ TensorRT conversion script
- ✅ Documentation

### Option 2 (Full Package) Includes:
- ✅ All Python source code (`agent/`, `tools/`, `scripts/`, `mcp_server/`)
- ✅ Configuration files (`requirements.txt`, `install.sh`)
- ✅ Documentation (`docs/`, `README.md`)
- ✅ Shell scripts
- ❌ Model files (excluded - download separately)
- ❌ Virtual environment (excluded)
- ❌ Python cache (excluded)

## Model File Transfer

The YOLOE model (`yoloe-26n-seg.pt`) is **not included** in the package. Transfer it separately:

```bash
# Transfer model (11MB)
scp models/yoloe-26n-seg.pt orin-nano:~/Documents/snowcrash/models/
```

Or download it directly on the Orin:

```bash
# On Orin Nano
cd ~/Documents/snowcrash/models
# Download from your source (Hugging Face, etc.)
```

## After Transfer: Convert to TensorRT

Once files are transferred and the model is in place, convert to TensorRT on the Orin:

```bash
# On Orin Nano
cd ~/Documents/snowcrash
python3 scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt --precision fp16 --imgsz 1280
```

This creates `models/yoloe-26n-seg.engine` which will be automatically used by the code.

## Verify Transfer

After transferring, verify on the Orin:

```bash
# On Orin Nano
cd ~/Documents/snowcrash

# Check updated files
ls -lh agent/simple_agent.py agent/vision_tool.py

# Check model exists
ls -lh models/yoloe-26n-seg.pt

# Check TensorRT script exists
ls -lh scripts/convert_yolo_to_tensorrt.py
```

## Quick Reference: All Files to Transfer

**Minimum (just updated code):**
```bash
scp agent/simple_agent.py agent/vision_tool.py orin-nano:~/Documents/snowcrash/agent/
scp models/yoloe-26n-seg.pt orin-nano:~/Documents/snowcrash/models/
```

**Complete (full project):**
```bash
bash package_for_orin.sh
scp snowcrash-orin-*.tar.gz orin-nano:~/Documents/snowcrash/
# Then extract on Orin
```

## Troubleshooting

**Permission Denied:**
```bash
# Fix permissions on Orin
ssh orin-nano
chmod +x ~/Documents/snowcrash/scripts/*.sh
```

**Model Not Found:**
- Ensure model is in `~/Documents/snowcrash/models/yoloe-26n-seg.pt`
- Check path matches the code (uses `MODELS_DIR` constant)

**Import Errors:**
- Run `bash install.sh` on Orin to install dependencies
- Ensure virtual environment is activated if using one

