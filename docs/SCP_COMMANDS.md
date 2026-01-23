# SCP Commands for Orin Nano

## Quick Transfer (Updated Files + Remote Old-Nano Agent)

```bash
# From your development machine
# Replace 'orin-nano' with your Orin's hostname or IP

# Updated agent files (detection history support)
scp agent/simple_agent.py orin-nano:~/Documents/snowcrash/agent/

# Updated viewer (detection history tracking)
scp tools/tracking_viewer_chatgui_V2.py orin-nano:~/Documents/snowcrash/tools/

# New remote old-nano health checker agent
scp remote_old_nano_agent.py orin-nano:~/Documents/snowcrash/

# Documentation (optional but helpful)
scp docs/REMOTE_OLD_NANO_AGENT.md orin-nano:~/Documents/snowcrash/docs/
```

## Complete Transfer (All Recent Updates)

```bash
# Agent files
scp agent/simple_agent.py agent/query_ontology.json orin-nano:~/Documents/snowcrash/agent/

# Viewer files
scp tools/tracking_viewer_chatgui_V2.py tools/color_detection.py orin-nano:~/Documents/snowcrash/tools/

# Main entry point
scp main_V2.py orin-nano:~/Documents/snowcrash/

# Remote old-nano agent
scp remote_old_nano_agent.py orin-nano:~/Documents/snowcrash/

# Documentation
scp docs/REMOTE_OLD_NANO_AGENT.md docs/TRANSFER_TO_ORIN.md orin-nano:~/Documents/snowcrash/docs/
```

## One-Liner Commands

**Minimum (just detection history + remote agent):**
```bash
scp agent/simple_agent.py tools/tracking_viewer_chatgui_V2.py remote_old_nano_agent.py orin-nano:~/Documents/snowcrash/
```

**Complete (all updates):**
```bash
scp agent/simple_agent.py agent/query_ontology.json tools/tracking_viewer_chatgui_V2.py tools/color_detection.py main_V2.py remote_old_nano_agent.py orin-nano:~/Documents/snowcrash/
```

## After Transfer: Verify

```bash
# SSH to Orin Nano
ssh orin-nano

# Check files are there
cd ~/Documents/snowcrash
ls -lh agent/simple_agent.py
ls -lh tools/tracking_viewer_chatgui_V2.py
ls -lh remote_old_nano_agent.py
ls -lh main_V2.py

# Check permissions
chmod +x remote_old_nano_agent.py
chmod +x main_V2.py
```

## What Each File Does

- **`agent/simple_agent.py`**: Updated with "Have any X been in the frame?" query handler
- **`tools/tracking_viewer_chatgui_V2.py`**: Updated with detection history tracking
- **`remote_old_nano_agent.py`**: New LlamaIndex-based agent for querying old-nano service
- **`main_V2.py`**: Main entry point (if updated)
- **`agent/query_ontology.json`**: Updated ontology for movement queries (if updated)

## Running the Remote Old-Nano Agent

After transferring, you can run the remote old-nano agent:

```bash
# On Orin Nano
cd ~/Documents/snowcrash

# Run with Llama-3.2 model (interactive mode)
python3 remote_old_nano_agent.py \
    --model-path models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --old-nano-ip 10.163.1.173 \
    --interactive
```

## Notes

- All paths use `~/Documents/snowcrash/` (not `snowcrash-orin3`)
- The remote old-nano agent requires `llama-index` packages (install separately if needed)
- Detection history tracking starts from when tracking begins (no retroactive data)

