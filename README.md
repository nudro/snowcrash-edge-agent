# Snowcrash - Edge AI Agent for Jetson Devices

Real-time object detection and analysis using SLMs (Small Language Models) on NVIDIA Jetson edge devices.

## 1. Diagram of SnowcrashAgent

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │ Web GUI     │  │ Chat GUI    │  │ Terminal/CLI        │   │
│  │ (Flask)     │  │ (Jan.ai     │  │                      │   │
│  │ Port 8080   │  │  style)     │  │                      │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘   │
└─────────┼─────────────────┼─────────────────────┼──────────────┘
          │                 │                     │
          └─────────────────┼─────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │    SimpleAgent (Agentic)     │
              │  ┌────────────────────────┐  │
              │  │  LLM (SLM)             │  │
              │  │  • Phi-3 Mini (3.8B)   │  │
              │  │  • Llama 3.2 (3B)      │  │
              │  │  • Gemma 2B            │  │
              │  └──────────┬─────────────┘  │
              │             │                │
              │  ┌──────────▼─────────────┐  │
              │  │ Tool Selection Engine  │  │
              │  │ • Keyword Matching     │  │
              │  │ • Hybrid LLM Reasoning │  │
              │  └──────────┬─────────────┘  │
              └─────────────┼────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│  YOLO26-seg     │ │  Parakeet STT    │ │  Tracking Tools │
│  Detection      │ │  (Primary)       │ │  (DeepSORT)     │
│  • Segmentation │ │  WhisperSTT      │ │  • Track IDs    │
│  • Bounding Box │ │  (Backup)        │ │  • Trajectories │
│  • Class IDs    │ └──────────────────┘ │  • Velocity     │
└────────┬────────┘                      └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│          Additional Tools                       │
│  • Color Detection (mask-based)                 │
│  • Distance Estimation                          │
│  • Statistics Aggregation                       │
│  • Geographic Positioning                       │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         Camera Feed (USB/CSI)                   │
│         /dev/video0 or /dev/video1              │
└─────────────────────────────────────────────────┘
```

## 2. Installation and Requirements

### Legacy Nano (Jetson Nano - No Agent)

**Requirements:**
- Ubuntu 18.04
- Python 3.7
- 8 GB unified memory
- ~2 GB SWAP

**Capabilities:**
- SLM only (no agentic AI)
- YOLOv8 object detection
- Direct tool usage (no MCP, no LangChain)

**Installation:**
```bash
# Install dependencies
pip3 install -r requirements.txt

# Download YOLOv8 model (if not included)
# Models are in /models/yolo-v8/ directory

# Run directly
python3 tools/yolo_detection.py
```

**Note:** Legacy Nano does not support:
- ❌ LangChain (not compatible with Python 3.7)
- ❌ MCP server
- ❌ Agentic agent
- ❌ Speech-to-Text

### Orin Nano (Full Agentic Pipeline)

**Requirements:**
- Ubuntu 20.04 or later
- Python 3.10+
- 8 GB unified memory
- CUDA support

**Capabilities:**
- ✅ Full agentic AI pipeline
- ✅ SLM support (Phi-3, Llama 3.2, Gemma 2B)
- ✅ YOLO26-seg (detection + segmentation)
- ✅ Speech-to-Text (optional, NVIDIA Parakeet TDT 0.6B primary, WhisperSTT backup)
- ✅ Web GUI with chat interface
- ✅ Real-time tracking and analysis

**⚠️ Model Access Requirement:**
To download and use Phi-3 Mini, Gemma 2B, and NVIDIA Parakeet models, you need a **Hugging Face account**:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model licenses for each model
3. Set up authentication using `huggingface-cli login` or environment token

**Installation:**
```bash
# Run installation script
./install.sh

# This will:
# 1. Install Python dependencies
# 2. Download YOLO26-seg model
# 3. Set up models directory structure
# 4. Configure system dependencies

# Start the main application with GUI
python3 main.py --model phi-3
```

**Available SLM Models:**
- `--model phi-3` - Phi-3 Mini (3.8B params, recommended - best quality, ~2-4 GB RAM)
- `--model llama` - Llama 3.2 (3B params, very efficient, ~1-2 GB RAM)
- `--model gemma` - Gemma 2B (2B params, lightweight, ~1.5 GB RAM)

**Available GUI Viewers:**
- `--gui-viewer default` - Standard web-based tracking viewer (default)
- `--gui-viewer chatgui` - Chat interface with text input and quick prompts
- `--gui-viewer audiogui` - Audio visualization interface with real-time transcription (requires STT, auto-initializes Parakeet)

See **Section 3. Usage** below for complete usage examples with all GUI viewers and models.

## 3. Usage

### Available Models

**SLM Models (select with `--model`):**
- `--model phi-3` - **Phi-3 Mini** (3.8B params, recommended - best quality, ~2-4 GB RAM)
- `--model llama` - **Llama 3.2** (3B params, very efficient, ~1-2 GB RAM)
- `--model gemma` - **Gemma 2B** (2B params, lightweight, ~1.5 GB RAM)

**GUI Viewers (select with `--gui-viewer`):**
- `--gui-viewer default` - Standard web-based tracking viewer (default)
- `--gui-viewer chatgui` - Chat interface with text input and quick prompts
- `--gui-viewer audiogui` - Audio visualization interface with real-time transcription (requires STT, auto-initializes Parakeet)

### Primary Usage: Chat GUI

The Chat GUI provides a text-based interface with quick-click tactical prompts and natural language queries.

**Basic Usage:**
```bash
# Start with Phi-3 model (recommended)
python3 main.py --model phi-3 --gui-viewer chatgui

# Start with Llama 3.2 model
python3 main.py --model llama --gui-viewer chatgui

# Start with Gemma 2B model
python3 main.py --model gemma --gui-viewer chatgui
```

**With Audio Input (Parakeet STT):**
```bash
# Enable audio input for chat GUI
python3 main.py --model phi-3 --gui-viewer chatgui --gui-stt-audio

# With custom Parakeet model path
python3 main.py --model phi-3 --gui-viewer chatgui --gui-stt-audio --stt-model-path /path/to/parakeet-tdt-0.6b-v2.nemo

# Custom microphone card number
python3 main.py --model phi-3 --gui-viewer chatgui --gui-stt-audio --stt-card 1
```

**Chat GUI Features:**
- **Quick-click tactical prompts**: STATUS CHECK, TARGET COUNT, DISTANCE REPORT, POSITION UPDATE, COLOR INTEL, TRACK IDs, ENVIRONMENT SCAN
- **Natural language queries**: Ask questions about detected objects
- **Real-time responses**: Get answers based on current video feed
- **Text input**: Type questions directly in the chat box
- **Optional audio input**: Enable with `--gui-stt-audio` flag

**Example Queries:**
- "What's the current situation?" → Returns detection statistics
- "How many objects detected?" → Returns count of all objects
- "Distances to all targets" → Returns distance estimates
- "Positions of all objects" → Returns location data
- "What color is the car?" → Color identification using segmentation masks
- "Based on the objects in the video, what kind of environment is this?" → Contextual reasoning

**Access the GUI:**
Once started, the application will print a URL like:
```
[MAIN] Starting Jan.ai chat viewer (experimental)...
[OK] Starting web server on http://<your-ip>:8080
```
Open this URL in your browser to view the chat interface.

### Primary Usage: Audio GUI

The Audio GUI provides an audio-first interface with real-time transcription visualization and automatic speech recognition.

**Basic Usage (STT auto-initialized):**
```bash
# Start Audio GUI with Phi-3 model
python3 main.py --model phi-3 --gui-viewer audiogui

# Start with Llama 3.2 model
python3 main.py --model llama --gui-viewer audiogui

# Start with Gemma 2B model
python3 main.py --model gemma --gui-viewer audiogui
```

**With Custom STT Settings:**
```bash
# Custom Parakeet model path
python3 main.py --model phi-3 --gui-viewer audiogui --stt-model-path /path/to/parakeet-tdt-0.6b-v2.nemo

# Custom microphone card number
python3 main.py --model phi-3 --gui-viewer audiogui --stt-card 1

# Custom audio chunk duration (default: 3.0 seconds)
python3 main.py --model phi-3 --gui-viewer audiogui --stt-chunk-duration 3.0
```

**Audio GUI Features:**
- **Real-time audio visualization**: Microphone icon and animated waveform
- **Live transcription display**: See your words transcribed in real-time
- **Automatic transcription**: Speech is automatically transcribed and sent to the agent
- **Agent responses**: View agent responses in the chat panel
- **Same tracking features**: Video stream, active tracks, quick prompts

**Note:** Audio GUI **requires** Parakeet STT. It will automatically initialize if not already loaded. If STT initialization fails, the application will exit with an error.

### Additional Options

```bash
# Change port (default: 8080)
python3 main.py --model phi-3 --gui-viewer chatgui --gui-port 8081

# Change camera device (default: 0)
python3 main.py --model phi-3 --gui-viewer chatgui --gui-device 1

# Disable GUI (terminal-only mode)
python3 main.py --model phi-3 --no-gui

# Terminal STT mode (terminal-only, no GUI - uses WhisperSTT as backup)
python3 main.py --model phi-3 --no-gui --stt --stt-model-size base

# Note: --stt flag is for terminal mode only. For GUI audio input:
#   - Use --gui-viewer audiogui (auto-initializes Parakeet STT)
#   - Use --gui-viewer chatgui --gui-stt-audio (enables audio input for chat GUI)
```

### Command Line Tools

```bash
# Interactive agent (no GUI)
python scripts/run_agent.py --model phi-3

# Test agent setup
python scripts/test_agent.py

# View webcam feed only
python scripts/view_webcam.py
```

## 4. Network

**Placeholder:** Air-gapped WiFi network configuration with Opal travel router.

*Network setup documentation will be added here for coordinating multiple edge devices in a mesh network.*

## 5. Device Specifications and Memory Estimates

### Legacy Nano (Jetson Nano)

| Spec | Value |
|------|-------|
| **OS** | Ubuntu 18.04 |
| **Memory (VRAM/Unified)** | 8 GB |
| **SWAP** | 1978 MB |
| **GPU** | Maxwell (128 CUDA cores) |
| **Tensor Cores** | No |
| **Architecture** | ARM Cortex-A57 |
| **CUDA Compute** | 5.3 |

**Memory Usage Breakdown:**

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **SLM Model** | ~2-4 GB | One model loaded at a time (Phi-3/Llama/Gemma) |
| **YOLOv8 Model** | ~6 MB | `yolov8n.pt` |
| **OS + System** | ~1-2 GB | Ubuntu 18.04, drivers |
| **Python Runtime** | ~200-500 MB | Python 3.7 interpreter |
| **Tool Overhead** | ~50-100 MB | Detection tools, image processing |
| **Total** | **~3.3-6.6 GB** | Leaves ~1.4-4.7 GB headroom |

**Note:** Models stored on disk don't consume RAM until loaded. Only one SLM loads at a time.

### Orin Nano (Jetson Orin Nano)

| Spec | Value |
|------|-------|
| **OS** | Ubuntu 20.04+ |
| **Memory (VRAM/Unified)** | 8 GB |
| **GPU** | Orin (1024 CUDA cores) |
| **Tensor Cores** | Yes (32) |
| **Architecture** | ARM Cortex-A78AE |
| **CUDA Compute** | 8.7 |

**Memory Usage Breakdown:**

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **SLM Model** | ~2-4 GB | One model loaded at a time (Phi-3/Llama/Gemma) |
| **YOLO26-seg Model** | ~15-20 MB | `yolo26n-seg.pt` (larger than YOLOv8) |
| **Parakeet STT Model** | ~200-300 MB | NVIDIA Parakeet TDT 0.6B (primary, loaded once if enabled) |
| **WhisperSTT Model** | ~150-500 MB | Whisper tiny/base/small (backup, loaded only if Parakeet unavailable) |
| **Agent Framework** | ~100-200 MB | LangChain, dependencies |
| **Web GUI Server** | ~50-100 MB | Flask, video streaming |
| **Tracking System** | ~50-100 MB | DeepSORT, trajectory data |
| **OS + System** | ~1-2 GB | Ubuntu 20.04+, drivers |
| **Python Runtime** | ~200-400 MB | Python 3.10+ interpreter |
| **Tool Overhead** | ~100-200 MB | Detection tools, segmentation masks, image processing |
| **Total (No STT)** | **~2.6-5.0 GB** | Leaves ~3.0-5.4 GB headroom |
| **Total (With Parakeet STT)** | **~2.9-5.3 GB** | Leaves ~2.7-5.1 GB headroom |
| **Total (With WhisperSTT backup)** | **~2.8-5.5 GB** | Leaves ~2.5-5.2 GB headroom |

**Memory Optimization:**
- Only one SLM model loads at a time (selected via `--model`)
- YOLO26-seg loaded once, shared across all detections
- Parakeet STT loaded once if enabled, shared across transcriptions
- WhisperSTT only loads if Parakeet unavailable (backup mode)
- Segmentation masks processed in-place (not stored long-term)

## 6. Agent Design

### Prompt Template: SOF (Special Operations Forces) Operator Style

The agent uses a tactical, concise communication style inspired by Special Operations Forces operators.

**System Prompt Template:**
```
You are a Special Operations Forces (SOF) tactical AI assistant operating in a real-time surveillance environment.

Communication Protocol:
- Respond in brief, direct operational style
- Use tactical terminology (target, status, confirm, negative, intel, situational awareness)
- Keep ALL responses under 45 words - be concise
- No explanations unless mission-critical
- Format: [Status] [Essential info] [Action/Result]
- Use military time references and operational brevity

Respond operationally:
```

**Response Length Enforcement:**
- All responses automatically truncated to 45 words maximum
- Truncation attempts to break at sentence boundaries when possible
- Fallback: Word-based truncation with ellipsis

**Tactical Quick Prompts:**
- `STATUS CHECK` → Detection statistics
- `TARGET COUNT` → Object count
- `DISTANCE REPORT` → Distance estimates
- `POSITION UPDATE` → Location data
- `COLOR INTEL` → Color identification
- `TRACK IDs` → Active track IDs

### Hybrid Tool Selection: Keywords + LLM Reasoning

The agent uses a **hybrid approach** combining keyword matching (fast path) with LLM reasoning (ambiguous cases).

**Flow:**
1. **Check keywords first** (fast path - no LLM call)
2. **If exactly 1 match** → use that tool immediately (fast path)
3. **If multiple matches or ambiguous** → ask LLM to choose the best tool
4. **If no matches but seems tool-related** → ask LLM if a tool is needed
5. **If no matches and not tool-related** → fallback to LLM for general response

**Benefits:**
- **Speed** - Single keyword matches are instant (no LLM call)
- **Accuracy** - LLM resolves ambiguous cases
- **Memory efficiency** - LLM only called when needed
- **Flexibility** - Handles complex queries that keywords miss

**Example Flows:**
- "Detect objects" → Keywords match → Use `yolo_object_detection` (no LLM call)
- "What objects are near the door?" → Ambiguous → LLM chooses tool based on context
- "Is this a kitchen?" → No keywords → Detection runs → LLM reasons with structured data

**Keyword Categories:**
- `STATS_KEYWORDS`: "status check", "statistics", "situational awareness"
- `DISTANCE_KEYWORDS`: "distance", "how far", "range to target"
- `POSITION_KEYWORDS`: "position update", "location report", "coordinates"
- `IMAGE_VIDEO_KEYWORDS`: "target count", "detect", "objects"
- `COLOR_QUERY_KEYWORDS`: "color intel", "color identification"
- `TRACKING_KEYWORDS`: "track", "tracking", "track ids"

### SLM, STT, and YOLO Versions

| Component | Legacy Nano | Orin Nano | Notes |
|-----------|-------------|-----------|-------|
| **SLM Options** | | | |
| • Phi-3 Mini | ✅ | ✅ | 3.8B params, ~2-4 GB RAM, Q4_K_M quantized |
| • Llama 3.2 | ✅ | ✅ | 3B params, ~1-2 GB RAM, Q4_K_M quantized |
| • Gemma 2B | ✅ | ✅ | 2B params, ~1.5 GB RAM, Q4_K_M quantized |
| **YOLO Model** | | | |
| • Model | YOLOv8n | YOLO26n-seg | |
| • File | `yolov8n.pt` | `yolo26n-seg.pt` | |
| • Size | ~6 MB | ~15-20 MB | |
| • Detection | ✅ Bounding boxes | ✅ Bounding boxes | |
| • Segmentation | ❌ No | ✅ Instance masks | |
| • Open-vocabulary | ❌ No | ✅ Yes | |
| **STT (Speech-to-Text)** | | | |
| • Parakeet TDT 0.6B | ❌ No | ✅ Primary | NVIDIA Parakeet (FastConformer-TDT, 600M params) |
| • WhisperSTT | ❌ No | ✅ Backup | tiny/base/small models (fallback if Parakeet unavailable) |
| • Memory (Parakeet) | N/A | ~200-300 MB | Requires at least 2GB RAM, supports up to 24min audio |
| • Memory (Whisper) | N/A | ~150-500 MB | Loaded only as backup |
| • Features | N/A | Timestamps, punctuation, capitalization | Parakeet: word/segment/char timestamps |
| **Agentic AI** | | | |
| • LangChain | ❌ No | ✅ Yes | Python 3.10+ required |
| • Tool Calling | ❌ Direct only | ✅ Hybrid (keywords + LLM) | |
| • MCP Server | ❌ No | ❌ No (not used) | See section 7 |

**Model Storage vs Memory:**
- Models stored on disk: **Do not consume RAM until loaded**
- Only one SLM model loads at a time (selected via `--model` flag)
- YOLO model loaded once, shared across all detections
- Parakeet STT loaded once if enabled, shared across transcriptions
- WhisperSTT only loads as backup if Parakeet unavailable

**⚠️ Hugging Face Account Required:**
The following models require a Hugging Face account to download:
- **Phi-3 Mini**: [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- **Gemma 2B**: [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
- **Parakeet TDT 0.6B V2**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

**Setup Instructions:**
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (requires account)
huggingface-cli login

# Accept model licenses via web interface, then models will download automatically
```

## 7. When and When Not to Use MCP

**Current Status: MCP is NOT used in Snowcrash.**

### Why MCP is Not Used

Snowcrash uses **direct tool invocation** instead of the MCP (Model Context Protocol) server approach:

**Architecture:**
```
main.py → SimpleAgent → LangChain Tools → Direct Tool Classes
```

**Benefits of Direct Tools:**
- ✅ **Lower overhead** - Direct function calls, no serialization/IPC overhead
- ✅ **Lower latency** - No protocol round-trips, faster response times (~2-8ms saved per call)
- ✅ **Lower memory** - Single process, shared memory space
- ✅ **Simpler architecture** - Fewer moving parts, easier debugging
- ✅ **Better for edge devices** - Optimal for Jetson's limited resources (8GB RAM)

**Memory Comparison:**

| Approach | Memory Usage | Overhead |
|----------|--------------|----------|
| **Direct Tools** | ~2.6-5.0 GB | Single process, shared memory |
| **MCP Server** | ~3.2-6.0 GB | Separate process, JSON serialization overhead |
| **Savings** | **~600 MB - 1 GB** | 15-20% reduction |

**When MCP Would Be Used (Not Applicable to Snowcrash):**

MCP server (`mcp_server/server.py`) would be appropriate for:

1. ✅ **Multiple agents** - Multiple agent instances sharing same tools (Snowcrash: single agent per device)
2. ✅ **Distributed tools** - Tools running on different hardware (Snowcrash: all local)
3. ✅ **Network access** - Tools exposed over network for remote clients (Snowcrash: local only)
4. ✅ **Process isolation** - Critical when tool stability is uncertain (Snowcrash: tools are stable)
5. ✅ **Language integration** - Using tools from non-Python codebases (Snowcrash: Python-only)

**Current Implementation:**

- `main.py` - Uses direct tools (current, recommended)
- `mcp_server/server.py` - Available but not used in production

**Conclusion:** For Snowcrash's single-agent, local-execution model on resource-constrained edge devices, direct tool invocation provides optimal performance and memory efficiency. MCP would add unnecessary overhead without providing benefits for this use case.

## Project Structure

```
snowcrash-orin/
├── models/              # SLM models (Phi-3 Mini, Llama 3.2, Gemma 2B)
│   ├── phi3-mini/
│   ├── llama3.2/
│   └── gemma2b/
├── agent/               # Agentic agent implementation
│   ├── simple_agent.py  # Main agentic agent with tool calling
│   ├── langchain_tools.py  # LangChain tool wrappers
│   ├── query_keywords.py   # Keyword definitions for tool selection
│   └── __init__.py
├── tools/               # Detection and analysis tools
│   ├── yolo_detection.py      # YOLO object detection tool
│   ├── tracking_tool.py       # Object tracking (DeepSORT)
│   ├── distance_tool.py       # Distance estimation
│   ├── color_detection.py     # Color identification
│   ├── statistics_tool.py     # Statistics aggregation
│   ├── geographic_tool.py     # Geographic positioning
│   ├── tracking_viewer_chatgui.py  # Web GUI with chat
│   └── tracking_viewer.py     # Basic tracking viewer
├── scripts/             # Setup and utility scripts
│   ├── download_models.py      # Download SLM models
│   ├── measure_vram.py         # Measure VRAM requirements
│   ├── package_for_transfer.py # Package for SCP transfer
│   ├── run_agent.py            # Run agentic agent interactively
│   ├── test_agent.py           # Test agent setup
│   ├── test_webcam.py          # Test camera access
│   └── view_webcam.py          # View webcam feed
├── mcp_server/          # MCP server (available but not used)
│   ├── server.py        # Main MCP server
│   └── __init__.py
├── requirements.txt     # Python dependencies
├── install.sh           # Installation script
├── main.py             # Main entry point
├── README.md           # This file
└── WORKFLOW.md         # Detailed workflow guide
```

## STT (Speech-to-Text) Details

### Model Comparison: Parakeet vs WhisperSTT

| Metric | NVIDIA Parakeet-TDT-0.6B-v2 | faster-whisper base |
|--------|------------------------------|---------------------|
| **Model Size** | 600 Million parameters (~1.2GB VRAM) | 74 Million parameters (~290MB) |
| **English WER** | ~6.05% (State-of-the-art) | ~10% - 14% (Lower precision) |
| **Inference Speed** | Extreme: RTFx up to 3380 on GPU | Moderate: ~16x real-time speed |
| **Processing Style** | Transducer (Fast & Streaming-friendly) | Encoder-Decoder (Segment-based) |
| **Language Support** | English only (v2) | Multilingual (99+ languages) |

**Why Parakeet-TDT-0.6B-v2 Wins for Snowcrash:**

1. **Zero Transcription Lag**: Parakeet's TDT (Token-and-Duration Transducer) architecture skips "blank" frames, allowing it to transcribe an hour of audio in less than a second on modern GPUs. In a Flask app where your GPU is already busy with YOLO, this "burst" processing ensures the chat response isn't delayed.

2. **Higher Accuracy for Agents**: With a WER of ~6.05%, it is significantly more accurate than the "base" Whisper model. This is critical for agentic tool calls, as small errors in transcribed commands (e.g., "track red car" vs. "track read car") can break your backend logic.

3. **Better Noise Handling**: Parakeet maintains lower error rates in noisy environments compared to small Whisper models, which is often the case with audio captured from a live camera feed.

### NVIDIA Parakeet TDT 0.6B V2 (Primary)

**Model Specifications:**
- **Architecture**: FastConformer-TDT (600 million parameters)
- **Memory Requirement**: At least 2GB RAM (supports larger audio with more RAM)
- **Audio Format**: 16kHz mono WAV/FLAC
- **Max Audio Length**: Up to 24 minutes in a single pass
- **Performance**: RTFx of 3380 (very fast inference)
- **Features**:
  - Word-level, segment-level, and character-level timestamps
  - Automatic punctuation and capitalization
  - Robust performance on spoken numbers and song lyrics
  - Noise robustness across various SNR levels

**Usage:**
```python
# Install NeMo toolkit
pip install -U nemo_toolkit["asr"]

# Load model
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# Transcribe
output = asr_model.transcribe(['audio.wav'])
print(output[0].text)

# Transcribe with timestamps
output = asr_model.transcribe(['audio.wav'], timestamps=True)
word_timestamps = output[0].timestamp['word']
segment_timestamps = output[0].timestamp['segment']
```

**Performance Metrics** (from [Hugging Face Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)):
- Average WER: 6.05% across multiple datasets
- LibriSpeech test-clean: 1.69% WER
- LibriSpeech test-other: 3.19% WER
- TEDLIUM-v3: 3.38% WER

**Hardware Compatibility:**
- NVIDIA Ampere, Blackwell, Hopper, Volta architectures
- Optimized for NVIDIA GPU-accelerated systems
- Linux preferred

**Adjustable Parameters:**

The Parakeet STT system supports several configurable parameters for fine-tuning transcription behavior:

| Parameter | Default | Description | Usage |
|-----------|---------|-------------|-------|
| `--stt-card` | `1` | ALSA card number for USB microphone. Use `arecord -l` to find your microphone's card number. | `--stt-card 1` (for USB lavalier mic on card 1) |
| `--stt-chunk-duration` | `3.0` (audiogui) / `5.0` (terminal) | Duration of each audio chunk in seconds. Shorter = faster response but more processing overhead. Longer = less overhead but may feel less responsive. | `--stt-chunk-duration 3.0` |
| `silence_chunks_after_speech` | `2` | **Critical for preventing cut-off**: Number of consecutive silent chunks required AFTER speech ends before finalizing transcription. Higher values = wait longer for you to finish speaking, preventing mid-sentence cut-offs. Lower values = faster response but may cut off sentences. This is configured internally in `listen_continuous()` method. | Internal parameter (see tuning tips below) |
| `max_silence_chunks` | `5` | Maximum consecutive silent chunks before showing status message (while waiting for speech to start). This is configured internally. | Internal parameter |

**Tuning Tips:**

- **Faster response**: Decrease `--stt-chunk-duration` to `2.0-3.0` seconds (may reduce accuracy slightly)
- **Better quality**: Increase `--stt-chunk-duration` to `5.0-7.0` seconds (slower but more accurate)
- **Prevent mid-sentence cut-off**: The `silence_chunks_after_speech=2` parameter ensures the system waits for 2 silent chunks (e.g., 6-10 seconds with 3-5s chunks) after you stop speaking before sending the transcription
- **Different microphone**: Use `arecord -l` to find your microphone's card number, then set `--stt-card` accordingly

**Example Usage:**
```bash
# AudioGUI with custom chunk duration and microphone card
python main.py --model phi-3 --gui-viewer audiogui --stt-card 1 --stt-chunk-duration 3.0

# ChatGUI with STT audio input
python main.py --model phi-3 --gui-viewer chatgui --gui-stt-audio --stt-chunk-duration 5.0
```

**License**: CC-BY-4.0

### WhisperSTT (Backup)

WhisperSTT is used as a fallback option if Parakeet is unavailable or if explicit WhisperSTT usage is requested.

**Model Options:**
- **tiny**: ~150 MB, fastest, lower accuracy
- **base**: ~290 MB, balanced
- **small**: ~500 MB, better accuracy

**Usage**: Automatically falls back to WhisperSTT if Parakeet model download fails or `--stt-backup whisper` flag is used.

## Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [NVIDIA Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [Jetson Hacks](https://www.jetsonhacks.com/)
