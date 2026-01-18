# Snowcrash - SLM Deployment on Jetson Devices

This repository contains information and recommendations for deploying Small Language Models (SLMs) on NVIDIA Jetson devices.

## Device Specifications

| Spec | Old Nano (Jetson Nano) | Orin Nano |
|------|------------------------|-----------|
| **OS** | Ubuntu 18.04 | - |
| **Memory (VRAM/Unified)** | 8 GB | - |
| **SWAP** | 1978 MB | - |
| **GPU** | Maxwell (128 CUDA cores) | Orin (1024 CUDA cores) |
| **Tensor Cores** | No | Yes (32) |
| **Architecture** | ARM Cortex-A57 | ARM Cortex-A78AE |
| **CUDA Compute** | 5.3 | 8.7 |

## SLM Recommendations for Old Nano

Given the constraints of the old Jetson Nano (Ubuntu 18.04, 8 GB unified memory, ~2 GB swap), here are the recommended Small Language Models:

### Top Recommendations

#### 1. **Phi-3 Mini (3.8B)** - ⭐ **Recommended**
- **Quantized Size**: ~2-4 GB
- **Memory Usage**: ~4-6 GB
- **Why**: Best quality-to-size ratio, performs well on resource-constrained devices
- **Format**: GGUF Q4_K_M or Q5_K_M
- **Framework**: llama.cpp
- **Best For**: General-purpose tasks, chat, code completion

#### 2. **Llama 3.2 (1B/3B)** - ⭐ **Very Efficient**
- **Quantized Size**: ~1-2 GB
- **Memory Usage**: ~2-4 GB  
- **Why**: Extremely lightweight, very fast inference
- **Format**: GGUF Q4_K_M or INT4
- **Framework**: llama.cpp or MLC
- **Best For**: Fast responses, simple tasks, minimal memory footprint

#### 3. **Mistral 7B (Quantized)** - **More Capable**
- **Quantized Size**: ~4-5 GB (Q4_K_M)
- **Memory Usage**: ~6-7 GB
- **Why**: Better quality but pushing memory limits
- **Format**: GGUF Q4_K_M (avoid Q8)
- **Framework**: llama.cpp
- **Best For**: Higher quality responses when memory allows

#### 4. **Gemma Models**
- **Gemma 2B**: ~1.5 GB quantized, very fast
- **Gemma 7B**: ~4-5 GB quantized, better quality
- **Framework**: llama.cpp

### Models to Avoid
- Models >7B unquantized - too large for 8GB
- FP16/BF16 full precision versions - too memory-heavy
- Vision-language models - require much more memory
- Models without quantization support

## Design Recommendations

### Memory Optimization

1. **Quantization**: Always use quantized models (Q4_K_M or Q5_K_M minimum)
   - Reduces memory footprint by 50-75%
   - Minimal quality loss

2. **Context Window**: Limit to 512-2048 tokens
   - Lower = less KV cache memory usage
   - Reduce if running into memory issues

3. **Batch Size**: Keep at 1
   - Prevents memory spikes during inference

4. **GPU Offloading**: Enable when possible
   - Offload layers to CPU if GPU memory is full
   - Helps avoid OOM errors

### Framework Setup

#### Using llama.cpp (Recommended)

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j4

# Run with GPU acceleration
./llama-cli -m model.gguf --gpu-layers 35 -c 2048
```

#### Key llama.cpp Parameters for Jetson Nano

- `--gpu-layers N`: Number of layers to offload to GPU (experiment with 25-35)
- `-c N`: Context size (start with 512-1024, increase if memory allows)
- `--batch-size 1`: Keep at 1
- `-ngl N`: Same as --gpu-layers

### Monitoring Resources

Use `tegrastats` to monitor GPU, CPU, and memory usage:
```bash
tegrastats
```

Or install `jtop` for a more detailed interface:
```bash
sudo pip3 install -U jetson-stats
sudo reboot
jtop
```

### Performance Tips

1. **SWAP Usage**: While swap helps avoid OOM, it's slow (~1978 MB on old Nano)
   - Prefer to stay within unified memory limits
   - Use swap as a safety net only

2. **Temperature Management**: Monitor GPU/CPU temps during inference
   - Jetson Nano can thermal throttle
   - Consider active cooling if running sustained loads

3. **Power Mode**: Set to maximum performance mode
   ```bash
   sudo nvpmodel -m 0  # Maximum performance
   sudo jetson_clocks  # Set clocks to max
   ```

### SLM Model Loading

**Key Principle**: Models stored on disk **do not automatically load into RAM/VRAM**. Only one model loads at a time when explicitly instantiated.

#### Disk Storage vs RAM/VRAM Usage

| Component | Disk Usage | RAM/VRAM Usage |
|-----------|------------|----------------|
| **Phi-3 Mini** (on disk) | ~2-4 GB | 0 GB (until loaded) |
| **Llama 3.2** (on disk) | ~1-2 GB | 0 GB (until loaded) |
| **Gemma 2B** (on disk) | ~1.5 GB | 0 GB (until loaded) |
| **Total on disk** | **~4.5-7.5 GB** | - |
| **Loaded model** (one at a time) | - | **~2-6 GB** |

#### Memory Footprint Analysis

For Old Nano (8 GB RAM/VRAM + ~2 GB SWAP = ~10 GB total):

- **Disk storage**: All 3 models can be shipped (~4.5-7.5 GB on disk)
- **RAM/VRAM**: Only 1 model loads at a time (~2-6 GB, depending on model)
- **Remaining**: ~4-8 GB available for YOLO, tools, OS, and other components

**Conclusion**: You can safely ship all 3 models in the package. They remain on disk until explicitly loaded, and only one loads into RAM at a time.

#### How Model Loading Works

Models are loaded **lazily** (on-demand) when you create a `SimpleAgent` instance:

```python
# Ship all 3 models on disk:
models/
  ├── phi3-mini/Phi-3-mini-4k-instruct-q4_K_M.gguf  (~3 GB on disk)
  ├── llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf    (~2 GB on disk)
  └── gemma2b/gemma-3n-E2B-it-Q4_K_M.gguf           (~1.5 GB on disk)

# At runtime, only ONE loads into RAM:
agent = SimpleAgent(model_type="phi-3")   # Loads Phi-3 (~4-6 GB RAM)
# OR
agent = SimpleAgent(model_type="llama")   # Loads Llama (~2-4 GB RAM)
# OR
agent = SimpleAgent(model_type="gemma")   # Loads Gemma (~1.5-3 GB RAM)
```

#### Switching Between Models

To switch models, create a new agent instance. The old model is unloaded when the previous agent instance is garbage collected:

```python
# Load Phi-3
agent1 = SimpleAgent(model_type="phi-3")  # Loads Phi-3 into RAM

# Switch to Llama (old model unloads when agent1 is garbage collected)
agent2 = SimpleAgent(model_type="llama")  # Loads Llama into RAM (replaces Phi-3)
```

#### Recommendations

✅ **Ship all 3 models** in the package for maximum flexibility:
- Users can choose the model at runtime
- No memory waste (only selected model loads)
- Disk space is less constrained than RAM/VRAM
- Easy switching between models

❌ **Don't load multiple models simultaneously** - each model consumes 2-6 GB RAM/VRAM. Loading multiple models would exceed the 10 GB limit.

## Project Structure

```
snowcrash/
├── models/              # SLM models (Phi-3 Mini, Llama 3.2, Gemma 2B)
│   ├── phi3-mini/
│   ├── llama3.2/
│   └── gemma2b/
├── agent/               # Agentic agent implementation
│   ├── simple_agent.py  # Main agentic agent with tool calling
│   ├── langchain_tools.py  # LangChain tool wrappers
│   └── __init__.py
├── mcp_server/          # MCP server for agentic SLM
│   ├── server.py        # Main MCP server
│   └── __init__.py
├── tools/               # MCP tools
│   ├── yolo_detection.py  # YOLO object detection tool
│   └── __init__.py
├── scripts/             # Setup and utility scripts
│   ├── download_models.py      # Download SLM models
│   ├── measure_vram.py         # Measure VRAM requirements
│   ├── package_for_transfer.py # Package for SCP transfer
│   ├── run_agent.py            # Run agentic agent interactively
│   ├── test_agent.py           # Test agent setup
│   └── setup.sh                # Initial setup script
├── package/             # Generated packages for transfer
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── WORKFLOW.md         # Detailed workflow guide
```

## Quick Start

### On DGX Spark

1. **Setup environment:**
   ```bash
   ./scripts/setup.sh
   ```

2. **Download models:**
   ```bash
   python scripts/download_models.py
   ```

3. **Measure VRAM:**
   ```bash
   python scripts/measure_vram.py
   ```

4. **Package for transfer:**
   ```bash
   python scripts/package_for_transfer.py
   ```

5. **Transfer to Old Nano:**
   ```bash
   scp package/snowcrash-*.tar.gz old-nano:~/snowcrash/
   ```

### On Old Nano

1. **Extract and setup:**
   ```bash
   cd ~/snowcrash
   tar -xzf snowcrash-*.tar.gz
   pip3 install -r requirements.txt
   ```

2. **Start MCP server:**
   ```bash
   python3 mcp_server/server.py
   ```

For detailed workflow, see [WORKFLOW.md](WORKFLOW.md).

## Getting Started (Detailed)

1. **Choose a model** based on your use case and memory constraints
2. **Download quantized GGUF** from HuggingFace or similar
3. **Build llama.cpp** with GPU support
4. **Test with small context** first, then increase if stable
5. **Monitor with tegrastats** during inference

## MCP Server

### What is MCP?

**Model Context Protocol (MCP)** is an open standard that enables AI agents to discover and use tools, resources, and data sources in a structured way. In Snowcrash, the MCP server exposes tools (like YOLO object detection) that can be used by agentic LLMs.

### MCP Server Architecture

The MCP server in Snowcrash:
- **Exposes tools** via MCP protocol (stdio transport)
- **Provides YOLO object detection** as a callable tool
- **Uses JSON-RPC** for communication
- **Can be extended** with additional tools

### Available Tools

#### YOLO Object Detection (`yolo_object_detection`)

Detects objects in images using YOLOv8.

**Parameters:**
- `image_path` (required): Path to image file
- `confidence_threshold` (optional, default 0.25): Confidence threshold (0.0-1.0)

**Returns:**
- List of detected objects with class names, confidence scores, and bounding boxes

#### Statistics Aggregation (`get_detection_statistics`)

Tracks detection statistics over time (counts, average confidence, most common objects).

**Example MCP Call:**
```json
{
  "tool": "get_detection_statistics",
  "arguments": {}
}
```

**Example Response:**
```json
{
  "total_detections": 142,
  "detection_count_by_class": {
    "person": 89,
    "car": 32,
    "chair": 15,
    "laptop": 6
  },
  "average_confidence_by_class": {
    "person": 0.87,
    "car": 0.82,
    "chair": 0.75,
    "laptop": 0.91
  },
  "most_common_object": "person",
  "detection_timestamp": "2024-01-15T14:30:25.123"
}
```

#### Distance Estimation (`estimate_object_distances`)

Estimates distance from camera to detected objects using bounding box height and known object sizes.

**Example MCP Call:**
```json
{
  "tool": "estimate_object_distances",
  "arguments": {
    "image_path": "webcam",
    "camera_device": 0
  }
}
```

**Example Response:**
```json
{
  "detections_with_distances": [
    {
      "class": "person",
      "bounding_box": [320, 180, 640, 480],
      "distance_meters": 3.2,
      "distance_feet": 10.5,
      "confidence": 0.89,
      "estimated_height_meters": 1.7
    },
    {
      "class": "car",
      "bounding_box": [100, 200, 300, 350],
      "distance_meters": 8.5,
      "distance_feet": 27.9,
      "confidence": 0.84,
      "estimated_height_meters": 1.5
    }
  ],
  "nearest_object": {
    "class": "person",
    "distance_meters": 3.2,
    "bounding_box": [320, 180, 640, 480]
  },
  "source": "webcam (device 0)",
  "timestamp": "2024-01-15T14:30:25.123"
}
```

#### Object Tracking (`track_objects`)

Tracks objects across frames with persistent IDs using DeepSORT algorithm.

**Example MCP Call:**
```json
{
  "tool": "track_objects",
  "arguments": {
    "image_path": "webcam",
    "camera_device": 0,
    "track_history_frames": 30
  }
}
```

**Example Response:**
```json
{
  "active_tracks": [
    {
      "track_id": 42,
      "class": "person",
      "bounding_box": [320, 180, 640, 480],
      "confidence": 0.89,
      "age_frames": 45,
      "position": {"x": 480, "y": 330},
      "velocity": {"vx": 2.3, "vy": -1.1},
      "trajectory_length": 45,
      "first_seen": "2024-01-15T14:29:40.500",
      "last_updated": "2024-01-15T14:30:25.123"
    },
    {
      "track_id": 17,
      "class": "car",
      "bounding_box": [100, 200, 300, 350],
      "confidence": 0.84,
      "age_frames": 23,
      "position": {"x": 200, "y": 275},
      "velocity": {"vx": -0.5, "vy": 0.2},
      "trajectory_length": 23,
      "first_seen": "2024-01-15T14:30:02.100",
      "last_updated": "2024-01-15T14:30:25.123"
    }
  ],
  "total_active_tracks": 2,
  "lost_tracks": 1,
  "frame_number": 533,
  "timestamp": "2024-01-15T14:30:25.123"
}
```

### Example Agent Prompts

After implementing these tools, you can use prompts like:

- **"Track statistics on detected objects"** → Uses `get_detection_statistics`
- **"How far away is the nearest person?"** → Uses `estimate_object_distances`
- **"Track objects in the video stream"** → Uses `track_objects`
- **"What's the most common object and how far is the nearest one?"** → Chain multiple tools

### Running the MCP Server

**Start the MCP server:**
```bash
python3 mcp_server/server.py
```

The server runs in stdio mode and waits for MCP protocol messages. It can be used with:
- MCP-compatible clients (e.g., Claude Desktop)
- LangChain agents
- Custom MCP clients

**Using with MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector python3 mcp_server/server.py
```

This opens a web UI to test tools interactively.

## Agentic Agent

### What is the Agent?

The **Snowcrash Agent** is an agentic LLM that can:
- **Interpret natural language prompts**
- **Decide when to use tools** (e.g., YOLO object detection)
- **Autonomously execute tools** based on user requests
- **Generate natural language responses** using tool results

This makes it "agentic" - it doesn't just respond to prompts, it actively uses tools to accomplish tasks.

### How the Agent Works

1. **User provides a prompt** (e.g., "Detect objects in image.jpg")
2. **Agent analyzes the prompt** and detects tool-related keywords
3. **Agent decides to use a tool** (e.g., `yolo_object_detection`)
4. **Agent executes the tool** with appropriate parameters
5. **Agent uses LLM** to generate a natural language response from tool results

### Model Selection

The agent supports three SLM models that can be selected:

- **Phi-3 Mini** (`phi-3`) - Recommended for best quality-to-size ratio
- **Llama 3.2** (`llama`) - Very efficient, fast inference
- **Gemma 2B** (`gemma`) - Lightweight, good for constrained environments

### Using the Agent

#### Command Line Usage

**Run interactively with model selection:**
```bash
# Use Phi-3 Mini
python scripts/run_agent.py --model phi-3

# Use Llama 3.2
python scripts/run_agent.py --model llama

# Use Gemma 2B
python scripts/run_agent.py --model gemma

# Use specific model path
python scripts/run_agent.py --model-path /path/to/model.gguf

# Adjust temperature
python scripts/run_agent.py --model phi-3 --temperature 0.5
```

**Example interaction:**
```bash
$ python scripts/run_agent.py --model phi-3
Snowcrash Agentic Agent
==================================================

✓ Using model: phi-3
✓ Loaded phi-3 model: models/phi3-mini/Phi-3-mini-4k-instruct-q4_K_M.gguf

Agent ready! Enter prompts (or 'quit' to exit):

You: Detect objects in test_image.jpg

🤖 Agent thinking...

🤖 Agent decided to use tool: yolo_object_detection

Agent: I found 3 objects in the image: a person (confidence: 95%), a car (confidence: 87%), and a dog (confidence: 72%).
```

#### Programmatic Usage

```python
from agent.simple_agent import SimpleAgent

# Create agent with Phi-3 Mini
agent = SimpleAgent(model_type="phi-3", temperature=0.7)

# Run a prompt
response = agent.run_sync("Detect objects in image.jpg")
print(response)

# Or use specific model path
agent = SimpleAgent(model_path="/path/to/model.gguf")

# Or use model constants
from agent.simple_agent import SimpleAgent
agent = SimpleAgent(model_type=SimpleAgent.MODEL_PHI3)
agent = SimpleAgent(model_type=SimpleAgent.MODEL_LLAMA)
agent = SimpleAgent(model_type=SimpleAgent.MODEL_GEMMA)
```

### Agent Features

- **Automatic tool selection** - Detects when to use tools from prompts
- **Model flexibility** - Easy switching between phi-3, llama, and gemma
- **Temperature control** - Adjustable creativity/randomness
- **Verbose mode** - See agent reasoning and tool decisions
- **Error handling** - Graceful failures with helpful messages

### Tool Selection: Keywords vs LLM-Based

The agent uses **keyword-based tool selection** rather than LLM-based function calling. This design decision has both practical and technical reasons:

**Why Keywords?**

1. **Deterministic behavior** - Same prompt always selects the same tool (no randomness)
2. **Faster inference** - No LLM call needed just to select a tool
3. **Lower latency** - Avoids LLM processing overhead for simple tool selection
4. **Reliability** - Small Language Models (SLMs) can be unreliable at function calling

**Could SLMs Do Function Calling?**

Yes! Many SLMs support function calling when properly configured:
- Phi-3 supports tool use/function calling
- Llama 3.2 supports function calling  
- However, they're less reliable than larger models like Claude or GPT-4

**Would Claude 3.7 Need Keywords?**

No. Claude 3.7 and other large models can:
- Understand context naturally ("track" vs "detect")
- Select tools autonomously via function calling
- Handle complex tool selection scenarios
- We'd just pass tool schemas and let Claude decide

**Architecture Trade-offs**

| Approach | Pros | Cons |
|----------|------|------|
| **LLM-based** | Understands context, flexible, works with large models | Unreliable with SLMs, slower, uses tokens |
| **Keyword-based** | Fast, deterministic, reliable with SLMs | Less flexible, can mis-match similar phrases |
| **Hybrid** | Best of both worlds | More complex code |

For production use with SLMs, keyword matching provides a pragmatic balance of reliability and speed. For testing or when using larger models, LLM-based tool selection would be more flexible.

### How MCP and Agent Work Together

```
┌─────────────┐
│   User      │
│   Prompt    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Agentic Agent  │  ← Interprets prompt, decides to use tool
│  (LangChain)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  LangChain      │  ← Wraps MCP tools as LangChain tools
│  Tool Wrappers  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  MCP Tools      │  ← Direct tool execution (YOLO, etc.)
│  (yolo_detection)│
└─────────────────┘
```

**Alternative Flow (Direct MCP):**
```
┌─────────────┐
│  MCP Client │  ← Connects via MCP protocol
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  MCP Server     │  ← Exposes tools via stdio/HTTP
│  (server.py)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  MCP Tools      │  ← Tool execution
│  (yolo_detection)│
└─────────────────┘
```

The agent uses MCP tools directly (via LangChain wrappers) for better integration and performance when running locally. The MCP server can also be used standalone with external MCP clients.

## Engineering Choices

### Direct Tools vs MCP Server

Snowcrash uses **direct tool invocation** (not MCP server protocol) when running locally via `main.py`. This is an intentional engineering decision based on deployment constraints.

#### Current Approach: Direct Tool Invocation

**Architecture:**
```
main.py → SimpleAgent → LangChain Tools → Direct Tool Classes
```

**Benefits:**
- ✅ **Lower overhead** - Direct function calls, no serialization/IPC overhead
- ✅ **Lower latency** - No protocol round-trips, faster response times
- ✅ **Lower memory** - Single process, shared memory space
- ✅ **Simpler architecture** - Fewer moving parts, easier debugging
- ✅ **Better for edge devices** - Optimal for Jetson Nano's limited resources (8GB RAM)

**Drawbacks:**
- ❌ Single agent per process (not an issue for single-device deployment)
- ❌ No process isolation (tool crashes can affect agent)
- ❌ Harder to share tools across multiple clients (not needed for current use case)

#### Alternative: MCP Server Protocol

**Architecture:**
```
main.py → SimpleAgent → MCP Client → MCP Server (separate process) → Tool Classes
```

**Benefits:**
- ✅ **Multiple clients** - Multiple agents/apps can share same tool server
- ✅ **Process isolation** - Tool failures don't crash agent process
- ✅ **Language agnostic** - Tools usable from any language via MCP protocol
- ✅ **Scalability** - Can run MCP server on different machine/GPU
- ✅ **Standard protocol** - Follows MCP spec for interoperability
- ✅ **Remote access** - Tools can run on different machine/network

**Drawbacks:**
- ❌ **Higher overhead** - JSON serialization, stdio IPC, process communication
- ❌ **Higher latency** - Protocol round-trips add significant delay
- ❌ **More complex** - Separate process management, more failure modes
- ❌ **Higher memory** - Separate process uses additional memory

#### Why Direct Tools for Snowcrash?

The **direct tool approach** is optimal for Snowcrash's deployment model:

1. **Single agent per device** - Each Jetson Nano runs one agent instance
2. **Resource constraints** - Edge devices have limited RAM (8GB unified memory)
3. **Local execution** - Everything runs on same device, no network needed
4. **Performance critical** - Lower latency important for real-time object detection
5. **Simplicity** - Easier to maintain and debug on edge devices

#### When to Use MCP Server?

The MCP server (`mcp_server/server.py`) is available as a **separate entry point** for these scenarios:

1. **Multiple agents** - Multiple agent instances sharing same tools
2. **Distributed tools** - Tools running on different hardware (e.g., DGX Spark hosting tools for Nano clients)
3. **Network access** - Tools exposed over network for remote clients
4. **Process isolation** - Critical when tool stability is uncertain
5. **Language integration** - Using tools from non-Python codebases

#### Snowcrash Hybrid Architecture

Snowcrash uses a **single agent with multiple tools** approach, rather than multiple separate agents:

```
SimpleAgent (single agent)
  ├── LLM (selectable: phi-3/llama/gemma)
  ├── YOLO Tools (direct - low latency needed)
  └── WhisperSTT Tools (direct - low latency needed)
```

**Why This Architecture?**

1. **Memory Efficiency** - Single LLM model (~2-4 GB) shared across all tools, instead of duplicating LLMs per agent
2. **Low Latency** - Direct tool invocation (no MCP protocol overhead of ~2-8ms per call)
3. **Resource Constraints** - Jetson Nano has 8GB total memory; can't afford model duplication
4. **Real-time Performance** - Video detection and audio transcription need low-latency direct calls
5. **Simplicity** - Single process, easier to debug and maintain on edge devices

**Memory Overhead:**

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **LLM Model** | ~2-4 GB | Single model (phi-3/llama/gemma), selectable |
| **YOLO Model** | ~6 MB | `yolov8n.pt` - loaded once, shared |
| **WhisperSTT Model** | ~150-500 MB | Whisper model (tiny/base/small), loaded once |
| **Tool Overhead** | ~1-2 MB | Python objects, data structures |
| **Agent Framework** | ~50-100 MB | LangChain, dependencies |
| **Total** | **~2.2-4.6 GB** | Single agent with all tools |

**Comparison with Multiple Agents:**

If using separate agents (YOLO agent + WhisperSTT agent):

| Approach | Memory Usage | Overhead |
|----------|--------------|----------|
| **2 Separate Agents** | ~4.5-8.5 GB | LLM models duplicated (wasteful) |
| **Single Agent (Hybrid)** | ~2.2-4.6 GB | **Saves ~2.3-3.9 GB** (50% reduction) |

The single-agent hybrid approach saves **50-60% memory** compared to multiple separate agents, while maintaining low latency through direct tool calls.

**When to Add New Tools vs New Agents:**

- ✅ **Add as Tool**: Voice transcription (WhisperSTT), additional CV models, data processing
- ✅ **Add as Tool**: Different modalities (audio, sensors) - all handled by one agent
- ❌ **New Agent Only**: If you need completely different LLM behavior/routing logic
- ❌ **New Agent Only**: If tools need different security/access levels

#### Implementation

The codebase supports both approaches:

- **`main.py`** - Uses direct tools (current, recommended for edge deployment)
- **`mcp_server/server.py`** - Standalone MCP server (available for advanced use cases)

Tool classes (e.g., `YOLODetectionTool`, `StatisticsTool`) are designed to work with both approaches, wrapped appropriately for each use case.

## SSH Access

Both devices are configured for SSH access via `~/.ssh/config`:

- `ssh old-nano` - Connect to Jetson Nano
- `ssh orin-nano` - Connect to Orin Nano

## Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [Jetson Hacks](https://www.jetsonhacks.com/)

