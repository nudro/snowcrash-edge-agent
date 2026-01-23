# Remote Old-Nano Agent (LlamaIndex)

A completely separate agent that uses LlamaIndex to interact with the old-nano Jetson Inference Service. This does NOT modify or interfere with existing agents.

## Installation

Install LlamaIndex dependencies:

```bash
pip install llama-index llama-index-llms-llama-cpp httpx
```

## Usage

### Basic Usage

```bash
python3 remote_old_nano_agent.py \
    --model-path models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --old-nano-ip 10.163.1.173 \
    --interactive
```

### Command Line Options

```bash
--model-path PATH          Path to GGUF model (required)
--old-nano-ip IP           old-nano hostname or IP (default: old-nano)
--old-nano-port PORT       Service port (default: 9000)
--temperature FLOAT        LLM temperature (default: 0.1)
--context-window INT        Context window size (default: 4096)
--n-gpu-layers INT         GPU layers (-1 = all, default: -1)
--interactive              Run in interactive chat mode
```

### Example Queries

Once running, you can ask:

- "Check if old-nano service is healthy"
- "What is the status of person detection on old-nano?"
- "Start person detection on old-nano"
- "Stop person detection on old-nano"
- "Get information about old-nano service"
- "Is old-nano detecting any people right now?"

## Available Tools

The agent has access to these tools:

1. **check_old_nano_health** - Check if service is responding
2. **get_old_nano_status** - Get person detection status
3. **start_old_nano_person_detection** - Start continuous person watch
4. **stop_old_nano_person_detection** - Stop person watch
5. **get_old_nano_info** - Get comprehensive service information

## Architecture

```
┌─────────────────────────────────┐
│  Remote Old-Nano Agent          │
│  (LlamaIndex ReActAgent)        │
│                                 │
│  ┌───────────────────────────┐  │
│  │  LlamaCPP LLM             │  │
│  │  (Llama-3.2-3B)          │  │
│  └───────────┬──────────────┘  │
│              │                   │
│  ┌───────────▼──────────────┐  │
│  │  ReActAgent              │  │
│  │  (Tool Selection)        │  │
│  └───────────┬──────────────┘  │
│              │                   │
│  ┌───────────▼──────────────┐  │
│  │  Old-Nano Tools          │  │
│  │  - Health Check          │  │
│  │  - Status Query          │  │
│  │  - Start/Stop Detection  │  │
│  └───────────┬──────────────┘  │
└──────────────┼──────────────────┘
               │ HTTP
               ▼
┌─────────────────────────────────┐
│  old-nano Service                │
│  (Jetson Inference Service)      │
│  Port: 9000                      │
└─────────────────────────────────┘
```

## Notes

- This agent is completely separate from `main_V2.py` and `SimpleAgent`
- Uses LlamaIndex ReActAgent pattern (not LangChain)
- Requires llama-index packages (separate from existing dependencies)
- Does not interfere with existing GUI or agents
- Can run simultaneously with other agents

## Troubleshooting

**Import Error: llama-index not found**
```bash
pip install llama-index llama-index-llms-llama-cpp
```

**Connection Error to old-nano**
- Check old-nano IP: `ping <old-nano-ip>`
- Verify service is running: `curl http://<old-nano-ip>:9000/health`
- Check firewall settings

**Model Not Found**
- Ensure GGUF model is in `models/` directory
- Use full path: `--model-path models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf`

