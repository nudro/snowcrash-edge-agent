# MCP Server Overhead Analysis

## Executive Summary

For Snowcrash's deployment model (edge devices with single agent), **direct tools are recommended**. MCP server overhead (~2-8ms per call) is acceptable for multiple agents sharing tools, but not necessary for single-agent deployments.

## MCP Server Overhead Breakdown

### Per-Tool-Call Overhead

| Component | Latency | Memory | Notes |
|-----------|---------|--------|-------|
| **JSON Serialization** | 0.1-1 ms | Negligible | Converting Python dict → JSON string |
| **stdio IPC** | 1-5 ms | ~100 bytes | Inter-process communication |
| **Process Context Switch** | 1-2 ms | Negligible | Switching between agent and server processes |
| **JSON Deserialization** | 0.1-1 ms | Negligible | Converting JSON → Python dict |
| **Tool Dispatch** | 0.1-0.5 ms | Negligible | Routing to correct tool handler |
| **Total Per Call** | **2-8 ms** | **~100 bytes** | For local same-machine MCP server |

### Accumulated Overhead

- **Single tool call**: ~2-8ms overhead (5-20% of 10-50ms YOLO detection)
- **10 tool calls/second**: ~20-80ms total overhead (minimal impact)
- **30 tool calls/second** (video frame rate): ~60-240ms total overhead (could affect real-time performance)

### Memory Overhead

- **MCP server process**: ~50-100 MB (Python + dependencies)
- **Separate process memory**: Agent and server don't share memory pools
- **Total overhead**: ~50-150 MB vs direct tools

## Multi-Agent Architecture

### Scenario: YOLO Agent + WhisperSTT Agent

**Question**: Is WhisperSTT a separate agent?

**Answer**: It depends on the architecture:

#### Option 1: Single Agent with Multiple Tools (Recommended)

```
SimpleAgent
  ├── YOLO Tools (yolo_detection, tracking, distance)
  └── WhisperSTT Tools (transcribe_audio, speech_to_text)
```

**Memory Usage:**
- 1x LLM model (~2-4 GB)
- 1x YOLO model (~6 MB)
- 1x Whisper model (~150-500 MB)
- **Total**: ~2.2-4.5 GB

**Overhead**: None (same process)

#### Option 2: Multiple Separate Agents

```
Agent 1 (YOLO Agent)
  ├── LLM model (~2-4 GB)
  └── YOLO tools (~6 MB)

Agent 2 (WhisperSTT Agent)
  ├── LLM model (~2-4 GB)
  └── WhisperSTT tools (~150-500 MB)
```

**Memory Usage (Direct Tools):**
- 2x LLM models (~4-8 GB)
- 1x YOLO model (~6 MB) - duplicated if both agents need it
- 1x Whisper model (~150-500 MB)
- **Total**: ~4.2-8.5 GB (duplication!)

**Memory Usage (MCP Server):**
- 2x LLM models (~4-8 GB) - still duplicated (agents need their own LLMs
- 1x YOLO model (~6 MB) - shared via MCP server
- 1x Whisper model (~150-500 MB) - shared via MCP server
- 1x MCP server process (~50-100 MB)
- **Total**: ~4.3-8.6 GB + protocol overhead

**Overhead**: ~2-8ms per tool call × number of calls

### Recommendation: Hybrid Architecture

**Best approach for Snowcrash:**

```
SimpleAgent (single agent)
  ├── LLM (selectable: phi-3/llama/gemma)
  ├── YOLO Tools (direct - low latency needed)
  └── WhisperSTT Tools (direct - low latency needed)
```

**Why:**
- Single agent = single LLM model (saves 2-4 GB)
- Direct tools = no protocol overhead (critical for real-time)
- All tools in same process = shared memory, faster communication
- Edge devices (Nano) have 8GB total - can't afford duplication

## Overhead Comparison: Direct vs MCP Server

### Single Agent Scenario

| Metric | Direct Tools | MCP Server | Difference |
|--------|--------------|------------|------------|
| **Tool call latency** | ~10-50ms | ~12-58ms | +2-8ms |
| **Memory per agent** | ~2-4 GB | ~2.1-4.1 GB | +50-100 MB |
| **Setup complexity** | Low | Medium | More moving parts |
| **Failure isolation** | Low | High | Tool crash = agent crash vs tool crash isolated |

### Multiple Agents Scenario (2-3 agents)

| Metric | Direct Tools | MCP Server | Difference |
|--------|--------------|------------|------------|
| **Tool call latency** | ~10-50ms | ~12-58ms | +2-8ms |
| **Memory (2 agents)** | ~4-8 GB | ~4.1-8.1 GB | +50-100 MB (but tools shared) |
| **Tool duplication** | Yes (each agent loads YOLO/Whisper) | No (shared via server) | Saves ~150-500 MB |
| **Setup complexity** | Low | Medium | More moving parts |

## When MCP Server Makes Sense

**Use MCP server if:**
1. **3+ agents** sharing same tools → saves ~150-500 MB per tool
2. **Distributed deployment** → Tools on DGX Spark, agents on Nanos
3. **Tool stability issues** → Need process isolation
4. **Language diversity** → Tools usable from JavaScript/Go/etc.

**Use direct tools if:**
1. **1-2 agents** → Overhead not worth it
2. **Edge device deployment** → Every millisecond/megabyte counts
3. **Real-time requirements** → Video detection needs low latency
4. **Simplicity** → Easier debugging and maintenance

## Quantitative Example: 3 Agents on Jetson Nano

### Direct Tools Approach
```
Agent 1 (YOLO):    2 GB (LLM) + 6 MB (YOLO) = 2.006 GB
Agent 2 (Whisper): 2 GB (LLM) + 500 MB (Whisper) = 2.5 GB
Agent 3 (Hybrid):  2 GB (LLM) + 6 MB (YOLO) + 500 MB (Whisper) = 2.506 GB
───────────────────────────────────────────────────────────────────────
Total: 7.012 GB (YOLO duplicated 2x, Whisper duplicated 2x)
```

### MCP Server Approach
```
Agent 1: 2 GB (LLM)
Agent 2: 2 GB (LLM)
Agent 3: 2 GB (LLM)
MCP Server: 50 MB + 6 MB (YOLO) + 500 MB (Whisper) = 556 MB
───────────────────────────────────────────────────────────────────────
Total: 6.556 GB (tools shared, but +50 MB server + 2-8ms latency per call)
```

**Savings**: ~456 MB (6.5% of 8GB), but adds latency overhead

## Conclusion

For Snowcrash's current architecture (single agent, edge deployment):

- **Direct tools are optimal** - No overhead, simpler, sufficient for single agent
- **MCP server would add ~2-8ms per call** with minimal benefit
- **If adding WhisperSTT** - Add as tool to existing agent, not separate agent
- **If scaling to 3+ agents** - Then MCP server makes sense for memory savings

The 2-8ms overhead becomes significant at video frame rates (30+ calls/second), but acceptable for occasional tool calls.

