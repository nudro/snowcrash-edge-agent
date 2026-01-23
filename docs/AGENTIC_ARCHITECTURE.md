# Agentic Architecture Diagram - SimpleAgent

## Overview

This document provides a detailed logic diagram of all agentic components in `simple_agent.py`, showing the roles of LangChain, custom routing systems, and tool execution.

---

## Complete Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY INPUT                                  │
│                    "How far away is the bench?"                             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 0: FRAME CONTEXT GATHERING                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ If web_viewer is running:                                            │  │
│  │  1. Get current frame (raw_frame or frame)                           │  │
│  │  2. Run YOLO detection on frame → current_detections                │  │
│  │  3. Store frame_path for later cleanup                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: EARLY DIRECT HANDLERS                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ These bypass all routing for speed:                                  │  │
│  │                                                                      │  │
│  │ 1. Detection History Queries                                        │  │
│  │    Pattern: "Have any <object> been in the frame/video?"            │  │
│  │    → Directly query web_viewer.get_detection_history()               │  │
│  │    → Return frame counts, duration, first/last frame                │  │
│  │                                                                      │  │
│  │ 2. Movement Queries                                                  │  │
│  │    Keywords: "are objects moving", "moving in video"                │  │
│  │    → Check web_viewer.tracks_data for velocity/speed                │  │
│  │    → Return list of moving objects                                  │  │
│  │                                                                      │  │
│  │ 3. Tool Capability Queries                                           │  │
│  │    Keywords: "what tools", "what can you do"                        │  │
│  │    → Return _get_tools_description()                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: QUERY ONTOLOGY MATCHER                            │
│                    (Fast Keyword-Based Routing)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Component: QueryOntologyMatcher                                      │  │
│  │ Source: agent/query_ontology.py                                     │  │
│  │ Data: query_ontology.json (JSON file)                                │  │
│  │                                                                      │  │
│  │ Process:                                                             │  │
│  │  1. Check cached queries (quick prompt buttons)                      │  │
│  │  2. Match keywords against tool keywords                             │  │
│  │  3. Extract object class from query                                 │  │
│  │  4. Determine query_type (detection, color, distance, etc.)          │  │
│  │  5. Calculate confidence score (0.0-1.0)                            │  │
│  │  6. Decide if LLM reasoning needed (needs_llm flag)                 │  │
│  │                                                                      │  │
│  │ Output:                                                              │  │
│  │  {                                                                   │  │
│  │    "tool": "estimate_object_distances",                              │  │
│  │    "object_class": "bench",                                          │  │
│  │    "query_type": "distance",                                         │  │
│  │    "confidence": 0.85,                                                │  │
│  │    "needs_llm": false,  # High confidence → skip LLM                 │  │
│  │    "direct_call": false                                              │  │
│  │  }                                                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │  Confidence > 0.8     │      │  Confidence 0.3-0.8   │
    │  OR direct_call=true   │      │  OR needs_llm=true     │
    │                       │      │                       │
    │  SKIP LLM             │      │  LLM VALIDATION       │
    │  (Fast Path)          │      │  (Agentic Check)      │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                               │
                │                               ▼
                │              ┌─────────────────────────────────────┐
                │              │  QueryUnderstandingRouter            │
                │              │  (LLM-Based Semantic Understanding)   │
                │              │  Source: agent/query_router.py       │
                │              │                                       │
                │              │  Uses LangChain Prompt Templates:     │
                │              │  - QUERY_ROUTER_TEMPLATE             │
                │              │  - OBJECT_DETECTION_TEMPLATE          │
                │              │  - COLOR_DETECTION_TEMPLATE           │
                │              │  - DISTANCE_SPATIAL_TEMPLATE         │
                │              │  - TRACKING_TEMPLATE                 │
                │              │                                       │
                │              │  Process:                             │
                │              │  1. Format prompt with template       │
                │              │  2. Call LLM (via LangChain)          │
                │              │  3. Parse JSON response               │
                │              │  4. Validate/override tool choice     │
                │              │  5. Extract semantic mappings         │
                │              │     (e.g., "sweater" → "person")      │
                │              │                                       │
                │              │  LLM Backend Options:                 │
                │              │  - DockerLLMAdapter (HTTP API)       │
                │              │  - LlamaCpp (direct bindings)        │
                │              └───────────────┬───────────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: TOOL SELECTION & EXECUTION                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ LangChain Tools (from agent/langchain_tools.py):                    │  │
│  │                                                                      │  │
│  │ 1. yolo_object_detection                                             │  │
│  │    → Wraps: tools/yolo_detection.py::YOLODetectionTool              │  │
│  │    → Uses: YOLO26-seg model (PyTorch or TensorRT)                  │  │
│  │    → Returns: JSON with detections, classes, confidence            │  │
│  │                                                                      │  │
│  │ 2. track_objects                                                     │  │
│  │    → Wraps: tools/tracking_tool.py::TrackingTool                    │  │
│  │    → Uses: DeepSORT for persistent track IDs                         │  │
│  │    → Returns: Track data with IDs, velocity, trajectories           │  │
│  │                                                                      │  │
│  │ 3. get_detection_statistics                                          │  │
│  │    → Wraps: tools/statistics_tool.py::StatisticsTool                │  │
│  │    → Aggregates: Counts, averages, most common objects               │  │
│  │    → Returns: Statistics summary                                    │  │
│  │                                                                      │  │
│  │ 4. estimate_object_distances                                         │  │
│  │    → Wraps: tools/distance_tool.py::DistanceTool                   │  │
│  │    → Uses: Bounding box height + known object sizes                 │  │
│  │    → Returns: Distance in meters/feet                               │  │
│  │                                                                      │  │
│  │ Custom Tools:                                                        │  │
│  │                                                                      │  │
│  │ 5. create_vision_tool (from agent/vision_tool.py)                   │  │
│  │    → Creates LangChain tool from current frame                      │  │
│  │    → Used for: Hazard detection, "what do you see" queries         │  │
│  │    → Can be bound to LLM for tool calling (if supported)            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: RESPONSE FORMATTING                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Component: RESPONSE_FORMATTING_TEMPLATE                             │  │
│  │ Source: agent/prompt_templates.py                                   │  │
│  │                                                                      │  │
│  │ Process:                                                             │  │
│  │  1. Take tool results (raw JSON/data)                               │  │
│  │  2. Format with LangChain ChatPromptTemplate                        │  │
│  │  3. Call LLM to convert to human-readable sentence                   │  │
│  │  4. Truncate to 20 words (SOF tactical style)                       │  │
│  │                                                                      │  │
│  │ Example:                                                             │  │
│  │  Tool Result: {"detections": [{"class": "bench", "distance": 5.2}]} │  │
│  │  → LLM formats → "The bench is approximately 5.2 meters away."      │  │
│  │  → Truncated → "The bench is approximately 5.2 meters away."        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINAL RESPONSE                                     │
│                    "The bench is approximately 5.2 meters away."              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Roles

### 1. LangChain Components

#### `LlamaCpp` (from `langchain_community.llms`)
- **Role**: Direct Python bindings to llama.cpp
- **Used When**: Docker mode fails or disabled
- **Location**: Fallback LLM initialization
- **Interface**: LangChain-compatible LLM (supports `ainvoke`, `invoke`)

#### `DockerLLMAdapter` (custom, from `agent/docker_llm.py`)
- **Role**: Wraps llama-server HTTP API for LangChain compatibility
- **Used When**: Docker mode enabled (default)
- **Interface**: Inherits from `langchain_core.language_models.llms.BaseLLM`
- **Connection**: HTTP requests to `http://localhost:8080/v1/chat/completions`
- **Benefits**: GPU-optimized, isolated, better memory management

#### LangChain Tools (`@tool` decorator)
- **Role**: Convert native Python tools to LangChain-compatible tools
- **Location**: `agent/langchain_tools.py`
- **Tools**:
  - `yolo_object_detection` → `YOLODetectionTool`
  - `track_objects` → `TrackingTool`
  - `get_detection_statistics` → `StatisticsTool`
  - `estimate_object_distances` → `DistanceTool`
- **Interface**: Async functions with `ainvoke()` method

#### `ChatPromptTemplate` (from `langchain_core.prompts`)
- **Role**: Format prompts with system/user messages
- **Used In**: 
  - `QueryUnderstandingRouter` (query routing)
  - `RESPONSE_FORMATTING_TEMPLATE` (response formatting)
- **Templates**: Defined in `agent/prompt_templates.py`

---

### 2. Custom Routing Components

#### `QueryOntologyMatcher` (from `agent/query_ontology.py`)
- **Role**: Fast keyword-based routing (no LLM needed)
- **Data Source**: `query_ontology.json` (JSON file)
- **Process**:
  1. Keyword matching against tool keywords
  2. Object class extraction
  3. Query type classification
  4. Confidence scoring
  5. LLM requirement decision
- **Speed**: Instant (no network/LLM calls)
- **Use Case**: High-confidence matches (>0.8), cached queries

#### `QueryUnderstandingRouter` (from `agent/query_router.py`)
- **Role**: LLM-based semantic understanding
- **Used When**: 
  - Low confidence matches (<0.3)
  - Semantic mapping needed (e.g., "sweater" → "person")
  - Spatial relationships ("wrt", "relative to")
  - Environment inference
- **LLM Backend**: Uses LangChain LLM (DockerLLMAdapter or LlamaCpp)
- **Templates**: Specialized prompts for different query types
- **Output**: JSON with tool, object_class, query_type, parameters

---

### 3. Tool Execution Layer

#### Native Tools (in `tools/` directory)
- **YOLODetectionTool**: Object detection using YOLO26-seg
- **TrackingTool**: DeepSORT tracking with persistent IDs
- **StatisticsTool**: Aggregation of detection data
- **DistanceTool**: Distance estimation from bounding boxes
- **GeographicTool**: GPS/positioning (if available)

#### LangChain Tool Wrappers
- **Location**: `agent/langchain_tools.py`
- **Purpose**: Convert native tools to LangChain format
- **Interface**: `@tool` decorator → async functions

#### Custom Tools
- **Vision Tool** (`agent/vision_tool.py`): Creates tool from current frame
- **Remote YOLO Tool** (`agent/remote_yolo_tool.py`): Remote detection service

---

## Decision Flow Logic

### Confidence-Based Routing

```
Query → Ontology Match
  │
  ├─ Confidence > 0.8
  │  └─→ Direct Tool Call (No LLM)
  │
  ├─ Confidence 0.3-0.8
  │  └─→ LLM Validation (Lightweight check)
  │      └─→ Tool Call
  │
  └─ Confidence < 0.3
     └─→ Full LLM Reasoning
         └─→ Tool Call
```

### LLM Usage Optimization

1. **Skip LLM** (Fast Path):
   - High confidence ontology matches (>0.8)
   - Cached queries (quick prompt buttons)
   - Direct tool calls (e.g., "Color Intel")

2. **Lightweight LLM** (Agentic Validation):
   - Medium confidence (0.3-0.8)
   - Validates tool choice
   - Maintains agentic nature while being fast

3. **Full LLM** (Semantic Reasoning):
   - Low confidence (<0.3)
   - Semantic mapping needed
   - Spatial relationships
   - Environment inference

---

## LangChain vs LlamaIndex

### LangChain (Used in `simple_agent.py`)
- **LLM Interface**: `LlamaCpp`, `DockerLLMAdapter` (BaseLLM)
- **Tools**: `@tool` decorator, LangChain tool wrappers
- **Prompts**: `ChatPromptTemplate`
- **Purpose**: Main agentic framework for SimpleAgent

### LlamaIndex (NOT used in `simple_agent.py`)
- **Used In**: `remote_old_nano_agent.py` only
- **Components**: `ReActAgent`, `FunctionTool`, `LlamaCPP`
- **Purpose**: Separate agent for querying old-nano service
- **Note**: SimpleAgent does NOT use LlamaIndex

---

## Special Query Handlers

### Detection History
- **Pattern**: "Have any <object> been in the frame/video?"
- **Handler**: Direct `web_viewer.get_detection_history()` call
- **Bypasses**: All routing, LLM, tools
- **Returns**: Frame counts, duration, first/last frame

### Movement Detection
- **Keywords**: "are objects moving", "moving in video"
- **Handler**: Direct `web_viewer.tracks_data` analysis
- **Bypasses**: All routing, LLM, tools
- **Returns**: List of moving objects with speeds

### Color Detection
- **Keywords**: "color", "color intel", "what color"
- **Process**:
  1. Ontology extracts object class (fast)
  2. If semantic mapping needed → LLM (e.g., "sweater" → "person")
  3. YOLOE-26 detection with text prompts
  4. Segmentation mask analysis
  5. Color extraction from pixels

---

## Memory & Performance

### Fast Path (No LLM)
- **Latency**: ~10-50ms (keyword matching only)
- **Examples**: "how many objects", "distance to bench", "status check"

### Medium Path (LLM Validation)
- **Latency**: ~200-500ms (lightweight LLM call)
- **Examples**: "how far away is the bench?" (medium confidence)

### Slow Path (Full LLM Reasoning)
- **Latency**: ~500-2000ms (full semantic analysis)
- **Examples**: "what kind of environment", "sweater color", spatial queries

---

## Tool Execution Flow

```
Tool Selected
    │
    ├─→ LangChain Tool (from langchain_tools.py)
    │   └─→ Native Tool (from tools/)
    │       └─→ YOLO Model / Tracking / Distance Calculation
    │           └─→ Results (JSON/data)
    │
    └─→ Response Formatting
        └─→ RESPONSE_FORMATTING_TEMPLATE
            └─→ LLM formats to human sentence
                └─→ Truncate to 20 words
                    └─→ Final Response
```

---

## Key Design Decisions

1. **Hybrid Routing**: Keyword matching (fast) + LLM reasoning (accurate)
2. **Confidence-Based LLM Usage**: Only use LLM when needed
3. **Early Handlers**: Bypass routing for common queries (history, movement)
4. **LangChain Integration**: All tools wrapped in LangChain format for consistency
5. **No LlamaIndex in SimpleAgent**: Uses LangChain exclusively
6. **Docker-First LLM**: Prefers HTTP API (GPU-optimized) over direct bindings

---

## Summary

**SimpleAgent Architecture:**
- **Framework**: LangChain (not LlamaIndex)
- **LLM**: DockerLLMAdapter (HTTP) or LlamaCpp (direct)
- **Routing**: QueryOntologyMatcher (fast) + QueryUnderstandingRouter (LLM)
- **Tools**: LangChain-wrapped native tools
- **Response**: LLM-formatted with 20-word truncation

**LlamaIndex Usage:**
- **Only in**: `remote_old_nano_agent.py`
- **Not in**: `simple_agent.py`
- **Purpose**: Separate agent for old-nano service queries

