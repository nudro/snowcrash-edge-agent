#!/usr/bin/env python3
"""
Simplified agentic agent with tool calling.
Works with local LLMs and MCP tools.
"""
import sys
import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Models directory path (relative to project root)
MODELS_DIR = PROJECT_ROOT / "models"

from langchain_core.prompts import ChatPromptTemplate
from agent.langchain_tools import (
    get_langchain_tools, 
    yolo_object_detection,
    track_objects,
    get_detection_statistics,
    estimate_object_distances
)
from agent.vision_tool import create_vision_tool
from agent.query_keywords import (
    TRACKING_KEYWORDS, STATS_KEYWORDS, DISTANCE_KEYWORDS, DISTANCE_COMPARISON_KEYWORDS,
    IMAGE_VIDEO_KEYWORDS, VIDEO_KEYWORDS,
    TOOL_QUERY_KEYWORDS, MOVEMENT_KEYWORDS, SPEED_KEYWORDS, HOW_LONG_KEYWORDS, TOOL_KEYWORDS,
    PRINT_DATA_KEYWORDS, TRACK_ID_KEYWORDS, COUNT_QUERY_KEYWORDS, DESCRIPTIVE_DATA_KEYWORDS,
    COLOR_QUERY_KEYWORDS, VIEW_GUI_KEYWORDS, COMMON_OBJECT_CLASSES, FOLLOW_UP_PATTERNS,
    TIMESTAMP_KEYWORDS, FRAME_KEYWORDS, BOTH_LOGGING_KEYWORDS,
    NUMERIC_DURATION_PATTERNS, WORD_TO_NUMBER, WORD_DURATION_PATTERNS,
    IMAGE_PATH_PATTERNS, TRACK_ID_PATTERN, OBJECT_CLASS_PATTERN
)

# For local LLM (llama.cpp)
try:
    from langchain_community.llms import LlamaCpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# For Docker-based LLM (llama-server)
try:
    from agent.docker_llm import DockerLLMAdapter
    from agent.container_manager import ContainerManager
    DOCKER_LLM_AVAILABLE = True
except ImportError as e:
    DOCKER_LLM_AVAILABLE = False
    DOCKER_IMPORT_ERROR = str(e)
    ContainerManager = None  # For type hints

# Query understanding router
try:
    from agent.query_router import QueryUnderstandingRouter
    QUERY_ROUTER_AVAILABLE = True
except ImportError as e:
    QUERY_ROUTER_AVAILABLE = False
    QUERY_ROUTER_IMPORT_ERROR = str(e)
    QueryUnderstandingRouter = None

# Query ontology matcher (fast keyword-based routing)
try:
    from agent.query_ontology import QueryOntologyMatcher
    QUERY_ONTOLOGY_AVAILABLE = True
except ImportError as e:
    QUERY_ONTOLOGY_AVAILABLE = False
    QUERY_ONTOLOGY_IMPORT_ERROR = str(e)
    QueryOntologyMatcher = None


class SimpleAgent:
    """Simple agentic agent with tool calling."""
    
    # Model type constants
    MODEL_PHI3 = "phi-3"
    MODEL_LLAMA = "llama"
    MODEL_GEMMA = "gemma"
    
    def __init__(
        self,
        llm=None,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        temperature: float = 0.7,
        verbose: bool = True,
        web_viewer=None,
        use_docker: bool = True,
        docker_port: Optional[int] = None,
        container_manager: Optional[ContainerManager] = None
    ):
        """
        Initialize the agent.
        
        Args:
            llm: Pre-initialized LLM (optional)
            model_path: Path to local GGUF model (optional, overrides model_type)
            model_type: Model to use - "phi-3", "llama", or "gemma" (optional)
            temperature: LLM temperature
            verbose: Whether to print agent reasoning
            use_docker: Use Docker llama-server (default: True, falls back to llama-cpp-python if fails)
            docker_port: Port for Docker llama-server (auto-assigned if None)
            container_manager: ContainerManager instance (created if None)
        """
        self.verbose = verbose
        self.tools = get_langchain_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.model_type = model_type
        self.web_viewer = web_viewer  # Reference to web viewer if already running
        self.use_docker = use_docker
        self.docker_port = docker_port
        self.container_manager = container_manager
        
        # Initialize LLM
        self.llm = None
        if llm is None:
            # Try Docker first (if enabled), fallback to llama-cpp-python
            if use_docker and DOCKER_LLM_AVAILABLE and model_type:
                try:
                    # Get or create container manager
                    if self.container_manager is None:
                        self.container_manager = ContainerManager(verbose=verbose)
                    
                    # Start container and get port
                    port = self.container_manager.start_model_container(model_type)
                    if port:
                        # Use Docker LLM adapter
                        self.llm = DockerLLMAdapter(
                            model_type=model_type,
                            port=port if docker_port is None else docker_port,
                            temperature=temperature,
                            verbose=verbose
                        )
                        if self.verbose:
                            print(f"[OK] Using Docker llama-server for {model_type} on port {port}")
                    else:
                        raise RuntimeError("Failed to start Docker container")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] Docker mode failed: {e}")
                        print(f"[WARNING] Falling back to llama-cpp-python...")
                    # Fall through to llama-cpp-python
                    use_docker = False
                    self.llm = None  # Ensure llm is reset so fallback works
            
            # Fallback to llama-cpp-python (or if Docker disabled)
            if not use_docker or self.llm is None:
                if model_path and LLAMA_CPP_AVAILABLE:
                    # Use explicit model path
                    self.llm = self._create_llama_llm(model_path, temperature)
                    self.model_type = self._detect_model_type(model_path)
                elif model_type and LLAMA_CPP_AVAILABLE:
                    # Auto-detect model path from model_type
                    model_path = self._find_model_by_type(model_type)
                    if model_path:
                        self.llm = self._create_llama_llm(model_path, temperature)
                        if self.verbose:
                            print(f"[OK] Loaded {model_type} model: {model_path}")
                    else:
                        print(f"[WARNING] {model_type} model not found. Install models first.")
                        self.llm = None
                else:
                    if use_docker:
                        print("[WARNING] Docker failed and llama-cpp-python not available.")
                    else:
                        print("[WARNING] No LLM available. Install llama-cpp-python")
                    self.llm = None
        else:
            self.llm = llm
        
        # Initialize query ontology matcher (fast keyword-based routing)
        self.query_ontology = None
        if QUERY_ONTOLOGY_AVAILABLE:
            try:
                self.query_ontology = QueryOntologyMatcher()
                if self.verbose:
                    print("[OK] Query ontology matcher initialized")
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Failed to initialize query ontology: {e}")
                self.query_ontology = None
        
        # Initialize query understanding router (LLM-based semantic understanding - only used when needed)
        self.query_router = None
        if self.llm is not None and QUERY_ROUTER_AVAILABLE:
            try:
                self.query_router = QueryUnderstandingRouter(self.llm, verbose=verbose)
                if self.verbose:
                    print("[OK] Query understanding router initialized")
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Failed to initialize query router: {e}")
                self.query_router = None
    
    def _detect_model_type(self, model_path: str) -> Optional[str]:
        """Detect model type from file path."""
        path_lower = model_path.lower()
        if "phi" in path_lower or "phi-3" in path_lower:
            return self.MODEL_PHI3
        elif "llama" in path_lower:
            return self.MODEL_LLAMA
        elif "gemma" in path_lower:
            return self.MODEL_GEMMA
        return None
    
    def _find_model_by_type(self, model_type: str) -> Optional[str]:
        """Find model file path by type."""
        # Use models directory relative to project root
        models_dir = MODELS_DIR
        
        # Map model types to directories
        model_dirs = {
            self.MODEL_PHI3: "phi3-mini",
            self.MODEL_LLAMA: "llama3.2",
            self.MODEL_GEMMA: "gemma2b",
        }
        
        model_dir_name = model_dirs.get(model_type.lower())
        if not model_dir_name:
            return None
        
        model_dir_path = models_dir / model_dir_name
        if not model_dir_path.exists():
            return None
        
        # Find first GGUF file
        gguf_files = list(model_dir_path.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])
        
        return None
    
    def _create_llama_llm(self, model_path: str, temperature: float):
        """Create a LlamaCpp LLM from GGUF file."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available")
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return LlamaCpp(
            model_path=str(model_path),
            temperature=temperature,
            n_ctx=2048,
            n_batch=512,
            verbose=False,
        )
    
    async def _should_use_tool(self, prompt: str) -> Optional[str]:
        """Decide if we should use a tool based on prompt."""
        prompt_lower = prompt.lower()
        
        # Check for tracking keywords first (more specific)
        if any(keyword in prompt_lower for keyword in TRACKING_KEYWORDS):
            return "track_objects"
        
        # Check for statistics keywords
        if any(keyword in prompt_lower for keyword in STATS_KEYWORDS):
            return "get_detection_statistics"
        
        # Check for distance keywords
        if any(keyword in prompt_lower for keyword in DISTANCE_KEYWORDS):
            return "estimate_object_distances"
        
        # Check for image/video-related keywords (detection)
        if any(keyword in prompt_lower for keyword in IMAGE_VIDEO_KEYWORDS):
            return "yolo_object_detection"
        
        return None
    
    def _get_tools_description(self) -> str:
        """Get description of available tools."""
        tools_description = """I have the following tools available:

1. **YOLO Object Detection** (`yolo_object_detection`)
   - Detect objects in images or webcam feeds
   - Examples: "Detect objects in image.jpg", "Detect objects in the video stream"

2. **Detection Statistics** (`get_detection_statistics`)
   - Get aggregated statistics about object detections over time
   - Shows counts, average confidence, and most common objects
   - Examples: "What's the most common object?", "Get detection statistics"

3. **Distance Estimation** (`estimate_object_distances`)
   - Estimate distance from camera to detected objects
   - Uses bounding box height and known object sizes
   - Compare distances: "Which car is closer?", "Order objects by distance"
   - Examples: "How far away is the nearest person?", "Estimate distances to objects", "Which car is closer?"

4. **Color Detection** (built-in capability)
   - Detect dominant colors of objects in video feed
   - Works on edge devices without internet
   - Examples: "What color are the cars?", "What color is that bench?"

6. **Object Tracking** (`track_objects`)
   - Track objects across frames with persistent IDs
   - Maintains track history, position, and velocity
   - Examples: "Track objects in the video stream", "Follow that person"

7. **GUI Viewer** (visual interface)
   - View detections and tracking in a live video window
   - Shows bounding boxes, labels, track IDs, and trajectory trails
   - Examples: "View tracking in the GUI", "Show me the detections", "Display tracking viewer"

You can ask me to use any of these tools by describing what you want to do in natural language."""
        return tools_description
    
    def _extract_duration(self, prompt: str) -> float:
        """Extract duration in seconds from prompt (supports '10' or 'ten', max 60 seconds)."""
        prompt_lower = prompt.lower()
        
        # First try to find numeric duration (e.g., "10 seconds", "for 5 sec", "for 10", "10 sec")
        for pattern in NUMERIC_DURATION_PATTERNS:
            match = re.search(pattern, prompt_lower)
            if match:
                duration = float(match.group(1))
                return min(60.0, max(0.0, duration))
        
        # Try word-based numbers (e.g., "ten seconds", "for ten", "ten sec")
        for word, number in WORD_TO_NUMBER.items():
            for pattern_template in WORD_DURATION_PATTERNS:
                pattern = pattern_template.format(word)
                if re.search(pattern, prompt_lower):
                    return min(60.0, max(0.0, float(number)))
        
        # Default: 0 seconds (single frame)
        return 0.0
    
    async def _extract_image_path(self, prompt: str) -> Optional[str]:
        """Extract image path from prompt."""
        # Look for file paths
        for pattern in IMAGE_PATH_PATTERNS:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                path = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
                if Path(path).exists():
                    return path
        
        return None
    
    def _is_open_question(self, prompt: str) -> bool:
        """Detect if prompt is an open-ended question that should have a concise response."""
        prompt_lower = prompt.lower().strip()
        
        # Check if it's a question (contains question words or ends with ?)
        is_question = (
            any(word in prompt_lower for word in ["what", "how", "why", "when", "where", "which", "who"]) or
            prompt_lower.endswith("?") or
            any(phrase in prompt_lower for phrase in ["what kind of", "what type of", "tell me about"])
        )
        
        # Check if it doesn't require tools (no detection/tracking/distance keywords)
        requires_tool = any(keyword in prompt_lower for keyword in TOOL_KEYWORDS)
        
        # It's an open question if it's a question but doesn't require tools
        return is_question and not requires_tool
    
    def _truncate_to_words(self, text: str, max_words: int = 20) -> str:
        """Truncate text to maximum number of words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        # Take first max_words words and add ellipsis if truncated
        truncated = ' '.join(words[:max_words])
        # Remove trailing punctuation that might look odd with ellipsis
        truncated = truncated.rstrip('.,;:')
        return truncated + "..."
    
    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt."""
        if self.llm is None:
            return "Error: No LLM available. Cannot run agent."
        
        # ALWAYS check video feed if web_viewer is available
        # Get current frame and run detection to provide context to query router
        current_detections = None
        current_frame_path = None
        
        if self.web_viewer is not None:
            try:
                with self.web_viewer.frame_lock:
                    if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                        current_frame = self.web_viewer.raw_frame.copy()
                    elif self.web_viewer.frame is not None:
                        current_frame = self.web_viewer.frame.copy()
                    else:
                        current_frame = None
                
                if current_frame is not None:
                    # Run detection on current frame to get context
                    import tempfile
                    import cv2
                    import os
                    from agent.langchain_tools import yolo_object_detection
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=PROJECT_ROOT) as tmp_file:
                        cv2.imwrite(tmp_file.name, current_frame)
                        current_frame_path = tmp_file.name
                    
                    try:
                        tool_result = await yolo_object_detection.ainvoke({
                            "image_path": current_frame_path,
                            "confidence_threshold": getattr(self.web_viewer, 'confidence_threshold', 0.50)
                        })
                        
                        # Parse detection results
                        if isinstance(tool_result, str):
                            import json
                            try:
                                current_detections = json.loads(tool_result)
                            except:
                                current_detections = {"detections": []}
                        else:
                            current_detections = tool_result
                    except Exception as e:
                        if self.verbose:
                            print(f"[AGENT] Error getting frame detections: {e}")
                        current_detections = None
            except Exception as e:
                if self.verbose:
                    print(f"[AGENT] Error accessing web_viewer frame: {e}")
        
        # Check if user is asking about available tools (check this FIRST before any tool calling)
        prompt_lower = prompt.lower().strip()
        
        # Check if the prompt is primarily asking about tools/capabilities
        if any(keyword in prompt_lower for keyword in TOOL_QUERY_KEYWORDS):
            # Clean up temp file if created
            if current_frame_path and os.path.exists(current_frame_path):
                os.unlink(current_frame_path)
            return self._get_tools_description()
        
        # STEP 1: Check for detection history queries FIRST (before ontology routing)
        prompt_lower = prompt.lower().strip()
        
        # Pattern 1: "Have any <object> been in the frame/video?"
        # More flexible pattern to catch variations
        have_any_pattern = re.compile(r'have\s+any\s+(\w+)\s+been\s+in\s+(?:the\s+)?(?:frame|video)', re.IGNORECASE)
        have_any_match = have_any_pattern.search(prompt_lower)
        
        # Pattern 2: "Give me the history of <object> in video"
        history_pattern = re.compile(r'give\s+me\s+the\s+history\s+of\s+(\w+)\s+in\s+(?:the\s+)?video', re.IGNORECASE)
        history_match = history_pattern.search(prompt_lower)
        
        # Pattern 3: "history of <object>" (shorter variant)
        history_short_pattern = re.compile(r'history\s+of\s+(\w+)\s+in\s+(?:the\s+)?video', re.IGNORECASE)
        history_short_match = history_short_pattern.search(prompt_lower)
        
        # Check if any pattern matches
        if (have_any_match or history_match or history_short_match) and self.web_viewer is not None:
            try:
                # Extract object class from whichever pattern matched
                if have_any_match:
                    requested_class = have_any_match.group(1).lower()
                elif history_match:
                    requested_class = history_match.group(1).lower()
                else:
                    requested_class = history_short_match.group(1).lower()
                
                # Handle plural forms (cars -> car, people -> person)
                if requested_class.endswith('s') and len(requested_class) > 1:
                    requested_class = requested_class[:-1]  # Remove 's'
                
                # Get detection history for this object class
                history = self.web_viewer.get_detection_history(object_class=requested_class)
                
                if history.get("detected", False):
                    frame_count = history.get("frame_count", 0)
                    total_frames = history.get("total_frames", 0)
                    first_frame = history.get("first_frame")
                    last_frame = history.get("last_frame")
                    
                    # Calculate duration in minutes (assuming ~30 FPS)
                    fps = 30.0  # Approximate FPS
                    if first_frame and last_frame:
                        frame_span = last_frame - first_frame + 1
                        duration_seconds = frame_span / fps
                        duration_minutes = duration_seconds / 60.0
                    else:
                        duration_minutes = 0.0
                    
                    if frame_count > 0:
                        if history_match or history_short_match:
                            # Detailed history response
                            response = f"History of {requested_class}(s) in video:\n"
                            response += f"  - Detected in {frame_count} out of {total_frames} frames\n"
                            if first_frame:
                                response += f"  - First detected at frame {first_frame}\n"
                            if last_frame and last_frame != first_frame:
                                response += f"  - Last detected at frame {last_frame}\n"
                            if duration_minutes > 0:
                                response += f"  - Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)\n"
                            response += f"  - Frame span: frames {first_frame} to {last_frame}"
                        else:
                            # Simple yes/no response
                            response = f"Yes, {requested_class}(s) have been detected in the video. "
                            response += f"They appeared in {frame_count} out of {total_frames} frames. "
                            if first_frame:
                                response += f"First detected at frame {first_frame}"
                            if last_frame and last_frame != first_frame:
                                response += f", last detected at frame {last_frame}"
                            if duration_minutes > 0:
                                response += f". Duration: {duration_minutes:.2f} minutes"
                            response += "."
                        return response
                    else:
                        return f"No, {requested_class}(s) have not been detected in the video yet."
                else:
                    return f"No, {requested_class}(s) have not been detected in the video yet."
            
            except Exception as e:
                if self.verbose:
                    import traceback
                    print(f"[AGENT] Detection history check error: {e}")
                    print(f"[AGENT] Traceback: {traceback.format_exc()}")
                # Fall through to other checks
        
        # STEP 2: Check for movement queries (before ontology routing)
        # Movement queries need tracking data, not object detection
        is_movement_query = any(keyword in prompt_lower for keyword in [
            "are objects moving", "are the objects moving", "which objects are moving",
            "what is moving", "what's moving", "objects moving", "moving objects",
            "is anything moving", "detect movement", "check movement",
            "are any objects moving", "any objects moving", "moving in video"
        ])
        
        if is_movement_query and self.web_viewer is not None:
            # Use tracking data to check for movement
            MOVEMENT_THRESHOLD = 1.0  # pixels per second
            
            try:
                with self.web_viewer.tracks_lock:
                    tracks = list(self.web_viewer.tracks_data.values())
                
                moving_objects = []
                for track in tracks:
                    # Check both average_speed_pixels_per_second and velocity magnitude
                    avg_speed = track.get("average_speed_pixels_per_second", 0.0)
                    velocity = track.get("velocity", {"vx": 0.0, "vy": 0.0})
                    velocity_magnitude = ((velocity.get("vx", 0.0) ** 2) + (velocity.get("vy", 0.0) ** 2)) ** 0.5
                    
                    # Object is moving if speed or velocity magnitude exceeds threshold
                    if avg_speed > MOVEMENT_THRESHOLD or velocity_magnitude > MOVEMENT_THRESHOLD:
                        moving_objects.append({
                            "track_id": track.get("track_id"),
                            "class": track.get("class"),
                            "speed": round(avg_speed, 2),
                            "velocity": velocity
                        })
                
                if moving_objects:
                    # Format response with moving objects - name them
                    object_names = [obj['class'] for obj in moving_objects]
                    unique_classes = list(set(object_names))
                    if len(unique_classes) == 1:
                        response = f"Yes, {unique_classes[0]}(s) are moving: "
                    else:
                        response = f"Yes, {', '.join(unique_classes)} are moving: "
                    
                    # Add details for each moving object
                    for obj in moving_objects:
                        response += f"{obj['class']} (ID: {obj['track_id']}), "
                    response = response.rstrip(", ") + "."
                    return response
                else:
                    return "No objects are currently moving."
            
            except Exception as e:
                if self.verbose:
                    print(f"[AGENT] Movement check error: {e}")
                # Fall through to ontology routing
        
        # STEP 2: Route through query ontology (after movement check)
        # This ensures queries like "How far away is the bench?" are properly routed
        # even if they don't explicitly mention "in the video"
        early_ontology_result = None
        early_tool_name = None
        skip_early_checks = False
        
        if self.query_ontology:
            try:
                early_ontology_result = self.query_ontology.quick_route(prompt)
                if self.verbose:
                    print(f"[AGENT] Ontology match (early): {early_ontology_result}")
                
                # Extract tool and object class from ontology
                early_tool_name = early_ontology_result.get("tool")
                early_confidence = early_ontology_result.get("confidence", 0.0)
                
                # If ontology found a tool match with reasonable confidence, skip early keyword checks
                # This ensures queries like "How far away is the bench?" go directly to distance tool
                if early_tool_name and early_confidence > 0.3:
                    skip_early_checks = True
                    if self.verbose:
                        print(f"[AGENT] Ontology routed to {early_tool_name} (confidence: {early_confidence:.2f}), skipping early keyword checks")
            except Exception as e:
                if self.verbose:
                    print(f"[AGENT] Query ontology error (early): {e}")
                early_ontology_result = None
        
        # PRIORITY CHECK: If asking "what objects" and we have detections, handle directly
        object_query_keywords = ["what objects", "list objects", "what can you see", "objects in the video", 
                                  "objects in video", "what's in the video", "what is in the video", 
                                  "what are in the video", "detect objects", "what objects are"]
        is_object_query = any(phrase in prompt_lower for phrase in object_query_keywords)
        
        if is_object_query and current_detections is not None:
            # Get raw frame from web viewer and run detection directly
            try:
                with self.web_viewer.frame_lock:
                    if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                        current_frame = self.web_viewer.raw_frame.copy()
                    elif self.web_viewer.frame is not None:
                        current_frame = self.web_viewer.frame.copy()
                    else:
                        current_frame = None
                
                if current_frame is None:
                    return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                
                # Run object detection directly
                import tempfile
                import cv2
                import os
                from agent.langchain_tools import yolo_object_detection
                
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=PROJECT_ROOT) as tmp_file:
                    cv2.imwrite(tmp_file.name, current_frame)
                    temp_path = tmp_file.name
                
                try:
                    tool_result = await yolo_object_detection.ainvoke({
                        "image_path": temp_path,
                        "confidence_threshold": getattr(self.web_viewer, 'confidence_threshold', 0.50)
                    })
                    
                    # Parse and format results
                    if isinstance(tool_result, str):
                        import json
                        try:
                            tool_result = json.loads(tool_result)
                        except:
                            pass
                    
                    if isinstance(tool_result, dict):
                        detections = tool_result.get("detections", [])
                        if detections:
                            class_counts = {}
                            for det in detections:
                                cls = det.get("class", "unknown")
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                            
                            total = len(detections)
                            response = f"I detected {total} object(s) in the current video frame:\n"
                            for cls, count in sorted(class_counts.items()):
                                response += f"  - {cls}: {count}\n"
                            return response.strip()
                        else:
                            return "I detected 0 objects in the current video frame."
                    else:
                        # Format raw result using Response Formatting Template
                        from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                        formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                            user_query=prompt,
                            tool_results=f"DETECTION: {str(tool_result)}"
                        )
                        
                        system_msg = None
                        user_msg = None
                        for msg in formatting_prompt:
                            if msg.type == "system":
                                system_msg = msg.content
                            elif msg.type == "human":
                                user_msg = msg.content
                        
                        if hasattr(self.llm, '_acall'):
                            response = await self.llm._acall(user_msg, system_prompt=system_msg)
                        else:
                            import asyncio
                            response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                            if not isinstance(response, str):
                                response = response.content if hasattr(response, 'content') else str(response)
                        
                        return self._truncate_to_words(response.strip(), max_words=20)
                finally:
                    # Keep temp file for router context, will clean up at end
                    pass
            except Exception as e:
                if self.verbose:
                    import traceback
                    print(f"[AGENT] Direct object detection error: {traceback.format_exc()}")
                # Clean up on error
                if current_frame_path and os.path.exists(current_frame_path):
                    try:
                        os.unlink(current_frame_path)
                    except:
                        pass
                return f"Error detecting objects in video frame: {str(e)}"
        
        # Skip early keyword checks if ontology already routed the query
        if not skip_early_checks:
            # EARLY CHECK: Detect non-vision queries to skip routing and answer directly
            # This prevents unnecessary LLM calls for general questions (fixes 30-second delay)
            vision_keywords = (
                VIDEO_KEYWORDS + IMAGE_VIDEO_KEYWORDS + TRACKING_KEYWORDS + 
                COLOR_QUERY_KEYWORDS + DISTANCE_KEYWORDS + STATS_KEYWORDS +
                ["detect", "detection", "object", "objects", "track", "tracking",
                 "color", "distance", "how many", "count", "what objects",
                 "environment", "what kind of place", "based on objects",
                 "hazard", "hazards", "obstruction", "obstacle", "obstacles",
                 "what do you see", "what's in the frame", "what's visible",
                 "is the path clear", "clear path", "safe to proceed",
                 "any dangers", "any threats", "risk", "risks"]
            )
            is_vision_query = any(keyword in prompt_lower for keyword in vision_keywords)
            
            # If not a vision query, answer directly without routing
            if not is_vision_query:
                if self.verbose:
                    print("[AGENT] Non-vision query detected, answering directly without routing...")
                try:
                    response = await self.llm.ainvoke(prompt)
                    return self._truncate_to_words(response.strip(), max_words=20)
                except Exception as e:
                    if self.verbose:
                        print(f"[AGENT] Direct LLM call failed: {e}")
                    return f"I encountered an error: {str(e)}"
        
        # Movement queries are now handled BEFORE ontology routing (see above)
        # This section is kept as fallback but should not be reached for movement queries
        
        # Check for speed queries - return speed estimates for objects
        # SKIP if ontology already routed
        is_speed_query = any(keyword in prompt_lower for keyword in SPEED_KEYWORDS) if not skip_early_checks else False
        
        if is_speed_query and self.web_viewer is not None:
            try:
                # Extract object class from prompt if specified
                object_class = None
                for obj_class in COMMON_OBJECT_CLASSES:
                    if obj_class in prompt_lower:
                        object_class = obj_class
                        break
                
                # Get tracks from web viewer
                with self.web_viewer.tracks_lock:
                    tracks = list(self.web_viewer.tracks_data.values())
                
                # Filter by object class if specified
                if object_class:
                    tracks = [t for t in tracks if t.get("class", "").lower() == object_class.lower()]
                
                if not tracks:
                    if object_class:
                        return f"No {object_class} objects are currently being tracked."
                    else:
                        return "No objects are currently being tracked."
                
                # Format response with speed estimates
                response = "Speed estimates:\n"
                for track in tracks:
                    track_id = track.get("track_id")
                    cls = track.get("class", "unknown")
                    avg_speed = track.get("average_speed_pixels_per_second", 0.0)
                    velocity = track.get("velocity", {"vx": 0.0, "vy": 0.0})
                    
                    response += f"  - {cls} (ID: {track_id}): {avg_speed:.2f} px/s"
                    if velocity.get("vx") != 0.0 or velocity.get("vy") != 0.0:
                        response += f" (vx: {velocity['vx']:.2f}, vy: {velocity['vy']:.2f} px/s)"
                    response += "\n"
                
                return response.strip()
            
            except Exception as e:
                return f"Error getting speed estimates: {str(e)}. Please ensure the web viewer is running."
        
        # Check for "how long" queries - calculate track duration
        # SKIP if ontology already routed
        is_how_long_query = any(keyword in prompt_lower for keyword in HOW_LONG_KEYWORDS) if not skip_early_checks else False
        
        if is_how_long_query and self.web_viewer is not None:
            try:
                # Extract track ID or object class from prompt
                track_id_match = re.search(TRACK_ID_PATTERN, prompt_lower)
                object_class_match = re.search(OBJECT_CLASS_PATTERN, prompt_lower)
                
                requested_track_id = int(track_id_match.group(1)) if track_id_match else None
                requested_class = object_class_match.group(1) if object_class_match and not requested_track_id else None
                
                # Call get_track_duration
                durations = self.web_viewer.get_track_duration(
                    track_id=requested_track_id,
                    object_class=requested_class if not requested_track_id else None
                )
                
                if not durations:
                    if requested_track_id:
                        return f"Track ID {requested_track_id} not found in tracking data."
                    elif requested_class:
                        return f"No {requested_class} objects found in tracking data."
                    else:
                        return "No tracking data available yet."
                
                # Format response
                response_lines = []
                for duration_info in durations:
                    track_id = duration_info["track_id"]
                    track_class = duration_info["class"]
                    duration_min = duration_info["duration_minutes"]
                    duration_sec = duration_info["duration_seconds"]
                    is_active = duration_info["is_active"]
                    status = "currently active" if is_active else "no longer active"
                    
                    response_lines.append(
                        f"{track_class} (ID: {track_id}) has been in the frame for {duration_min:.2f} minutes ({duration_sec:.1f} seconds) - {status}"
                    )
                
                return "\n".join(response_lines)
            
            except Exception as e:
                return f"Error calculating track duration: {str(e)}. Please ensure the web viewer is running and tracking is active."
        
        # Detection history queries are now handled BEFORE ontology routing (see above)
        # This section is kept as fallback but should not be reached for detection history queries
        
        # Check for color queries - detect colors of objects in the video
        # SKIP if ontology already routed to a different tool
        is_color_query = any(keyword in prompt_lower for keyword in COLOR_QUERY_KEYWORDS) if not skip_early_checks or early_tool_name == "color_detection" else False
        
        if is_color_query:
            # Check if this is the "Color Intel" cached query (should analyze ALL objects, not filter)
            # The prompt is "Color identification of targets" - should NOT extract object class
            is_color_intel_query = (
                "color identification of targets" in prompt_lower or
                prompt_lower.strip() == "color identification of targets" or
                (prompt_lower.strip().startswith("color") and "identification" in prompt_lower and "targets" in prompt_lower)
            )
            
            # FAST PATH: Use ontology first to extract object class (skip slow LLM call)
            # BUT: Skip object class extraction for "Color Intel" - it should analyze ALL objects
            requested_class = None
            
            if not is_color_intel_query:
                # Only extract object class if NOT the "Color Intel" button query
                # Try ontology matcher first (fast, no LLM)
                if self.query_ontology:
                    try:
                        requested_class = self.query_ontology.extract_object_class(prompt)
                        if self.verbose and requested_class:
                            print(f"[AGENT] Ontology extracted object class: {requested_class}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[AGENT] Ontology error for color query: {e}")
                
                # Fallback to keyword matching (fast, no LLM)
                if requested_class is None:
                    for cls in COMMON_OBJECT_CLASSES:
                        if cls in prompt_lower or f"{cls}s" in prompt_lower:
                            requested_class = cls
                            break
            else:
                # "Color Intel" query - analyze ALL objects, no filtering
                # Skip object class extraction entirely
                requested_class = None
                if self.verbose:
                    print(f"[AGENT] Color Intel query detected - analyzing ALL objects (no class filter)")
            
            # Only use LLM if ontology/keywords failed AND we need semantic mapping (e.g., "sweater" → "person")
            # Check if query contains semantic mapping keywords
            semantic_keywords = ["sweater", "shirt", "jacket", "pants", "dress", "face", "hand", "foot", "head"]
            needs_semantic_mapping = any(kw in prompt_lower for kw in semantic_keywords)
            
            if requested_class is None and needs_semantic_mapping and self.query_router:
                # Only now call LLM for semantic mapping (e.g., "sweater" → "person")
                try:
                    color_understanding = await self.query_router.understand_color_query(prompt)
                    requested_class = color_understanding.get("object_class")
                    if self.verbose and requested_class:
                        semantic_mapping = color_understanding.get("semantic_mapping")
                        if semantic_mapping:
                            print(f"[AGENT] LLM semantic mapping: {semantic_mapping}")
                except Exception as e:
                    if self.verbose:
                        print(f"[AGENT] Query router error for color query: {e}")
            
            # Try to get frame from web viewer if running
            if self.web_viewer is not None:
                try:
                    from tools.color_detection import detect_colors_from_yolo_results
                    from tools.yolo_utils import load_yolo_model
                    import cv2
                    import os
                    
                    # Get frame from web viewer
                    with self.web_viewer.frame_lock:
                        if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                            current_frame = self.web_viewer.raw_frame.copy()
                        elif self.web_viewer.frame is not None:
                            current_frame = self.web_viewer.frame.copy()
                        else:
                            current_frame = None
                    
                    if current_frame is None:
                        return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                    
                    # Use YOLOE with text prompts for better object detection (like environment detection)
                    model_path = str(MODELS_DIR / "yolo26n-seg.pt")
                    
                    # Check if YOLOE-26 model is available (with text prompt support)
                    yoloe_models_with_prompts = [
                        str(MODELS_DIR / "yoloe-26n-seg.pt"),
                        str(MODELS_DIR / "yoloe-26s-seg.pt"),
                        str(MODELS_DIR / "yoloe-26m-seg.pt"),
                        str(MODELS_DIR / "yoloe-26l-seg.pt"),
                    ]
                    
                    # Fallback to prompt-free versions if text-prompt versions not available
                    yoloe_models_prompt_free = [
                        str(MODELS_DIR / "yoloe-26n-seg-pf.pt"),
                        str(MODELS_DIR / "yoloe-26s-seg-pf.pt"),
                        str(MODELS_DIR / "yoloe-26m-seg-pf.pt"),
                    ]
                    
                    use_yoloe = False
                    use_text_prompts = False
                    for yoloe_path in yoloe_models_with_prompts:
                        if os.path.exists(yoloe_path):
                            model_path = yoloe_path
                            use_yoloe = True
                            use_text_prompts = True
                            if self.verbose:
                                print(f"[AGENT] Using YOLOE-26 with text prompts for color detection: {yoloe_path}")
                            break
                    
                    # Fallback to prompt-free if text-prompt versions not found
                    if not use_yoloe:
                        for yoloe_path in yoloe_models_prompt_free:
                            if os.path.exists(yoloe_path):
                                model_path = yoloe_path
                                use_yoloe = True
                                if self.verbose:
                                    print(f"[AGENT] Using YOLOE-26 prompt-free model for color detection: {yoloe_path}")
                                break
                    
                    # Load YOLO model (auto-detects TensorRT .engine or PyTorch .pt)
                    # Use YOLOE if available, otherwise fallback to regular YOLO
                    if use_yoloe:
                        yolo_model = load_yolo_model(model_path, verbose=self.verbose)
                    else:
                        # Fallback to regular YOLO or web viewer's model
                        if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                            yolo_model = self.web_viewer.model
                        else:
                            from ultralytics import YOLO
                            yolo_model = YOLO(model_path)
                    
                    # Check if model is TensorRT engine (requires specific imgsz)
                    from pathlib import Path
                    engine_path = Path(model_path).with_suffix('.engine')
                    is_tensorrt_engine = engine_path.exists()
                    if is_tensorrt_engine:
                        imgsz_for_detection = 640  # TensorRT engines are compiled for specific size (640x640)
                    else:
                        imgsz_for_detection = 640  # PyTorch models - use smaller size to save memory
                    
                    # If using YOLOE with text prompts and we have a requested class, use it as text prompt
                    # BUT: Skip text prompts for "Color Intel" - we want ALL objects
                    if use_yoloe and use_text_prompts and requested_class and not is_color_intel_query:
                        try:
                            # Use the requested object class as a text prompt for targeted detection
                            # This ensures YOLOE focuses on detecting the specific object we want color info for
                            color_classes = [requested_class]
                            
                            # Also include common variations/synonyms for better detection
                            if requested_class == "person":
                                color_classes.extend(["person", "people", "human"])
                            elif requested_class == "car":
                                color_classes.extend(["car", "vehicle", "auto", "automobile"])
                            elif requested_class == "bicycle":
                                color_classes.extend(["bicycle", "bike", "cycle"])
                            elif requested_class == "motorcycle":
                                color_classes.extend(["motorcycle", "motorbike", "bike"])
                            
                            # Set text prompts for targeted object detection
                            yolo_model.set_classes(color_classes, yolo_model.get_text_pe(color_classes))
                            if self.verbose:
                                print(f"[AGENT] Set YOLOE text prompts for color detection: {color_classes}")
                        except Exception as e:
                            if self.verbose:
                                print(f"[AGENT] Warning: Could not set YOLOE text prompts for color detection: {e}")
                            use_text_prompts = False  # Fallback to regular detection
                    
                    # Run detection with appropriate resolution
                    # TensorRT engines require imgsz=640 (compiled size)
                    # PyTorch models use imgsz=640 (memory efficient)
                    results = yolo_model(current_frame, conf=0.15, verbose=False, imgsz=imgsz_for_detection)
                    
                    # Debug: Check if we got any detections
                    if self.verbose:
                        result = results[0]
                        num_detections = len(result.boxes) if result.boxes is not None else 0
                        has_masks = result.masks is not None if hasattr(result, 'masks') else False
                        print(f"[AGENT] Detection results: {num_detections} objects, masks={has_masks}")
                    
                    # Detect colors using segmentation masks (fast, direct pixel analysis)
                    # This function uses masks if available, which is more accurate than bounding boxes
                    try:
                        detections_with_colors = detect_colors_from_yolo_results(
                            current_frame, results, object_class=requested_class
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"[AGENT] Error in color detection: {e}")
                        # Fallback: check if we have any detections at all
                        result = results[0]
                        if result.boxes is None or len(result.boxes) == 0:
                            obj_text = f"{requested_class} " if requested_class else ""
                            return f"No {obj_text}objects detected in the current frame. (Using YOLOE-26 for detection)"
                        else:
                            # We have detections but color detection failed - return basic info
                            detections = []
                            for box in result.boxes:
                                cls_id = int(box.cls)
                                cls_name = result.names[cls_id]
                                if requested_class and cls_name != requested_class:
                                    continue
                                detections.append(f"{cls_name} (confidence: {box.conf:.2f})")
                            if detections:
                                return f"Detected: {', '.join(detections)}. Color analysis failed - please try again."
                            else:
                                obj_text = f"{requested_class} " if requested_class else ""
                                return f"No {obj_text}objects detected in the current frame. (Using YOLOE-26 for detection)"
                    
                    if not detections_with_colors:
                        # Check if we actually got detections but they were filtered out
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            # We have detections but they were filtered (wrong class or color detection failed)
                            detected_classes = [result.names[int(box.cls)] for box in result.boxes]
                            if requested_class and requested_class not in detected_classes:
                                return f"No {requested_class} objects detected. Found: {', '.join(set(detected_classes))}. (Using YOLOE-26 for detection)"
                            else:
                                return f"Objects detected but color analysis failed. Detected: {', '.join(set(detected_classes))}. (Using YOLOE-26 for detection)"
                        else:
                            obj_text = f"{requested_class} " if requested_class else ""
                            if use_yoloe:
                                return f"No {obj_text}objects detected in the current frame. (Using YOLOE-26 for detection)"
                            else:
                                return f"No {obj_text}objects detected in the current frame."
                    
                    # Format response
                    response_lines = []
                    for det in detections_with_colors:
                        class_name = det.get("class", "unknown")
                        color_name = det.get("color_name", "unknown")
                        color_rgb = det.get("color_rgb", (0, 0, 0))
                        
                        response_lines.append(f"{class_name.capitalize()}: {color_name} (RGB: {color_rgb})")
                    
                    response = "\n".join(response_lines)
                    
                    # Add note about YOLOE usage
                    if use_yoloe:
                        if use_text_prompts:
                            response += "\n\n(Using YOLOE-26 with text prompts for targeted object detection)"
                        else:
                            response += "\n\n(Using YOLOE-26 prompt-free mode)"
                    
                    return response
                    
                except Exception as e:
                    import traceback
                    return f"Error detecting colors: {str(e)}\n{traceback.format_exc()}"
            else:
                # Web viewer not running - need to capture frame
                # This should trigger YOLO detection which will handle color detection
                # But for now, return a helpful message
                return "Color detection requires access to the video feed. Please start the tracking viewer first, or ask me to detect objects in the video."
        
        # Check for distance comparison queries - "which car is closer?", "order by distance", etc.
        # SKIP if ontology already routed to distance tool (handles both simple and comparison queries)
        is_distance_comparison = any(keyword in prompt_lower for keyword in DISTANCE_COMPARISON_KEYWORDS) if not skip_early_checks or early_tool_name != "estimate_object_distances" else False
        has_distance_keyword = any(keyword in prompt_lower for keyword in DISTANCE_KEYWORDS + DISTANCE_COMPARISON_KEYWORDS)
        
        if is_distance_comparison and self.web_viewer is not None:
            try:
                # Extract object class from prompt (e.g., "which car" -> "car")
                requested_class = None
                for cls in COMMON_OBJECT_CLASSES:
                    if cls in prompt_lower or f"{cls}s" in prompt_lower:
                        requested_class = cls
                        break
                
                # Get frame from web viewer
                with self.web_viewer.frame_lock:
                    if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                        current_frame = self.web_viewer.raw_frame.copy()
                    elif self.web_viewer.frame is not None:
                        current_frame = self.web_viewer.frame.copy()
                    else:
                        current_frame = None
                
                if current_frame is None:
                    return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                
                # Run distance estimation
                from tools.distance_tool import DistanceTool
                distance_tool = DistanceTool()
                
                viewer_confidence = getattr(self.web_viewer, 'confidence_threshold', 0.15)
                result = await distance_tool.execute({
                    "frame": current_frame,
                    "confidence_threshold": viewer_confidence
                })
                
                # Filter by class if specified, otherwise get all objects with distances
                detections_with_distances = result.get("detections_with_distances", [])
                
                # Filter objects with valid distances and by class if specified
                valid_distances = []
                for det in detections_with_distances:
                    if det.get("distance_meters") is not None:
                        # Filter by class if specified
                        if requested_class is None or det.get("class") == requested_class:
                            valid_distances.append(det)
                
                if not valid_distances:
                    obj_text = f"{requested_class} " if requested_class else ""
                    return f"No {obj_text}objects with valid distances detected in the current frame."
                
                # Sort by distance (closest to farthest)
                valid_distances.sort(key=lambda x: x["distance_meters"])
                
                # Format response
                response_lines = []
                if len(valid_distances) == 1:
                    obj = valid_distances[0]
                    response_lines.append(
                        f"The {obj['class']} is approximately {obj['distance_meters']:.1f} meters "
                        f"({obj['distance_feet']:.1f} feet) away."
                    )
                else:
                    # Show ordered list from closest to farthest
                    response_lines.append(f"Objects ordered from closest to farthest ({len(valid_distances)} total):\n")
                    for i, obj in enumerate(valid_distances, 1):
                        obj_class = obj.get("class", "unknown")
                        distance_m = obj.get("distance_meters")
                        distance_ft = obj.get("distance_feet")
                        response_lines.append(
                            f"  {i}. {obj_class.capitalize()}: {distance_m:.1f} meters ({distance_ft:.1f} feet)"
                        )
                    
                    # Also mention which one is closest
                    closest = valid_distances[0]
                    response_lines.append(
                        f"\nThe closest {requested_class if requested_class else 'object'} is "
                        f"the {closest['class']} at {closest['distance_meters']:.1f} meters "
                        f"({closest['distance_feet']:.1f} feet)."
                    )
                
                return "\n".join(response_lines)
                
            except Exception as e:
                import traceback
                return f"Error comparing distances: {str(e)}\n{traceback.format_exc()}"
        
        # Check if user wants to print/get track data in chat (not GUI)
        # BUT skip this if they're asking for distance - distance queries should calculate distance
        distance_keywords_in_prompt = any(word in prompt_lower for word in DISTANCE_KEYWORDS)
        
        # Check if user wants to see track data in chat (but not if asking for distance)
        wants_data_in_chat = any(keyword in prompt_lower for keyword in PRINT_DATA_KEYWORDS)
        mentions_track_id = any(keyword in prompt_lower for keyword in TRACK_ID_KEYWORDS)
        
        # Only return track data if NOT asking for distance
        if wants_data_in_chat and mentions_track_id and not distance_keywords_in_prompt and self.web_viewer is not None:
            # Extract track ID from prompt (look for "ID:1", "track 1", "ID 1", etc.)
            track_id_match = re.search(TRACK_ID_PATTERN, prompt_lower)
            if track_id_match:
                requested_id = int(track_id_match.group(1))
                
                # Get track data from web viewer
                with self.web_viewer.tracks_lock:
                    tracks = list(self.web_viewer.tracks_data.values())
                
                # Find the requested track
                track_data = None
                for track in tracks:
                    if track.get("track_id") == requested_id:
                        track_data = track
                        break
                
                if track_data:
                    # Format track data for chat
                    response = f"Track ID: {track_data['track_id']}\n"
                    response += f"Class: {track_data['class']}\n"
                    response += f"Velocity: vx={track_data['velocity']['vx']}, vy={track_data['velocity']['vy']} px/s\n"
                    response += f"Distance Moved: {track_data['distance_moved_pixels']} px\n"
                    response += f"Average Speed: {track_data['average_speed_pixels_per_second']} px/s\n"
                    response += f"First Seen: {track_data['first_seen']}\n"
                    response += f"Last Seen: {track_data['last_seen']}"
                    return response
                else:
                    return f"Track ID {requested_id} not found in current tracking data. Available track IDs: {[t.get('track_id') for t in tracks] if tracks else 'none'}"
        
        # Check if user wants to view detections/tracking in GUI
        is_gui_request = any(keyword in prompt_lower for keyword in VIEW_GUI_KEYWORDS)
        is_tracking_request = any(keyword in prompt_lower for keyword in [
            "track", "tracking", "trajectory", "trajectories"
        ])
        
        # If user wants to view tracking, open tracking viewer (web-based GUI)
        if is_gui_request and is_tracking_request:
            try:
                from tools.tracking_web_viewer import TrackingWebViewer
                import threading
                
                # Extract duration if specified
                duration = self._extract_duration(prompt) if self._extract_duration(prompt) > 0 else 0
                
                # Get device IP for display
                import socket
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    device_ip = s.getsockname()[0]
                    s.close()
                except:
                    device_ip = "localhost"
                
                viewer = TrackingWebViewer(
                    model_path=str(MODELS_DIR / "yolo26n-seg.pt"),
                    device=0,
                    confidence_threshold=0.25,
                    use_gstreamer=True,
                    port=8080
                )
                
                def run_viewer():
                    viewer.run(duration_seconds=duration)
                
                print("[AGENT] Starting web-based tracking viewer...")
                if duration > 0:
                    print(f"  Will run for {duration} seconds.")
                print(f"  Open browser to: http://{device_ip}:8080")
                
                viewer_thread = threading.Thread(target=run_viewer, daemon=True)
                viewer_thread.start()
                
                response_msg = f"I've started the web-based tracking viewer. Open your browser to http://{device_ip}:8080 to see the live video stream and track data. The viewer shows bounding boxes, track IDs, trajectories, velocity, distance moved, and more."
                if duration > 0:
                    response_msg += f" The viewer will run for {duration} seconds."
                
                return response_msg
            except ImportError as e:
                # Fallback to desktop viewer if Flask not available
                from tools.tracking_viewer import run_tracking_viewer
                import threading
                
                duration = self._extract_duration(prompt) if self._extract_duration(prompt) > 0 else 0
                
                def run_viewer():
                    run_tracking_viewer(
                        model_path=str(MODELS_DIR / "yolo26n-seg.pt"),
                        device=0,
                        confidence_threshold=0.25,
                        use_gstreamer=True,
                        duration_seconds=duration,
                        show_trajectories=True
                    )
                
                print("[AGENT] Opening tracking viewer window (desktop mode)...")
                print("  Press 'q' in the video window to close it.")
                if duration > 0:
                    print(f"  Will run for {duration} seconds.")
                
                viewer_thread = threading.Thread(target=run_viewer, daemon=True)
                viewer_thread.start()
                
                response_msg = "I've opened the tracking viewer window. You should see a live video feed with object tracking. Press 'q' in the video window to close it."
                if duration > 0:
                    response_msg += f" The viewer will automatically close after {duration} seconds."
                
                return response_msg
        
        # If user wants to view detections (without tracking), open detection viewer
        elif is_gui_request:
            from tools.yolo_live import run_live_detection
            import threading
            
            # Extract duration if specified
            duration = self._extract_duration(prompt) if self._extract_duration(prompt) > 0 else 0
            
            def run_viewer():
                # Note: yolo_live doesn't support duration parameter, but we can add it if needed
                # For now, just run until 'q' is pressed
                run_live_detection(
                    device=0,
                    confidence_threshold=0.25,
                    use_gstreamer=True
                )
            
            print("[AGENT] Opening detection viewer window...")
            print("  Press 'q' in the video window to close it.")
            
            viewer_thread = threading.Thread(target=run_viewer, daemon=True)
            viewer_thread.start()
            
            response_msg = "I've opened the detection viewer window. You should see a live video feed with real-time object detection overlays showing bounding boxes and labels. Press 'q' in the video window to close it."
            
            return response_msg
        
        # STEP 1: Fast keyword-based routing using JSON ontology
        # Use early ontology result if available, otherwise route again
        ontology_result = early_ontology_result
        tool_name = early_tool_name
        router_result = None
        
        if not ontology_result and self.query_ontology:
            try:
                ontology_result = self.query_ontology.quick_route(prompt)
                if self.verbose:
                    print(f"[AGENT] Ontology match: {ontology_result}")
                
                # Extract tool and object class from ontology
                tool_name = ontology_result.get("tool")
                object_class = ontology_result.get("object_class")
                query_type = ontology_result.get("query_type")
                needs_llm = ontology_result.get("needs_llm", True)
                direct_call = ontology_result.get("direct_call", False)
                cached_query = ontology_result.get("cached_query")
                confidence = ontology_result.get("confidence", 0.0)
                
                # AGENTIC VALIDATION: Confidence-based LLM usage
                # High confidence (>0.8): Skip LLM (simple, unambiguous queries - not truly agentic, but fast)
                # Medium confidence (0.3-0.8): LLM validates (lightweight check - maintains agentic nature)
                # Low confidence (<0.3): Full LLM reasoning (truly agentic)
                
                use_llm_validation = False
                if confidence > 0.8:
                    # Very high confidence - skip LLM entirely (simple queries like "how many objects")
                    needs_llm = False
                    if self.verbose:
                        print(f"[AGENT] Very high confidence ({confidence:.2f}) - skipping LLM (simple query)")
                elif confidence > 0.3:
                    # Medium-high confidence - use lightweight LLM validation (maintains agentic nature)
                    needs_llm = True  # Will use lightweight validation
                    use_llm_validation = True
                    if self.verbose:
                        print(f"[AGENT] Medium confidence ({confidence:.2f}) - using LLM validation (agentic check)")
                else:
                    # Low confidence - full LLM reasoning (truly agentic)
                    needs_llm = True
                    if self.verbose:
                        print(f"[AGENT] Low confidence ({confidence:.2f}) - using full LLM reasoning")
                
                # SPECIAL CASE: Color queries with direct_call should skip LLM entirely
                # Color detection uses ontology + YOLOE + segmentation masks (no LLM needed)
                if tool_name == "color_detection" and direct_call:
                    needs_llm = False  # Override - color detection doesn't need LLM
                    use_llm_validation = False
                    if self.verbose:
                        print(f"[AGENT] Color detection with direct_call - skipping LLM, using ontology + YOLOE + masks")
                
                # If ontology found a cached query or match and doesn't need LLM, use it directly
                if tool_name and (not needs_llm or direct_call):
                    if self.verbose:
                        if cached_query:
                            print(f"[AGENT] Using cached query '{cached_query}' (direct tool call, no LLM): {tool_name}")
                        else:
                            print(f"[AGENT] Using fast ontology match (no LLM needed): {tool_name}")
                    # Store ontology result for later use
                    router_result = {
                        "tool": tool_name,
                        "query_type": query_type,
                        "object_class": object_class,
                        "reasoning": f"cached_query:{cached_query}" if cached_query else "ontology_keyword_match",
                        "direct_call": direct_call
                    }
                elif needs_llm and self.query_router:
                    # AGENTIC VALIDATION: Use LLM to validate/confirm tool choice
                    if use_llm_validation:
                        # Lightweight validation (maintains agentic nature, but faster than full reasoning)
                        if self.verbose:
                            print(f"[AGENT] Using lightweight LLM validation for agentic tool confirmation...")
                        try:
                            validation_result = await self.query_router.validate_tool_choice(
                                prompt, tool_name, object_class
                            )
                            # Override with validated tool if LLM suggests different
                            validated_tool = validation_result.get("tool")
                            validated_object_class = validation_result.get("object_class")
                            validation_confidence = validation_result.get("confidence", 0.7)
                            
                            if validated_tool and validated_tool != tool_name:
                                if self.verbose:
                                    print(f"[AGENT] LLM validation overrode tool: {tool_name} → {validated_tool}")
                                tool_name = validated_tool
                                object_class = validated_object_class or object_class
                            
                            router_result = {
                                "tool": tool_name,
                                "query_type": query_type,
                                "object_class": object_class,
                                "reasoning": f"ontology_match + llm_validation (confidence: {validation_confidence:.2f})",
                                "direct_call": False
                            }
                        except Exception as e:
                            if self.verbose:
                                print(f"[AGENT] LLM validation error: {e}, using ontology suggestion")
                            # Fallback to ontology suggestion
                            router_result = {
                                "tool": tool_name,
                                "query_type": query_type,
                                "object_class": object_class,
                                "reasoning": "ontology_keyword_match (validation failed)",
                                "direct_call": False
                            }
                    else:
                        # STEP 2: Use LLM reasoning only if ontology says it's needed
                        if self.verbose:
                            print(f"[AGENT] Ontology requires LLM reasoning, calling router...")
                        try:
                            # Pass detection context to router if available
                            router_result = await self.query_router.understand_query(
                                prompt, 
                                detection_context=current_detections if current_detections else None
                            )
                            if self.verbose:
                                print(f"[AGENT] Router result: {router_result}")
                            
                            # Check if router says this is NOT a vision query
                            router_query_type = router_result.get("query_type", "")
                            if router_query_type == "general" or not router_result.get("tool"):
                                # Non-vision query - answer directly
                                if self.verbose:
                                    print("[AGENT] Router identified non-vision query, answering directly...")
                                try:
                                    response = await self.llm.ainvoke(prompt)
                                    return self._truncate_to_words(response.strip(), max_words=20)
                                except Exception as e:
                                    if self.verbose:
                                        print(f"[AGENT] Direct LLM call failed: {e}")
                                    return f"I encountered an error: {str(e)}"
                            
                            # Override tool selection if router suggests a tool
                            if router_result.get("tool"):
                                tool_name_from_router = router_result.get("tool")
                                # Map router tool names to our tool names
                                tool_name_mapping = {
                                    "yolo_object_detection": "yolo_object_detection",
                                    "get_detection_statistics": "get_detection_statistics",
                                    "estimate_object_distances": "estimate_object_distances",
                                    "track_objects": "track_objects",
                                    "color_detection": None,  # Handled separately
                                    "gui_viewer": None  # Handled separately
                                }
                                mapped_tool = tool_name_mapping.get(tool_name_from_router)
                                if mapped_tool:
                                    tool_name = mapped_tool
                        except Exception as e:
                            if self.verbose:
                                print(f"[AGENT] Query router error: {e}")
                            router_result = None
            except Exception as e:
                if self.verbose:
                    print(f"[AGENT] Query ontology error: {e}")
                ontology_result = None
        
        # STEP 3: Fallback to keyword-based tool selection if ontology/router didn't work
        if tool_name is None:
            tool_name = await self._should_use_tool(prompt)
        
        if tool_name and tool_name in self.tool_map:
            if self.verbose:
                print(f"[AGENT] Agent decided to use tool: {tool_name}")
            
            # Handle tracking tool
            if tool_name == "track_objects":
                prompt_lower = prompt.lower()
                is_webcam_request = any(word in prompt_lower for word in [
                    "webcam", "camera", "cam", "video stream", "video", "stream", "live"
                ])
                
                # If web viewer is already running and user wants webcam tracking,
                # tell them tracking is already happening in the GUI
                # BUT skip this if it's a detection history query (already handled above)
                is_history_query = bool(re.search(r'(have\s+any|history\s+of|give\s+me\s+the\s+history)', prompt_lower, re.IGNORECASE))
                if is_webcam_request and self.web_viewer is not None and not is_history_query:
                    try:
                        import socket
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        device_ip = s.getsockname()[0]
                        s.close()
                    except:
                        device_ip = "localhost"
                    port = getattr(self.web_viewer, 'port', 8080)
                    return f"Object tracking is already running in the web-based viewer! The web viewer is continuously tracking objects with persistent IDs, velocity, and trajectories. Open your browser to http://{device_ip}:{port} to see the live tracking with bounding boxes, track IDs, and trajectory trails in real-time."
                
                image_path = "webcam" if is_webcam_request else await self._extract_image_path(prompt)
                
                if not image_path:
                    return "I can track objects in images or video streams. Please provide a path to an image file, or ask me to use the webcam/video stream."
                
                # Extract duration from prompt (support "10", "ten", etc., max 60 seconds)
                duration_seconds = self._extract_duration(prompt)
                
                try:
                    tool_result = await track_objects.ainvoke({
                        "image_path": image_path,
                        "camera_device": 0,
                        "confidence_threshold": 0.15,  # Lower threshold for better small object detection
                        "track_history_frames": 30,
                        "duration_seconds": duration_seconds
                    })
                    return tool_result
                except Exception as e:
                    return f"Error running tracking tool: {str(e)}"
            
            # Handle statistics tool
            elif tool_name == "get_detection_statistics":
                try:
                    tool_result = await get_detection_statistics.ainvoke({
                        "reset": False
                    })
                    
                    # Format using Response Formatting Template for human sentence
                    from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                    formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                        user_query=prompt,
                        tool_results=f"STATISTICS: {tool_result}"
                    )
                    
                    # Extract system and user messages
                    system_msg = None
                    user_msg = None
                    for msg in formatting_prompt:
                        if msg.type == "system":
                            system_msg = msg.content
                        elif msg.type == "human":
                            user_msg = msg.content
                    
                    # Call LLM to format response
                    if hasattr(self.llm, '_acall'):
                        response = await self.llm._acall(user_msg, system_prompt=system_msg)
                    else:
                        import asyncio
                        response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                        if not isinstance(response, str):
                            response = response.content if hasattr(response, 'content') else str(response)
                    
                    return self._truncate_to_words(response.strip(), max_words=20)
                except Exception as e:
                    return f"Error running statistics tool: {str(e)}"
            
            # Handle distance estimation tool
            elif tool_name == "estimate_object_distances":
                prompt_lower = prompt.lower()
                
                # Use query router for spatial relationship understanding (e.g., "person wrt dog")
                distance_understanding = None
                if self.query_router and router_result:
                    try:
                        distance_understanding = await self.query_router.understand_distance_query(prompt)
                        if self.verbose:
                            print(f"[AGENT] Distance understanding: {distance_understanding}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[AGENT] Distance query router error: {e}")
                
                # Check if this is a follow-up query with just an object class (e.g., "do it for bench")
                # These should be treated as distance queries, not general prompts
                is_follow_up_class_query = any(pattern in prompt_lower for pattern in FOLLOW_UP_PATTERNS) and \
                                          any(cls in prompt_lower for cls in COMMON_OBJECT_CLASSES) and \
                                          "distance" not in prompt_lower
                
                # If it's a follow-up class query, ensure we treat it as a distance query
                if is_follow_up_class_query:
                    # Continue with distance estimation logic below
                    pass
                
                # Check for webcam/video keywords, including "me" in context of distance queries
                is_webcam_request = any(word in prompt_lower for word in [
                    "webcam", "camera", "cam", "video stream", "video", "stream", "live",
                    "me", "myself", "here"  # User referring to themselves/camera position
                ]) or is_follow_up_class_query  # Follow-up queries are also webcam requests
                
                # Default to webcam if no explicit image path found and it's a distance query
                image_path = "webcam" if is_webcam_request else await self._extract_image_path(prompt)
                
                # If still no image path, default to webcam for distance queries (user likely wants current view)
                if not image_path and tool_name == "estimate_object_distances":
                    image_path = "webcam"
                    is_webcam_request = True
                
                # If webcam requested and web viewer is running, use its frame instead of opening camera
                if is_webcam_request and self.web_viewer is not None:
                    # Get raw frame from web viewer (without annotations) for better detection
                    with self.web_viewer.frame_lock:
                        # Prefer raw_frame if available (no annotations), fall back to annotated frame
                        if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                            current_frame = self.web_viewer.raw_frame.copy()
                        elif self.web_viewer.frame is not None:
                            current_frame = self.web_viewer.frame.copy()
                        else:
                            current_frame = None
                    
                    if current_frame is None:
                        return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                    
                    # Use the web viewer's frame for distance estimation
                    try:
                        # Extract requested object class and track ID from prompt
                        prompt_lower = prompt.lower()
                        
                        # Check for distance between two track IDs (e.g., "Distance from ID:2 to ID:3")
                        two_track_id_pattern = re.search(r'(?:distance|dist|from).*?(?:id|track)[:\s]*(\d+).*?(?:to|and).*?(?:id|track)[:\s]*(\d+)', prompt_lower)
                        if two_track_id_pattern and self.web_viewer is not None:
                            track_id1 = int(two_track_id_pattern.group(1))
                            track_id2 = int(two_track_id_pattern.group(2))
                            
                            # Get track metadata from web viewer
                            from tools.spatial_utils import compute_distance
                            
                            # Access track metadata (stored in tracking loop, accessible via lock)
                            track_metadata = {}
                            if hasattr(self.web_viewer, 'track_metadata_lock'):
                                with self.web_viewer.track_metadata_lock:
                                    track_metadata = getattr(self.web_viewer, 'track_metadata_history', {})
                            else:
                                # Fallback: try to get from tracks_data
                                with self.web_viewer.tracks_lock:
                                    tracks = list(self.web_viewer.tracks_data.values())
                                    # Build metadata from tracks_data
                                    for track in tracks:
                                        track_id = track.get("track_id")
                                        if track_id:
                                            track_metadata[track_id] = {
                                                "last_position": {"x": 0, "y": 0},  # Approximate
                                                "class": track.get("class", "unknown")
                                            }
                            
                            # Get positions for both tracks
                            track1_info = track_metadata.get(track_id1)
                            track2_info = track_metadata.get(track_id2)
                            
                            if not track1_info:
                                return self._truncate_to_words(f"Track ID {track_id1} not found.", max_words=20)
                            if not track2_info:
                                return self._truncate_to_words(f"Track ID {track_id2} not found.", max_words=20)
                            
                            # Get last positions
                            pos1 = track1_info.get("last_position", {})
                            pos2 = track2_info.get("last_position", {})
                            
                            if not pos1 or not pos2 or "x" not in pos1 or "y" not in pos1 or "x" not in pos2 or "y" not in pos2:
                                # Try to get positions from current frame tracking
                                if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                                    track_model = self.web_viewer.model
                                    track_results = track_model.track(current_frame, conf=0.15, persist=True, verbose=False)
                                    
                                    pos1 = None
                                    pos2 = None
                                    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                                        for box, track_id in zip(track_results[0].boxes, track_results[0].boxes.id):
                                            if int(track_id) == track_id1:
                                                bbox = box.xyxy[0].tolist()
                                                pos1 = {"x": (bbox[0] + bbox[2]) / 2, "y": (bbox[1] + bbox[3]) / 2}
                                            elif int(track_id) == track_id2:
                                                bbox = box.xyxy[0].tolist()
                                                pos2 = {"x": (bbox[0] + bbox[2]) / 2, "y": (bbox[1] + bbox[3]) / 2}
                                    
                                    if not pos1 or not pos2:
                                        return self._truncate_to_words(f"Could not find positions for one or both tracks in current frame.", max_words=20)
                                else:
                                    return self._truncate_to_words(f"Position data not available for one or both tracks.", max_words=20)
                            
                            # Calculate pixel distance
                            pixel_distance = compute_distance((pos1["x"], pos1["y"]), (pos2["x"], pos2["y"]))
                            
                            # Get frame to estimate real-world distance
                            # We'll use the average distance from camera to both objects
                            if current_frame is not None:
                                # Estimate real-world distance using distance tool
                                from tools.distance_tool import DistanceTool
                                distance_tool = DistanceTool()
                                
                                # Get bounding boxes for both tracks from current frame
                                if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                                    track_model = self.web_viewer.model
                                    track_results = track_model.track(current_frame, conf=0.15, persist=True, verbose=False)
                                    
                                    bbox1 = None
                                    bbox2 = None
                                    class1 = track1_info.get("class", "unknown")
                                    class2 = track2_info.get("class", "unknown")
                                    
                                    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                                        for box, track_id in zip(track_results[0].boxes, track_results[0].boxes.id):
                                            if int(track_id) == track_id1:
                                                bbox1 = box.xyxy[0].tolist()
                                                cls_id = int(box.cls)
                                                class1 = track_model.names[cls_id]
                                            elif int(track_id) == track_id2:
                                                bbox2 = box.xyxy[0].tolist()
                                                cls_id = int(box.cls)
                                                class2 = track_model.names[cls_id]
                                    
                                    # Calculate distances from camera to each object
                                    image_height = current_frame.shape[0]
                                    dist1_m = distance_tool._estimate_distance(bbox1, class1, image_height) if bbox1 else None
                                    dist2_m = distance_tool._estimate_distance(bbox2, class2, image_height) if bbox2 else None
                                    
                                    if dist1_m and dist2_m:
                                        # Approximate distance between objects
                                        # Use pixel distance and average depth for rough conversion
                                        avg_depth = (dist1_m + dist2_m) / 2.0
                                        
                                        # Rough conversion: pixel distance to meters (approximate)
                                        # Assume ~100 pixels per meter at average distance (rough estimate)
                                        pixels_per_meter = 100.0 / max(avg_depth, 1.0)  # Rough estimate
                                        estimated_distance_m = pixel_distance / pixels_per_meter
                                        estimated_distance_ft = estimated_distance_m * 3.28084
                                        
                                        return self._truncate_to_words(
                                            f"Distance from {class1} (ID:{track_id1}) to {class2} (ID:{track_id2}): "
                                            f"{estimated_distance_m:.2f}m ({estimated_distance_ft:.1f}ft). "
                                            f"Object 1: {dist1_m:.1f}m. Object 2: {dist2_m:.1f}m.",
                                            max_words=20
                                        )
                            
                            # Fallback: return pixel distance only
                            return self._truncate_to_words(
                                f"Distance from {track1_info.get('class', 'object')} (ID:{track_id1}) to "
                                f"{track2_info.get('class', 'object')} (ID:{track_id2}): {pixel_distance:.1f}px.",
                                max_words=20
                            )
                        
                        # Extract track ID if specified (e.g., "ID:9", "track 9") - SINGLE track distance
                        requested_track_id = None
                        track_id_match = re.search(TRACK_ID_PATTERN, prompt_lower)
                        if track_id_match:
                            requested_track_id = int(track_id_match.group(1))
                        
                        # If track ID specified, use tracking data from web viewer directly
                        if requested_track_id and self.web_viewer is not None:
                            # Get tracking data with bounding boxes from web viewer's current tracking state
                            # We need to access the tracking loop's metadata which has bounding boxes
                            # For now, let's use the distance tool with the frame and match by position
                            # But better: access track metadata from web viewer if available
                            
                            # Get track metadata from web viewer (if accessible)
                            # Otherwise, use distance tool and match by bounding box overlap
                            from tools.distance_tool import DistanceTool
                            distance_tool = DistanceTool()
                            
                            # Get tracking data to find bounding box for this track ID
                            # The web viewer stores track metadata in _tracking_loop, but it's not directly accessible
                            # We'll use distance tool and then match detections to track IDs using bounding box overlap
                            
                            # First, get tracking information from web viewer's track data
                            with self.web_viewer.tracks_lock:
                                tracks = list(self.web_viewer.tracks_data.values())
                            
                            requested_track_data = None
                            for track in tracks:
                                if track.get("track_id") == requested_track_id:
                                    requested_track_data = track
                                    break
                            
                            if requested_track_data:
                                requested_class = requested_track_data.get("class")
                            else:
                                # Track ID not found in current tracks
                                return f"Track ID {requested_track_id} not found in current tracking data. Available track IDs: {[t.get('track_id') for t in tracks] if tracks else 'none'}"
                            
                            # Run distance estimation on the frame
                            result = await distance_tool.execute({
                                "frame": current_frame,
                                "confidence_threshold": 0.15
                            })
                            
                            # Match detections to track ID by using the web viewer's own tracking model
                            # This ensures we use the same tracker instance and consistent track IDs
                            # Reuse the web viewer's model instead of creating a new one
                            if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                                # Use web viewer's model to maintain tracker state consistency
                                track_model = self.web_viewer.model
                            else:
                                # Fallback: create new model (less ideal for track ID consistency)
                                from ultralytics import YOLO
                                track_model = YOLO(str(MODELS_DIR / "yolo26n-seg.pt"))
                            
                            # Run tracking with the same model instance
                            track_results = track_model.track(
                                current_frame,
                                conf=0.15,
                                persist=True,  # Maintain track persistence
                                verbose=False
                            )
                            
                            # Find the bounding box for requested_track_id
                            track_bbox = None
                            track_class = None
                            if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                                for box, track_id in zip(track_results[0].boxes, track_results[0].boxes.id):
                                    if int(track_id) == requested_track_id:
                                        track_bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                                        cls_id = int(box.cls)
                                        track_class = track_model.names[cls_id]
                                        break
                            
                            # If we found the bounding box, calculate distance for it
                            if track_bbox and track_class:
                                # Use distance tool's distance estimation method
                                image_height = current_frame.shape[0]
                                distance_m = distance_tool._estimate_distance(track_bbox, track_class, image_height)
                                
                                if distance_m is not None:
                                    distance_ft = distance_m * 3.28084
                                    return f"The {track_class} (track ID: {requested_track_id}) is approximately {distance_m:.1f} meters ({distance_ft:.1f} feet) away."
                                else:
                                    return f"Track ID {requested_track_id} ({track_class}) is visible, but distance estimation is not available for this object type."
                            else:
                                # Track ID not found in current frame's tracking results
                                # Fall back to class-based matching
                                matching_distances = []
                                for det in result.get("detections_with_distances", []):
                                    if det.get("distance_meters") is not None and det.get("class") == requested_class:
                                        matching_distances.append({
                                            "class": det.get("class"),
                                            "distance_meters": det.get("distance_meters"),
                                            "distance_feet": det.get("distance_feet")
                                        })
                                
                                if matching_distances:
                                    nearest = min(matching_distances, key=lambda x: x["distance_meters"])
                                    return f"The {nearest['class']} (track ID: {requested_track_id}) is approximately {nearest['distance_meters']:.1f} meters ({nearest['distance_feet']:.1f} feet) away."
                                else:
                                    return f"Track ID {requested_track_id} ({requested_class}) is tracked but not detected in the current frame for distance calculation. The object may have moved out of view."
                        
                        # No track ID specified, use normal distance estimation
                        # Use the same confidence threshold and model as the web viewer for consistency
                        from tools.distance_tool import DistanceTool
                        distance_tool = DistanceTool()
                        
                        # Use web viewer's confidence threshold (defaults to 0.50 if not set)
                        viewer_confidence = getattr(self.web_viewer, 'confidence_threshold', 0.15)
                        
                        # If web viewer has a model, we could use it, but distance_tool has its own model
                        # For now, just use the same confidence threshold
                        # The annotated frame should still work for detection
                        result = await distance_tool.execute({
                            "frame": current_frame,
                            "confidence_threshold": viewer_confidence  # Match web viewer's confidence threshold
                        })
                        
                        # Debug: Log what was detected (if verbose)
                        if self.verbose and result.get("detections_with_distances"):
                            print(f"[DEBUG] Distance tool detected {len(result['detections_with_distances'])} objects")
                            for det in result["detections_with_distances"][:3]:  # First 3
                                print(f"  - {det.get('class')}: conf={det.get('confidence')}, dist={det.get('distance_meters')}")
                        
                        # Extract object class from prompt if specified
                        # Also check if this is a follow-up query (e.g., "do it for bench" after asking for distance)
                        requested_class = None
                        for cls in COMMON_OBJECT_CLASSES:
                            if cls in prompt_lower:
                                requested_class = cls
                                break
                        
                        # Handle follow-up queries like "do it for bench" - prioritize object class keywords
                        # Only if no track ID was specified
                        if not requested_track_id and requested_class:
                            # This is a class-based distance request
                            pass  # requested_class is already set
                        
                        # Find matching objects in distance results
                        matching_distances = []
                        for det in result.get("detections_with_distances", []):
                            if det.get("distance_meters") is not None:
                                # If class specified, match it; otherwise accept all
                                if requested_class is None or det.get("class") == requested_class:
                                    matching_distances.append({
                                        "class": det.get("class"),
                                        "distance_meters": det.get("distance_meters"),
                                        "distance_feet": det.get("distance_feet"),
                                        "confidence": det.get("confidence", 0)
                                    })
                        
                        if matching_distances:
                            # Return all detected objects with distances (detailed format like terminal)
                            # Sort by distance (closest to farthest)
                            matching_distances.sort(key=lambda x: x["distance_meters"])
                            
                            response_lines = []
                            if requested_class:
                                # If specific class requested, show all instances of that class
                                response_lines.append(f"Distance to {requested_class}s ({len(matching_distances)} detected):")
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class']}: {obj['distance_meters']:.2f}m ({obj['distance_feet']:.1f}ft) - conf={obj.get('confidence', 0):.2f}"
                                    )
                            else:
                                # Show all detected objects
                                response_lines.append(f"Distances to all detected objects ({len(matching_distances)} total):")
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class']}: {obj['distance_meters']:.2f}m ({obj['distance_feet']:.1f}ft) - conf={obj.get('confidence', 0):.2f}"
                                    )
                            
                            return "\n".join(response_lines)
                        elif result.get("detections_with_distances"):
                            # Objects detected but not matching the request
                            classes = [d.get("class") for d in result.get("detections_with_distances", [])]
                            if requested_class:
                                return f"No {requested_class} detected in the current frame. Detected objects: {', '.join(set(classes))}. Please ensure a {requested_class} is visible."
                            else:
                                return f"Objects detected but distance calculation unavailable. Detected objects: {', '.join(set(classes))}."
                        else:
                            return "No objects detected in the current frame. Please ensure objects are visible to the camera."
                            
                    except Exception as e:
                        import traceback
                        return f"Error estimating distances from web viewer frame: {str(e)}\n{traceback.format_exc()}"
                
                # Otherwise, use normal path (capture from camera or load from file)
                try:
                    tool_result_str = await estimate_object_distances.ainvoke({
                        "image_path": image_path,
                        "camera_device": 0,
                        "confidence_threshold": 0.15  # Lower threshold for better small object detection
                    })
                    
                    # Parse the JSON result to find distances
                    try:
                        tool_result = json.loads(tool_result_str)
                        
                        # Extract requested object class and track ID from prompt
                        prompt_lower = prompt.lower()
                        
                        # Check for distance between two track IDs (e.g., "Distance from ID:2 to ID:3")
                        two_track_id_pattern = re.search(r'(?:distance|dist|from).*?(?:id|track)[:\s]*(\d+).*?(?:to|and).*?(?:id|track)[:\s]*(\d+)', prompt_lower)
                        if two_track_id_pattern and self.web_viewer is not None:
                            track_id1 = int(two_track_id_pattern.group(1))
                            track_id2 = int(two_track_id_pattern.group(2))
                            
                            # Get track metadata from web viewer
                            from tools.spatial_utils import compute_distance
                            
                            # Access track metadata (stored in tracking loop, accessible via lock)
                            track_metadata = {}
                            if hasattr(self.web_viewer, 'track_metadata_lock'):
                                with self.web_viewer.track_metadata_lock:
                                    track_metadata = getattr(self.web_viewer, 'track_metadata_history', {})
                            else:
                                # Fallback: try to get from tracks_data
                                with self.web_viewer.tracks_lock:
                                    tracks = list(self.web_viewer.tracks_data.values())
                                    # Build metadata from tracks_data
                                    for track in tracks:
                                        track_id = track.get("track_id")
                                        if track_id:
                                            track_metadata[track_id] = {
                                                "last_position": {"x": 0, "y": 0},  # Approximate
                                                "class": track.get("class", "unknown")
                                            }
                            
                            # Get positions for both tracks
                            track1_info = track_metadata.get(track_id1)
                            track2_info = track_metadata.get(track_id2)
                            
                            if not track1_info:
                                return f"Track ID {track_id1} not found in tracking data."
                            if not track2_info:
                                return f"Track ID {track_id2} not found in tracking data."
                            
                            # Get last positions
                            pos1 = track1_info.get("last_position", {})
                            pos2 = track2_info.get("last_position", {})
                            
                            if not pos1 or not pos2:
                                return f"Position data not available for one or both tracks."
                            
                            # Calculate pixel distance
                            pixel_distance = compute_distance((pos1["x"], pos1["y"]), (pos2["x"], pos2["y"]))
                            
                            # Get frame to estimate real-world distance
                            # We'll use the average distance from camera to both objects
                            current_frame = None
                            if hasattr(self.web_viewer, 'get_latest_raw_frame'):
                                current_frame = self.web_viewer.get_latest_raw_frame()
                            elif hasattr(self.web_viewer, 'raw_frame'):
                                with self.web_viewer.frame_lock:
                                    current_frame = self.web_viewer.raw_frame.copy() if self.web_viewer.raw_frame is not None else None
                            
                            if current_frame is not None:
                                # Estimate real-world distance using distance tool
                                from tools.distance_tool import DistanceTool
                                distance_tool = DistanceTool()
                                
                                # Get bounding boxes for both tracks from current frame
                                if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                                    track_model = self.web_viewer.model
                                    track_results = track_model.track(current_frame, conf=0.15, persist=True, verbose=False)
                                    
                                    bbox1 = None
                                    bbox2 = None
                                    class1 = track1_info.get("class", "unknown")
                                    class2 = track2_info.get("class", "unknown")
                                    
                                    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                                        for box, track_id in zip(track_results[0].boxes, track_results[0].boxes.id):
                                            if int(track_id) == track_id1:
                                                bbox1 = box.xyxy[0].tolist()
                                            elif int(track_id) == track_id2:
                                                bbox2 = box.xyxy[0].tolist()
                                    
                                    # Calculate distances from camera to each object
                                    image_height = current_frame.shape[0]
                                    dist1_m = distance_tool._estimate_distance(bbox1, class1, image_height) if bbox1 else None
                                    dist2_m = distance_tool._estimate_distance(bbox2, class2, image_height) if bbox2 else None
                                    
                                    if dist1_m and dist2_m:
                                        # Use law of cosines to estimate distance between objects
                                        # We need the angle between the two objects from camera's perspective
                                        # For simplicity, use pixel distance as approximation
                                        # More accurate: use depth estimation if available
                                        
                                        # Approximate: assume objects are at similar depth
                                        # Distance between objects ≈ pixel_distance * (avg_depth / focal_length_pixels)
                                        # For now, provide pixel distance and estimated real-world distance
                                        avg_depth = (dist1_m + dist2_m) / 2.0
                                        
                                        # Rough conversion: pixel distance to meters (approximate)
                                        # This is a simplified conversion - actual requires camera calibration
                                        # Assume ~100 pixels per meter at average distance (rough estimate)
                                        pixels_per_meter = 100.0 / max(avg_depth, 1.0)  # Rough estimate
                                        estimated_distance_m = pixel_distance / pixels_per_meter
                                        estimated_distance_ft = estimated_distance_m * 3.28084
                                        
                                        return self._truncate_to_words(
                                            f"Distance from {class1} (ID:{track_id1}) to {class2} (ID:{track_id2}): "
                                            f"approximately {estimated_distance_m:.2f}m ({estimated_distance_ft:.1f}ft) "
                                            f"(pixel distance: {pixel_distance:.1f}px). "
                                            f"Object 1 is {dist1_m:.1f}m away, Object 2 is {dist2_m:.1f}m away.",
                                            max_words=20
                                        )
                            
                            # Fallback: return pixel distance only
                            return self._truncate_to_words(
                                f"Distance from {track1_info.get('class', 'object')} (ID:{track_id1}) to "
                                f"{track2_info.get('class', 'object')} (ID:{track_id2}): {pixel_distance:.1f} pixels.",
                                max_words=20
                            )
                        
                        # Extract track ID if specified (single track distance query)
                        requested_track_id = None
                        track_id_match = re.search(r'(?:id|track)[:\s]*(\d+)', prompt_lower)
                        if track_id_match:
                            requested_track_id = int(track_id_match.group(1))
                        
                        # Extract object class from prompt
                        requested_class = None
                        for cls in COMMON_OBJECT_CLASSES:
                            if cls in prompt_lower:
                                requested_class = cls
                                break
                        
                        # Find matching objects in distance results
                        matching_distances = []
                        for det in tool_result.get("detections_with_distances", []):
                            if det.get("distance_meters") is not None:
                                # If class specified, match it; otherwise accept all
                                if requested_class is None or det.get("class") == requested_class:
                                    matching_distances.append({
                                        "class": det.get("class"),
                                        "distance_meters": det.get("distance_meters"),
                                        "distance_feet": det.get("distance_feet"),
                                        "confidence": det.get("confidence", 0)
                                    })
                        
                        if matching_distances:
                            # Return all detected objects with distances (detailed format like terminal)
                            # Sort by distance (closest to farthest)
                            matching_distances.sort(key=lambda x: x["distance_meters"])
                            
                            response_lines = []
                            if requested_track_id:
                                # Track ID specified - show that specific track
                                matching_track = next((d for d in matching_distances if d.get("track_id") == requested_track_id), None)
                                if matching_track:
                                    return f"Track ID {requested_track_id} ({matching_track['class']}): {matching_track['distance_meters']:.2f}m ({matching_track['distance_feet']:.1f}ft) - conf={matching_track.get('confidence', 0):.2f}"
                                else:
                                    return f"Track ID {requested_track_id} not found in distance results."
                            elif requested_class:
                                # If specific class requested, show all instances of that class
                                response_lines.append(f"Distance to {requested_class}s ({len(matching_distances)} detected):")
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class']}: {obj['distance_meters']:.2f}m ({obj['distance_feet']:.1f}ft) - conf={obj.get('confidence', 0):.2f}"
                                    )
                            else:
                                # Show all detected objects
                                response_lines.append(f"Distances to all detected objects ({len(matching_distances)} total):")
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class']}: {obj['distance_meters']:.2f}m ({obj['distance_feet']:.1f}ft) - conf={obj.get('confidence', 0):.2f}"
                                    )
                            
                            # Format using Response Formatting Template for human sentence
                            from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                            formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                                user_query=prompt,
                                tool_results=f"DISTANCE: {chr(10).join(response_lines)}"
                            )
                            
                            # Extract system and user messages
                            system_msg = None
                            user_msg = None
                            for msg in formatting_prompt:
                                if msg.type == "system":
                                    system_msg = msg.content
                                elif msg.type == "human":
                                    user_msg = msg.content
                            
                            # Call LLM to format response
                            if hasattr(self.llm, '_acall'):
                                response = await self.llm._acall(user_msg, system_prompt=system_msg)
                            else:
                                import asyncio
                                response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                                if not isinstance(response, str):
                                    response = response.content if hasattr(response, 'content') else str(response)
                            
                            return self._truncate_to_words(response.strip(), max_words=20)
                        elif tool_result.get("detections_with_distances"):
                            # Objects detected but not matching the request
                            classes = [d.get("class") for d in tool_result.get("detections_with_distances", [])]
                            if requested_class:
                                return f"No {requested_class} detected in the current frame. Detected objects: {', '.join(set(classes))}. Please ensure a {requested_class} is visible."
                            else:
                                return f"Objects detected but distance calculation unavailable. Detected objects: {', '.join(set(classes))}."
                        else:
                            return "No objects detected in the current frame. Please ensure objects are visible to the camera."
                    except json.JSONDecodeError:
                        # If parsing fails, format the raw result using Response Formatting Template
                        from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                        formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                            user_query=prompt,
                            tool_results=f"DISTANCE: {tool_result_str}"
                        )
                        
                        # Extract system and user messages
                        system_msg = None
                        user_msg = None
                        for msg in formatting_prompt:
                            if msg.type == "system":
                                system_msg = msg.content
                            elif msg.type == "human":
                                user_msg = msg.content
                        
                        # Call LLM to format response
                        if hasattr(self.llm, '_acall'):
                            response = await self.llm._acall(user_msg, system_prompt=system_msg)
                        else:
                            import asyncio
                            response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                            if not isinstance(response, str):
                                response = response.content if hasattr(response, 'content') else str(response)
                        
                        return self._truncate_to_words(response.strip(), max_words=20)
                        
                except Exception as e:
                    return f"Error running distance estimation tool: {str(e)}"
            
            # Handle YOLO detection tool
            elif tool_name == "yolo_object_detection":
                # Use query router for semantic understanding (e.g., "sweater" → "person")
                detection_understanding = None
                if self.query_router:
                    try:
                        detection_understanding = await self.query_router.understand_detection_query(prompt)
                        if self.verbose:
                            print(f"[AGENT] Detection understanding: {detection_understanding}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[AGENT] Detection query router error: {e}")
                
                # Check if this is an environment inference query
                is_environment_query = (
                    detection_understanding and 
                    detection_understanding.get("is_environment_query", False)
                ) or any(keyword in prompt.lower() for keyword in [
                    "what kind of environment", "what kind of place", "what environment",
                    "what type of environment", "based on objects", "what kind of location",
                    "what setting", "what scene"
                ])
                
                # Handle environment inference query
                if is_environment_query and self.web_viewer is not None:
                    try:
                        # Get frame from web viewer
                        with self.web_viewer.frame_lock:
                            if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                                current_frame = self.web_viewer.raw_frame.copy()
                            elif self.web_viewer.frame is not None:
                                current_frame = self.web_viewer.frame.copy()
                            else:
                                current_frame = None
                        
                        if current_frame is None:
                            return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                        
                        # FORCE YOLOE-26 usage for environment detection
                        from tools.yolo_utils import load_yolo_model
                        import tempfile
                        import cv2
                        import os
                        
                        # FORCE YOLOE-26 - no fallback to regular YOLO26
                        # Check if YOLOE-26 model is available (with text prompt support)
                        # Prefer non-prompt-free versions for text prompting
                        yoloe_models_with_prompts = [
                            str(MODELS_DIR / "yoloe-26n-seg.pt"),
                            str(MODELS_DIR / "yoloe-26s-seg.pt"),
                            str(MODELS_DIR / "yoloe-26m-seg.pt"),
                            str(MODELS_DIR / "yoloe-26l-seg.pt"),
                        ]
                        
                        # Fallback to prompt-free versions if text-prompt versions not available
                        yoloe_models_prompt_free = [
                            str(MODELS_DIR / "yoloe-26n-seg-pf.pt"),
                            str(MODELS_DIR / "yoloe-26s-seg-pf.pt"),
                            str(MODELS_DIR / "yoloe-26m-seg-pf.pt"),
                        ]
                        
                        model_path = None
                        use_yoloe = False
                        use_text_prompts = False
                        
                        # Try text-prompt versions first
                        for yoloe_path in yoloe_models_with_prompts:
                            if os.path.exists(yoloe_path):
                                model_path = yoloe_path
                                use_yoloe = True
                                use_text_prompts = True
                                if self.verbose:
                                    print(f"[AGENT] Using YOLOE-26 with text prompts: {yoloe_path}")
                                break
                        
                        # Fallback to prompt-free if text-prompt versions not found
                        if not use_yoloe:
                            for yoloe_path in yoloe_models_prompt_free:
                                if os.path.exists(yoloe_path):
                                    model_path = yoloe_path
                                    use_yoloe = True
                                    if self.verbose:
                                        print(f"[AGENT] Using YOLOE-26 prompt-free model: {yoloe_path}")
                                    break
                        
                        # REQUIRE YOLOE - raise error if not found
                        if not use_yoloe or model_path is None:
                            available_models = yoloe_models_with_prompts + yoloe_models_prompt_free
                            return f"ERROR: YOLOE-26 model required for environment detection. Please download one of: {available_models}"
                        
                        # Load YOLO model (auto-detects TensorRT .engine or PyTorch .pt)
                        yolo_model = load_yolo_model(model_path, verbose=self.verbose)
                        
                        # Check if model is TensorRT engine (requires specific imgsz)
                        from pathlib import Path
                        engine_path = Path(model_path).with_suffix('.engine')
                        is_tensorrt_engine = engine_path.exists()
                        if is_tensorrt_engine:
                            imgsz_for_detection = 640  # TensorRT engines are compiled for specific size (640x640)
                        else:
                            imgsz_for_detection = 640  # PyTorch models - use smaller size to save memory
                        
                        # If using YOLOE with text prompts, set environment-related classes
                        if use_yoloe and use_text_prompts:
                            # Define comprehensive environment object classes
                            env_classes = [
                                # Outdoor/Street
                                "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", 
                                "stop sign", "road sign", "sidewalk", "crosswalk", "street",
                                # Park/Nature
                                "tree", "grass", "bench", "park", "playground", "fountain",
                                # Indoor/Commercial
                                "chair", "table", "desk", "computer", "monitor", "keyboard",
                                "restaurant", "cafe", "store", "shop",
                                # Residential
                                "bed", "sofa", "couch", "television", "refrigerator", "oven",
                                "kitchen", "bathroom", "window", "door",
                                # People/Animals
                                "person", "dog", "cat", "bird",
                                # Other
                                "building", "house", "office", "school"
                            ]
                            try:
                                # Set text prompts for environment detection
                                yolo_model.set_classes(env_classes, yolo_model.get_text_pe(env_classes))
                                if self.verbose:
                                    print(f"[AGENT] Set YOLOE text prompts for {len(env_classes)} environment classes")
                            except Exception as e:
                                if self.verbose:
                                    print(f"[AGENT] Warning: Could not set YOLOE text prompts: {e}")
                                use_text_prompts = False  # Fallback to regular detection
                        
                        # Run detection with appropriate resolution
                        # TensorRT engines require imgsz=640 (compiled size)
                        # PyTorch models use imgsz=640 (memory efficient)
                        results = yolo_model(current_frame, conf=0.15, verbose=False, imgsz=imgsz_for_detection)
                        
                        # Extract all detected objects
                        all_detections = []
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    cls_name = result.names[cls_id]
                                    all_detections.append({
                                        "class": cls_name,
                                        "confidence": conf
                                    })
                        
                        if not all_detections:
                            return "I couldn't detect any objects in the current frame. Please ensure objects are visible to the camera."
                        
                        # Count objects by class
                        class_counts = {}
                        for det in all_detections:
                            cls = det["class"]
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        # Create object list for LLM analysis
                        objects_list = ", ".join([f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in sorted(class_counts.items())])
                        
                        # Use LLM to infer environment based on detected objects
                        environment_prompt = f"""Based on the objects detected in the video frame, what kind of environment or location is this?

Detected objects: {objects_list}

Analyze the combination of objects and infer the environment type. Examples:
- cars + traffic lights + road signs → street/road
- benches + trees + dogs + grass → park
- chairs + tables + people + plates → restaurant/cafe
- desks + computers + monitors → office
- beds + dressers + windows → bedroom
- kitchen appliances + countertops → kitchen

Respond with ONLY the environment type (e.g., "street", "park", "restaurant", "office", "bedroom", "kitchen", "outdoor", "indoor", etc.). Be concise."""
                        
                        environment_response = await self.llm.ainvoke(environment_prompt)
                        environment_type = environment_response.strip()
                        
                        # Format final response
                        response = f"Based on the objects detected in the video, this appears to be a **{environment_type}** environment.\n\n"
                        response += f"Detected objects: {objects_list}\n"
                        response += f"\n(Total: {len(all_detections)} objects across {len(class_counts)} different classes)"
                        
                        if use_yoloe:
                            if use_text_prompts:
                                response += "\n\n(Using YOLOE-26 with text prompts for targeted environment object detection)"
                            else:
                                response += "\n\n(Using YOLOE-26 prompt-free mode for comprehensive object recognition)"
                        
                        return response
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"Error analyzing environment: {str(e)}"
                        if self.verbose:
                            print(f"[AGENT] Environment query error: {traceback.format_exc()}")
                        return error_msg
                
                # Check if webcam/video stream is requested
                prompt_lower = prompt.lower()
                
                # Check for webcam/video-related keywords - these should open live window
                is_webcam_request = any(word in prompt_lower for word in VIDEO_KEYWORDS)
                
                # Check if user wants data/results in chat (not just viewing GUI)
                # Keywords like "how many", "count", "what objects", "list", "tell me", etc.
                wants_data_in_chat = any(phrase in prompt_lower for phrase in COUNT_QUERY_KEYWORDS + PRINT_DATA_KEYWORDS + DESCRIPTIVE_DATA_KEYWORDS + [
                    "what objects", "list objects", "what can you see", "detect and tell", 
                    "detect and report", "detected objects", "objects in the video", "objects in video",
                    "what's in the video", "what is in the video", "what are in the video"
                ])
                
                # PRIORITY: If asking about objects in video AND web viewer is running, run detection FIRST
                # This bypasses query router to ensure we get object detection, not color detection
                if wants_data_in_chat and self.web_viewer is not None:
                    # Get raw frame from web viewer and run detection
                    try:
                        with self.web_viewer.frame_lock:
                            if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                                current_frame = self.web_viewer.raw_frame.copy()
                            elif self.web_viewer.frame is not None:
                                current_frame = self.web_viewer.frame.copy()
                            else:
                                current_frame = None
                        
                        if current_frame is None:
                            return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                        
                        # Save frame to temp file for detection
                        import tempfile
                        import cv2
                        import os
                        
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=PROJECT_ROOT) as tmp_file:
                            cv2.imwrite(tmp_file.name, current_frame)
                            temp_path = tmp_file.name
                        
                        try:
                            # Run detection using the temp file
                            tool_result = await yolo_object_detection.ainvoke({
                                "image_path": temp_path,
                                "confidence_threshold": getattr(self.web_viewer, 'confidence_threshold', 0.50)
                            })
                            
                            # Format response based on what user asked
                            if "how many" in prompt_lower or "count" in prompt_lower:
                                # Count objects
                                if isinstance(tool_result, str):
                                    # Try to parse if it's JSON string
                                    import json
                                    try:
                                        tool_result = json.loads(tool_result)
                                    except:
                                        pass
                                
                                if isinstance(tool_result, dict):
                                    detections = tool_result.get("detections", [])
                                    if detections:
                                        # Count by class
                                        class_counts = {}
                                        for det in detections:
                                            cls = det.get("class", "unknown")
                                            class_counts[cls] = class_counts.get(cls, 0) + 1
                                        
                                        total = len(detections)
                                        response = f"I detected {total} object(s) in the current frame:\n"
                                        for cls, count in sorted(class_counts.items()):
                                            response += f"  - {cls}: {count}\n"
                                        return response.strip()
                                    else:
                                        return "I detected 0 objects in the current frame."
                                else:
                                    # Format raw result using Response Formatting Template
                                    from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                                    formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                                        user_query=prompt,
                                        tool_results=f"DETECTION: {str(tool_result)}"
                                    )
                                    
                                    system_msg = None
                                    user_msg = None
                                    for msg in formatting_prompt:
                                        if msg.type == "system":
                                            system_msg = msg.content
                                        elif msg.type == "human":
                                            user_msg = msg.content
                                    
                                    if hasattr(self.llm, '_acall'):
                                        response = await self.llm._acall(user_msg, system_prompt=system_msg)
                                    else:
                                        import asyncio
                                        response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                                        if not isinstance(response, str):
                                            response = response.content if hasattr(response, 'content') else str(response)
                                    
                                    return self._truncate_to_words(response.strip(), max_words=20)
                            elif any(keyword in prompt_lower for keyword in DESCRIPTIVE_DATA_KEYWORDS):
                                # Descriptive data request - return formatted detection data with timestamps
                                if isinstance(tool_result, str):
                                    import json
                                    try:
                                        tool_result = json.loads(tool_result)
                                    except:
                                        pass
                                
                                if isinstance(tool_result, dict):
                                    detections = tool_result.get("detections", [])
                                    if detections:
                                        # Count by class
                                        class_counts = {}
                                        for det in detections:
                                            cls = det.get("class", "unknown")
                                            class_counts[cls] = class_counts.get(cls, 0) + 1
                                        
                                        # Get timestamp
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        
                                        total = len(detections)
                                        response = f"Detection data for {timestamp}:\n"
                                        response += f"Total objects detected: {total}\n\n"
                                        response += "Objects by class:\n"
                                        for cls, count in sorted(class_counts.items()):
                                            avg_conf = sum(d.get("confidence", 0.0) for d in detections if d.get("class") == cls) / count
                                            response += f"  - {cls}: {count} (avg confidence: {avg_conf:.1%})\n"
                                        
                                        # Include first_seen/last_seen from tracking data if available
                                        with self.web_viewer.tracks_lock:
                                            tracks = list(self.web_viewer.tracks_data.values())
                                        
                                        if tracks:
                                            response += "\nTracking data:\n"
                                            for track in tracks[:5]:  # Show first 5 tracks
                                                track_id = track.get("track_id", "?")
                                                track_class = track.get("class", "unknown")
                                                first_seen = track.get("first_seen", "N/A")
                                                last_seen = track.get("last_seen", "N/A")
                                                response += f"  - {track_class} (ID: {track_id}): first seen {first_seen}, last seen {last_seen}\n"
                                        
                                        return response.strip()
                                    else:
                                        return "I detected 0 objects in the current frame."
                                else:
                                    # Format raw result using Response Formatting Template
                                    from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                                    formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                                        user_query=prompt,
                                        tool_results=f"DETECTION: {str(tool_result)}"
                                    )
                                    
                                    system_msg = None
                                    user_msg = None
                                    for msg in formatting_prompt:
                                        if msg.type == "system":
                                            system_msg = msg.content
                                        elif msg.type == "human":
                                            user_msg = msg.content
                                    
                                    if hasattr(self.llm, '_acall'):
                                        response = await self.llm._acall(user_msg, system_prompt=system_msg)
                                    else:
                                        import asyncio
                                        response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                                        if not isinstance(response, str):
                                            response = response.content if hasattr(response, 'content') else str(response)
                                    
                                    return self._truncate_to_words(response.strip(), max_words=20)
                            else:
                                # Generic response - format using Response Formatting Template
                                from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                                formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                                    user_query=prompt,
                                    tool_results=f"DETECTION: {tool_result}"
                                )
                                
                                system_msg = None
                                user_msg = None
                                for msg in formatting_prompt:
                                    if msg.type == "system":
                                        system_msg = msg.content
                                    elif msg.type == "human":
                                        user_msg = msg.content
                                
                                if hasattr(self.llm, '_acall'):
                                    response = await self.llm._acall(user_msg, system_prompt=system_msg)
                                else:
                                    import asyncio
                                    response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                                    if not isinstance(response, str):
                                        response = response.content if hasattr(response, 'content') else str(response)
                                
                                return self._truncate_to_words(response.strip(), max_words=20)
                                
                                # OLD CODE - replaced with Response Formatting Template
                                # response_prompt = f"""User asked: {prompt}

# I ran object detection and got: {tool_result}

# Respond directly to the user based on these results. Do not give examples or explain your process. Just answer naturally."""
                                
                                llm_response = await self.llm.ainvoke(response_prompt)
                                return self._truncate_to_words(llm_response.strip(), max_words=20)
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    
                    except Exception as e:
                        return f"Error running detection on current frame: {str(e)}"
                
                # If web viewer is already running and user just wants to view (not get data), redirect to GUI
                # BUT skip this if it's a detection history query (already handled above)
                is_history_query = bool(re.search(r'(have\s+any|history\s+of|give\s+me\s+the\s+history)', prompt_lower, re.IGNORECASE))
                if is_webcam_request and self.web_viewer is not None and not is_history_query:
                    try:
                        import socket
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        device_ip = s.getsockname()[0]
                        s.close()
                    except:
                        device_ip = "localhost"
                    port = getattr(self.web_viewer, 'port', 8080)
                    return f"The web-based tracking viewer is already running and showing live object detection with bounding boxes, track IDs, and trajectories. Open your browser to http://{device_ip}:{port} to see the video stream and detection results in real-time."
                
                # For ANY video/webcam request, always open live detection window (not single frame)
                if is_webcam_request:
                    # Launch live detection window
                    from tools.yolo_live import run_live_detection
                    import threading
                    import tempfile
                    
                    # Check what kind of logging user wants
                    prompt_lower = prompt.lower()
                    
                    # Check for explicit requests
                    wants_both = any(phrase in prompt_lower for phrase in BOTH_LOGGING_KEYWORDS)
                    wants_timestamps = any(phrase in prompt_lower for phrase in TIMESTAMP_KEYWORDS)
                    wants_frames = any(phrase in prompt_lower for phrase in FRAME_KEYWORDS) and not wants_timestamps  # Don't trigger if user said "frames and timestamps"
                    
                    # If user says "both", override individual flags
                    if wants_both:
                        wants_timestamps = False
                        wants_frames = False
                    
                    # Create log file if any logging is requested
                    log_file = None
                    if wants_timestamps or wants_frames or wants_both:
                        log_file = tempfile.NamedTemporaryFile(
                            mode='w', 
                            suffix='_detections.csv', 
                            delete=False,
                            dir=PROJECT_ROOT
                        ).name
                    
                    def run_live():
                        run_live_detection(
                            device=0, 
                            confidence_threshold=0.25, 
                            use_gstreamer=True,
                            log_timestamps=wants_timestamps,
                            log_frames=wants_frames,
                            log_both=wants_both,
                            log_file=log_file
                        )
                    
                    print("[AGENT] Opening live webcam detection window...")
                    print("  Press 'q' in the video window to close it.")
                    
                    if wants_both:
                        print(f"  Logging mode: Both timestamps and frame numbers. Log file: {log_file}")
                    elif wants_timestamps:
                        print(f"  Logging mode: Wall-clock timestamps. Log file: {log_file}")
                    elif wants_frames:
                        print(f"  Logging mode: Frame numbers. Log file: {log_file}")
                    
                    # Run in a separate thread so agent can continue
                    live_thread = threading.Thread(target=run_live, daemon=True)
                    live_thread.start()
                    
                    response_msg = "I've opened the live webcam detection window. You should see a video feed with real-time object detection overlays showing bounding boxes and labels. Press 'q' in the video window to close it."
                    
                    if wants_both:
                        response_msg += f" Detections are being logged with both wall-clock timestamps and frame numbers to: {log_file}"
                    elif wants_timestamps:
                        response_msg += f" Detections are being logged with wall-clock timestamps (device time) to: {log_file}"
                    elif wants_frames:
                        response_msg += f" Detections are being logged with frame numbers to: {log_file}"
                    
                    return response_msg
                else:
                    # Extract image path from prompt
                    image_path = await self._extract_image_path(prompt)
                
                if not image_path:
                    return "I can help detect objects in images or video streams. Please provide a path to an image file, or ask me to use the webcam/video stream."
                
                # Call the tool
                try:
                    tool_result = await yolo_object_detection.ainvoke({
                        "image_path": image_path,
                        "confidence_threshold": 0.25
                    })
                    
                    # Check if user wants count/numbers - handle explicitly before LLM
                    prompt_lower = prompt.lower()
                    wants_count = any(phrase in prompt_lower for phrase in [
                        "how many", "count", "number of"
                    ])
                    
                    if wants_count:
                        # Parse result and count objects
                        if isinstance(tool_result, str):
                            import json
                            try:
                                tool_result = json.loads(tool_result)
                            except:
                                pass
                        
                        if isinstance(tool_result, dict):
                            detections = tool_result.get("detections", [])
                            if detections:
                                # Count by class
                                class_counts = {}
                                for det in detections:
                                    cls = det.get("class", "unknown")
                                    class_counts[cls] = class_counts.get(cls, 0) + 1
                                
                                # Check if user asked about a specific class (e.g., "how many cars")
                                # Extract class from prompt
                                requested_class = None
                                for cls in COMMON_OBJECT_CLASSES:
                                    if cls in prompt_lower or f"{cls}s" in prompt_lower:
                                        requested_class = cls
                                        break
                                
                                if requested_class:
                                    # User asked about a specific class
                                    count = class_counts.get(requested_class, 0)
                                    response = f"I detected {count} {requested_class}(s)"
                                else:
                                    # User asked for general count - show breakdown
                                    total = len(detections)
                                    response = f"I detected {total} object(s) in the current frame:\n"
                                    for cls, count in sorted(class_counts.items()):
                                        response += f"  - {cls}: {count}\n"
                                return response.strip()
                            else:
                                return "I detected 0 objects in the current frame."
                        else:
                            # Fall back to LLM formatting if parsing fails
                            pass
                    
                    # Generate response using LLM
                    # Determine if it was from video stream or image
                    source_type = "video stream" if is_webcam_request else "image"
                    
                    # Build a clear prompt with tool results
                    response_prompt = f"""User asked: {prompt}

I ran object detection and got: {tool_result}

Respond directly to the user based on these results. Maximum 20 words. Do not give examples or explain your process. Just answer naturally."""
                    
                    llm_response = await self.llm.ainvoke(response_prompt)
                    
                    # Clean up response - remove any example text or extra formatting
                    response = llm_response.strip()
                    
                    # Remove common patterns that indicate example responses
                    lines = response.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        # Skip lines that look like examples or instructions
                        if any(skip in line.lower() for skip in ['example response:', 'my response should be:', 'note that']):
                            continue
                        # Skip quoted examples
                        if line.startswith('"') and line.endswith('"') and len(cleaned_lines) == 0:
                            # Might be an example, but could also be the actual response
                            cleaned_lines.append(line.strip('"'))
                        else:
                            cleaned_lines.append(line)
                    
                    cleaned_response = ' '.join(cleaned_lines).strip()
                    # Enforce 20-word limit
                    final_response = cleaned_response if cleaned_response else response
                    return self._truncate_to_words(final_response, max_words=20)
                    
                except Exception as e:
                    return f"Error running tool {tool_name}: {str(e)}"
        
        # Check if this is a hazard/vision query that should use LLM tool binding
        # Examples: "Are there any hazards in the frame?", "What do you see?", "Is the path clear?"
        hazard_vision_keywords = [
            "hazard", "hazards", "obstruction", "obstacle", "obstacles",
            "what do you see", "what's in the frame", "what's visible",
            "is the path clear", "clear path", "safe to proceed",
            "any dangers", "any threats", "risk", "risks"
        ]
        is_hazard_vision_query = any(keyword in prompt_lower for keyword in hazard_vision_keywords)
        
        # If hazard/vision query and web viewer is running, use LLM tool binding
        if is_hazard_vision_query and self.web_viewer is not None:
            try:
                # Get current frame from web viewer
                with self.web_viewer.frame_lock:
                    if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                        current_frame = self.web_viewer.raw_frame.copy()
                    elif self.web_viewer.frame is not None:
                        current_frame = self.web_viewer.frame.copy()
                    else:
                        current_frame = None
                
                if current_frame is not None:
                    # Create vision tool with current frame
                    vision_tool = create_vision_tool(current_frame)
                    
                    # Bind tool to LLM (LLM can now call this tool)
                    try:
                        # Try to bind tools (if LLM supports it)
                        if hasattr(self.llm, 'bind_tools'):
                            llm_with_tools = self.llm.bind_tools([vision_tool])
                        elif hasattr(self.llm, 'with_tools'):
                            # Alternative method name
                            llm_with_tools = self.llm.with_tools([vision_tool])
                        else:
                            # Fallback: Manual tool calling via prompt
                            if self.verbose:
                                print("[AGENT] LLM doesn't support bind_tools, using manual tool calling")
                            llm_with_tools = self.llm
                            
                            # Manual approach: Call tool first, then format with Response Formatting Template
                            tool_result = vision_tool.invoke({})
                            
                            # Format using Response Formatting Template for human sentence
                            from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                            formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                                user_query=prompt,
                                tool_results=f"HAZARD DETECTION: {tool_result}"
                            )
                            
                            # Extract system and user messages
                            system_msg = None
                            user_msg = None
                            for msg in formatting_prompt:
                                if msg.type == "system":
                                    system_msg = msg.content
                                elif msg.type == "human":
                                    user_msg = msg.content
                            
                            # Call LLM to format response
                            if hasattr(self.llm, '_acall'):
                                response = await self.llm._acall(user_msg, system_prompt=system_msg)
                            else:
                                response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                                if not isinstance(response, str):
                                    response = response.content if hasattr(response, 'content') else str(response)
                            
                            return self._truncate_to_words(response.strip(), max_words=20)
                    except AttributeError:
                        # LLM doesn't support tool binding, use manual approach
                        if self.verbose:
                            print("[AGENT] LLM doesn't support tool binding, using manual tool calling")
                        # Always use manual approach for reliability
                        tool_result = vision_tool.invoke({})
                        
                        # Format using Response Formatting Template for human sentence
                        from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                        formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                            user_query=prompt,
                            tool_results=f"HAZARD DETECTION: {tool_result}"
                        )
                        
                        # Extract system and user messages
                        system_msg = None
                        user_msg = None
                        for msg in formatting_prompt:
                            if msg.type == "system":
                                system_msg = msg.content
                            elif msg.type == "human":
                                user_msg = msg.content
                        
                        # Call LLM to format response
                        if hasattr(self.llm, '_acall'):
                            response = await self.llm._acall(user_msg, system_prompt=system_msg)
                        else:
                            response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                            if not isinstance(response, str):
                                response = response.content if hasattr(response, 'content') else str(response)
                        
                        return self._truncate_to_words(response.strip(), max_words=20)
                    
                    # If we got here, tool binding was attempted but may not work reliably
                    # Force manual approach instead
                    if self.verbose:
                        print("[AGENT] Forcing manual tool calling for reliability")
                    tool_result = vision_tool.invoke({})
                    
                    # Format using Response Formatting Template for human sentence
                    from agent.prompt_templates import RESPONSE_FORMATTING_TEMPLATE
                    formatting_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
                        user_query=prompt,
                        tool_results=f"HAZARD DETECTION: {tool_result}"
                    )
                    
                    # Extract system and user messages
                    system_msg = None
                    user_msg = None
                    for msg in formatting_prompt:
                        if msg.type == "system":
                            system_msg = msg.content
                        elif msg.type == "human":
                            user_msg = msg.content
                    
                    # Call LLM to format response
                    if hasattr(self.llm, '_acall'):
                        response = await self.llm._acall(user_msg, system_prompt=system_msg)
                    else:
                        response = await asyncio.to_thread(self.llm.invoke, formatting_prompt)
                        if not isinstance(response, str):
                            response = response.content if hasattr(response, 'content') else str(response)
                    
                    return self._truncate_to_words(response.strip(), max_words=20)
                    
                    # If tool binding worked, invoke LLM (it will call the tool if needed)
                    if self.verbose:
                        print("[AGENT] Using LLM with vision tool binding for hazard detection")
                    
                    # Create a prompt that encourages tool use
                    tool_prompt = f"""You are a tactical AI assistant analyzing real-time video.

User query: {prompt}

You have access to a vision tool that can detect objects in the current frame.
If the user is asking about hazards, obstacles, or what's visible, use the detect_objects_in_frame tool first.

Call the tool, then provide a concise response (under 20 words) based on the results."""
                    
                    # Invoke LLM (it should call the tool)
                    response = await llm_with_tools.ainvoke(tool_prompt)
                    
                    # Check if response contains tool calls (for OpenAI-compatible APIs)
                    # For now, if response is just text, return it
                    # In future, we can parse tool calls from response
                    if isinstance(response, str):
                        return response.strip()
                    else:
                        # Handle structured response with tool calls
                        # This depends on the LLM implementation
                        return str(response).strip()
                else:
                    if self.verbose:
                        print("[AGENT] Web viewer running but no frame available")
            except Exception as e:
                if self.verbose:
                    print(f"[AGENT] Vision tool binding error: {e}")
                # Fall through to regular LLM call
        
        # No tool needed, just use LLM
        try:
            response = await self.llm.ainvoke(prompt)
        except RuntimeError as e:
            # Check if it's a Docker connection error
            error_str = str(e).lower()
            if "connection attempts failed" in error_str or "connection error" in error_str:
                # Docker server not available - try to fallback to llama-cpp-python
                if self.verbose:
                    print(f"[AGENT] Docker LLM connection failed, attempting fallback to llama-cpp-python...")
                
                # Try to initialize llama-cpp-python fallback
                if LLAMA_CPP_AVAILABLE and self.model_type:
                    try:
                        model_path = self._find_model_by_type(self.model_type)
                        if model_path:
                            if self.verbose:
                                print(f"[AGENT] Initializing llama-cpp-python fallback...")
                            self.llm = self._create_llama_llm(model_path, temperature=0.7)
                            response = await self.llm.ainvoke(prompt)
                        else:
                            raise RuntimeError(f"Docker failed and model not found for fallback: {e}")
                    except Exception as fallback_error:
                        raise RuntimeError(f"Docker LLM failed and fallback also failed: {fallback_error}")
                else:
                    raise RuntimeError(f"Docker LLM failed and llama-cpp-python not available: {e}")
            else:
                # Other error, re-raise
                raise
        
        # Enforce 20-word limit on ALL responses (not just open questions)
        # This ensures concise, tactical responses
        response = self._truncate_to_words(response, max_words=20)
        
        # Clean up temp file if it still exists
        if current_frame_path and os.path.exists(current_frame_path):
            try:
                os.unlink(current_frame_path)
            except:
                pass
        
        return response
    
    def run_sync(self, prompt: str) -> str:
        """Synchronous wrapper."""
        return asyncio.run(self.run(prompt))


# Alias for compatibility
SnowcrashAgent = SimpleAgent

