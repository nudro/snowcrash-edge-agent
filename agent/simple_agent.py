#!/usr/bin/env python3
"""
Simplified agentic agent with tool calling.
Works with local LLMs and MCP tools.
"""
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from agent.langchain_tools import (
    get_langchain_tools, 
    yolo_object_detection,
    track_objects,
    get_detection_statistics,
    estimate_object_distances
)
from agent.query_keywords import (
    TRACKING_KEYWORDS, STATS_KEYWORDS, DISTANCE_KEYWORDS, DISTANCE_COMPARISON_KEYWORDS,
    IMAGE_VIDEO_KEYWORDS, VIDEO_KEYWORDS,
    TOOL_QUERY_KEYWORDS, MOVEMENT_KEYWORDS, SPEED_KEYWORDS, HOW_LONG_KEYWORDS, TOOL_KEYWORDS,
    PRINT_DATA_KEYWORDS, TRACK_ID_KEYWORDS, COUNT_QUERY_KEYWORDS, DESCRIPTIVE_DATA_KEYWORDS,
    COLOR_QUERY_KEYWORDS, VIEW_GUI_KEYWORDS, COMMON_OBJECT_CLASSES, FOLLOW_UP_PATTERNS,
    TIMESTAMP_KEYWORDS, FRAME_KEYWORDS, BOTH_LOGGING_KEYWORDS,
    NUMERIC_DURATION_PATTERNS, WORD_TO_NUMBER, WORD_DURATION_PATTERNS,
    IMAGE_PATH_PATTERNS, TRACK_ID_PATTERN, OBJECT_CLASS_PATTERN, POSITION_KEYWORDS
)

# For local LLM (llama.cpp)
try:
    from langchain_community.llms import LlamaCpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


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
        web_viewer=None
    ):
        """
        Initialize the agent.
        
        Args:
            llm: Pre-initialized LLM (optional)
            model_path: Path to local GGUF model (optional, overrides model_type)
            model_type: Model to use - "phi-3", "llama", or "gemma" (optional)
            temperature: LLM temperature
            verbose: Whether to print agent reasoning
        """
        self.verbose = verbose
        self.tools = get_langchain_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.model_type = model_type
        self.web_viewer = web_viewer  # Reference to web viewer if already running
        
        # Initialize LLM
        if llm is None:
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
                print("[WARNING] No LLM available. Install llama-cpp-python")
                self.llm = None
        else:
            self.llm = llm
    
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
        models_dir = PROJECT_ROOT / "models"
        
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
        """
        Decide if we should use a tool based on prompt (hybrid: keywords + LLM reasoning).
        
        Flow:
        1. Check keywords first (fast path)
        2. If exactly 1 match → use that tool immediately (fast path)
        3. If multiple matches or ambiguous → ask LLM to choose the best tool
        4. If no matches but seems tool-related → ask LLM if tool needed
        5. If no matches and not tool-related → return None (will fallback to LLM)
        """
        prompt_lower = prompt.lower()
        
        # First pass: Check for clear keyword matches (fast path)
        # Priority order: distance > tracking > stats > detection
        keyword_matches = []
        
        has_distance = any(keyword in prompt_lower for keyword in DISTANCE_KEYWORDS)
        has_position = any(keyword in prompt_lower for keyword in POSITION_KEYWORDS)
        has_tracking = any(keyword in prompt_lower for keyword in TRACKING_KEYWORDS)
        has_stats = any(keyword in prompt_lower for keyword in STATS_KEYWORDS)
        
        # For detection keywords, be more selective - don't match generic "targets" when distance keywords are present
        # Only match detection keywords that don't conflict with distance queries
        detection_keywords_filtered = [kw for kw in IMAGE_VIDEO_KEYWORDS if kw not in ["targets", "target count", "target identification"]]
        has_detection = any(keyword in prompt_lower for keyword in detection_keywords_filtered)
        
        # If distance keywords are explicitly present, strongly prioritize distance tool
        # Even if "targets" is mentioned, it's clearly a distance query
        if has_distance or has_position:
            keyword_matches.append("estimate_object_distances")
            # If distance is present, ignore conflicting detection matches from "targets"
            # This prevents "Distances to all targets" from matching both distance and detection
            if has_detection and any(kw in prompt_lower for kw in ["targets", "target count", "target identification"]):
                has_detection = False
        if has_tracking:
            keyword_matches.append("track_objects")
        # Only add stats if distance is NOT present (avoid confusion)
        if has_stats and not (has_distance or has_position):
            keyword_matches.append("get_detection_statistics")
        if has_detection:
            keyword_matches.append("yolo_object_detection")
        
        # If distance keyword is present, prioritize it strongly - use it even with multiple matches
        if has_distance or has_position:
            # Return distance tool immediately if distance keywords are present
            # This prevents LLM from incorrectly choosing statistics
            return "estimate_object_distances"
        
        # If exactly one match, use it (fast path - no LLM call)
        if len(keyword_matches) == 1:
            return keyword_matches[0]
        
        # If multiple matches or ambiguous, use LLM reasoning (only if LLM available)
        if len(keyword_matches) > 1 or (len(keyword_matches) == 0 and any(kw in prompt_lower for kw in TOOL_KEYWORDS)):
            try:
                # Ask LLM to choose the best tool (only if we have an LLM available)
                if self.llm is not None:
                    tools_list = "\n".join([f"- {tool}" for tool in keyword_matches] if keyword_matches 
                                          else ["- yolo_object_detection", "- track_objects", 
                                                "- get_detection_statistics", "- estimate_object_distances"])
                    
                    reasoning_prompt = f"""Given this user prompt, which tool should be used? Respond with ONLY the tool name (no explanation).

IMPORTANT: If the user asks about "distances", "distance", "how far", or "meters away", use estimate_object_distances.
If the user asks about "statistics", "stats", or "count", use get_detection_statistics.

User prompt: "{prompt}"

Available tools:
{tools_list}

Respond with only the tool name, or "none" if no tool is needed."""
                    
                    # Wrap with SOF template for consistent style (no detection context for tool selection)
                    wrapped_prompt = self._wrap_prompt_with_sof_template(reasoning_prompt, has_detection_context=False)
                    
                    # Get LLM response (truncate to reasonable length for efficiency)
                    response = await self.llm.ainvoke(wrapped_prompt)
                    tool_name = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
                    
                    # Map response to tool name
                    if "track" in tool_name or tool_name == "track_objects":
                        return "track_objects"
                    elif "stat" in tool_name or tool_name == "get_detection_statistics":
                        return "get_detection_statistics"
                    elif "distance" in tool_name or tool_name == "estimate_object_distances":
                        return "estimate_object_distances"
                    elif "detect" in tool_name or "yolo" in tool_name or tool_name == "yolo_object_detection":
                        return "yolo_object_detection"
                    elif "none" in tool_name or tool_name == "none":
                        return None
            except Exception as e:
                # If LLM reasoning fails, fall back to first keyword match or most common
                if keyword_matches:
                    return keyword_matches[0]  # Use first match as fallback
                # If no keywords and LLM failed, check if it looks tool-related
                if any(kw in prompt_lower for kw in TOOL_KEYWORDS):
                    # Default to detection for ambiguous tool-related queries
                    return "yolo_object_detection"
        
        # If no keyword matches, return None (will fallback to LLM for general response)
        return None if not keyword_matches else keyword_matches[0]
    
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
        import re
        
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
        import re
        
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
    
    def _truncate_to_words(self, text: str, max_words: int = 30) -> str:
        """Truncate text to maximum number of words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        # Take first max_words words and add ellipsis if truncated
        truncated = ' '.join(words[:max_words])
        # Remove trailing punctuation that might look odd with ellipsis
        truncated = truncated.rstrip('.,;:')
        return truncated + "..."
    
    def _get_sof_prompt_template(self, has_detection_context: bool = False) -> str:
        """
        Get SOF (Special Operations Forces) operator prompt template.
        Enforces tactical, concise communication style with 45-word limit.
        
        Args:
            has_detection_context: If True, agent is describing actual detected objects.
                                  If False, use normal language to prevent hallucinations.
        """
        if has_detection_context:
            # Use tactical style only when describing actual detections
            return """You are a Special Operations Forces (SOF) tactical AI assistant describing real-time surveillance data.

Communication Protocol:
- Respond in brief, direct operational style
- Use tactical terminology ONLY for actual detected objects (target, status, intel)
- Keep ALL responses under 45 words - be concise
- DO NOT invent scenarios, hostile situations, or objects that were not detected
- Only describe what was actually detected in the video feed
- Format: [Status] [Essential info] [Action/Result]

Respond operationally based on actual detection data:"""
        else:
            # Use normal, factual language for general questions
            return """You are a helpful AI assistant. Respond briefly and factually.

Keep responses under 45 words. Be concise and accurate. Do not invent scenarios or objects that don't exist.

Respond:"""
    
    def _enforce_response_length(self, response: str, max_words: int = 45) -> str:
        """
        Enforce response length limit by truncating to max_words.
        Tries to truncate at sentence boundary if possible.
        """
        words = response.split()
        if len(words) <= max_words:
            return response
        
        # Try to truncate at sentence boundary
        truncated = ' '.join(words[:max_words])
        
        # Find last sentence-ending punctuation
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        # If we found a sentence boundary within reasonable distance, use it
        if last_sentence_end > len(truncated) * 0.7:  # At least 70% through
            return truncated[:last_sentence_end + 1]
        
        # Otherwise truncate and add ellipsis
        truncated = truncated.rstrip('.,;:')
        return truncated + "..."
    
    def _wrap_prompt_with_sof_template(self, user_prompt: str, has_detection_context: bool = False) -> str:
        """
        Wrap user prompt with SOF tactical template.
        
        Args:
            user_prompt: The user's prompt
            has_detection_context: If True, use tactical style (for actual detections).
                                  If False, use normal language (for general questions).
        """
        sof_template = self._get_sof_prompt_template(has_detection_context=has_detection_context)
        return f"{sof_template}\n\nUser: {user_prompt}\nAssistant:"
    
    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt."""
        if self.llm is None:
            return "Error: No LLM available. Cannot run agent."
        
        # Check if user is asking about available tools (check this FIRST before any tool calling)
        prompt_lower = prompt.lower().strip()
        
        # Check if the prompt is primarily asking about tools/capabilities
        if any(keyword in prompt_lower for keyword in TOOL_QUERY_KEYWORDS):
            return self._get_tools_description()
        
        # Check if asking about agent capabilities/agentic nature (meta questions)
        agentic_keywords = ["agentic", "is snowcrash", "what is snowcrash", "capabilities", "what can you do"]
        if any(keyword in prompt_lower for keyword in agentic_keywords):
            if "agentic" in prompt_lower or "is snowcrash" in prompt_lower:
                return "Snowcrash is agentic. Uses hybrid keyword + LLM reasoning for tool selection. Capabilities: object detection (YOLO26-seg), tracking, distance estimation, color detection, spatial reasoning."
            elif "capabilities" in prompt_lower or "what can you do" in prompt_lower:
                return self._get_tools_description()
        
        # Check if user is asking about moving objects (using SpeedEstimator/velocity data)
        is_movement_query = any(keyword in prompt_lower for keyword in MOVEMENT_KEYWORDS)
        
        if is_movement_query and self.web_viewer is not None:
            # Use SpeedEstimator approach: check velocity/speed from tracking data
            # Threshold: object is "moving" if average speed > 1.0 px/s or velocity magnitude > 1.0 px/s
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
                    # Format response with moving objects
                    response = "Moving objects detected:\n"
                    for obj in moving_objects:
                        response += f"  - {obj['class']} (ID: {obj['track_id']}): {obj['speed']} px/s\n"
                    return response.strip()
                else:
                    return "No objects are currently moving."
            
            except Exception as e:
                return f"Error checking movement: {str(e)}. Please ensure the web viewer is running."
        
        # Check for speed queries - return speed estimates for objects
        is_speed_query = any(keyword in prompt_lower for keyword in SPEED_KEYWORDS)
        
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
        is_how_long_query = any(keyword in prompt_lower for keyword in HOW_LONG_KEYWORDS)
        
        if is_how_long_query and self.web_viewer is not None:
            try:
                # Extract track ID or object class from prompt
                import re
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
        
        # Check for spatial/contextual queries - these need detection + LLM reasoning
        from agent.query_keywords import SPATIAL_RELATIONSHIP_KEYWORDS, CONTEXTUAL_QUERY_KEYWORDS
        has_spatial = any(kw in prompt_lower for kw in SPATIAL_RELATIONSHIP_KEYWORDS)
        has_contextual = any(kw in prompt_lower for kw in CONTEXTUAL_QUERY_KEYWORDS)
        
        # Handle track ID-based "relative to" queries first (e.g., "Where is ID1 car relative to ID5 car?")
        if "relative to" in prompt_lower and self.web_viewer is not None:
            import re
            # Extract track IDs from prompt (e.g., "ID1", "id:1", "track 1", "ID5")
            track_id_pattern = r'(?:id|track)[:\s]*(\d+)'
            track_id_matches = re.findall(track_id_pattern, prompt_lower)
            
            if len(track_id_matches) >= 2:
                try:
                    target_id = int(track_id_matches[0])
                    reference_id = int(track_id_matches[1])
                    
                    # Get tracking data
                    with self.web_viewer.tracks_lock:
                        tracks = list(self.web_viewer.tracks_data.values())
                    
                    target_track = None
                    reference_track = None
                    
                    for track in tracks:
                        track_id = track.get("track_id")
                        if track_id == target_id:
                            target_track = track
                        elif track_id == reference_id:
                            reference_track = track
                    
                    if target_track is None:
                        return f"Track ID {target_id} not found in current tracking data."
                    if reference_track is None:
                        return f"Track ID {reference_id} not found in current tracking data."
                    
                    # Get bounding boxes
                    target_bbox = target_track.get("bbox")
                    reference_bbox = reference_track.get("bbox")
                    
                    if not target_bbox or not reference_bbox:
                        return f"Bounding box data not available for track IDs {target_id} or {reference_id}."
                    
                    # Convert bbox dict to center point
                    from tools.spatial_utils import compute_object_center, compute_cardinal_direction
                    
                    if isinstance(target_bbox, dict):
                        target_center = compute_object_center(target_bbox)
                    else:
                        # Assume it's [x1, y1, x2, y2]
                        target_center = (
                            (target_bbox[0] + target_bbox[2]) / 2.0,
                            (target_bbox[1] + target_bbox[3]) / 2.0
                        )
                    
                    if isinstance(reference_bbox, dict):
                        reference_center = compute_object_center(reference_bbox)
                    else:
                        reference_center = (
                            (reference_bbox[0] + reference_bbox[2]) / 2.0,
                            (reference_bbox[1] + reference_bbox[3]) / 2.0
                        )
                    
                    # Compute cardinal direction from reference to target
                    direction = compute_cardinal_direction(reference_center, target_center)
                    
                    target_class = target_track.get("class", "object")
                    reference_class = reference_track.get("class", "object")
                    
                    return f"Track ID {target_id} ({target_class}) is {direction} of Track ID {reference_id} ({reference_class})."
                    
                except Exception as e:
                    import traceback
                    return f"Error computing relative position: {str(e)}\n{traceback.format_exc()}"
        
        if (has_spatial or has_contextual) and self.web_viewer is not None:
            # Complex spatial/contextual query - run detection and pass to LLM with spatial context
            try:
                from tools.spatial_utils import compute_spatial_relationships, format_spatial_context
                import cv2
                import json
                
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
                
                # Run YOLO detection with segmentation
                if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                    yolo_model = self.web_viewer.model
                else:
                    from ultralytics import YOLO
                    yolo_model = YOLO("yolo26n-seg.pt")
                
                # Run detection with segmentation
                results = yolo_model(current_frame, conf=0.50, task='segment', verbose=False)
                
                # Extract detections with full info
                detections = []
                result = results[0]
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls)
                        cls_name = result.names[cls_id]
                        bbox = box.xyxy[0].tolist()
                        
                        detection = {
                            "class": cls_name,
                            "confidence": float(box.conf),
                            "bbox": {
                                "x1": float(bbox[0]),
                                "y1": float(bbox[1]),
                                "x2": float(bbox[2]),
                                "y2": float(bbox[3])
                            },
                            "center": {
                                "x": float((bbox[0] + bbox[2]) / 2),
                                "y": float((bbox[1] + bbox[3]) / 2)
                            }
                        }
                        
                        # Add color if masks available (will be computed below)
                        # Color detection happens after all detections are collected
                        
                        detections.append(detection)
                
                if not detections:
                    return "No objects detected in the current frame. Please ensure objects are visible to the camera."
                
                # Add colors to detections using mask-based color detection
                from tools.color_detection import detect_colors_from_yolo_results
                color_dets = detect_colors_from_yolo_results(current_frame, results)
                # Match colors to detections by index
                for i, det in enumerate(detections):
                    if i < len(color_dets):
                        det["color_name"] = color_dets[i].get("color_name")
                        det["color_rgb"] = color_dets[i].get("color_rgb")
                
                # Compute spatial relationships
                relationships = compute_spatial_relationships(detections)
                
                # Format spatial context for LLM
                spatial_context = format_spatial_context(detections, relationships)
                
                # Build enhanced prompt for LLM reasoning
                reasoning_prompt = f"""User asked: "{prompt}"

{spatial_context}

Based on the detected objects and their spatial relationships, answer the user's question directly and naturally."""
                
                # Get LLM response with SOF template
                if self.llm is None:
                    return "Error: No LLM available for reasoning."
                
                # Use tactical style for spatial queries (describing actual detections)
                wrapped_prompt = self._wrap_prompt_with_sof_template(reasoning_prompt, has_detection_context=True)
                llm_response = await self.llm.ainvoke(wrapped_prompt)
                response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                
                # Enforce 45-word limit
                response = self._enforce_response_length(response.strip(), max_words=45)
                return response
                
            except Exception as e:
                import traceback
                return f"Error processing spatial/contextual query: {str(e)}\n{traceback.format_exc()}"
        
        # Check for color queries - detect colors of objects in the video
        is_color_query = any(keyword in prompt_lower for keyword in COLOR_QUERY_KEYWORDS)
        
        if is_color_query:
            # Extract object class from prompt (e.g., "color of cars" -> "car")
            # Use word boundaries to avoid false matches (e.g., "identification" shouldn't match "cat")
            requested_class = None
            import re
            for cls in COMMON_OBJECT_CLASSES:
                # Match whole words only (word boundaries)
                pattern = r'\b' + re.escape(cls) + r's?\b'
                if re.search(pattern, prompt_lower):
                    requested_class = cls
                    break
            
            # Try to get frame from web viewer if running
            if self.web_viewer is not None:
                try:
                    from tools.color_detection import detect_colors_from_yolo_results
                    import cv2
                    
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
                    
                    # Run YOLO detection on the frame (use segmentation for mask-based color detection)
                    if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                        yolo_model = self.web_viewer.model
                    else:
                        from ultralytics import YOLO
                        yolo_model = YOLO("yolo26n-seg.pt")
                    
                    # Run detection with segmentation task to get masks
                    results = yolo_model(current_frame, conf=0.50, task='segment', verbose=False)
                    
                    # Detect colors using masks (more accurate)
                    detections_with_colors = detect_colors_from_yolo_results(
                        current_frame, results, object_class=requested_class
                    )
                    
                    if not detections_with_colors:
                        obj_text = f"{requested_class} " if requested_class else ""
                        return f"No {obj_text}objects detected in the current frame."
                    
                    # Format response
                    response_lines = []
                    for det in detections_with_colors:
                        class_name = det.get("class", "unknown")
                        color_name = det.get("color_name", "unknown")
                        color_rgb = det.get("color_rgb", (0, 0, 0))
                        
                        response_lines.append(f"{class_name.capitalize()}: {color_name} (RGB: {color_rgb})")
                    
                    return "\n".join(response_lines)
                    
                except Exception as e:
                    import traceback
                    return f"Error detecting colors: {str(e)}\n{traceback.format_exc()}"
            else:
                # Web viewer not running - need to capture frame
                # This should trigger YOLO detection which will handle color detection
                # But for now, return a helpful message
                return "Color detection requires access to the video feed. Please start the tracking viewer first, or ask me to detect objects in the video."
        
        # Check for distance comparison queries - "which car is closer?", "order by distance", etc.
        is_distance_comparison = any(keyword in prompt_lower for keyword in DISTANCE_COMPARISON_KEYWORDS)
        has_distance_keyword = any(keyword in prompt_lower for keyword in DISTANCE_KEYWORDS + DISTANCE_COMPARISON_KEYWORDS)
        
        if is_distance_comparison and self.web_viewer is not None:
            try:
                # Extract object class from prompt (e.g., "which car" -> "car")
                # Use word boundaries to avoid false matches
                requested_class = None
                import re
                for cls in COMMON_OBJECT_CLASSES:
                    # Match whole words only (word boundaries)
                    pattern = r'\b' + re.escape(cls) + r's?\b'
                    if re.search(pattern, prompt_lower):
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
                
                viewer_confidence = getattr(self.web_viewer, 'confidence_threshold', 0.50)
                result = await distance_tool.execute({
                    "frame": current_frame,
                    "confidence_threshold": viewer_confidence
                })
                
                # Get tracking data to match detections to track IDs
                track_data_for_matching = {}
                with self.web_viewer.tracks_lock:
                    tracks = list(self.web_viewer.tracks_data.values())
                    for track in tracks:
                        track_id = track.get("track_id")
                        track_class = track.get("class")
                        track_bbox = track.get("bbox")
                        if track_id is not None and track_bbox:
                            track_data_for_matching[track_id] = {
                                "class": track_class,
                                "bbox": track_bbox
                            }
                
                # Helper function to compute IoU (Intersection over Union)
                def compute_iou(bbox1, bbox2):
                    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
                    x1_1, y1_1, x2_1, y2_1 = bbox1
                    x1_2, y1_2, x2_2, y2_2 = bbox2
                    
                    # Intersection
                    x1_i = max(x1_1, x1_2)
                    y1_i = max(y1_1, y1_2)
                    x2_i = min(x2_1, x2_2)
                    y2_i = min(y2_1, y2_2)
                    
                    if x2_i <= x1_i or y2_i <= y1_i:
                        return 0.0
                    
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union = area1 + area2 - intersection
                    
                    return intersection / union if union > 0 else 0.0
                
                # Match detections to track IDs using bounding box overlap (IoU)
                def match_detection_to_track_id(det_bbox, det_class):
                    """Match a detection bounding box to a track ID."""
                    best_match_id = None
                    best_iou = 0.3  # Minimum IoU threshold
                    
                    for track_id, track_info in track_data_for_matching.items():
                        if track_info["class"] != det_class:
                            continue
                        
                        track_bbox = track_info["bbox"]
                        if isinstance(track_bbox, dict):
                            track_bbox_list = [track_bbox.get("x1", 0), track_bbox.get("y1", 0),
                                             track_bbox.get("x2", 0), track_bbox.get("y2", 0)]
                        else:
                            track_bbox_list = track_bbox
                        
                        iou = compute_iou(det_bbox, track_bbox_list)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_id = track_id
                    
                    return best_match_id
                
                # Filter by class if specified, otherwise get all objects with distances
                detections_with_distances = result.get("detections_with_distances", [])
                
                # Filter objects with valid distances and by class if specified, and match to track IDs
                valid_distances = []
                for det in detections_with_distances:
                    if det.get("distance_meters") is not None:
                        # Filter by class if specified
                        if requested_class is None or det.get("class") == requested_class:
                            # Match to track ID
                            det_bbox = det.get("bounding_box", [])
                            det_class = det.get("class")
                            track_id = match_detection_to_track_id(det_bbox, det_class) if det_bbox else None
                            det["track_id"] = track_id
                            valid_distances.append(det)
                
                if not valid_distances:
                    obj_text = f"{requested_class} " if requested_class else ""
                    return f"No {obj_text}objects with valid distances detected in the current frame."
                
                # Sort by distance (closest to farthest)
                valid_distances.sort(key=lambda x: x["distance_meters"])
                
                # Format response
                response_lines = []
                
                # Check if user asked specifically for track ID of closest
                asks_for_track_id = any(keyword in prompt_lower for keyword in ["track id", "track id of", "track id:", "id of"])
                
                if len(valid_distances) == 1:
                    obj = valid_distances[0]
                    track_id_text = f" (Track ID: {obj['track_id']})" if obj.get('track_id') else ""
                    response_lines.append(
                        f"The {obj['class']}{track_id_text} is approximately {obj['distance_meters']:.1f} meters "
                        f"({obj['distance_feet']:.1f} feet) away."
                    )
                else:
                    # Show ordered list from closest to farthest
                    response_lines.append(f"Objects ordered from closest to farthest ({len(valid_distances)} total):\n")
                    for i, obj in enumerate(valid_distances, 1):
                        obj_class = obj.get("class", "unknown")
                        distance_m = obj.get("distance_meters")
                        distance_ft = obj.get("distance_feet")
                        track_id = obj.get("track_id")
                        track_id_text = f" (Track ID: {track_id})" if track_id else ""
                        response_lines.append(
                            f"  {i}. {obj_class.capitalize()}{track_id_text}: {distance_m:.1f} meters ({distance_ft:.1f} feet)"
                        )
                    
                    # Also mention which one is closest
                    closest = valid_distances[0]
                    track_id_text = f" (Track ID: {closest.get('track_id')})" if closest.get('track_id') else ""
                    closest_text = f"\nThe closest {requested_class if requested_class else 'object'} is "
                    if asks_for_track_id and closest.get('track_id'):
                        closest_text += f"Track ID {closest['track_id']}: the {closest['class']} at {closest['distance_meters']:.1f} meters ({closest['distance_feet']:.1f} feet)."
                    else:
                        closest_text += f"the {closest['class']}{track_id_text} at {closest['distance_meters']:.1f} meters ({closest['distance_feet']:.1f} feet)."
                    response_lines.append(closest_text)
                
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
            import re
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
                    model_path="yolo26n-seg.pt",
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
                        model_path="yolo26n-seg.pt",
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
        
        # Check if we should use a tool
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
                if is_webcam_request and self.web_viewer is not None:
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
                        "confidence_threshold": 0.50,  # Updated to match main.py
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
                    return tool_result
                except Exception as e:
                    return f"Error running statistics tool: {str(e)}"
            
            # Handle distance estimation tool
            elif tool_name == "estimate_object_distances":
                prompt_lower = prompt.lower()
                
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
                        import re
                        prompt_lower = prompt.lower()
                        
                        # Extract track ID if specified (e.g., "ID:9", "track 9")
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
                                "confidence_threshold": 0.50
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
                                track_model = YOLO("yolo26n-seg.pt")
                            
                            # Run tracking with the same model instance
                            track_results = track_model.track(
                                current_frame,
                                conf=0.50,
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
                        viewer_confidence = getattr(self.web_viewer, 'confidence_threshold', 0.50)
                        
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
                                        "distance_feet": det.get("distance_feet")
                                    })
                        
                        if matching_distances:
                            # Check if user asked for "each", "all", "every", "list", or "targets"
                            prompt_lower = prompt.lower()
                            wants_all = any(word in prompt_lower for word in ["each", "all", "every", "list", "targets"])
                            
                            # If query explicitly mentions "distances" (plural) or "all", return all distances
                            if ("distances" in prompt_lower and not requested_class) or (wants_all and len(matching_distances) > 1):
                                # Return all objects with distances
                                response_lines = [f"Distances to all detected objects ({len(matching_distances)} total):\n"]
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class'].capitalize()}: {obj['distance_meters']:.1f} meters "
                                        f"({obj['distance_feet']:.1f} feet) away"
                                    )
                                return "\n".join(response_lines)
                            elif len(matching_distances) == 1:
                                # Only one object, return it
                                obj = matching_distances[0]
                                return f"The {obj['class']} is approximately {obj['distance_meters']:.1f} meters ({obj['distance_feet']:.1f} feet) away."
                            else:
                                # Return nearest matching object distance in simple format
                                nearest = min(matching_distances, key=lambda x: x["distance_meters"])
                                obj_class = requested_class or nearest["class"]
                                distance_m = nearest["distance_meters"]
                                distance_ft = nearest["distance_feet"]
                                
                                return f"The {obj_class} is approximately {distance_m:.1f} meters ({distance_ft:.1f} feet) away."
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
                        "confidence_threshold": 0.50  # Match confidence threshold used elsewhere
                    })
                    
                    # Parse the JSON result to find distances
                    try:
                        tool_result = json.loads(tool_result_str)
                        
                        # Extract requested object class and track ID from prompt
                        import re
                        prompt_lower = prompt.lower()
                        
                        # Extract track ID if specified
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
                                        "distance_feet": det.get("distance_feet")
                                    })
                        
                        if matching_distances:
                            # Check if user asked for "each", "all", "every", "list", or "targets"
                            prompt_lower = prompt.lower()
                            wants_all = any(word in prompt_lower for word in ["each", "all", "every", "list", "targets"])
                            
                            # If query explicitly mentions "distances" (plural) or "all", return all distances
                            if ("distances" in prompt_lower and not requested_class) or (wants_all and len(matching_distances) > 1):
                                # Return all objects with distances
                                response_lines = [f"Distances to all detected objects ({len(matching_distances)} total):\n"]
                                for i, obj in enumerate(matching_distances, 1):
                                    response_lines.append(
                                        f"  {i}. {obj['class'].capitalize()}: {obj['distance_meters']:.1f} meters "
                                        f"({obj['distance_feet']:.1f} feet) away"
                                    )
                                return "\n".join(response_lines)
                            elif len(matching_distances) == 1:
                                # Only one object, return it
                                obj = matching_distances[0]
                                if requested_track_id:
                                    return f"The {obj['class']} (track ID: {requested_track_id}) is approximately {obj['distance_meters']:.1f} meters ({obj['distance_feet']:.1f} feet) away."
                                else:
                                    return f"The {obj['class']} is approximately {obj['distance_meters']:.1f} meters ({obj['distance_feet']:.1f} feet) away."
                            else:
                                # Return nearest matching object distance in simple format
                                nearest = min(matching_distances, key=lambda x: x["distance_meters"])
                                obj_class = requested_class or nearest["class"]
                                distance_m = nearest["distance_meters"]
                                distance_ft = nearest["distance_feet"]
                                
                                if requested_track_id:
                                    return f"The {obj_class} (track ID: {requested_track_id}) is approximately {distance_m:.1f} meters ({distance_ft:.1f} feet) away."
                                else:
                                    return f"The {obj_class} is approximately {distance_m:.1f} meters ({distance_ft:.1f} feet) away."
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
                        # If parsing fails, return the raw result
                        return tool_result_str
                        
                except Exception as e:
                    return f"Error running distance estimation tool: {str(e)}"
            
            # Handle YOLO detection tool
            elif tool_name == "yolo_object_detection":
                # Check if webcam/video stream is requested
                prompt_lower = prompt.lower()
                
                # Check for webcam/video-related keywords - these should open live window
                is_webcam_request = any(word in prompt_lower for word in VIDEO_KEYWORDS)
                
                # Check if user wants data/results in chat (not just viewing GUI)
                # Keywords like "how many", "count", "what objects", "list", "tell me", etc.
                wants_data_in_chat = any(phrase in prompt_lower for phrase in COUNT_QUERY_KEYWORDS + PRINT_DATA_KEYWORDS + DESCRIPTIVE_DATA_KEYWORDS + [
                    "what objects", "list objects", "what can you see", "detect and tell", 
                    "detect and report", "detected objects"
                ])
                
                # If user wants data in chat AND web viewer is running, run detection and return results
                # Even if not explicitly asking for video, if web viewer is running and they want data, give it
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
                                    # Return the raw result if we can't parse it
                                    return str(tool_result)
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
                                    # Return the raw result if we can't parse it
                                    return str(tool_result)
                            else:
                                # Generic response - let LLM format it with SOF template
                                response_prompt = f"""User asked: {prompt}

I ran object detection and got: {tool_result}

Respond directly to the user based on these results. Do not give examples or explain your process. Just answer naturally."""
                                
                                # Use tactical style when describing actual detection results
                                wrapped_prompt = self._wrap_prompt_with_sof_template(response_prompt, has_detection_context=True)
                                llm_response = await self.llm.ainvoke(wrapped_prompt)
                                response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                                
                                # Enforce 45-word limit with SOF style
                                return self._enforce_response_length(response.strip(), max_words=45)
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    
                    except Exception as e:
                        return f"Error running detection on current frame: {str(e)}"
                
                # Check if user is asking for positions (not just viewing)
                prompt_lower = prompt.lower()
                is_position_query = any(word in prompt_lower for word in POSITION_KEYWORDS)
                
                # If asking for positions, return actual position data instead of redirecting to GUI
                if is_position_query and self.web_viewer is not None:
                    try:
                        # Get current frame and run detection
                        with self.web_viewer.frame_lock:
                            if hasattr(self.web_viewer, 'raw_frame') and self.web_viewer.raw_frame is not None:
                                current_frame = self.web_viewer.raw_frame.copy()
                            elif self.web_viewer.frame is not None:
                                current_frame = self.web_viewer.frame.copy()
                            else:
                                current_frame = None
                        
                        if current_frame is None:
                            return "The web viewer is running but no frame is available yet. Please wait a moment and try again."
                        
                        # Run detection with segmentation
                        if hasattr(self.web_viewer, 'model') and self.web_viewer.model is not None:
                            yolo_model = self.web_viewer.model
                        else:
                            from ultralytics import YOLO
                            yolo_model = YOLO("yolo26n-seg.pt")
                        
                        results = yolo_model(current_frame, conf=0.50, task='segment', verbose=False)
                        
                        # Extract positions
                        detections = []
                        result = results[0]
                        if result.boxes is not None:
                            for i, box in enumerate(result.boxes):
                                cls_id = int(box.cls)
                                cls_name = result.names[cls_id]
                                bbox = box.xyxy[0].tolist()
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2
                                
                                detections.append({
                                    "class": cls_name,
                                    "confidence": float(box.conf),
                                    "center": {"x": round(center_x, 1), "y": round(center_y, 1)},
                                    "bbox": {"x1": round(bbox[0], 1), "y1": round(bbox[1], 1), 
                                            "x2": round(bbox[2], 1), "y2": round(bbox[3], 1)}
                                })
                        
                        if not detections:
                            return "No objects detected in the current frame. Please ensure objects are visible to the camera."
                        
                        # Format response
                        response_lines = [f"Positions of all detected objects ({len(detections)} total):\n"]
                        for i, det in enumerate(detections, 1):
                            response_lines.append(
                                f"  {i}. {det['class'].capitalize()} (confidence: {det['confidence']:.1%}): "
                                f"center at ({det['center']['x']:.1f}, {det['center']['y']:.1f}), "
                                f"bbox: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) to "
                                f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})"
                            )
                        return "\n".join(response_lines)
                    except Exception as e:
                        import traceback
                        return f"Error getting object positions: {str(e)}\n{traceback.format_exc()}"
                
                # If web viewer is already running and user just wants to view (not get data), redirect to GUI
                if is_webcam_request and self.web_viewer is not None:
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

Respond directly to the user based on these results. Do not give examples or explain your process. Just answer naturally."""
                    
                    wrapped_prompt = self._wrap_prompt_with_sof_template(response_prompt)
                    llm_response_obj = await self.llm.ainvoke(wrapped_prompt)
                    llm_response = llm_response_obj.content if hasattr(llm_response_obj, 'content') else str(llm_response_obj)
                    
                    # Clean up response - remove any example text or extra formatting
                    response = llm_response.strip()
                    
                    # Enforce 45-word limit
                    response = self._enforce_response_length(response, max_words=45)
                    
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
                    return cleaned_response if cleaned_response else response
                    
                except Exception as e:
                    return f"Error running tool {tool_name}: {str(e)}"
        
        # No tool needed, check if this is about actual detections or a general question
        # Use tactical style only when describing actual detected objects
        # Use normal language for general questions to prevent hallucinations
        is_detection_related = any(keyword in prompt_lower for keyword in [
            "detected", "detection", "objects in", "what objects", "what do you see",
            "tracking", "distance", "color", "in the video", "in the frame"
        ])
        has_active_viewer = self.web_viewer is not None
        
        # Only use tactical style if user is asking about actual detections AND viewer is active
        use_tactical = is_detection_related and has_active_viewer
        
        wrapped_prompt = self._wrap_prompt_with_sof_template(prompt, has_detection_context=use_tactical)
        response_obj = await self.llm.ainvoke(wrapped_prompt)
        response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        
        # Enforce 45-word limit (override the 30-word limit for open questions)
        response = self._enforce_response_length(response.strip(), max_words=45)
        
        return response
    
    def run_sync(self, prompt: str) -> str:
        """Synchronous wrapper."""
        return asyncio.run(self.run(prompt))


# Alias for compatibility
SnowcrashAgent = SimpleAgent

