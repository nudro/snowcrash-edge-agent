"""
Centralized keyword and pattern definitions for query parsing.

This module contains all keyword lists and regex patterns used for
detecting user intent and routing queries to appropriate tools.
"""

# ============================================================================
# Tool Detection Keywords
# These keywords help determine which tool should be used
# ============================================================================

TRACKING_KEYWORDS = [
    "track", "tracking", "follow", "track objects", "track object",
    "track the", "follow that", "persistent id", "track id"
]

STATS_KEYWORDS = [
    "statistics", "stats", "most common", "detection count",
    "average confidence", "aggregate", "summary"
]

DISTANCE_KEYWORDS = [
    "distance", "how far", "far away", "estimate distance",
    "meters away", "feet away", "how close"
]

DISTANCE_COMPARISON_KEYWORDS = [
    "which", "closer", "closest", "farther", "farthest", "nearest", "furthest",
    "compare distances", "order by distance", "closest to farthest"
]

IMAGE_VIDEO_KEYWORDS = [
    "image", "picture", "photo", "detect", "objects", "yolo", 
    "analyze image", "webcam", "camera", "cam",
    "video stream", "video", "stream", "live", 
    "detect objects in the video",
    # Count/detection queries that need object detection
    "how many", "count", "number of", "what objects", "list objects"
]

VIDEO_KEYWORDS = [
    "webcam", "camera", "cam", "video stream", "video", 
    "stream", "live", "detect objects in the video",
    "detect objects in video", "open the webcam", "show webcam",
    "find objects in video", "find objects"
]

# ============================================================================
# Query Intent Keywords
# These help determine what the user is asking about
# ============================================================================

TOOL_QUERY_KEYWORDS = [
    "what tools", "available tools", "what can you do", "list tools",
    "what tools do you have", "what are your tools", "tools available",
    "show me tools", "help", "capabilities", "what are you capable of",
    "other tools", "what other tools", "tools do you", "do you have tools",
    "what tools are", "tools you have"
]

MOVEMENT_KEYWORDS = [
    "are objects moving", "are the objects moving", "which objects are moving",
    "what is moving", "what's moving", "objects moving", "moving objects",
    "is anything moving", "detect movement", "check movement"
]

HOW_LONG_KEYWORDS = [
    "how long", "how long has", "how long been", "duration", "been in video",
    "been in frame", "time in", "time on video", "minutes in"
]

# Keywords that indicate tool usage (for filtering open questions)
TOOL_KEYWORDS = [
    "detect", "track", "distance", "count", "how many", "how long", "show", 
    "display", "view", "gui", "image", "video", "camera", "webcam",
    "objects", "statistics", "data", "information about"
]

# ============================================================================
# Data Extraction Keywords
# These help determine what data the user wants
# ============================================================================

PRINT_DATA_KEYWORDS = [
    "print", "show me", "tell me", "what is", "get data", "data for",
    "information about", "details for", "stats for", "descriptive data",
    "describe", "data on", "information on"
]

TRACK_ID_KEYWORDS = ["id:", "track id", "track", "tracking id"]

COUNT_QUERY_KEYWORDS = [
    "how many", "count", "number of"
]

DESCRIPTIVE_DATA_KEYWORDS = [
    "descriptive data", "data on", "information on"
]

COLOR_QUERY_KEYWORDS = [
    "color", "colour", "what color", "what colour", "colors", "colours",
    "color of", "colour of", "paint", "painted"
]

# ============================================================================
# UI/GUI Keywords
# These help determine if user wants to view something in GUI
# ============================================================================

VIEW_GUI_KEYWORDS = [
    "view", "show", "display", "open viewer", "show me", "let me see",
    "gui", "visualize", "visualization", "show tracking", "show detections",
    "tracking viewer", "view tracking", "show tracking viewer"
]

# ============================================================================
# Object Classes
# Common COCO object classes for prompt parsing
# ============================================================================

COMMON_OBJECT_CLASSES = [
    "car", "person", "bicycle", "motorcycle", "bus", "truck", 
    "bench", "chair", "couch", "bed", "dog", "cat", "bird", 
    "bottle", "cup", "laptop", "mouse", "keyboard", "phone", "book"
]

# ============================================================================
# Follow-up Query Patterns
# These patterns help detect follow-up queries
# ============================================================================

FOLLOW_UP_PATTERNS = [
    "do it for", "do it", "for", "find distance to", "distance to", "how far to"
]

# ============================================================================
# Logging Keywords
# Keywords for detecting logging preferences
# ============================================================================

TIMESTAMP_KEYWORDS = [
    "timestamp", "by time", "by timestamp", "clock time", 
    "wall clock", "device time", "current time"
]

FRAME_KEYWORDS = [
    "by frame", "frame number", "frame numbers", "frames"
]

BOTH_LOGGING_KEYWORDS = [
    "both", "frames and timestamps", "timestamps and frames", 
    "frames and time", "time and frames"
]

# ============================================================================
# Regex Patterns
# Patterns for extracting data from prompts
# ============================================================================

# Duration extraction patterns
NUMERIC_DURATION_PATTERNS = [
    r'\b(\d+)\s*(?:second|sec|s)\b',
    r'for\s+(\d+)\s*(?:second|sec|s)?',
    r'(\d+)\s*(?:second|sec|s)?\s+(?:long|duration)'
]

# Word-to-number mapping for duration extraction
WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60
}

# Word-based duration patterns (template, will be formatted with word)
WORD_DURATION_PATTERNS = [
    r'\b({})\s*(?:second|sec|s)\b',
    r'for\s+({})\s*(?:second|sec|s)?',
    r'\b({})\s+(?:second|sec|s)?\s+(?:long|duration)'
]

# Image path extraction patterns
IMAGE_PATH_PATTERNS = [
    r'["\']([^"\']+\.(jpg|jpeg|png|gif|bmp))["\']',
    r'([^\s]+\.(jpg|jpeg|png|gif|bmp))',
]

# Track ID extraction pattern
TRACK_ID_PATTERN = r'(?:id|track)[:\s]*(\d+)'

# Object class extraction pattern (uses COMMON_OBJECT_CLASSES)
OBJECT_CLASS_PATTERN = r'\b(car|person|bicycle|motorcycle|bus|truck|bench|chair|couch|bed|dog|cat|bird|bottle|cup|laptop|mouse|keyboard|phone|book|object)\b'
