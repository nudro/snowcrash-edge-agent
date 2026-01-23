# Prompt Template Analysis

## Tools Available
1. `yolo_object_detection` - Basic object detection
2. `get_detection_statistics` - Aggregated statistics
3. `estimate_object_distances` - Distance estimation
4. `track_objects` - Object tracking with persistent IDs
5. `estimate_object_geography` - GPS-based location (rarely used)
6. Color detection (built-in, not separate tool)
7. GUI viewers (tracking viewer, detection viewer)

## Query Types Identified

### Category 1: Object Detection & Identification
- **Keywords**: `IMAGE_VIDEO_KEYWORDS`, `COUNT_QUERY_KEYWORDS`
- **Queries**: "detect objects", "how many cars", "what objects", "count persons"
- **Tool**: `yolo_object_detection`
- **Semantic challenges**: 
  - Object parts (sweater → person)
  - Synonyms (vehicle → car)
  - Plural forms (cars → car)

### Category 2: Color Detection
- **Keywords**: `COLOR_QUERY_KEYWORDS`
- **Queries**: "what color sweater", "color of cars", "what color is that person"
- **Tool**: Color detection (built-in)
- **Semantic challenges**:
  - Object parts (sweater → person)
  - Clothing items (shirt, jacket, pants → person)
  - Body parts (face, hand → person)

### Category 3: Distance Estimation
- **Keywords**: `DISTANCE_KEYWORDS`
- **Queries**: "how far away is the person", "distance to car", "how close"
- **Tool**: `estimate_object_distances`
- **Semantic challenges**:
  - Object identification
  - Spatial references ("the person" vs "person")

### Category 4: Distance Comparison
- **Keywords**: `DISTANCE_COMPARISON_KEYWORDS`
- **Queries**: "which car is closer", "order by distance", "closest person"
- **Tool**: `estimate_object_distances` + comparison logic
- **Semantic challenges**:
  - Object class extraction
  - Comparative reasoning

### Category 5: Spatial Relationships
- **Keywords**: `SPATIAL_RELATIONSHIP_KEYWORDS`, "wrt", "relative to"
- **Queries**: "person wrt dog", "person relative to dog", "how far is person from dog"
- **Tool**: `estimate_object_distances` + spatial analysis
- **Semantic challenges**:
  - Understanding "wrt" = "with respect to"
  - Relative positioning
  - Multi-object queries

### Category 6: Object Tracking
- **Keywords**: `TRACKING_KEYWORDS`
- **Queries**: "track objects", "follow that person", "track the car"
- **Tool**: `track_objects`
- **Semantic challenges**:
  - Object identification
  - Duration extraction

### Category 7: Movement Detection
- **Keywords**: `MOVEMENT_KEYWORDS`
- **Queries**: "are objects moving", "what is moving", "which objects are moving"
- **Tool**: Tracking data analysis (no separate tool)
- **Semantic challenges**: None (uses existing tracking data)

### Category 8: Speed Estimation
- **Keywords**: `SPEED_KEYWORDS`
- **Queries**: "how fast is the car", "speed of person", "velocity"
- **Tool**: Tracking data analysis (no separate tool)
- **Semantic challenges**:
  - Object identification
  - Speed vs velocity distinction

### Category 9: Duration Queries
- **Keywords**: `HOW_LONG_KEYWORDS`
- **Queries**: "how long has person been in frame", "duration of track ID 5"
- **Tool**: Tracking data analysis (no separate tool)
- **Semantic challenges**:
  - Track ID extraction
  - Object class extraction

### Category 10: Statistics
- **Keywords**: `STATS_KEYWORDS`
- **Queries**: "most common object", "detection statistics", "status check"
- **Tool**: `get_detection_statistics`
- **Semantic challenges**: None (aggregated data)

### Category 11: Context/Environment Inference
- **Keywords**: `CONTEXTUAL_QUERY_KEYWORDS`
- **Queries**: "what is the environment", "what kind of place", "based on objects"
- **Tool**: `yolo_object_detection` + reasoning
- **Semantic challenges**:
  - Multi-object reasoning
  - Environment type inference (street, park, restaurant, etc.)

### Category 12: Position/Location
- **Keywords**: `POSITION_KEYWORDS`
- **Queries**: "where is the person", "location of car", "coordinates"
- **Tool**: `estimate_object_distances` or `estimate_object_geography`
- **Semantic challenges**:
  - Object identification
  - Position vs distance distinction

### Category 13: Track ID Data
- **Keywords**: `TRACK_ID_KEYWORDS`, `PRINT_DATA_KEYWORDS`
- **Queries**: "data for track ID 5", "show me track 3", "print track ID 1"
- **Tool**: Tracking data access (no separate tool)
- **Semantic challenges**:
  - Track ID extraction
  - Data formatting

### Category 14: GUI/Viewer Requests
- **Keywords**: `VIEW_GUI_KEYWORDS`
- **Queries**: "show tracking", "view detections", "open viewer"
- **Tool**: GUI viewers (no separate tool)
- **Semantic challenges**: None (direct action)

### Category 15: Follow-up Queries
- **Keywords**: `FOLLOW_UP_PATTERNS`
- **Queries**: "do it for bench", "distance to car" (after previous query)
- **Tool**: Context-dependent (reuses previous tool)
- **Semantic challenges**:
  - Context retention
  - Object class extraction

## Prompt Templates Required

### Template 1: Query Understanding (Master Router)
**Purpose**: Understand user intent and route to appropriate tool
**Handles**: All query types
**Output**: JSON with tool name, object class, parameters

### Template 2: Object Detection Queries
**Purpose**: Handle detection queries with semantic understanding
**Handles**: Categories 1, 11
**Key features**:
- Object part mapping (sweater → person)
- Synonym handling (vehicle → car)
- Plural normalization

### Template 3: Color Detection Queries
**Purpose**: Handle color queries with object part understanding
**Handles**: Category 2
**Key features**:
- Clothing → person mapping
- Body parts → person mapping
- Object part extraction

### Template 4: Distance & Spatial Queries
**Purpose**: Handle distance and spatial relationship queries
**Handles**: Categories 3, 4, 5, 12
**Key features**:
- "wrt" understanding
- Relative positioning
- Multi-object queries
- Comparison logic

### Template 5: Tracking Queries
**Purpose**: Handle tracking-related queries
**Handles**: Categories 6, 7, 8, 9, 13
**Key features**:
- Object identification
- Track ID extraction
- Duration extraction

### Template 6: Response Formatting
**Purpose**: Format tool results into natural language
**Handles**: All categories
**Key features**:
- Context-aware formatting
- Tactical/operational style (SOF)
- Concise responses

## Summary: **6 Prompt Templates Required**

1. **Query Understanding Router** (master template)
2. **Object Detection** (detection + context inference)
3. **Color Detection** (color queries)
4. **Distance & Spatial** (distance, comparison, spatial relationships)
5. **Tracking** (tracking, movement, speed, duration, track data)
6. **Response Formatting** (format all tool results)

Note: Categories 10 (statistics) and 14 (GUI) don't need special templates as they're straightforward keyword matches.

