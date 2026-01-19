# Creative Tool Recommendations for Snowcrash

This document outlines potential creative tools to add to the Snowcrash agentic system, with implementation details and memory analysis.

## Recommended Priority Order

### Tier 1: Low Memory, High Value

#### 1. **Statistics Aggregation Tool** ⭐⭐⭐⭐⭐
- **What**: Tracks detection statistics over time (most common objects, avg confidence, detection counts)
- **Memory**: ~1-2 KB (just counters and running averages)
- **Implementation**: Simple dict tracking, update on each detection
- **Value**: High - enables "what's the most common object?" queries
- **Complexity**: Low (2-3 hours)

#### 2. **Zone/Region Analysis Tool** ⭐⭐⭐⭐
- **What**: Detects objects in specific regions (e.g., "entrance", "parking lot", "shelf")
- **Memory**: ~100 bytes per zone (just coordinates)
- **Implementation**: Point-in-polygon checks for bbox centers
- **Value**: High - enables spatial queries ("what's in the entrance?")
- **Complexity**: Low (3-4 hours)

#### 3. **Distance Estimation Tool** ⭐⭐⭐⭐
- **What**: Estimates distance from camera to detected objects
- **Memory**: Minimal (just calculations, no storage)
- **Implementation**: Pinhole camera model with known object sizes
  ```python
  # Formula: distance = (known_height * focal_length) / (detected_height_px * pixel_size)
  # Default assumptions: person=1.7m, car=1.5m height, etc.
  ```
- **Value**: High - enables "how far away is the person?" queries
- **Complexity**: Medium (4-6 hours, needs calibration or defaults)

#### 4. **Object Counting Tool** ⭐⭐⭐
- **What**: Counts unique objects (people in/out via virtual lines)
- **Memory**: ~1 KB (counters + line definitions)
- **Implementation**: Track centroid crossing of virtual lines
- **Value**: Medium - useful for traffic/crowd analysis
- **Complexity**: Medium (5-7 hours, needs tracking)

### Tier 2: Medium Memory, High Value

#### 5. **Object Frequency Chart Tool** ⭐⭐⭐⭐
- **What**: Generates line charts showing object counts over time
- **Memory**: ~1-2 KB per minute of data (timestamped counts only)
- **Implementation**: 
  - Store: `{"person": [(timestamp, count), ...]}`
  - Generate chart with matplotlib on-demand
  - Auto-clear old data (>10 minutes)
- **Value**: High - visual trend analysis
- **Complexity**: Low-Medium (4-5 hours, matplotlib dependency)

#### 6. **Motion Detection Tool** ⭐⭐⭐
- **What**: Detects movement in the scene
- **Memory**: Medium (~3-6 MB for previous frame storage)
  - Store 1-2 previous frames (640x480 RGB = ~1.5 MB each)
- **Implementation**: Frame differencing or background subtraction
- **Value**: Medium - useful for activity detection
- **Complexity**: Medium (6-8 hours)

### Tier 3: Higher Memory, Specialized

#### 7. **Object Tracking Tool** ⭐⭐⭐⭐
- **What**: Tracks specific objects across frames with unique IDs
- **Memory**: Medium (~10-50 KB per tracked object, manageable with limits)
  - Store trajectory history: `{track_id: [(frame, bbox, class), ...]}`
  - Clear tracks after N frames or when lost
- **Implementation Options**:
  - **StoneSoup** (recommended): Mature, MIT-licensed framework (~400-500 KB package)
    - Supports Kalman filters, PHD filters, track stitching
    - Flexible measurement models, data association
    - More robust than simple IoU, better for occlusion/missed detections
    - Memory: ~10-20 KB per track (state vectors + covariance)
    - Configure pruning/merging thresholds to limit memory growth
  - **DeepSORT/SORT**: Simpler, lighter (~100 KB package)
    - Faster, less sophisticated
    - Good for basic tracking needs
- **Value**: High - enables "follow that person" queries, track persistence
- **Complexity**: 
  - StoneSoup: Medium-High (10-15 hours, but well-documented)
  - DeepSORT: Medium (6-8 hours)

## Memory Impact Summary

| Tool | Memory Usage | Impact |
|------|--------------|--------|
| Statistics | ~1-2 KB | Negligible |
| Zone Analysis | ~100 bytes | Negligible |
| Distance | 0 KB | None |
| Counting | ~1 KB | Negligible |
| Charts | ~1-2 KB/min | Low |
| Motion | ~3-6 MB | Medium |
| Tracking (StoneSoup) | ~10-20 KB/track + 400-500 KB package | Medium |
| Tracking (DeepSORT) | ~5-10 KB/track + ~100 KB package | Low-Medium |

**Total with all tools**: ~10-15 MB for typical scenarios (vs current ~0 MB for tools alone)

**StoneSoup vs DeepSORT for tracking:**
- **StoneSoup**: Better for complex scenarios (occlusion, missed detections, multiple hypotheses)
- **DeepSORT**: Simpler, faster, lighter - good for basic tracking needs
- **Recommendation**: Use StoneSoup if you need robust tracking with state estimation; DeepSORT if you want simpler, faster tracking

## Implementation Architecture

Each tool should follow the pattern:

```python
# tools/statistics_tool.py
class StatisticsTool:
    """Tracks detection statistics over time."""
    
    def __init__(self):
        self.stats = {}  # {"person": {"count": 0, "avg_conf": 0.0, ...}}
    
    async def update(self, detections: List[Dict]):
        """Update statistics with new detections."""
        # Update counters, averages, etc.
    
    async def get_stats(self) -> Dict:
        """Return current statistics."""
        return self.stats
    
    def get_tool_schema(self) -> Tool:
        """MCP tool schema."""
        return Tool(...)
```

Then register in `mcp_server/server.py` and wrap in `agent/langchain_tools.py`.

## Recommended Starting Point

**Start with Tier 1 tools (Statistics + Zone + Distance)**
- Total memory: ~2-3 KB
- Total implementation time: ~10-12 hours
- Maximum value for minimum cost
- All can be added incrementally without breaking existing code

## Example Agent Prompts After Implementation

- "Track statistics on detected objects"
- "What objects are in the entrance zone?"
- "How far away is the nearest person?"
- "Show me a chart of person detections over the last 5 minutes"
- "Count how many people enter the room"
- "Is there motion in the scene?"
- "Track that person and tell me their trajectory"

