# StoneSoup Tracking Analysis

## Overview

**StoneSoup** is an open-source Python framework for multi-target tracking and state estimation, developed by the UK Defence Science and Technology Laboratory (DSTL). It's MIT-licensed and actively maintained.

## Why StoneSoup is a Good Fit

### ✅ Advantages

1. **Sophisticated Tracking**
   - State estimation filters (Kalman, Extended Kalman, Particle filters)
   - Multi-target tracking (PHD filters, Linear Complexity with Cumulants)
   - Handles occlusion, missed detections, clutter
   - Track stitching (reconnects track fragments)

2. **Flexible Integration**
   - Works with YOLO detections (bounding boxes → measurements)
   - Multiple measurement models (position, velocity, bounding box)
   - Customizable data association (nearest neighbor, global nearest neighbor, JPDA)

3. **Well-Documented & Maintained**
   - Comprehensive documentation with examples
   - Active development community
   - MIT license (free for commercial use)

4. **Moderate Memory Footprint**
   - Package size: ~400-500 KB (from PyPI)
   - Per-track memory: ~10-20 KB (state vector + covariance matrix)
   - Configurable pruning/merging to limit track count

### ⚠️ Considerations

1. **Complexity**
   - More moving parts than simple IoU tracking
   - Requires tuning (false alarm rate, detection probability, gating thresholds)
   - Learning curve for configuring trackers

2. **Performance**
   - Additional CPU overhead for predictions and data association
   - O(n_tracks × n_detections) complexity for association
   - May need to limit maximum tracks for real-time performance

3. **Memory Management**
   - Each track stores state + history (if logging)
   - Need to configure pruning/merging to prevent unbounded growth
   - History buffers can accumulate if not cleared periodically

## Memory Analysis

### StoneSoup Package
- **Wheel size**: ~400-500 KB (from PyPI)
- **Installed size**: ~2-3 MB (with dependencies)

### Per-Track Memory
- **State vector**: 4-8 floats (position, velocity) = ~32-64 bytes
- **Covariance matrix**: 4×4 or 8×8 floats = ~64-256 bytes
- **Track metadata**: ID, timestamp, class = ~100-200 bytes
- **Total per track**: ~200-500 bytes (state only)

### With History Logging
- **Per position logged**: ~50-100 bytes (timestamp, bbox, class)
- **For 1000 position history**: ~50-100 KB per track
- **10 active tracks with history**: ~500 KB - 1 MB

### Typical Configuration (10 active tracks, no history)
- **StoneSoup overhead**: ~2-3 MB
- **Active tracks**: ~2-5 KB (10 × 200-500 bytes)
- **Total**: ~2-5 MB

### With Pruning/Merging Limits
- Max tracks: 20-50 (configurable)
- Prune tracks after: 5-10 frames of no detections
- Memory capped at: ~5-10 MB for tracks + overhead

## Integration with YOLO

### Architecture Sketch

```
YOLO Detection → Convert to StoneSoup Measurement → Tracker Update → Track Output
     ↓                              ↓                      ↓              ↓
[Bbox, Class, Conf]    [StateVector, Covariance]    [Predict + Associate]  [Track ID, Position, Velocity]
```

### Implementation Steps

1. **Convert YOLO detections to measurements**
   ```python
   from stonesoup.types.detection import Detection
   from stonesoup.types.state import StateVector
   
   # YOLO gives: bbox (x, y, w, h), class, confidence
   # Convert to: center (cx, cy), size (w, h) + velocity (if available)
   detection = Detection(
       StateVector([[cx], [cy], [w], [h]]),  # or with velocity [vx, vy]
       timestamp=current_time
   )
   ```

2. **Configure tracker**
   ```python
   from stonesoup.tracker.simple import MultiTargetTracker
   from stonesoup.predictor.kalman import KalmanPredictor
   from stonesoup.updater.kalman import KalmanUpdater
   # ... configure predictor, updater, hypothesiser
   ```

3. **Process each frame**
   ```python
   tracks = tracker.tracks  # Get all active tracks
   for detection in detections:
       tracker.update(detection)
   ```

4. **Extract track information**
   - Track ID (persistent across frames)
   - Position, velocity, bounding box
   - Confidence/state probability

## Comparison with Alternatives

### StoneSoup vs DeepSORT

| Aspect | StoneSoup | DeepSORT |
|--------|-----------|----------|
| **Package size** | ~400-500 KB | ~100 KB |
| **Complexity** | High (sophisticated) | Medium (appearance + motion) |
| **Occlusion handling** | Excellent (state estimation) | Good (appearance features) |
| **Memory per track** | ~200-500 bytes | ~500-1000 bytes (with appearance features) |
| **Tuning required** | Yes (filters, thresholds) | Yes (association thresholds) |
| **Documentation** | Excellent | Good |
| **Best for** | Complex scenarios, state estimation | Appearance-based tracking |

### StoneSoup vs Simple IoU Tracking

| Aspect | StoneSoup | IoU Tracking |
|--------|-----------|--------------|
| **Package size** | ~400-500 KB | 0 KB (custom code) |
| **Complexity** | High | Low |
| **Occlusion handling** | Excellent | Poor (loses track) |
| **Memory per track** | ~200-500 bytes | ~50-100 bytes |
| **Tuning required** | Yes | Minimal |
| **Best for** | Production tracking | Simple, fast tracking |

## Recommendation

**Use StoneSoup if:**
- ✅ You need robust tracking (occlusion, missed detections)
- ✅ You want state estimation (velocity, predictions)
- ✅ You can accept ~2-5 MB memory overhead
- ✅ You're willing to tune configuration parameters

**Use DeepSORT if:**
- ✅ You want simpler, faster tracking
- ✅ Appearance features are important (re-ID after occlusion)
- ✅ You want lighter memory footprint (~1-2 MB)
- ✅ You need basic multi-object tracking

**Use Simple IoU if:**
- ✅ You have simple use cases (few objects, no occlusion)
- ✅ You want zero dependencies
- ✅ You need maximum performance
- ✅ Memory is extremely constrained

## Implementation Recommendation

**For Snowcrash, I recommend StoneSoup** because:
1. It handles edge cases better (occlusion, clutter)
2. State estimation enables future features (prediction, velocity-based filtering)
3. Memory overhead is acceptable (~2-5 MB for typical scenarios)
4. Well-documented and maintained
5. Flexible enough to grow with your needs

**Configuration to limit memory:**
- Max tracks: 20-50
- Prune after 5-10 missed detections
- Don't log full history (just current state)
- Use simpler predictors/updaters if needed (Linear vs Extended Kalman)

## Next Steps

If you want to proceed with StoneSoup:
1. Add `stonesoup>=1.8` to `requirements.txt`
2. Create `tools/object_tracking.py` with StoneSoup integration
3. Convert YOLO detections → StoneSoup measurements
4. Configure a multi-target tracker (start with simple Kalman)
5. Expose as MCP tool and LangChain wrapper
6. Test with live webcam feed

Should I implement StoneSoup tracking integration?

