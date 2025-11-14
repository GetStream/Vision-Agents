# Debugging Guide

This guide helps you understand the logs and debug timing issues.

## Understanding the Logs

When you run the app, you'll now see detailed logs with timestamps showing exactly when things happen.

### Log Format

```
HH:MM:SS.mmm | module_name | LEVEL | message
```

Example:
```
14:23:45.123 | squat_counter_processor | INFO | 🎯 REP COUNTED! #1 | Timestamp: 1699876425.123 | Knee angle: 162.3°
```

## Key Log Messages

### 1. Rep Counted (Most Important!)

```
🎯 REP COUNTED! #X | Timestamp: XXXXX.XXX | Knee angle: XXX.X°
✅ Total reps: X
```

**What it means**: A squat rep was successfully detected and counted.

**When you see this**: Immediately when you complete a squat (transition from ascending to standing).

**What to check**:
- Does the timestamp match when you actually stood up?
- Is the knee angle > 160° (standing position)?

### 2. Message Generated

```
📢 Message generated: 'Rep X complete! 🎉' at XXXXX.XXX
```

**What it means**: The system created a message to send to the LLM.

**When you see this**: Right after rep is counted.

**What to check**:
- How long between this and when you hear the audio?
- This shows when data is sent to LLM, not when audio plays.

### 3. Processing Latency

```
⏱️  Processing latency: XX.Xms | Rep #X detected
```

**What it means**: How long it took to process the frame and detect the rep.

**When you see this**: After each rep is counted.

**What to check**:
- Should be < 100ms for good performance
- If > 200ms, processing is slow (see optimization tips)

### 4. Phase Transitions

```
🔄 Phase transition: ascending → standing (angle: XXX.X°)
```

**What it means**: The squat phase changed.

**When you see this**: During your squat movement.

**What to check**:
- Do the transitions match your actual movement?
- standing → descending → bottom → ascending → standing

### 5. Current State (Every 10 frames)

```
📊 Current state: phase=descending, knee=120.5°, reps=3
```

**What it means**: Periodic status update.

**When you see this**: Every ~300ms (at 30 FPS).

**What to check**:
- Is the knee angle accurate for your position?
- Is the phase correct?

### 6. Form Issues

```
⚠️  Form issues detected: Go deeper, Keep knees behind toes
```

**What it means**: Form problems detected.

**When you see this**: When form issues are present.

**What to check**:
- Are the warnings accurate?
- Adjust thresholds in config if too sensitive.

## Debugging Audio Latency

If audio seems late, here's how to track down the issue:

### Step 1: Find the Rep Count Log

Do a squat and look for:
```
14:23:45.123 | squat_counter_processor | INFO | 🎯 REP COUNTED! #1
```

Note the timestamp: `14:23:45.123`

### Step 2: Find When You Hear Audio

Note the time when you actually hear the AI say something about the rep.

Example: `14:23:47.500` (you hear "Nice! That's 1 squat!")

### Step 3: Calculate Latency

```
Audio time - Rep count time = Total latency
14:23:47.500 - 14:23:45.123 = 2.377 seconds
```

### Step 4: Understand the Breakdown

Total latency includes:

1. **Processing time** (shown in logs): ~50-100ms
   ```
   ⏱️  Processing latency: 75.3ms
   ```

2. **LLM processing time**: ~500-1500ms
   - Time for Gemini to generate response
   - Depends on LLM FPS setting (default: 3 FPS)

3. **Audio generation time**: ~200-500ms
   - Time to generate speech
   - Depends on TTS service

4. **Network latency**: ~50-200ms
   - Varies with connection quality

**Total expected latency**: 800ms - 2300ms

### Step 5: Optimize if Needed

If latency > 3 seconds:

1. **Increase LLM FPS** (sends data more frequently):
   ```python
   # config.py
   LLM_CONFIG = {
       "fps": 5,  # Was 3 - more frequent updates
   }
   ```

2. **Reduce processing load**:
   ```python
   # config.py
   YOLO_CONFIG = {
       "imgsz": 384,  # Was 512 - faster processing
       "max_workers": 12,  # Was 24 - less CPU
   }
   ```

3. **Check network connection**:
   - Slow internet = higher latency
   - Use wired connection if possible

## Common Issues

### Issue: Reps Not Being Counted

**Check logs for**:
```
📊 Current state: phase=ascending, knee=155.2°, reps=0
```

**Problem**: Knee angle not reaching threshold (160°).

**Solution**: Lower `max_standing_angle` in config.py:
```python
"max_standing_angle": 155.0,  # Was 160.0
```

### Issue: Too Many Reps Counted

**Check logs for**:
```
🔄 Phase transition: descending → standing (angle: 161.0°)
```

**Problem**: Skipping phases (not going through bottom).

**Solution**: Raise `min_squat_angle` to require deeper squats:
```python
"min_squat_angle": 110.0,  # Was 100.0
```

### Issue: Audio is Very Late (> 3 seconds)

**Check logs for**:
```
⏱️  Processing latency: 450.2ms
```

**Problem**: Processing is slow.

**Solution**: Reduce image size and workers:
```python
YOLO_CONFIG = {
    "imgsz": 320,  # Smaller = faster
    "max_workers": 8,  # Fewer workers
}
```

### Issue: No Pose Detection

**Check logs for**:
```
❌ Error in pose processing: ...
```

**Problem**: YOLO model not loading or person not visible.

**Solution**:
1. Ensure `yolo11n-pose.pt` exists in directory
2. Check camera shows full body
3. Improve lighting

## Advanced Debugging

### Enable Debug Logging

For even more detailed logs:

```python
# main.py
logging.basicConfig(
    level=logging.DEBUG,  # Was INFO
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
```

This will show:
- Every phase transition
- Periodic state updates
- All form issues
- Detailed processing info

### Log to File

To save logs for analysis:

```python
# main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('squat_counter.log'),
        logging.StreamHandler()
    ]
)
```

Now logs are saved to `squat_counter.log` for review.

### Analyze Timing

Create a script to analyze log timing:

```python
# analyze_logs.py
import re
from datetime import datetime

with open('squat_counter.log', 'r') as f:
    lines = f.readlines()

rep_times = []
for line in lines:
    if 'REP COUNTED' in line:
        # Extract timestamp
        match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if match:
            rep_times.append(match.group(1))

# Calculate intervals between reps
for i in range(1, len(rep_times)):
    print(f"Rep {i} to Rep {i+1}: {rep_times[i]} - {rep_times[i-1]}")
```

## Performance Benchmarks

Expected performance on different hardware:

### MacBook Pro M1/M2
- Processing: 30-50ms per frame
- FPS: 30
- Total latency: 1-2 seconds

### MacBook Air Intel
- Processing: 80-120ms per frame
- FPS: 20-25
- Total latency: 2-3 seconds

### Windows Desktop (GPU)
- Processing: 20-40ms per frame
- FPS: 30+
- Total latency: 1-2 seconds

### Windows Laptop (CPU only)
- Processing: 100-200ms per frame
- FPS: 15-20
- Total latency: 2-4 seconds

## Getting Help

If you're still having issues:

1. **Collect logs**: Run with logging enabled
2. **Note timestamps**: When rep happens vs when you hear audio
3. **Check system**: CPU usage, network speed
4. **Share logs**: Include relevant log excerpts

## Example Log Sequence

Here's what a successful squat looks like in logs:

```
14:23:40.100 | squat_counter_processor | DEBUG | 📊 Current state: phase=standing, knee=165.2°, reps=0
14:23:41.200 | squat_counter_processor | DEBUG | 🔄 Phase transition: standing → descending (angle: 155.3°)
14:23:42.300 | squat_counter_processor | DEBUG | 🔄 Phase transition: descending → bottom (angle: 98.5°)
14:23:43.400 | squat_counter_processor | DEBUG | 🔄 Phase transition: bottom → ascending (angle: 105.2°)
14:23:44.500 | squat_counter_processor | DEBUG | 🔄 Phase transition: ascending → standing (angle: 162.3°)
14:23:44.501 | squat_counter_processor | INFO | 🎯 REP COUNTED! #1 | Timestamp: 1699876424.501 | Knee angle: 162.3°
14:23:44.502 | squat_counter_processor | INFO | ✅ Total reps: 1
14:23:44.503 | squat_counter_processor | INFO | 📢 Message generated: 'Rep 1 complete! 🎉' at 1699876424.503
14:23:44.550 | squat_yolo_processor | INFO | ⏱️  Processing latency: 50.2ms | Rep #1 detected
```

Total processing time: **50ms** (very fast!)

Then you wait for:
- LLM to process: ~1000ms
- Audio to generate: ~500ms
- Network: ~200ms

**Total latency**: ~1750ms (1.75 seconds)

