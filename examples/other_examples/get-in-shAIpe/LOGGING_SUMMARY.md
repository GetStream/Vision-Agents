# Logging Improvements Summary

## What Was Added

Added comprehensive logging to help debug timing issues and verify the squat counter is working correctly.

## New Log Messages

### 1. Rep Counted (🎯)
```
🎯 REP COUNTED! #1 | Timestamp: 1699876425.123 | Knee angle: 162.3°
✅ Total reps: 1
```
**When**: Immediately when a squat rep is completed
**Purpose**: Confirm counter is working and show exact timing

### 2. Message Generated (📢)
```
📢 Message generated: 'Rep 1 complete! 🎉' at 1699876425.125
```
**When**: Right after rep is counted
**Purpose**: Show when data is sent to LLM (before audio is generated)

### 3. Processing Latency (⏱️)
```
⏱️  Processing latency: 50.2ms | Rep #1 detected
```
**When**: After each rep is detected
**Purpose**: Monitor processing performance (should be < 100ms)

### 4. Phase Transitions (🔄)
```
🔄 Phase transition: ascending → standing (angle: 162.3°)
```
**When**: During squat movement when phase changes
**Purpose**: Verify phase detection is working correctly

### 5. Current State (📊)
```
📊 Current state: phase=descending, knee=120.5°, reps=3
```
**When**: Every 10 frames (~300ms at 30 FPS)
**Purpose**: Periodic status updates for monitoring

### 6. Form Issues (⚠️)
```
⚠️  Form issues detected: Go deeper, Keep knees behind toes
```
**When**: Form problems are detected
**Purpose**: Show what form corrections are being flagged

## Timestamp Format

All logs now include millisecond-precision timestamps:

```
HH:MM:SS.mmm | module_name | LEVEL | message
14:23:45.123 | squat_counter_processor | INFO | 🎯 REP COUNTED! #1
```

This allows you to:
- Track exact timing of events
- Calculate latency between detection and audio
- Debug performance issues

## How to Use

### Quick Check: Is Counter Working?

Run the app and do a squat. You should see:

```
🎯 REP COUNTED! #1 | Timestamp: XXXXX.XXX | Knee angle: XXX.X°
✅ Total reps: 1
```

If you see this, the counter is working! 

### Check Audio Latency

1. Do a squat
2. Note the timestamp when you see `🎯 REP COUNTED!`
3. Note the time when you hear the audio
4. Calculate difference

Example:
- Rep counted: `14:23:45.123`
- Audio heard: `14:23:47.000`
- Latency: `1.877 seconds` ✅ (good!)

### Monitor Performance

Watch for `⏱️ Processing latency` messages:
- < 100ms: Excellent ✅
- 100-200ms: Good 👍
- 200-500ms: Acceptable ⚠️
- > 500ms: Slow, needs optimization ❌

### Debug Phase Detection

Enable DEBUG logging to see all phase transitions:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

You'll see the complete squat cycle:
```
🔄 Phase transition: standing → descending
🔄 Phase transition: descending → bottom
🔄 Phase transition: bottom → ascending
🔄 Phase transition: ascending → standing
🎯 REP COUNTED! #1
```

## Files Modified

1. **squat_counter_processor.py**
   - Added logging in `_update_phase()` for rep counting
   - Added logging in `process_pose_data()` for state updates
   - Added timestamps to all events

2. **squat_yolo_processor.py**
   - Added processing latency logging
   - Added timing measurements

3. **main.py**
   - Configured logging format with timestamps
   - Set default log level to INFO

## Configuration

### Change Log Level

In `main.py`:

```python
# Show everything (very verbose)
logging.basicConfig(level=logging.DEBUG, ...)

# Show important events only (default)
logging.basicConfig(level=logging.INFO, ...)

# Show only warnings and errors
logging.basicConfig(level=logging.WARNING, ...)
```

### Save Logs to File

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('squat_counter.log'),  # Save to file
        logging.StreamHandler()  # Also print to console
    ]
)
```

## Understanding Audio Latency

The logs help you understand where time is spent:

```
14:23:45.123 | 🎯 REP COUNTED!           <- Detection happens (instant)
14:23:45.125 | 📢 Message generated      <- Data sent to LLM (2ms later)
14:23:45.175 | ⏱️  Processing: 50.2ms    <- Total processing time
[LLM processing: ~1000ms]                <- Gemini generates response
[Audio generation: ~500ms]               <- TTS creates audio
[Network: ~200ms]                        <- Audio transmitted
14:23:47.000 | You hear: "Nice squat!"  <- Total: ~1.9 seconds
```

### Breakdown:
- **Local processing**: 50ms (shown in logs)
- **LLM processing**: 1000ms (not shown, happens on server)
- **Audio generation**: 500ms (not shown, TTS service)
- **Network**: 200ms (not shown, varies)
- **Total**: ~1750ms (1.75 seconds)

This is **normal and expected**! The local processing is fast (50ms), but waiting for the LLM and audio takes time.

## Troubleshooting with Logs

### Problem: No "REP COUNTED" messages

**Check logs for**:
```
📊 Current state: phase=ascending, knee=155.2°, reps=0
```

**Diagnosis**: Knee angle not reaching standing threshold (160°)

**Solution**: Lower `max_standing_angle` in config.py

### Problem: Too many "REP COUNTED" messages

**Check logs for**:
```
🔄 Phase transition: descending → standing (angle: 161.0°)
```

**Diagnosis**: Skipping bottom phase (not going deep enough)

**Solution**: Raise `min_squat_angle` in config.py

### Problem: Slow performance

**Check logs for**:
```
⏱️  Processing latency: 450.2ms
```

**Diagnosis**: Processing is taking too long

**Solution**: Reduce `imgsz` and `max_workers` in config.py

## Next Steps

1. **Run the app** and watch the logs
2. **Do some squats** and verify you see `🎯 REP COUNTED!`
3. **Check timing** between detection and audio
4. **Adjust config** if needed based on log feedback

For detailed debugging instructions, see [DEBUGGING.md](DEBUGGING.md).

