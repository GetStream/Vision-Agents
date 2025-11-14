# Squat Counter Tuning Guide

This guide helps you optimize the squat counter for your specific needs.

## Quick Start

All configuration is in `config.py`. Start by adjusting these two main parameters:

```python
"min_squat_angle": 100.0,    # How deep squats need to be
"max_standing_angle": 160.0,  # How upright standing needs to be
```

## Common Issues & Solutions

### 🔴 Problem: Counter is too strict (missing valid squats)

**Symptoms:**
- You do a squat but it doesn't count
- Counter seems to "skip" some reps

**Solutions:**

1. **Lower the min_squat_angle** (make squats easier to count)
   ```python
   "min_squat_angle": 110.0,  # Was 100.0 - allows shallower squats
   ```

2. **Lower the max_standing_angle** (don't require full standing)
   ```python
   "max_standing_angle": 155.0,  # Was 160.0 - less upright needed
   ```

3. **Lower confidence threshold** (accept less certain detections)
   ```python
   "confidence_threshold": 0.4,  # Was 0.5 - more lenient
   ```

### 🟡 Problem: Counter is too lenient (counting partial reps)

**Symptoms:**
- Small movements are being counted as squats
- Counter increments when you're just shifting weight

**Solutions:**

1. **Raise the min_squat_angle** (require deeper squats)
   ```python
   "min_squat_angle": 90.0,  # Was 100.0 - requires deeper squats
   ```

2. **Raise the max_standing_angle** (require fuller standing)
   ```python
   "max_standing_angle": 165.0,  # Was 160.0 - more upright required
   ```

3. **Raise confidence threshold** (only count confident detections)
   ```python
   "confidence_threshold": 0.6,  # Was 0.5 - more strict
   ```

### 🟢 Problem: Form feedback is annoying/not helpful

**Symptoms:**
- Getting "go deeper" when you think you're deep enough
- Form warnings appearing too often

**Solutions:**

1. **Adjust form thresholds in config.py:**
   ```python
   FORM_CONFIG = {
       "shallow_squat_threshold": 120,  # Raise to be less strict about depth
       "knee_forward_threshold": 70,    # Raise to allow knees more forward
   }
   ```

2. **Disable specific form checks in squat_counter_processor.py:**
   ```python
   def _detect_form_issues(self, keypoints, knee_angle):
       issues = []
       # Comment out checks you don't want:
       # if knee_angle > 110:
       #     issues.append("Go deeper")
       return issues
   ```

### 🔵 Problem: Counter is laggy or slow

**Symptoms:**
- Video is choppy
- Delay between movement and count

**Solutions:**

1. **Reduce image size:**
   ```python
   YOLO_CONFIG = {
       "imgsz": 384,  # Was 512 - faster processing
   }
   ```

2. **Reduce worker threads:**
   ```python
   "max_workers": 12,  # Was 24 - less CPU usage
   ```

3. **Lower LLM FPS:**
   ```python
   LLM_CONFIG = {
       "fps": 1,  # Was 3 - less frequent AI feedback
   }
   ```

## Understanding the Angles

### Knee Angle Visualization

```
Standing (160°+):
    |  <- shoulder
    |  <- hip
    |  <- knee (almost straight)
    |  <- ankle

Bottom (90-100°):
    |  <- shoulder
   /   <- hip
  /    <- knee (bent ~90°)
 |     <- ankle
```

### How to Find Your Perfect Settings

1. **Start with defaults:**
   - min_squat_angle: 100°
   - max_standing_angle: 160°

2. **Test with 5 squats** and observe:
   - How many were counted?
   - Were any missed?
   - Were any false positives?

3. **Adjust based on results:**
   - If 3/5 counted → Too strict, lower min_squat_angle by 10°
   - If 7/5 counted → Too lenient, raise min_squat_angle by 10°
   - If standing not detected → Lower max_standing_angle by 5°

4. **Fine-tune in 5° increments** until you get 5/5 accuracy

## Advanced Tuning

### For Different User Types

**Beginners / Less Flexible:**
```python
"min_squat_angle": 110.0,    # Shallower squats OK
"max_standing_angle": 155.0,  # Don't need full extension
```

**Athletes / Advanced:**
```python
"min_squat_angle": 90.0,     # Require deep squats
"max_standing_angle": 165.0,  # Require full standing
```

**Rehabilitation / Elderly:**
```python
"min_squat_angle": 120.0,    # Very shallow squats
"max_standing_angle": 150.0,  # Partial standing OK
"confidence_threshold": 0.4,  # More forgiving detection
```

### Camera Setup Tips

For best results:

1. **Position camera at hip height** - not too high or low
2. **Stand 6-8 feet from camera** - full body visible
3. **Side profile works best** - 90° angle to camera
4. **Good lighting** - avoid backlighting
5. **Solid background** - avoid busy patterns

### Debugging

Enable detailed logging in main.py:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Detected angles in real-time
- Phase transitions
- Why reps are/aren't counted

## Testing Your Configuration

Create a test script to validate settings:

```python
# test_config.py
from config import SQUAT_CONFIG

print("Current Configuration:")
print(f"Min squat angle: {SQUAT_CONFIG['min_squat_angle']}°")
print(f"Max standing angle: {SQUAT_CONFIG['max_standing_angle']}°")
print(f"Range of motion required: {SQUAT_CONFIG['max_standing_angle'] - SQUAT_CONFIG['min_squat_angle']}°")

# Recommendations
rom = SQUAT_CONFIG['max_standing_angle'] - SQUAT_CONFIG['min_squat_angle']
if rom < 50:
    print("⚠️  Warning: Range of motion is small - may count partial reps")
elif rom > 70:
    print("⚠️  Warning: Range of motion is large - may miss reps")
else:
    print("✅ Range of motion looks good!")
```

## Getting Help

If you're still having issues:

1. Check the logs for error messages
2. Verify camera can see full body
3. Test with different lighting
4. Try the "Beginners" preset first
5. Gradually make it more strict

## Preset Configurations

Copy these into your `config.py`:

### Preset 1: Lenient (Good for testing)
```python
SQUAT_CONFIG = {
    "min_squat_angle": 110.0,
    "max_standing_angle": 155.0,
    "confidence_threshold": 0.4,
}
```

### Preset 2: Standard (Recommended default)
```python
SQUAT_CONFIG = {
    "min_squat_angle": 100.0,
    "max_standing_angle": 160.0,
    "confidence_threshold": 0.5,
}
```

### Preset 3: Strict (For athletes)
```python
SQUAT_CONFIG = {
    "min_squat_angle": 90.0,
    "max_standing_angle": 165.0,
    "confidence_threshold": 0.6,
}
```

