# Get-in-shAIpe: AI Squat Coach 💪

An intelligent squat counting and coaching application that uses computer vision and biomechanical analysis to accurately track squats and provide real-time form feedback.

## Features

### 🎯 Accurate Squat Counting
- **Biomechanical Analysis**: Uses joint angle calculations (hip, knee, ankle) to detect squat phases
- **State Machine Logic**: Tracks squat phases (standing → descending → bottom → ascending → standing)
- **No Manual Counting**: The LLM doesn't count - a dedicated processor handles it with precision

### 📊 Real-time Form Analysis
- **Depth Detection**: Ensures squats reach proper depth (thighs parallel to ground)
- **Knee Alignment**: Checks that knees stay behind toes
- **Phase Tracking**: Shows current squat phase on video overlay

### 🎥 Visual Feedback
- **Rep Counter**: Large, visible rep count on video
- **Phase Indicator**: Color-coded phase display
- **Angle Display**: Shows knee angle in real-time
- **Form Warnings**: On-screen alerts for form issues

### 🤖 AI Coaching
- **Motivational Feedback**: Encouraging comments after each rep
- **Form Corrections**: Gentle guidance when form issues detected
- **Milestone Celebrations**: Special feedback at 5, 10, 20 reps
- **Phase-Specific Tips**: Coaching based on current squat phase

## How It Works

### Architecture

1. **YOLO Pose Detection** (`YOLOPoseProcessor`)
   - Detects body keypoints using YOLO11n-pose model
   - Provides 17 keypoint coordinates with confidence scores

2. **Squat Analysis** (`SquatCounterProcessor`)
   - Calculates joint angles from keypoints
   - Implements state machine for phase detection
   - Counts reps when full cycle completes
   - Detects form issues

3. **Combined Processor** (`SquatYOLOProcessor`)
   - Integrates YOLO + squat counter
   - Adds visual overlays to video
   - Provides structured data to LLM

4. **AI Coach** (Gemini Realtime)
   - Receives squat data from processor
   - Provides voice feedback and motivation
   - Reacts to rep counts and form issues

### Squat Detection Algorithm

The system uses a **biomechanical approach** to detect squats:

1. **Angle Calculation**: 
   - Knee angle: hip → knee → ankle
   - Hip angle: shoulder → hip → knee

2. **Phase Detection**:
   - **Standing**: knee angle > 160°
   - **Descending**: angle decreasing
   - **Bottom**: knee angle < 100°
   - **Ascending**: angle increasing

3. **Rep Counting**:
   - Rep counted when transitioning from ascending → standing
   - Ensures full range of motion

4. **Form Analysis**:
   - Depth check: knee angle at bottom position
   - Knee alignment: horizontal distance knee-to-ankle

## Configuration

### Tuning Parameters

Adjust these in `main.py` when creating `SquatYOLOProcessor`:

```python
SquatYOLOProcessor(
    model_path="yolo11n-pose.pt",
    min_squat_angle=100.0,      # Lower = deeper squat required
    max_standing_angle=160.0,    # Higher = more upright required
    conf_threshold=0.5,          # Keypoint confidence threshold
    enable_hand_tracking=False,  # Disable for better performance
)
```

### Parameter Guide

- **min_squat_angle** (default: 100°)
  - Lower values require deeper squats
  - Recommended: 90-110° depending on user flexibility
  - Too low: may miss reps from less flexible users
  - Too high: may count partial squats

- **max_standing_angle** (default: 160°)
  - Higher values require more upright standing
  - Recommended: 155-165°
  - Too low: may count reps prematurely
  - Too high: may require hyperextension

- **conf_threshold** (default: 0.5)
  - Minimum confidence for keypoint detection
  - Recommended: 0.4-0.6
  - Lower: more detections but less accurate
  - Higher: fewer false positives but may miss frames

## Installation

```bash
cd examples/other_examples/get-in-shAIpe
uv sync
```

## Usage

1. Set up environment variables:
```bash
cp .env.example .env
# Add your API keys:
# - STREAM_API_KEY
# - STREAM_API_SECRET
# - GOOGLE_API_KEY (for Gemini)
```

2. Run the application:
```bash
uv run main.py
```

3. Join the video call and start doing squats!

## Logging and Debugging

The app now includes detailed logging to help you understand timing and debug issues.

### What You'll See in Logs

When you run the app, you'll see timestamped logs like:

```
14:23:45.123 | squat_counter_processor | INFO | 🎯 REP COUNTED! #1 | Timestamp: 1699876425.123 | Knee angle: 162.3°
14:23:45.124 | squat_counter_processor | INFO | ✅ Total reps: 1
14:23:45.125 | squat_counter_processor | INFO | 📢 Message generated: 'Rep 1 complete! 🎉' at 1699876425.125
14:23:45.175 | squat_yolo_processor | INFO | ⏱️  Processing latency: 50.2ms | Rep #1 detected
```

### Key Log Messages

- **🎯 REP COUNTED!** - A squat was successfully detected and counted
- **⏱️ Processing latency** - How long it took to process the frame (should be < 100ms)
- **📢 Message generated** - When data is sent to LLM for voice feedback
- **🔄 Phase transition** - When squat phase changes (standing → descending → bottom → ascending)
- **📊 Current state** - Periodic updates every 10 frames showing phase, angle, and rep count

### Understanding Audio Latency

If audio seems late, check the logs:

1. **Find when rep was counted**: Look for `🎯 REP COUNTED!` timestamp
2. **Note when you hear audio**: Check your clock
3. **Calculate difference**: This is your total latency

**Expected latency**: 1-2 seconds
- Processing: ~50-100ms (shown in logs)
- LLM response: ~500-1500ms
- Audio generation: ~200-500ms
- Network: ~50-200ms

**If latency > 3 seconds**: See [DEBUGGING.md](DEBUGGING.md) for optimization tips.

### Enable More Detailed Logs

For debugging, enable DEBUG level logging:

```python
# main.py
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO
    ...
)
```

This shows:
- Every phase transition
- Periodic state updates (every 10 frames)
- Form issues detected
- Detailed processing information

See [DEBUGGING.md](DEBUGGING.md) for complete debugging guide.

## Troubleshooting

### Counter Not Working

**Problem**: Squats not being counted

**Solutions**:
1. Ensure full body is visible in camera
2. Check lighting - need clear view of joints
3. Lower `min_squat_angle` if you can't go deep enough
4. Increase `max_standing_angle` if not standing fully upright
5. Check logs for keypoint detection confidence

### Counting Too Many Reps

**Problem**: Partial movements being counted

**Solutions**:
1. Increase `min_squat_angle` to require deeper squats
2. Increase `max_standing_angle` to require fuller standing
3. Ensure smooth movements (no bouncing)

### Counting Too Few Reps

**Problem**: Valid squats not being counted

**Solutions**:
1. Decrease `min_squat_angle` for shallower squats
2. Decrease `max_standing_angle` for less upright standing
3. Ensure you're completing full range of motion
4. Check that all body parts are visible

### Form Issues Not Detected

**Problem**: Bad form not being flagged

**Solutions**:
1. Ensure side profile is visible (not front-facing)
2. Check camera angle - should see full body
3. Adjust form detection thresholds in `squat_counter_processor.py`

## Customization

### Adjusting Form Detection

Edit `squat_counter_processor.py`:

```python
def _detect_form_issues(self, keypoints, knee_angle):
    issues = []
    
    # Adjust depth threshold
    if self.current_phase == SquatPhase.BOTTOM and knee_angle > 110:  # Change 110
        issues.append("Go deeper")
    
    # Adjust knee-to-toe distance
    if left_knee[0] > left_ankle[0] + 50:  # Change 50 (pixels)
        issues.append("Keep knees behind toes")
    
    return issues
```

### Changing Visual Overlay

Edit `squat_yolo_processor.py` in `_draw_squat_info()` method to customize:
- Text size and position
- Colors
- Information displayed

### Modifying AI Coaching

Edit `squat-coach.md` to change:
- Coaching style and tone
- Milestone thresholds
- Types of feedback provided

## Technical Details

### Performance

- **FPS**: Processes at ~30 FPS on CPU
- **Latency**: ~50-100ms from movement to count
- **Accuracy**: >95% rep counting accuracy with proper setup

### Requirements

- Python 3.12+
- Webcam or video input
- Good lighting conditions
- Full body visible in frame

### Dependencies

- `vision-agents`: Core framework
- `vision-agents-plugins-getstream`: Video streaming
- `vision-agents-plugins-ultralytics`: YOLO pose detection
- `vision-agents-plugins-gemini`: AI coaching
- `opencv-python`: Image processing
- `numpy`: Numerical computations

## Future Improvements

- [ ] Add support for other exercises (push-ups, lunges, etc.)
- [ ] Track workout history and progress
- [ ] Add rep speed analysis
- [ ] Implement set tracking
- [ ] Add rest timer between sets
- [ ] Support multiple users simultaneously
- [ ] Add exercise recommendations based on form

## Credits

Built with Vision Agents framework using:
- YOLO11n-pose for pose detection
- Gemini Realtime for AI coaching
- GetStream for video infrastructure

