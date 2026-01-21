# Golf Coach Example

This example shows you how to build a real-time golf coaching AI using [Vision Agents](https://visionagents.ai/). The
agent uses video processing to watch golf swings and provide feedback through voice conversation.

In this example, the AI golf coach will:

- Watches video of the user's golf swing
- Uses [YOLO](https://www.ultralytics.com/yolo) pose detection to analyze body position and movement
- Processes the video in real-time with an LLM (Large Language Model)
- Provides voice feedback on the swing technique
- Runs on Stream's low-latency edge network

This approach combines a fast object detection model (YOLO) with a full realtime AI. You can apply this pattern to other
video AI use cases like sports coaching, physical therapy, workout coaching, or drone monitoring.

## Prerequisites

- Python 3.13 or higher
- API keys for:
    - [Gemini](https://ai.google.dev/) (for realtime LLM with vision)
    - [Stream](https://getstream.io/) (for video/audio infrastructure)
    - Alternatively: [OpenAI](https://openai.com) (if using OpenAI Realtime instead)

## Installation

1. Go to the example's directory
    ```bash
    cd examples/02_golf_coach_example
    ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

   If using OpenAI instead of Gemini, also add:
   ```
   OPENAI_API_KEY=your_openai_key
   ```

## Running the Example

Run the agent:

```bash
uv run golf_coach_example.py run
```

The agent will:

1. Create a video call
2. Open a demo UI in your browser
3. Join the call and start watching
4. Ask you to do a golf swing
5. Analyze your swing and provide feedback

## Code Walkthrough

### Setting Up the Agent

The code creates an agent with video processing capabilities:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="AI golf coach"),
    instructions="Read @golf_coach.md",
    llm=gemini.Realtime(fps=10),
    processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")],
)
```

**Components:**

- `edge`: Handles low-latency audio/video transport
- `agent_user`: Sets the agent's name and ID
- `instructions`: Loads coaching instructions from `golf_coach.md`
- `llm`: The language model that powers the conversation and video analysis
- `processors`: Video processing pipeline that runs alongside the LLM

### Understanding Processors

Processors enable the agent to analyze video in real-time. The `YOLOPoseProcessor` detects human poses and body
positions in each video frame. This information is sent to the LLM so it can understand the user's body movement during
the golf swing.

The `fps=10` parameter means the LLM processes 10 frames per second. Higher FPS gives more detail but costs more.

### Instructions File

The `golf_coach.md` file contains detailed coaching guidelines. It tells the agent:

- How to behave (personality and tone)
- What to look for in a golf swing
- How to provide feedback
- Golf coaching best practices

You can modify this file to change the coaching style or add more specific guidance.

## Customization

### Change the FPS

Adjust how many frames per second the LLM processes:

```python
llm = gemini.Realtime(fps=5)  # Lower FPS = less expensive
llm = gemini.Realtime(fps=15)  # Higher FPS = more detailed analysis
```

### Use OpenAI Instead of Gemini

Switch to OpenAI's realtime API:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My happy AI friend", id="agent"),
    instructions="You're a video AI assistant...",
    llm=openai.Realtime(fps=10)
)
```

Both models support video processing with YOLO.

### Modify the Coaching Style

Edit the `golf_coach.md` file to change:

- The agent's personality
- The coaching focus areas
- The level of detail in feedback
- The voice and tone

### Use Different YOLO Models

Try other YOLO models for different use cases:

```python
# For general object detection
ultralytics.YOLOProcessor(model_path="yolo11n.pt")

# For pose detection (current)
ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
```

## How It Works

1. **Video Capture**: The user's camera feeds video to the agent
2. **Pose Detection**: YOLO analyzes each frame and extracts body position data
3. **LLM Processing**: The realtime LLM receives both the video and pose data
4. **Analysis**: The LLM watches the swing and evaluates technique
5. **Feedback**: The agent speaks feedback based on coaching guidelines

## Learn More

- [Building a Voice AI app](https://visionagents.ai/introduction/voice-agents)
- [Building a Video AI app](https://visionagents.ai/introduction/video-agents)
- [Main Vision Agents README](../../README.md)
- [Simple Agent Example](../01_simple_agent_example) - Start here if you're new to Vision Agents

