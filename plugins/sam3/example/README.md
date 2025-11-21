# SAM3 Video Segmentation Example

This example demonstrates real-time video segmentation using SAM3 via Hugging Face Transformers with dynamic prompt changing.

## What This Does

- Creates a voice AI agent that can see your video feed
- Segments objects in real-time using Meta's SAM3 model
- AI can dynamically change what's being segmented via function calls
- Displays colored masks and bounding boxes on detected objects

## Setup

1. **Install dependencies:**
```bash
cd example
uv sync
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - HF_TOKEN (for SAM3 model access)
# - GEMINI_API_KEY (for Gemini LLM)
# - GETSTREAM_API_KEY (for video transport)
# - GETSTREAM_API_SECRET
```

3. **Get SAM3 access:**
```bash
# Request access at https://huggingface.co/facebook/sam3
# Then authenticate:
export HF_TOKEN=your_token_here
# or
huggingface-cli login
```

## Running

```bash
uv run sam3_example.py
```

This will:
1. Start the agent with SAM3 video segmentation
2. Initial segmentation target is "person"
3. Connect to GetStream's edge network
4. Wait for you to join the call

## Using the Agent

Once connected, you can talk to the agent:

**Example conversations:**

**You:** "Segment people in the video"  
**Agent:** *Calls change_prompt("person")* "Sure! I'm now segmenting all people in your video."

**You:** "Now segment cars"  
**Agent:** *Calls change_prompt("car")* "Switched to car detection!"

**You:** "Find all the dogs"  
**Agent:** *Calls change_prompt("dog")* "Now looking for dogs in the video!"

**You:** "Show me basketballs"  
**Agent:** *Calls change_prompt("basketball")* "I'll highlight all basketballs!"

## How It Works

### 1. SAM3 Processor
```python
sam3_processor = sam3.VideoSegmentationProcessor(
    text_prompt="person",  # Initial target
    threshold=0.5,         # Confidence threshold
    fps=30,                # Processing rate
)
```

### 2. Function Registration
The `change_prompt` function is registered with the LLM so the AI can call it:
```python
@agent.llm.register_function(description="Change what to segment")
async def change_prompt(prompt: str) -> dict:
    return await sam3_processor.change_prompt(prompt)
```

### 3. AI Makes Decisions
When you ask to segment something, the AI:
1. Understands your request
2. Calls `change_prompt` with the appropriate object
3. Confirms the change

### 4. Real-time Processing
- Video frames are processed at 30 FPS
- SAM3 finds ALL instances of the specified object
- Results are overlaid on the video with:
  - Colored masks (30% transparent)
  - Bounding boxes
  - Labels with confidence scores

## Supported Prompts

SAM3 supports open-vocabulary segmentation. Try:

**Common Objects:**
- "person", "car", "dog", "cat", "bicycle"

**Specific Items:**
- "basketball", "laptop", "coffee cup", "smartphone"

**Complex Concepts:**
- "person wearing red shirt"
- "black car"
- "sports equipment"

## Configuration

Edit `sam3_example.py` to adjust:

**Processing Rate:**
```python
sam3_processor = sam3.VideoSegmentationProcessor(
    fps=15,  # Lower for better performance
)
```

**Detection Thresholds:**
```python
sam3_processor = sam3.VideoSegmentationProcessor(
    threshold=0.3,      # Lower = more detections
    mask_threshold=0.5,  # Mask quality
)
```

**LLM:**
```python
# Use OpenAI instead of Gemini
from vision_agents.plugins import openai
agent = Agent(
    llm=openai.Realtime(fps=3),
    # ...
)
```

## Troubleshooting

### "Model requires authentication"
- Request access at https://huggingface.co/facebook/sam3
- Run `huggingface-cli login` or set `HF_TOKEN`

### Slow performance
- Lower FPS: `fps=15` instead of `fps=30`
- Check GPU: The processor should use CUDA automatically
- Lower thresholds if too many false positives

### No detections
- Try simpler prompts: "person" instead of "person with blue hat"
- Lower threshold: `threshold=0.3`
- Check if objects are actually visible

### API Key Errors
- Make sure all keys are set in `.env`
- Check GetStream dashboard for valid credentials
- Verify Gemini API key is active

## Architecture

```
User Video Feed
     ↓
SAM3 Processor (segments objects)
     ↓
Annotated Video (with masks/boxes)
     ↓
Gemini Realtime LLM (sees video + receives function calls)
     ↓
GetStream Edge (video transport)
     ↓
Client (browser/mobile app)
```

## Next Steps

- Try different prompts and see what SAM3 can segment
- Adjust thresholds for your use case
- Combine with other processors (e.g., pose detection)
- Build a custom UI for your application

## Learn More

- [SAM3 Plugin Documentation](../README.md)
- [Vision Agents Documentation](https://visionagents.ai)
- [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)
