# SAM3 Plugin for Vision Agents

Real-time video segmentation using Meta's Segment Anything Model 3 (SAM3) via Hugging Face Transformers.

## Installation

```bash
pip install transformers>=4.40.0 torch>=2.0.0
```

## Authentication

Request access at https://huggingface.co/facebook/sam3, then:

```bash
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

## Usage

See the `example/` directory for a complete working example.

```python
from vision_agents.plugins import sam3

# Create processor
sam3_processor = sam3.VideoSegmentationProcessor(
    text_prompt="person",  # Segments all people in the video
    threshold=0.5,
    fps=30
)

# Register with agent so AI can change what to segment
@agent.llm.register_function(description="Change segmentation target")
async def change_prompt(prompt: str) -> dict:
    return await sam3_processor.change_prompt(prompt)
```

## Features

- **Open vocabulary segmentation**: Segment any object using text prompts
- **Real-time processing**: Works with live video streams
- **Dynamic prompts**: AI can change what to segment via function calls
- **GPU accelerated**: Automatic CUDA support

## License

MIT
