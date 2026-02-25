# HuggingFace Plugin for Vision Agents

HuggingFace integration for Vision Agents. Supports cloud-based inference via HuggingFace's Inference Providers API and local on-device inference via Transformers.

## Installation

```bash
# Cloud inference (HuggingFace Inference API)
uv add vision-agents-plugins-huggingface

# Local inference (Transformers - LLM, VLM, object detection)
uv add "vision-agents-plugins-huggingface[transformers]"

# Local inference with quantization (4-bit / 8-bit)
uv add "vision-agents-plugins-huggingface[transformers-quantized]"
```

## Cloud Inference (API-based)

### Configuration

```bash
export HF_TOKEN=your_huggingface_token
```

### Text-only LLM

```python
from vision_agents.plugins import huggingface

llm = huggingface.LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="together",  # or "groq", "cerebras", etc.
)

response = await llm.simple_response("Hello, how are you?")
print(response.text)
```

### Vision Language Model (VLM)

```python
from vision_agents.plugins import huggingface

vlm = huggingface.VLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    fps=1,
    frame_buffer_seconds=10,
)

response = await vlm.simple_response("What do you see?")
print(response.text)
```

## Local Inference (Transformers)

Runs models directly on your hardware (GPU/CPU/MPS). Requires the `[transformers]` extra.

### Local LLM

```python
from vision_agents.plugins import huggingface

llm = huggingface.TransformersLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
)

# With 4-bit quantization (~4x memory reduction)
llm = huggingface.TransformersLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    quantization="4bit",
)
```

**Parameters:**
- `model` (str): HuggingFace model ID
- `device`: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`
- `quantization`: `"none"`, `"4bit"`, or `"8bit"`
- `torch_dtype`: `"auto"`, `"float16"`, `"bfloat16"`, or `"float32"`
- `max_new_tokens` (int): Max tokens per response (default: 512)

### Local VLM

```python
from vision_agents.plugins import huggingface

vlm = huggingface.TransformersVLM(
    model="Qwen/Qwen2-VL-2B-Instruct",
)
```

**Parameters:**
- `model` (str): HuggingFace model ID
- `device`: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`
- `quantization`: `"none"`, `"4bit"`, or `"8bit"`
- `fps` (int): Frames per second to capture (default: 1)
- `frame_buffer_seconds` (int): Seconds of video to buffer (default: 10)
- `max_frames` (int): Max frames per inference (default: 4)

### Local Object Detection

Runs detection models like RT-DETRv2 on video frames and emits `DetectionCompletedEvent` with bounding boxes.

```python
from vision_agents.core import Agent
from vision_agents.plugins import huggingface

processor = huggingface.TransformersDetectionProcessor(
    model="PekingU/rtdetr_v2_r101vd",
    conf_threshold=0.5,
    fps=5,
)

agent = Agent(processors=[processor], ...)

@agent.events.subscribe
async def on_detection(event: huggingface.DetectionCompletedEvent):
    for obj in event.objects:
        print(f"{obj['label']} ({obj['confidence']:.0%})")
```

**Parameters:**
- `model` (str): HuggingFace model ID (default: `"PekingU/rtdetr_v2_r101vd"`)
- `conf_threshold` (float): Confidence threshold 0-1 (default: 0.5)
- `fps` (int): Frame processing rate (default: 10)
- `classes` (list[str], optional): Filter to specific class names
- `device`: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`
- `annotate` (bool): Draw bounding boxes on output video (default: True)
