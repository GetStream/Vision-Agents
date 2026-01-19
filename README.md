<img width="1280" height="360" alt="Readme" src="assets/repo_image.png" />

# Open Vision Agents by Stream

[![build](https://github.com/GetStream/Vision-Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/GetStream/Vision-Agents/actions)
[![PyPI version](https://badge.fury.io/py/vision-agents.svg)](http://badge.fury.io/py/vision-agents)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vision-agents.svg)
[![License](https://img.shields.io/github/license/GetStream/Vision-Agents)](https://github.com/GetStream/Vision-Agents/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1108586339550638090)](https://discord.gg/RkhX9PxMS6)

---

## Build Real-Time Vision AI Agents

https://github.com/user-attachments/assets/d9778ab9-938d-4101-8605-ff879c29b0e4


### Multi-modal AI agents that watch, listen, and understand video.

Vision Agents give you the building blocks to create intelligent, low-latency video experiences powered by your models, your infrastructure, and your use cases.

### Key Highlights

- **Video AI:** Built for real-time video AI. Combine YOLO, Roboflow, and others with Gemini/OpenAI in real-time.
- **Low Latency:** Join quickly (500ms) and maintain audio/video latency under 30ms using [Stream's edge network](https://getstream.io/video/).
- **Open:** Built by Stream, but works with any video edge network.
- **Native APIs:** Native SDK methods from OpenAI (`create response`), Gemini (`generate`), and Claude (`create message`) — always access the latest LLM capabilities.
- **SDKs:** SDKs for React, Android, iOS, Flutter, React Native, and Unity, powered by Stream's ultra-low-latency network.

https://github.com/user-attachments/assets/d66587ea-7af4-40c4-9966-5c04fbcf467c

---

## See It In Action

### Sports Coaching

https://github.com/user-attachments/assets/d1258ac2-ca98-4019-80e4-41ec5530117e

This example shows you how to build golf coaching AI with YOLO and Gemini Live.
Combining a fast object detection model (like YOLO) with a full realtime AI is useful for many different video AI use cases.
For example: Drone fire detection, sports/video game coaching, physical therapy, workout coaching, just dance style games etc.

```python
# partial example, full example: examples/02_golf_coach_example/golf_coach_example.py
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    instructions="Read @golf_coach.md",
    llm=gemini.Realtime(fps=10),
    #llm=openai.Realtime(fps=1), # Careful with FPS can get expensive
    processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt", device="cuda")],
)
```

### Security Camera with Package Theft Detection

https://github.com/user-attachments/assets/92a2cdd8-909c-46d8-aab7-039a90efc186

This example shows a security camera system that detects faces, tracks packages and detects when a package is stolen. It automatically generates "WANTED" posters, posting them to X in real-time.

It combines face recognition, YOLOv11 object detection, Nano Banana and Gemini for a complete security workflow with voice interaction.

```python
# partial example, full example: examples/04_security_camera_example/security_camera_example.py
security_processor = SecurityCameraProcessor(
    fps=5,
    model_path="weights_custom.pt",  # YOLOv11 for package detection
    package_conf_threshold=0.7,
)

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Security AI", id="agent"),
    instructions="Read @instructions.md",
    processors=[security_processor],
    llm=gemini.LLM("gemini-2.5-flash-lite"),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
)
```

### Cluely style Invisible Assistant (coming soon)

Apps like Cluely offer realtime coaching via an invisible overlay. This example shows you how you can build your own invisible assistant.
It combines Gemini realtime (to watch your screen and audio), and doesn't broadcast audio (only text). This approach
is quite versatile and can be used for: Sales coaching, job interview cheating, physical world/ on the job coaching with glasses

Demo video

```python
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

## Quick Start

**Step 1: Install via uv**

`uv add vision-agents`

**Step 2: (Optional) Install with extra integrations**

`uv add "vision-agents[getstream, openai, elevenlabs, deepgram]"`

**Step 3: Obtain your Stream API credentials**

Get a free API key from [Stream](https://getstream.io/). Developers receive **333,000 participant minutes** per month, plus extra credits via the Maker Program.

## Features

| **Feature**                        | **Description**                                                                                                                                       |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **True real-time via WebRTC**      | Stream directly to model providers that support it for instant visual understanding.                                                                  |
| **Interval/processor pipeline**    | For providers without WebRTC, process frames with pluggable video processors (e.g., YOLO, Roboflow, or custom PyTorch/ONNX) before/after model calls. |
| **Turn detection & diarization**   | Keep conversations natural; know when the agent should speak or stay quiet and who's talking.                                                         |
| **Voice activity detection (VAD)** | Trigger actions intelligently and use resources efficiently.                                                                                          |
| **Speech↔Text↔Speech**             | Enable low-latency loops for smooth, conversational voice UX.                                                                                         |
| **Tool/function calling**          | Execute arbitrary code and APIs mid-conversation. Create Linear issues, query weather, trigger telephony, or hit internal services.                   |
| **Built-in memory via Stream Chat** | Agents recall context naturally across turns and sessions.                                                                                            |
| **Text back-channel**              | Message the agent silently during a call.                                                                                                             |
| **Phone and RAG**                  | Interact with the Agent via inbound or outbound phone calls using Twilio and Turbopuffer                                                              |


## Out-of-the-Box Integrations

| **Plugin Name** | **Description** | **Docs Link** |
|-------------|-------------|-----------|
| AWS Bedrock | Realtime speech-to-speech plugin using Amazon Nova models with automatic reconnection | [AWS](https://visionagents.ai/integrations/aws-bedrock) |
| AWS Polly | TTS plugin using Amazon's cloud-based service with natural-sounding voices and neural engine support | [AWS Polly](https://visionagents.ai/integrations/aws-polly) |
| Cartesia | TTS plugin for realistic voice synthesis in real-time voice applications | [Cartesia](https://visionagents.ai/integrations/cartesia) |
| Decart | Real-time AI video transformation service for applying artistic styles and effects to video streams | [Decart](https://visionagents.ai/integrations/decart) |
| Deepgram | STT plugin for fast, accurate real-time transcription with speaker diarization | [Deepgram](https://visionagents.ai/integrations/deepgram) |
| ElevenLabs | TTS plugin with highly realistic and expressive voices for conversational agents | [ElevenLabs](https://visionagents.ai/integrations/elevenlabs) |
| Fast-Whisper | High-performance STT plugin using OpenAI's Whisper model with CTranslate2 for fast inference | [Fast-Whisper](https://visionagents.ai/integrations/fast-whisper) |
| Fish Audio | STT and TTS plugin with automatic language detection and voice cloning capabilities | [Fish Audio](https://visionagents.ai/integrations/fish) |
| Gemini | Realtime API for building conversational agents with support for both voice and video | [Gemini](https://visionagents.ai/integrations/gemini) |
| HeyGen | Realtime interactive avatars powered by [HeyGen](https://heygen.com/) | [HeyGen](https://visionagents.ai/integrations/heygen) |
| Inworld | TTS plugin with high-quality streaming voices for real-time conversational AI agents | [Inworld](https://visionagents.ai/integrations/inworld) |
| Kokoro | Local TTS engine for offline voice synthesis with low latency | [Kokoro](https://visionagents.ai/integrations/kokoro) |
| Moondream | Moondream provides realtime detection and VLM capabilities. Developers can choose from using the hosted API or running locally on their CUDA devices. Vision Agents supports Moondream's Detect, Caption and VQA skills out-of-the-box. | [Moondream](https://visionagents.ai/integrations/moondream) |
| NVIDIA Cosmos 2 | VLM plugin using NVIDIA's Cosmos 2 models for video understanding with automatic frame buffering and streaming responses | [NVIDIA](https://visionagents.ai/integrations/nvidia) |
| OpenAI | Realtime API for building conversational agents with out of the box support for real-time video directly over WebRTC, LLMs and Open AI TTS | [OpenAI](https://visionagents.ai/integrations/openai) |
| OpenRouter | LLM plugin providing access to multiple providers (Anthropic, Google, OpenAI) through a unified API | [OpenRouter](https://visionagents.ai/integrations/openrouter) |
| Qwen | Realtime audio plugin using Alibaba's Qwen3 with native audio output and built-in speech recognition | [Qwen](https://visionagents.ai/integrations/qwen) |
| Roboflow | Object detection processor using Roboflow's hosted API or local RF-DETR models | [Roboflow](https://visionagents.ai/integrations/roboflow) |
| Smart Turn | Advanced turn detection system combining Silero VAD, Whisper, and neural models for natural conversation flow | [Smart Turn](https://visionagents.ai/integrations/smart-turn) |
| TurboPuffer | RAG plugin using TurboPuffer for hybrid search (vector + BM25) with Gemini embeddings for retrieval augmented generation | [TurboPuffer](https://visionagents.ai/guides/rag) |
| Twilio | Voice call integration plugin enabling bidirectional audio streaming via Twilio Media Streams with call registry and audio conversion | [Twilio](https://github.com/GetStream/Vision-Agents/tree/main/examples/03_phone_and_rag_example) |
| Ultralytics | Real-time pose detection processor using YOLO models with skeleton overlays | [Ultralytics](https://visionagents.ai/integrations/ultralytics) |
| Vogent | Neural turn detection system for intelligent turn-taking in voice conversations | [Vogent](https://visionagents.ai/integrations/vogent) |
| Wizper | STT plugin with real-time translation capabilities powered by Whisper v3 | [Wizper](https://visionagents.ai/integrations/wizper) |
| xAI | LLM plugin using xAI's Grok models with advanced reasoning and real-time knowledge | [xAI](https://visionagents.ai/integrations/xai) |


## Processors

Processors let your agent **manage state** and **handle audio/video** in real-time.

They take care of the hard stuff, like:

- Running smaller models
- Making API calls
- Transforming media

… so you can focus on your agent logic.

## Documentation

Check out our getting started guide at [VisionAgents.ai](https://visionagents.ai/).

**Quickstart:** [Building a Voice AI app](https://visionagents.ai/introduction/voice-agents)
**Quickstart:** [Building a Video AI app](https://visionagents.ai/introduction/video-agents)
**Tutorial:** [Building real-time sports coaching](https://github.com/GetStream/Vision-Agents/tree/main/examples/02_golf_coach_example)
**Tutorial:** [Building a real-time meeting assistant](https://github.com/GetStream/Vision-Agents#)

## Examples

| 🔮 Demo Applications |                                                                                |
|:-----|--------------------------------------------------------------------------------|
|  <br><h3>Cartesia</h3>Using Cartesia's Sonic 3 model to visually look at what's in the frame and tell a story with emotion.<br><br>• Real-time visual understanding<br>• Emotional storytelling<br>• Frame-by-frame analysis<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/plugins/cartesia/example) | <img src="assets/demo_gifs/cartesia.gif" width="320" alt="Cartesia Demo">      |
|  <br><h3>Realtime Stable Diffusion</h3>Realtime stable diffusion using Vision Agents and Decart's Mirage 2 model to create interactive scenes and stories.<br><br>• Real-time video restyling<br>• Interactive scene generation<br>• Stable diffusion integration<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/plugins/decart/example) | <img src="assets/demo_gifs/mirage.gif" width="320" alt="Mirage Demo">          |
|  <br><h3>Golf Coach</h3>Using Gemini Live together with Vision Agents and Ultralytics YOLO, we're able to track the user's pose and provide realtime actionable feedback on their golf game.<br><br>• Real-time pose tracking<br>• Actionable coaching feedback<br>• YOLO pose detection<br>• Gemini Live integration<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/02_golf_coach_example) | <img src="assets/demo_gifs/golf.gif" width="320" alt="Golf Coach Demo">        |
|  <br><h3>GeoGuesser</h3>Together with OpenAI Realtime and Vision Agents, we can take GeoGuesser to the next level by asking it to identify places in our real world surroundings.<br><br>• Real-world location identification<br>• OpenAI Realtime integration<br>• Visual scene understanding<br><br> [>Source Code and tutorial](https://visionagents.ai/integrations/openai#openai-realtime)| <img src="assets/demo_gifs/geoguesser.gif" width="320" alt="GeoGuesser Demo">  |
|  <br><h3>Phone and RAG</h3>Interact with your Agent over the phone using Twilio. This example demonstrates how to use TurboPuffer for Retrieval Augmented Generation (RAG) to give your agent specialized knowledge.<br><br>• Inbound/Outbound telephony<br>• Twilio Media Streams integration<br>• Vector search with TurboPuffer<br>• Retrieval Augmented Generation<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/03_phone_and_rag_example) | <img src="assets/demo_gifs/va_phone.png" width="320" alt="Phone and RAG Demo"> |
|  <br><h3>Security Camera</h3>A security camera with face recognition, package detection and automated theft response. Generates WANTED posters with Nano Banana and posts them to X when packages disappear.<br><br>• Face detection & named recognition<br>• YOLOv11 package detection<br>• Automated WANTED poster generation<br>• Real-time X posting<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/04_security_camera_example) | <img src="assets/demo_gifs/security_camera.gif" width="320" alt="Security Camera Demo"> |


## Development

See [DEVELOPMENT.md](DEVELOPMENT.md)

## Open Platform

Want to add your platform or provider? Reach out to **nash@getstream.io**.

## Awesome Video AI

Our favorite people & projects to follow for vision AI

|  [<img src="https://github.com/user-attachments/assets/9149e871-cfe8-4169-a4ce-4073417e645c" width="80"/>](https://x.com/demishassabis)  |       [<img src="https://github.com/user-attachments/assets/2e1335d3-58af-4988-b879-1db8d862cd34" width="80"/>](https://x.com/OfficialLoganK)        |            [<img src="https://github.com/user-attachments/assets/c9249ae9-e66a-4a70-9393-f6fe4ab5c0b0" width="80"/>](https://x.com/ultralytics)            |
| :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| [@demishassabis](https://x.com/demishassabis)<br>CEO @ Google DeepMind<br><sub>Won a Nobel prize</sub> | [@OfficialLoganK](https://x.com/OfficialLoganK)<br>Product Lead @ Gemini<br><sub>Posts about robotics vision</sub> | [@ultralytics](https://x.com/ultralytics)<br>Various fast vision AI models<br><sub>Pose, detect, segment, classify</sub> |

|         [<img src="https://github.com/user-attachments/assets/c1fe873d-6f41-4155-9be1-afc287ca9ac7" width="80"/>](https://x.com/skalskip92)         |            [<img src="https://github.com/user-attachments/assets/43359165-c23d-4d5d-a5a6-1de58d71fabd" width="80"/>](https://x.com/moondreamai)            |  [<img src="https://github.com/user-attachments/assets/490d349c-7152-4dfb-b705-04e57bb0a4ca" width="80"/>](https://x.com/kwindla)   |
| :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| [@skalskip92](https://x.com/skalskip92)<br>Open Source Lead @ Roboflow<br><sub>Building tools for vision AI</sub> | [@moondreamai](https://x.com/moondreamai)<br>The tiny vision model that could<br><sub>Lightweight, fast, efficient</sub> | [@kwindla](https://x.com/kwindla)<br>Pipecat / Daily<br><sub>Sharing AI and vision insights</sub> |

|   [<img src="https://github.com/user-attachments/assets/d7ade584-781f-4dac-95b8-1acc6db4a7c4" width="80"/>](https://x.com/juberti)    |            [<img src="https://github.com/user-attachments/assets/00a1ed37-3620-426d-b47d-07dd59c19b28" width="80"/>](https://x.com/romainhuet)            | [<img src="https://github.com/user-attachments/assets/eb5928c7-83b9-4aaa-854f-1d4f641426f2" width="80"/>](https://x.com/thorwebdev) |
| :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| [@juberti](https://x.com/juberti)<br>Head of Realtime AI @ OpenAI<br><sub>Realtime AI systems</sub> | [@romainhuet](https://x.com/romainhuet)<br>Head of DX @ OpenAI<br><sub>Developer tooling & APIs</sub> |   [@thorwebdev](https://x.com/thorwebdev)<br>Eleven Labs<br><sub>Voice and AI experiments</sub>   |

|    [<img src="https://github.com/user-attachments/assets/ab5ef918-7c97-4c6d-be10-2e2aeefec015" width="80"/>](https://x.com/mervenoyann)    |        [<img src="https://github.com/user-attachments/assets/af936e13-22cf-4000-a35b-bfe30d44c320" width="80"/>](https://x.com/stash_pomichter)         |            [<img src="https://pbs.twimg.com/profile_images/1893061651152121856/Op4W8mza_400x400.jpg" width="80"/>](https://x.com/Mentraglass)            |
| :------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| [@mervenoyann](https://x.com/mervenoyann)<br>Hugging Face<br><sub>Posts extensively about Video AI</sub> | [@stash_pomichter](https://x.com/stash_pomichter)<br>Spatial memory for robots<br><sub>Robotics & AI navigation</sub> | [@Mentraglass](https://x.com/Mentraglass)<br>Open-source smart glasses<br><sub>Open-Source, hackable AR glasses with AI capabilities built in</sub> |

|            [<img src="https://pbs.twimg.com/profile_images/1664559115581145088/UMD1vtMw_400x400.jpg" width="80"/>](https://x.com/vikhyatk)            |
| :----------------------------------------------------------------------------------------------------------------------: |
| [@vikhyatk](https://x.com/vikhyatk)<br>AI Engineer<br><sub>Open-source AI projects, Creator of Moondream AI</sub> |

## Inspiration

- Livekit Agents: Great syntax, Livekit only
- Pipecat: Flexible, but more verbose.
- OpenAI Agents: Focused on openAI only

## Roadmap

### 0.1 – First Release - Oct

- Working TTS, Gemini & OpenAI

### 0.2 - Simplification - Nov

- Simplified the library & improved code quality
- Deepgram Nova 3, Elevenlabs Scribe 2, Fish, Moondream, QWen3, Smart turn, Vogent, Inworld, Heygen, AWS and more
- Improved openAI & Gemini realtime performance
- Audio & Video utilities

### 0.3 - Examples and Deploys- Jan
- Production-grade HTTP API for agent deployment (`uv run <agent.py> serve`)
- Metrics & Observability stack
- Phone/voice integration with RAG capabilities
- 10 new LLM plugins ([AWS Nova 2](plugins/aws), [Qwen 3 Realtime](plugins/qwen), [NVIDIA Cosmos 2](plugins/nvidia), [Pocket TTS](plugins/pocket), [Deepgram TTS](plugins/deepgram), [OpenRouter](plugins/openrouter), [HuggingFace Inference](plugins/huggingface), [Roboflow](plugins/roboflow), [Twilio](plugins/twilio), [Turbopuffer](plugins/turbopuffer))
- Real-world examples ([security camera](examples/05_security_camera_example), [phone integration](examples/03_phone_and_rag_example), [football commentator](examples/04_football_commentator_example), [Docker deployment with GPU support](examples/07_deploy_example), [agent server](examples/08_agent_server_example))
- Stability: Fixes for participant sync, video frame handling, agent lifecycle, and screen sharing

### 0.4 Documentation/polish
- Excellence on documentation/polish
- Better Roboflow annotation docs
- Automated workflows for maintenance
- Local camera/audio support AND/OR WebRTC connection
- Embedded/robotics examples


## Vision AI limitations

Video AI is the frontier of AI. The state of the art is changing daily to help models understand live video.
While building the integrations, here are the limitations we've noticed (Dec 2025)

* Video AI struggles with small text. If you want the AI to read the score in a game it will often get it wrong and hallucinate
* Longer videos can cause the AI to lose context. For instance if it's watching a soccer match it will get confused after 30 seconds
* Most applications require a combination of small specialized models like Yolo/Roboflow/Moondream, API calls to get more context and larger models like gemini/openAI
* Image size & FPS need to stay relatively low due to performance constraints
* Video doesn’t trigger responses in realtime models. You always need to send audio/text to trigger a response.


## We are hiring

Join the team behind this project - we’re hiring a Staff Python Engineer to architect, build, and maintain a powerful toolkit for developers integrating voice and video AI into their products.

[Apply here](https://jobs.ashbyhq.com/stream/3bea7dba-54e1-4c71-aa02-712a075842df?utm_source=Jmv9QOkznl)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GetStream/vision-agents&type=timeline&legend=top-left)](https://www.star-history.com/#GetStream/vision-agents&type=timeline&legend=top-left)
