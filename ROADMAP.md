# Roadmap

## 0.5 Documentation/polish - Planned

- Excellence on documentation
- Segmentation examples with HuggingFace and Roboflow
- Automated workflows for maintenance
- Local camera/audio support AND/OR WebRTC connection
- Embedded/robotics examples

## 0.4 - Production Polish & Scalability - Feb

- Horizontal scaling support via Redis-based session store for multi-node deployments
- New model/provider expansions: [XAI realtime](https://visionagents.ai/integrations/xai), [Mistral/Voxtral](https://visionagents.ai/integrations), [Gemini 3 Vision](https://visionagents.ai/integrations/gemini), [Hugging Face Transformers plugin](https://visionagents.ai/integrations), Qwen, [OpenRouter VLM](https://visionagents.ai/integrations/openrouter) + upgraded defaults (e.g., GPT-Realtime 1.5)
- Enforced async-only APIs, agent testing framework, authentication flow updates, limits in AgentLauncher, and reduced GetStream dependency coupling for standalone use
- Multi-speaker call support, LemonSlice Avatar plugin, TTS/audio fixes, VLM message deduplication (Gemini/NVIDIA/Anthropic), and improved video/screen-sharing handling
- Lot of polish: full [CHANGELOG.md](https://github.com/GetStream/Vision-Agents/blob/main/CHANGELOG.md), [Grafana/Prometheus examples](https://visionagents.ai/core/telemetry), and numerous stability/bug fixes

## 0.3 - Examples and Deploys - Jan

- Production-grade HTTP API for agent deployment (`uv run <agent.py> serve`)
- Metrics & Observability stack
- Phone/voice integration with RAG capabilities
- 10 new LLM
  plugins ([AWS Nova 2](plugins/aws), [Qwen 3 Realtime](plugins/qwen), [NVIDIA Cosmos 2](plugins/nvidia), [Pocket TTS](plugins/pocket), [Deepgram TTS](plugins/deepgram), [OpenRouter](plugins/openrouter), [HuggingFace Inference](plugins/huggingface), [Roboflow](plugins/roboflow), [Twilio](plugins/twilio), [Turbopuffer](plugins/turbopuffer))
- Real-world
  examples ([security camera](examples/05_security_camera_example), [phone integration](examples/03_phone_and_rag_example), [football commentator](examples/04_football_commentator_example), [Docker deployment with GPU support](examples/07_deploy_example), [agent server](examples/08_agent_server_example))
- Stability: Fixes for participant sync, video frame handling, agent lifecycle, and screen sharing

## 0.2 - Simplification - Nov

- Simplified the library & improved code quality
- Deepgram Nova 3, Elevenlabs Scribe 2, Fish, Moondream, QWen3, Smart turn, Vogent, Inworld, Heygen, AWS and more
- Improved openAI & Gemini realtime performance
- Audio & Video utilities

## 0.1 – First Release - Oct

- Working TTS, Gemini & OpenAI
