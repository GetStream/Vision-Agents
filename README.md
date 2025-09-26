# Open Vision Agents by Stream

Video/Vision AI agents running on [Stream's edge network](https://getstream.io/video/).
Open Agent library, adding support for other realtime video providers.

-  **Video AI**: Built for real-time video AI. Combine Yolo, Roboflow and others with realtime models
-  **Low Latency**: Join quickly (500ms) and low audio/video latency (30ms)
-  **Open**: Built by Stream, but use any video edge network that you like
-  **Native APIs**: Native SDK methods from OpenAI (create response), Gemini (generate) and Claude (create message). So you're never behind on the latest features
-  **SDKs**: SDKs for React, Android, iOS, Flutter, React, React Native and Unity.

## Examples

### Sports Coaching

This example shows you how to build golf coaching AI with YOLO and OpenAI realtime.
Combining a fast object detection model (like YOLO) with a full realtime AI is useful for many different video AI use cases.
For example: Drone fire detection. Sports/video game coaching. Physical therapy. Workout coaching, Just dance style games etc.

TODO: Demo video

```python
# partial example, full example: examples/03_golf_coach_example/golf_coach_example.py
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="Help users perfect their swing. Read @golf_coach.md",
    # openai realtime, no need to set tts, or sst (though that's also supported)
    llm=openai.Realtime(model="gpt-realtime"),
    processors=[
        ultralytics.YOLOPoseProcessor()
    ],  # processors can fetch extra data, check images/audio data or transform video
)
```

### Cluely style Invisible Assistant

Apps like Cluely offer realtime coaching via an invisible overlay. This example shows you how you can build your own invisible assistant.
It combines Gemini realtime (to watch your screen and audio), and doesn't broadcast audio (only text). Again this approach
is quite versatile and can be used for: Sales coaching, job interview cheating, physical world/ on the job coaching with glasses

Demo video

```python
# partial example, full example: examples/...
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

### Dota Coaching Example

This example combines OpenDota, Stratz, valve game state together with Gemini to improve your Dota skills.
It showcases how you can combine API calls for additional context with vision AI.

Demo video

```python
# partial example, full example: examples/...
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

## Processors

Processors make it easy to combine the video & LLM with additional state. Here are some built-in examples

* YoloPose
* YoloObjectDetection
* YoloSpeedMeasurement
* ImageCapture
* BufferedVideoCapture
* TavusAvatar

## Docs

TODO Link to docs


## Competitors & Partners

Reach out to nash@getstream.io, and we'll collaborate on getting you added
We'd like to add support for and are reaching out to:

* Mediasoup
* Janus
* Cloudflare
* Twilio
* AWS IVS
* Vonage
* And others.

## Dev / Contributor Guidelines

### Light wrapping

AI is changing daily. This makes it important to use light wrapping. IE

```python
tts = ElevenLabsTTS(client=ElevenLabs())
```

Note how the ElevenLabsTTS handles standardization.
But if the init for ElevenLabs changes, nothing breaks.
If features are added to the client, you can use them easily via tts.client

### Typing

Avoid using Union types or complicated composite types.
Keep typing simple. Use the PcmAudio type instead of bytes when passing around audio.
This prevents mistakes related to different audio formats.

### Testing

Many of the underlying APIs change daily. To ensure things work we keep 2 sets of tests. Integration tests and unit tests.
Integration tests run once a day to verify that changes to underlying APIs didn't break the framework. Some testing guidelines

- Every plugin needs an integration test
- Limit usage of response capturing style testing. (since they diverge from reality)

### Observability

- Traces and metrics go to Prometheus and OpenTelemetry
- Metrics on performance of TTS, STT, LLM, Turn detection and connection to realtime edge.
- Integration with external LLM observability solutions

### Queuing

- Video: There is no reason to publish old video. So you want to cap the queue to x latest frames
- Audio: Writing faster than 1x causes audio glitches. So we need a queue.
- Audio: Writing slower than 1x also causes glitches. You need to write 0 frames
- Audio generated by LLM: The LLM -> TTS can generate a lot of audio. This has to be stopped when interrupt happens
- Gemini & Google generate at what pace?

### Tasks

- Short running tasks should check if the connection is closed before doing work
- Long running tasks are should be cancelled when calling agent.close()

### Video Frames & Tracks

- Track.recv errors will fail silently. The API is to return a frame. Never return None. and wait till the next frame is available
- When using frame.to_ndarray(format="rgb24") specify the format. Typically you want rgb24 when connecting/sending to Yolo etc

## Awesome Video AI

Our favorite people & projects to follow for vision AI

* https://x.com/demishassabis. CEO google deepmind, won a nobel prize
* https://x.com/OfficialLoganK. Product lead gemini, posts about robotics vision
* https://x.com/ultralytics. various fast vision AI models. Pose, detect objects, segment, classify etc.
* https://x.com/skalskip92. roboflow open source lead
* https://x.com/moondreamai. the tiny vision model that could
* https://x.com/kwindla. pipecat/daily
* https://x.com/juberti. head of realtime AI openai
* https://x.com/romainhuet head of developer experience openAI
* https://x.com/thorwebdev eleven labs
* https://x.com/mervenoyann huggingface, quite some posts about Video AI

## Inspiration

- Livekit Agents: Great syntax, Livekit only
- Pipecat: Flexible, but more verbose. Open, we will add support for Stream
- OpenAI Agents: Focused on openAI only, but we will try to add support

## Stream Agents or Proxy

The proxy mode which handles the openAI/Stream connection is a good option if you don't need to run any additional AI models.
If all you need is low latency integration between stream and openAI, that's a good option.
It's available for JS & Python.

## OpenAI Proxy mode vs Stream agents

** OpenAI Proxy mode **:

For use cases that only require you to connect openAI & Stream you can consider
https://getstream.io/video/voice-agents/
This is a direct proxy written in Go and maintained by Stream.
You can use it with any programming language/ SDK.

Benefits:

* Hosted
* Faster/lowest latency
* All programming languages

**Stream agents**

This python framework gives you full control.