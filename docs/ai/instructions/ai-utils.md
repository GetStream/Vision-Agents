# Audio management

1. Audio is received as signed integer 16-bit little-endian, PCM 48khz stereo
2. The transport layer takes care of converting to WebRTC opus, when building plugins or working on the agent class you don't need to think about it
3. Audio is passed around within the SDK and between plugins using the `PcmData` type
4. PCM data should be loaded from raw bytes (or similar) into `PcmData` immediately
5. When manipulating audio, use the methods from `PcmData` to resample, mix channels (eg. stereo > mono) and change format

Audio resampling code lives in getstream library (https://github.com/GetStream/stream-py/blob/main/getstream/video/rtc/track_util.py)

## Creating PcmData

### from_bytes
Build from raw PCM bytes

```python
from getstream.video.rtc import PcmData

PcmData.from_bytes(audio_bytes, sample_rate=16000, format=AudioFormat.S16, channels=1)
```

### from_numpy
Build from numpy arrays with automatic dtype/shape conversion

```python
# Automatically converts dtype and handles channel reshaping
PcmData.from_numpy(np.array([1, 2], np.int16), sample_rate=16000, format=AudioFormat.S16, channels=1)
```

### from_response
Construct from API response (bytes, iterators, async iterators, objects with .data)

```python
# Handles streaming responses
PcmData.from_response(
    response, sample_rate=16000, channels=1, format=AudioFormat.S16
)
```

### from_av_frame
Create from PyAV AudioFrame

```python
PcmData.from_av_frame(frame)
```

## Converting Format

### to_float32
Convert samples to float32 in [-1, 1]

```python
pcm_f32 = pcm.to_float32()
```

### to_int16
Convert samples to int16 PCM format

```python
pcm_s16 = pcm.to_int16()
```

### to_bytes
Return interleaved PCM bytes

```python
audio_bytes = pcm.to_bytes()
```

### to_wav_bytes
Return WAV file bytes (header + frames)

```python
wav_bytes = pcm.to_wav_bytes()
```

## Resampling and Channels

### resample

Resample to target sample rate and/or channels

```python
pcm = pcm.resample(16000, target_channels=1)  # to 16khz, mono
```

## Manipulating Audio

### append
Append another PcmData in-place (adjusts format/rate automatically)

```python
pcm.append(other_pcm)
```

### copy
Create a deep copy

```python
pcm_copy = pcm.copy()
```

### clear
Clear all samples in-place (keeps metadata)

```python
pcm.clear()
```

## Slicing and Chunking

### head
Keep only the first N seconds

```python
pcm_head = pcm.head(duration_s=3.0)
```

### tail
Keep only the last N seconds

```python
pcm_tail = pcm.tail(duration_s=5.0)
```

### chunks
Iterate over fixed-size chunks with optional overlap

```python
for chunk in pcm.chunks(chunk_size=4000, overlap=200):
    process(chunk)
```

# Audio Queue

Use `AudioQueue` from `utils.audio_queue` if you need to enqueue PCM and dequeue audio with a given format and duration.

```python
from vision_agents.core.utils.audio_queue import AudioQueue

queue = AudioQueue(buffer_limit_ms=1000)

# enqueue PcmData into the queue
queue.put(pcm)

# dequeue the oldest PcmData entry from the queue
await queue.get()

# dequeue 100ms of audio
pcm = await queue.get_duration(100)
```

# AudioTrack

Use `getstream.video.rtc.AudioTrack` if you need to publish audio using PyAV, this class ensures that `recv` paces audio correctly every 20ms.

- Use `.write()` method to enqueue audio (PcmData)
- Use `.flush()` to empty all the enqueued audio (eg. barge-in event)

By default AudioTrack holds 30s of audio in the buffer.

# Video track

* VideoForwarder to forward video. see video_forwarder.py
* AudioForwarder to forward audio. See audio_forwarder.py
* QueuedVideoTrack to have a writable video track
* QueuedAudioTrack to have a writable audio track
* AudioQueue enables you to buffer audio, and read a certain number of ms or number of samples of audio
