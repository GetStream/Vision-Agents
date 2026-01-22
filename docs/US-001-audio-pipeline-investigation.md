# US-001: Audio Pipeline Configuration Investigation

**Investigation Date:** 2026-01-22
**Purpose:** Document the current audio format flow from API response to speaker output and identify format/sample rate mismatches

## Executive Summary

The Vision Agents Stream audio pipeline uses a flexible, multi-stage architecture that supports both remote WebRTC participants (via GetStream) and local device playback (via LocalRTC). Audio flows through several transformation stages with automatic resampling and format conversion at key points.

**Key Finding:** The pipeline has **built-in format normalization** at the output stage, but potential mismatches can occur if:
1. Audio queue receives inconsistent sample rates from different sources
2. STT pipeline expects 16kHz mono but receives different formats
3. Output tracks are misconfigured with incompatible sample rates

---

## Audio Pipeline Architecture

### 1. API Response Stage

**Entry Point:** GetStream WebRTC Connection
**File:** `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py:394-402`

**Format Specifications:**
- **Source:** Remote participant audio via WebRTC
- **Data Type:** `StreamPcmData` (GetStream library format)
- **Sample Rate:** Variable (depends on remote participant's encoder)
- **Channels:** Variable (typically mono or stereo)
- **Bit Depth:** Variable (16-bit or 32-bit float)

**Processing:**
```python
# Line 394-402 in stream_edge_transport.py
self._connection.on("audio", lambda event: self._on_audio_received(event))

def _on_audio_received(self, event):
    stream_pcm = event.data  # StreamPcmData from GetStream
    core_pcm = adapters.adapt_pcm_data(stream_pcm)  # Convert to core PcmData
    audio_event = AudioReceivedEvent(data=core_pcm)
    self._event_manager.emit(audio_event)
```

**Adapter Conversion:**
**File:** `plugins/getstream/vision_agents/plugins/getstream/adapters.py`

The `adapt_pcm_data()` function converts GetStream's `StreamPcmData` to the core framework's `PcmData` format, preserving:
- Sample rate
- Channel count
- Audio format (S16 or F32)
- Raw audio data
- Timestamps

---

### 2. Processing Stage

**Component:** AudioQueue
**File:** `agents-core/vision_agents/core/utils/audio_queue.py`

**Format Specifications:**
- **Sample Rate:** Tracked per-queue (validates consistency)
- **Channels:** Preserved from input
- **Bit Depth:** Preserved from input (S16 or F32)
- **Buffer Size:** Configurable (default: 8000ms in `agents.py:164`)

**Key Features:**
- Validates that all incoming audio chunks have the same sample rate
- Throws error if sample rate changes mid-stream
- Supports three retrieval modes:
  - `get()` - next chunk
  - `get_samples(num_samples)` - exact sample count
  - `get_duration(duration_ms)` - time-based retrieval

**Component:** AudioForwarder (STT Pipeline)
**File:** `agents-core/vision_agents/core/utils/audio_forwarder.py:64`

**Format Specifications:**
- **Input:** Any sample rate/channels
- **Output:** **16kHz mono** (hardcoded)
- **Conversion:** Automatic via `pcm.resample(target_sample_rate=16000, target_channels=1)`

**Purpose:** Standardizes audio for Speech-to-Text processing

---

### 3. WebRTC Tracks Stage

#### A. GetStream AudioStreamTrack (Remote Participants)

**File:** `.venv/lib/python3.12/site-packages/getstream/video/rtc/audio_track.py`
**Configuration:** `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py:426-428`

**Format Specifications:**
- **Sample Rate:** **48kHz** (default, configurable)
- **Channels:** **Stereo** (2 channels, default)
- **Bit Depth:** 16-bit signed int or 32-bit float (configurable)
- **Frame Size:** 20ms frames for WebRTC
- **Buffer Size:** 300 seconds (default)

**Normalization Process:**
```python
# AudioStreamTrack._normalize_pcm() (line 245-264)
def _normalize_pcm(self, pcm: PcmData) -> PcmData:
    # Resample to track's target sample rate and channels
    pcm = pcm.resample(self.sample_rate, target_channels=self.channels)

    # Convert format if needed
    if self.format == "s16" and pcm.format != "s16":
        pcm = pcm.to_int16()
    elif self.format == "f32" and pcm.format != "f32":
        pcm = pcm.to_float32()

    return pcm
```

**Key Method:**
- `write(data: PcmData)` - Accepts any format, normalizes automatically
- `recv()` - Returns 20ms frames at configured sample rate/channels
- `flush()` - Clears pending audio buffer

#### B. LocalRTC AudioOutputTrack (Local Speakers)

**File:** `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:255-600`

**Format Specifications:**
- **Sample Rate:** **16kHz** (default, configurable at line 69)
- **Channels:** **Mono** (1 channel, default)
- **Bit Depth:** 16-bit signed int (default)
- **Frame Size:** ~50ms chunks for smooth playback
- **Library:** sounddevice.OutputStream

**Key Features:**
- Persistent OutputStream with background thread (line 418-455)
- Callback-based playback for low latency
- Device selection by name, index, or default
- Automatic channel adjustment based on device capabilities

---

### 4. Output Device Stage

#### A. Remote Participant Playback (GetStream Path)

**Flow:**
```
AudioStreamTrack.write(PcmData)
    ↓ [normalize to 48kHz stereo]
AudioStreamTrack.recv() [20ms frames]
    ↓ [WebRTC transmission]
Remote Participant's Browser/App
    ↓
Speaker Playback (remote device)
```

**Format at Output:**
- **Sample Rate:** 48kHz (WebRTC standard)
- **Channels:** Stereo
- **Bit Depth:** 16-bit or 32-bit float

#### B. Local Speaker Playback (LocalRTC Path)

**Flow:**
```
AudioOutputTrack.write(PcmData)
    ↓ [resample to 16kHz mono via recv()]
sounddevice.OutputStream callback
    ↓ [background thread]
Local Speaker Device
```

**Format at Output:**
- **Sample Rate:** 16kHz (configurable)
- **Channels:** Mono (configurable)
- **Bit Depth:** 16-bit signed int

**Device Configuration:**
```python
# Line 69 in tracks.py
DEFAULT_SAMPLE_RATE = 16000
```

---

## Libraries/APIs Mapping

### Stage 1: API Response → Core Format
| Component | Library/API | File | Purpose |
|-----------|------------|------|---------|
| WebRTC Audio Reception | GetStream SDK | `stream_edge_transport.py` | Receive audio from remote participants |
| Format Adaptation | Custom Adapter | `adapters.py` | Convert StreamPcmData → PcmData |
| Event Emission | Core Event Manager | `agents.py` | Dispatch AudioReceivedEvent |

### Stage 2: Processing & Buffering
| Component | Library/API | File | Purpose |
|-----------|------------|------|---------|
| Audio Queue | asyncio.Queue | `audio_queue.py` | Buffer incoming audio chunks |
| Audio Forwarder | PyAV (av library) | `audio_forwarder.py` | Resample to 16kHz mono for STT |
| MediaStreamTrack Reader | aiortc | `audio_forwarder.py` | Read frames from WebRTC tracks |

### Stage 3: Format Conversion & Resampling
| Component | Library/API | File | Purpose |
|-----------|------------|------|---------|
| PcmData.resample() | PyAV + Custom | `track_util.py` | Resample sample rate/channels |
| PcmData.to_int16() | NumPy | `track_util.py` | Convert float32 → int16 |
| PcmData.to_float32() | NumPy | `track_util.py` | Convert int16 → float32 |
| PcmData.from_av_frame() | PyAV | `track_util.py` | Convert PyAV frames → PcmData |

### Stage 4: Output Tracks
| Component | Library/API | File | Purpose |
|-----------|------------|------|---------|
| AudioStreamTrack | GetStream SDK + aiortc | `audio_track.py` | Output to WebRTC (remote) |
| AudioOutputTrack | sounddevice | `tracks.py` | Output to local speakers |
| QueuedAudioTrack | asyncio.Queue | `audio_track.py` | Base class for buffered tracks |

### Stage 5: Device Playback
| Component | Library/API | File | Purpose |
|-----------|------------|------|---------|
| WebRTC Transmission | GetStream SDK | `stream_edge_transport.py` | Send audio to remote participants |
| Speaker Output | sounddevice.OutputStream | `tracks.py` | Play audio on local device |

---

## PcmData Format Conversion Methods

**File:** `.venv/lib/python3.12/site-packages/getstream/video/rtc/track_util.py`

### Supported Audio Formats
- `AudioFormat.S16` - 16-bit signed integer (int16)
- `AudioFormat.F32` - 32-bit floating point (float32)

### Conversion Methods

#### 1. from_av_frame() [Line 447-551]
**Purpose:** Convert PyAV AudioFrame to PcmData

**Supported PyAV Formats:**
- `s16` - Signed 16-bit (packed)
- `s16p` - Signed 16-bit (planar)
- `flt` - 32-bit float (packed)
- `fltp` - 32-bit float (planar)

**Process:**
1. Detect format from frame.format
2. Convert planar → packed if needed
3. Deinterleave multi-channel audio
4. Extract timestamps (pts, dts)

#### 2. from_bytes() [Line 299-373]
**Purpose:** Convert raw PCM bytes to PcmData

**Process:**
1. Determine dtype from format parameter
2. Convert bytes to NumPy array
3. Deinterleave if multi-channel: `[L0,R0,L1,R1,...]` → `[[L0,L1,...], [R0,R1,...]]`
4. Normalize shape to `(channels, samples)`

#### 3. from_numpy() [Line 376-444]
**Purpose:** Convert NumPy arrays to PcmData

**Process:**
1. Auto-convert dtype to match format (int16 or float32)
2. Normalize array shape to `(channels, samples)`
3. Validate data integrity

#### 4. from_g711() [Line 554-638]
**Purpose:** Decode G.711 compressed audio (μ-law/A-law)

**Supported Encodings:**
- μ-law (PCMU)
- A-law (PCMA)

**Process:**
1. Decode from raw bytes or base64
2. Decompress to 16-bit PCM
3. Create PcmData with 8kHz sample rate (G.711 standard)

#### 5. resample() [Line 640-667]
**Purpose:** Resample to target sample rate and/or channels

**Algorithm Selection:**
- **Long audio (>500ms):** PyAV resampler (higher quality, FFmpeg-based)
- **Short audio (≤500ms):** Custom resampler (lower latency)

**Supports:**
- Any sample rate conversion (e.g., 48kHz → 16kHz)
- Any channel conversion (e.g., stereo → mono)

#### 6. to_bytes() [Line 669-685]
**Purpose:** Convert PcmData to raw PCM bytes

**Process:**
1. Interleave channels if needed
2. Convert NumPy array to bytes
3. Return raw PCM data

#### 7. to_int16() / to_float32()
**Purpose:** Convert between audio formats

**Process:**
- Float32 → Int16: Scale by 32767, clip, cast to int16
- Int16 → Float32: Cast to float32, divide by 32767

---

## WebRTC Track Involvement in Playback

### GetStream Path (Remote Participants)
**WebRTC Track:** YES - Full involvement

```
Agent TTS/LLM Output
    ↓
await _audio_track.write(event.data)  # agents.py:311 or :1356
    ↓
AudioStreamTrack._normalize_pcm()  # Resample + format conversion
    ↓
AudioStreamTrack._buffer (Queue)  # Buffer as bytes
    ↓
AudioStreamTrack.recv()  # Return 20ms frames
    ↓
WebRTC RTP/RTCP Transmission
    ↓
Remote Participant's Browser
    ↓
Remote Speaker Playback
```

**Key Points:**
- WebRTC track formats audio as 20ms frames
- Automatic normalization to 48kHz stereo (configurable)
- Network transmission via RTP protocol
- Playback occurs on remote device, not local

### LocalRTC Path (Local Speakers)
**WebRTC Track:** PARTIAL - MediaStreamTrack protocol, but not network transmission

```
Agent TTS/LLM Output
    ↓
await _audio_track.write(event.data)  # agents.py:311 or :1356
    ↓
AudioOutputTrack._queue.put(data)  # Buffer PcmData
    ↓
AudioOutputTrack.recv()  # Return audio frames (no normalization here)
    ↓
sounddevice.OutputStream callback  # Background thread
    ↓
Local Speaker Device
```

**Key Points:**
- Implements MediaStreamTrack protocol for consistency
- No network transmission (local only)
- Uses sounddevice for direct hardware access
- Lower latency than WebRTC path
- Resample/normalization happens in recv() method

---

## Identified Format/Sample Rate Mismatches

### 1. STT Pipeline Hardcoded Sample Rate
**Location:** `audio_forwarder.py:64`

**Issue:** AudioForwarder always resamples to 16kHz mono, regardless of input.

**Impact:**
- If incoming audio is already 16kHz mono, unnecessary resampling adds latency
- If STT model expects different rate (e.g., 8kHz), mismatch occurs

**Recommendation:** Make target sample rate configurable

### 2. GetStream vs LocalRTC Default Mismatch
**GetStream Default:** 48kHz stereo
**LocalRTC Default:** 16kHz mono

**Issue:** If agent switches between GetStream and LocalRTC output, audio characteristics change drastically.

**Impact:**
- Quality degradation when switching to LocalRTC
- Potential playback speed issues if not properly resampled

**Recommendation:** Standardize defaults or make configuration explicit

### 3. AudioQueue Sample Rate Validation
**Location:** `audio_queue.py`

**Issue:** Queue validates that all chunks have the same sample rate, but initial rate is set by first chunk.

**Impact:**
- If first chunk has wrong sample rate, entire queue will reject correct chunks
- No mechanism to reset or reconfigure queue sample rate mid-stream

**Recommendation:** Add method to reset queue with new sample rate expectation

### 4. Missing Format Validation in Agent
**Location:** `agents.py:311, :1356`

**Issue:** Agent writes PcmData to audio track without validating format compatibility.

**Current Behavior:**
```python
await self._audio_track.write(event.data)  # No format checking
```

**Impact:**
- Relies on track's automatic normalization (which works, but has performance cost)
- No early warning if format is completely incompatible

**Recommendation:** Add format validation or explicit conversion before write

---

## Complete Audio Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: API RESPONSE                                           │
├─────────────────────────────────────────────────────────────────┤
│ Remote Participant Audio                                        │
│         ↓                                                        │
│ GetStream RTC Connection.on("audio")                            │
│   Sample Rate: Variable (typically 16kHz, 24kHz, or 48kHz)      │
│   Channels: Variable (mono or stereo)                           │
│   Format: StreamPcmData (S16 or F32)                            │
│         ↓                                                        │
│ adapt_pcm_data() [adapters.py]                                  │
│         ↓                                                        │
│ Core PcmData (framework standard format)                        │
│         ↓                                                        │
│ AudioReceivedEvent                                              │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PROCESSING                                             │
├─────────────────────────────────────────────────────────────────┤
│ Agent._incoming_audio_queue (AudioQueue)                        │
│   Sample Rate: Tracked (must be consistent)                     │
│   Channels: Preserved from input                                │
│   Format: Preserved (S16 or F32)                                │
│   Buffer: Up to 8000ms                                          │
│         ↓                                                        │
│ ┌───────────────────────┬─────────────────────────┐             │
│ │ STT Path              │ Direct Output Path      │             │
│ │ AudioForwarder        │ No Processing           │             │
│ │ ↓ resample()          │                         │             │
│ │ 16kHz mono (forced)   │ Original format         │             │
│ └───────────────────────┴─────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: OUTPUT GENERATION                                      │
├─────────────────────────────────────────────────────────────────┤
│ TTS/LLM Response Generation                                     │
│   TTSAudioEvent or RealtimeAudioOutputEvent                     │
│   Contains: PcmData with TTS output                             │
│         ↓                                                        │
│ await _audio_track.write(event.data) [agents.py:311, :1356]     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: WEBRTC TRACKS                                          │
├─────────────────────────────────────────────────────────────────┤
│ ┌────────────────────────┬───────────────────────────┐          │
│ │ GetStream Path         │ LocalRTC Path             │          │
│ │ AudioStreamTrack       │ AudioOutputTrack          │          │
│ │ ↓ _normalize_pcm()     │ ↓ recv()                  │          │
│ │ 48kHz stereo (default) │ 16kHz mono (default)      │          │
│ │ S16 or F32             │ S16                       │          │
│ │ 20ms frames            │ ~50ms chunks              │          │
│ └────────────────────────┴───────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: OUTPUT DEVICE                                          │
├─────────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────┬─────────────────────────┐          │
│ │ GetStream WebRTC         │ sounddevice             │          │
│ │ RTP/RTCP Transmission    │ OutputStream callback   │          │
│ │         ↓                │         ↓               │          │
│ │ Remote Browser/App       │ Local Speaker Device    │          │
│ │         ↓                │         ↓               │          │
│ │ Remote Speaker Playback  │ Local Speaker Playback  │          │
│ │ (48kHz stereo)           │ (16kHz mono)            │          │
│ └──────────────────────────┴─────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary of Key Findings

### Audio Format Specifications by Stage

| Stage | Sample Rate | Channels | Bit Depth | Format Type |
|-------|-------------|----------|-----------|-------------|
| API Response (GetStream) | Variable | Variable | 16/32-bit | StreamPcmData |
| Core PcmData (Adapted) | Preserved | Preserved | Preserved | PcmData (S16/F32) |
| AudioQueue (Buffer) | Consistent | Preserved | Preserved | PcmData |
| AudioForwarder (STT) | **16kHz** ⚠️ | **Mono** ⚠️ | Preserved | PcmData |
| AudioStreamTrack (GetStream) | **48kHz** ⚠️ | **Stereo** ⚠️ | 16/32-bit | PcmData |
| AudioOutputTrack (LocalRTC) | **16kHz** ⚠️ | **Mono** ⚠️ | **16-bit** | PcmData (S16) |
| WebRTC Output (Remote) | 48kHz | Stereo | 16/32-bit | RTP frames |
| Local Speaker Output | 16kHz | Mono | 16-bit | PCM |

⚠️ = Hardcoded or default value with potential mismatch risk

### Critical Libraries by Function

| Function | Library | Version Location |
|----------|---------|------------------|
| WebRTC Implementation | GetStream SDK | `.venv/lib/.../getstream/` |
| Audio Resampling | PyAV (FFmpeg) | `track_util.py:640-667` |
| Local Speaker Output | sounddevice | `tracks.py:255-600` |
| Async Media Streaming | aiortc | `audio_forwarder.py` |
| Audio Data Manipulation | NumPy | `track_util.py` (throughout) |

### WebRTC Track Involvement

- **GetStream Path:** Full WebRTC stack with network transmission
- **LocalRTC Path:** MediaStreamTrack protocol only, no network (direct sounddevice output)

---

## Recommendations for Future Work

1. **Standardize Default Sample Rates**
   - Consider using 16kHz across all components OR
   - Make sample rate a global configuration parameter

2. **Add Format Validation Layer**
   - Validate audio format before writing to tracks
   - Provide clear error messages for mismatches

3. **Make STT Target Rate Configurable**
   - Remove hardcoded 16kHz in AudioForwarder
   - Allow STT models to specify required sample rate

4. **Implement Dynamic AudioQueue Reset**
   - Allow sample rate changes mid-stream
   - Support multiple concurrent queues with different rates

5. **Add Audio Format Monitoring**
   - Log format conversions for debugging
   - Track resampling overhead in metrics

6. **Document Configuration Options**
   - Create clear documentation for audio track configuration
   - Provide examples for common use cases

---

## References

### Key Files Analyzed
- `agents-core/vision_agents/core/agents/agents.py`
- `agents-core/vision_agents/core/utils/audio_queue.py`
- `agents-core/vision_agents/core/utils/audio_forwarder.py`
- `agents-core/vision_agents/core/edge/types.py`
- `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py`
- `plugins/getstream/vision_agents/plugins/getstream/adapters.py`
- `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py`
- `.venv/lib/python3.12/site-packages/getstream/video/rtc/audio_track.py`
- `.venv/lib/python3.12/site-packages/getstream/video/rtc/track_util.py`

### Related Documentation
- WebRTC Specification: https://www.w3.org/TR/webrtc/
- PyAV Documentation: https://pyav.org/
- sounddevice Documentation: https://python-sounddevice.readthedocs.io/

---

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Investigator:** Claude Sonnet 4.5
