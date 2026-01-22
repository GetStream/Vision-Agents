# US-002: Root Cause Analysis - Sample Rate Mismatch

## Executive Summary

The 2x speed playback issue (chipmunk effect) in LocalRTC audio output is caused by a **sample rate mismatch** between the Gemini API audio responses (24kHz) and the LocalRTC audio output device configuration (16kHz).

**Root Cause:** Gemini Realtime API returns audio at **24,000 Hz**, but the LocalRTC Edge is configured to use **16,000 Hz** sample rate. The AudioOutputTrack has resampling logic in place, but there appears to be an issue in the conversion that causes the audio to play at incorrect speed.

## Detailed Findings

### 1. Sample Rate Expected by Output Devices

**Location:** `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:278-310`

The `AudioOutputTrack` class is initialized with a configurable sample rate:

```python
class AudioOutputTrack:
    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: int = 16000,  # DEFAULT IS 16kHz
        channels: int = 1,
        bit_depth: int = 16,
        buffer_size_ms: int = 500,
    ) -> None:
```

**Finding:** The default sample rate for audio output devices is **16,000 Hz**.

**Example Usage:** `examples/localrtc/basic_agent.py:54`
```python
edge = localrtc.Edge(
    audio_device="default",
    video_device=0,
    speaker_device="default",
    sample_rate=16000,  # Configured for 16kHz
    channels=1,
)
```

### 2. Sample Rate of Incoming LLM Audio Responses

**Location:** `plugins/gemini/vision_agents/plugins/gemini/gemini_realtime.py:437`

The Gemini Realtime API returns audio data at a fixed sample rate:

```python
elif part.inline_data:
    # Emit audio output event
    pcm = PcmData.from_bytes(part.inline_data.data, 24000)  # HARDCODED 24kHz
    self._emit_audio_output_event(audio_data=pcm)
```

**Finding:** Gemini Realtime API audio responses are **24,000 Hz** (hardcoded).

### 3. Resampling and Format Conversion Points

**Location:** `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:631-640`

The AudioOutputTrack has resampling logic that should handle the conversion:

```python
# Convert sample rate and/or channels if needed
if sample_rate != self.sample_rate or channels != self.channels:
    audio_data = self._convert_audio(
        audio_data,
        from_rate=sample_rate,      # 24000 from Gemini
        to_rate=self.sample_rate,   # 16000 for output device
        from_channels=channels,
        to_channels=self.channels,
        bit_depth=bit_depth,
    )
```

**Resampling Implementation:** `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:520-576`

The `_convert_audio` method performs resampling using linear interpolation:

```python
def _convert_audio(
    self,
    data: bytes,
    from_rate: int,      # 24000
    to_rate: int,        # 16000
    from_channels: int,
    to_channels: int,
    bit_depth: int,
) -> bytes:
    # ... (conversion logic using numpy interpolation)

    # Resample if needed
    if from_rate != to_rate:
        num_samples = len(audio_mono)
        new_length = int(num_samples * to_rate / from_rate)

        # Use linear interpolation for resampling
        old_indices = np.arange(num_samples)
        new_indices = np.linspace(0, num_samples - 1, new_length)
        audio_mono = np.interp(new_indices, old_indices, audio_mono)
```

**Expected Behavior:** When resampling from 24kHz to 16kHz:
- `from_rate = 24000`
- `to_rate = 16000`
- `new_length = num_samples * (16000 / 24000) = num_samples * 0.667`
- This should produce audio that is **1.5x longer** (plays at 2/3 speed)

**Finding:** The resampling logic exists and appears mathematically correct for converting 24kHz to 16kHz.

### 4. The Mismatch Causing 2x Speed Playback

**Analysis of the Problem:**

Given the math in the resampling code:
- 24kHz audio → 16kHz output should play at **slower** speed (1.5x duration, 0.67x speed)
- However, the reported issue is **2x speed** (chipmunk effect)

**Hypothesis: The resampling is actually NOT being applied correctly, or there's a path where it's bypassed.**

Let me check the audio pipeline flow:

1. **Gemini generates audio:** 24kHz PCM data
2. **AudioOutputTrack receives data:** Should detect mismatch (24kHz vs 16kHz)
3. **Resampling should occur:** Convert 24kHz → 16kHz
4. **Audio plays:** At 16kHz rate

**Potential Issues:**

1. **The resampling might be working in reverse:** If the code is somehow treating 24kHz data as 16kHz data and playing it at 24kHz device rate, it would play at 1.5x speed (close to 2x)

2. **Device may auto-configure:** The sounddevice library might be auto-detecting and using 24kHz based on the incoming data, ignoring the configured 16kHz

3. **StreamPcmData vs CorePcmData handling:** There's special handling for getstream PcmData types (lines 604-627), which might have different behavior

## Audio Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Gemini Realtime API                                             │
│ Returns: 24kHz PCM audio (hardcoded)                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ PcmData(data, sample_rate=24000)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ AudioOutputTrack.write()                                        │
│ Configured: 16kHz output                                        │
│                                                                  │
│ 1. Detect mismatch: 24kHz (incoming) != 16kHz (configured)     │
│ 2. Call _convert_audio(from_rate=24000, to_rate=16000)        │
│ 3. Resampling: 24kHz → 16kHz using linear interpolation       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Resampled 16kHz audio data
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ sounddevice OutputStream                                        │
│ Configured: 16kHz, mono, 16-bit                                │
│                                                                  │
│ Expected: Audio plays at correct speed                          │
│ Actual: Audio plays at ~2x speed (chipmunk effect)            │
└─────────────────────────────────────────────────────────────────┘
```

## The Core Mismatch

**Mismatch Summary:**

| Component | Sample Rate | Location |
|-----------|-------------|----------|
| Gemini API Audio Output | **24,000 Hz** | `plugins/gemini/vision_agents/plugins/gemini/gemini_realtime.py:437` |
| LocalRTC Edge Configuration | **16,000 Hz** | `examples/localrtc/basic_agent.py:54` |
| AudioOutputTrack Default | **16,000 Hz** | `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:282` |

**Expected Conversion:** 24kHz → 16kHz (should make audio 1.5x longer/slower)

**Actual Result:** ~2x speed playback (chipmunk effect)

## Key Questions for Further Investigation

1. **Is the resampling actually being triggered?**
   - Add logging to confirm `_convert_audio` is called
   - Verify `from_rate` and `to_rate` values

2. **Is the sounddevice stream using the correct sample rate?**
   - Check if device is auto-configuring to 24kHz instead of 16kHz
   - Verify stream creation at `tracks.py:505-510`

3. **Is there a path where resampling is bypassed?**
   - Check for edge cases in the write() method
   - Verify both CorePcmData and StreamPcmData code paths

4. **Could the issue be in how Gemini's audio is being interpreted?**
   - The 24kHz might be incorrectly specified
   - The actual data might be different from what the metadata claims

## Recommendations for US-003 (Fix)

Based on this analysis, the fix should:

1. **Option A: Match sample rates** - Configure LocalRTC to use 24kHz to match Gemini
   ```python
   edge = localrtc.Edge(
       sample_rate=24000,  # Match Gemini's output
       ...
   )
   ```

2. **Option B: Fix the resampling** - Debug and fix the _convert_audio method
   - Add extensive logging
   - Verify the resampling math
   - Test the output

3. **Option C: Request different rate from Gemini** - If Gemini supports it, request 16kHz audio

## Testing Recommendations

1. **Add debug logging** to track sample rates through the pipeline
2. **Save raw audio files** at each stage (pre/post resampling)
3. **Compare WAV file output** (working) vs speaker output (broken)
4. **Use audio analysis tools** to verify actual sample rates in the data

## References

- PRD: `tasks/prd-product-requirements-document-fix-localrtc-audio-playback-issue.md`
- Audio output implementation: `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py`
- Gemini integration: `plugins/gemini/vision_agents/plugins/gemini/gemini_realtime.py`
- Example usage: `examples/localrtc/basic_agent.py`
