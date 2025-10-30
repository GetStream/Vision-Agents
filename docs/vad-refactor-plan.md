# VAD Refactor Plan

## Goals
- Centralize all common VAD responsibilities in a single base class.
- Make VAD implementations minimal: only inference-specific logic (is_speech).
- Normalize incoming audio to the model’s needs (rate, channels, format) in the base.
- Emit all VAD events from the base class only.
- Simplify state and buffering; use PcmData consistently.
- Provide clear, testable behavior with uv run, no mocks.

## High-Level Architecture
- Base VAD orchestrates the entire pipeline:
  - Receives arbitrary PCM frames.
  - Normalizes frames to the model spec (rate, channels, format).
  - Windows normalized audio to fixed-size model windows.
  - Calls the single abstract method: `is_speech(window: PcmData) -> float`.
  - Runs the speech state machine (activation/deactivation, padding, min/max).
  - Buffers detected speech and emits all events (start/partial/final/end).
- Implementations (e.g., Silero):
  - Define model input spec via base constructor params.
  - Implement `is_speech` only.
  - No buffering, no resampling, no event emission.

## Base VAD API (Proposed)
Constructor (model spec + thresholds):
- `sample_rate: int` — model input rate (Hz)
- `window_samples: int` — model window size (samples)
- `channels: int = 1` — model channel count (default mono)
- `audio_format: AudioFormat = AudioFormat.PCM_S16` — model PCM format (default s16)
- `activation_th: float`
- `deactivation_th: float`
- `speech_pad_ms: int`
- `min_speech_ms: int`
- `max_speech_ms: int`
- `partial_frames: int`
- `provider_name: Optional[str]`

Abstract method:
- `async def is_speech(self, window: PcmData) -> float`
  - window is exactly `window_samples` long at `sample_rate`, `channels`, `audio_format`.

Public methods (unchanged surface):
- `async def process_audio(self, pcm_data: PcmData, participant: Optional[Participant])`
- `async def flush(self, participant: Optional[Participant])`
- `async def reset(self)`
- `async def close(self)`

## Base VAD Responsibilities
- Normalize incoming frames to model spec using PcmData:
  - `frame -> PcmData(...).resample(self.sample_rate, self.channels, target_format=audio_format)`
  - Note: normalization is based on the VAD implementation’s model needs (not always int16).
- Accumulate into `_model_buffer: Optional[PcmData]` (model spec).
- Windowing:
  - While `_model_buffer` has ≥ `window_samples`, pop a window `PcmData` and call `is_speech(window)`.
- Speech state machine:
  - Asymmetric thresholds with `activation_th` and `deactivation_th`.
  - Padding: `speech_pad_ms` to join turns.
  - Limits: `min_speech_ms`, `max_speech_ms`.
- Accumulation for output:
  - `self.speech_buffer: Optional[PcmData]` in model spec; uses `.append()` to grow.
- Event emission (only in base):
  - `VADSpeechStartEvent` on first activation.
  - `VADPartialEvent` every `partial_frames` while active.
  - `VADAudioEvent` on turn emission (bytes + duration + counts).
  - `VADSpeechEndEvent` when speech ends.

## Implementation Responsibilities (e.g., Silero)
- Call base with model spec:
  - `sample_rate=model_rate`, `window_samples=512/256`, `channels=1`, `audio_format=PCM_S16` (or PCM_F32 for float models).
- Implement `async def is_speech(self, window: PcmData) -> float`:
  - Convert window to the model’s tensor format (e.g., float32 in [-1,1])
  - Run inference and return probability.
- Optional: override `reset()` only for model internal state.
- Must not: buffer audio, resample, or emit events.

## PcmData Usage
- Single audio carrier everywhere (bytes or ndarray):
  - `.resample(target_rate, target_channels, target_format=...)` (extend as needed for f32)
  - `.append(other)` — auto-resamples to match and concatenates
  - `.to_bytes()`, `.to_wav_bytes()`, `.duration_ms`, `.channels`

## Event Semantics
- Start: emitted on first activation for a segment.
- Partial: emitted every `partial_frames` frames during active speech (model windows).
- Audio: emitted when silence exceeds `speech_pad_ms` or `max_speech_ms` reached and `min_speech_ms` satisfied.
- End: emitted after audio event for the completed turn.

## Tests
- Use uv run; no mocks.
- Reuse existing Silero VAD tests; they assert durations and turn counts via real assets.
- Add base-VAD specific tests (future work):
  - Normalization to model spec (rate/channels/format).
  - 20ms/512-sample windowing behavior.
  - Event emission timing and counts with small synthetic inputs.

Commands:
- `PYTHONPATH=stream-py:. uv run pytest -q plugins/silero/tests/test_vad.py`
- Add similar command for any new base-VAD tests.

## Migration Plan
1) Introduce model spec in base VAD constructor and shift normalization + windowing there.
2) Update implementations (Silero) to only implement `is_speech` and pass model spec via super().__init__.
3) Replace any implementation-level event emission with base emission.
4) Run Silero tests with uv; fix regressions.
5) Optional: extend `PcmData.resample` to support `target_format` (PCM_F32) for float models; wire base to request f32 windows when `audio_format=PCM_F32`.

## Acceptance Criteria
- All VAD implementations rely on the base for buffering/windowing/events.
- Base normalizes audio to the model’s sample_rate/channels/format.
- Events (start/partial/audio/end) originate from the base only.
- Silero test suite passes with `uv run`.
- Plan documented and discoverable in `docs/`.
