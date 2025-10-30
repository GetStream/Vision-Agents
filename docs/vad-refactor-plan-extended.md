# VAD Refactor Plan (Extended with Testing)

## Goals
- Centralize all common VAD responsibilities in a single base class.
- Make VAD implementations minimal: only inference-specific logic (is_speech).
- Normalize incoming audio to the model's needs (rate, channels, format) in the base.
- Emit all VAD events from the base class only.
- Simplify state and buffering; use PcmData consistently.
- Provide clear, testable behavior with uv run, no mocks.
- **NEW:** Ensure comprehensive testing with deterministic patterns and real audio.

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

### Simplified Constructor with Dataclasses
```python
@dataclass
class VADModelSpec:
    """Model requirements - what the VAD model expects."""
    sample_rate: int = 16000
    window_samples: int = 512
    channels: int = 1
    audio_format: AudioFormat = AudioFormat.PCM_S16

@dataclass
class VADThresholds:
    """Configurable detection thresholds."""
    activation_th: float = 0.5
    deactivation_th: float = 0.3
    speech_pad_ms: int = 300
    min_speech_ms: int = 250
    max_speech_ms: int = 60000
    partial_frames: int = 5

class VAD(abc.ABC):
    def __init__(
        self,
        model_spec: VADModelSpec,
        thresholds: VADThresholds = VADThresholds(),
        provider_name: Optional[str] = None
    ):
        ...
```

### Abstract method:
- `async def is_speech(self, window: PcmData) -> float`
  - window is exactly `window_samples` long at `sample_rate`, `channels`, `audio_format`.

### Public methods (unchanged surface):
- `async def process_audio(self, pcm_data: PcmData, participant: Optional[Participant])`
- `async def flush(self, participant: Optional[Participant])`
- `async def reset(self)`
- `async def close(self)`

## Base VAD Responsibilities
- Normalize incoming frames to model spec using PcmData:
  - `frame -> PcmData(...).resample(self.sample_rate, self.channels, target_format=audio_format)`
  - Note: normalization is based on the VAD implementation's model needs (not always int16).
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
  - Convert window to the model's tensor format (e.g., float32 in [-1,1])
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

## Testing Strategy (NEW)

### Test Infrastructure

#### 1. Test Audio Generator (`agents-core/vision_agents/core/vad/testing.py`)
```python
class VADTestAudioGenerator:
    """Generate synthetic audio for deterministic VAD testing."""

    @staticmethod
    def generate_tone(freq_hz: int, duration_ms: int, sample_rate: int = 16000) -> PcmData
        """Pure tone to simulate speech (440Hz typically triggers VADs)."""

    @staticmethod
    def generate_silence(duration_ms: int, sample_rate: int = 16000) -> PcmData
        """Generate silence samples."""

    @staticmethod
    def generate_noise(duration_ms: int, amplitude: float, sample_rate: int = 16000) -> PcmData
        """Generate white noise at specified amplitude."""

    @staticmethod
    def generate_speech_pattern(
        pattern: List[Tuple[str, int]],  # [("speech", 500), ("silence", 200), ...]
        sample_rate: int = 16000
    ) -> PcmData
        """Generate complex patterns for testing state transitions."""
```

#### 2. Mock VAD Implementation (`tests/test_vad_base.py`)
```python
class MockVAD(VAD):
    """Controllable VAD for testing base class logic independently."""

    def set_speech_pattern(self, pattern: List[Tuple[bool, int]]):
        """Set deterministic is_speech() responses: [(is_speech, num_windows), ...]"""

    async def is_speech(self, window: PcmData) -> float:
        """Return predetermined probabilities based on pattern."""
```

#### 3. VAD Test Session Helper (`agents-core/vision_agents/core/vad/testing.py`)
```python
class VADTestSession:
    """Helper for collecting and asserting VAD events in tests."""

    def __init__(self, vad: VAD)
    async def process_pattern(self, pattern: List[Tuple[str, int]])
    def assert_turn_count(self, expected: int)
    def assert_total_speech_duration(self, min_ms: int, max_ms: int)
    def assert_event_sequence(self, expected: List[Type[Event]])
    def get_audio_events(self) -> List[VADAudioEvent]
```

### Test Categories

#### 1. Base VAD Logic Tests (`tests/test_vad_base.py`)
These test the base class behavior using MockVAD:

- **State Machine Tests:**
  - `test_state_transitions`: QUIET → SPEAKING → QUIET with deterministic patterns
  - `test_activation_threshold`: Speech only detected above activation_th
  - `test_deactivation_threshold`: Speech ends below deactivation_th
  - `test_hysteresis`: Activation at 0.5, deactivation at 0.3 prevents flapping

- **Duration Tests:**
  - `test_min_speech_duration`: Rejects speech < min_speech_ms
  - `test_max_speech_duration`: Cuts off at max_speech_ms
  - `test_speech_padding`: Bridges gaps < speech_pad_ms

- **Event Emission Tests:**
  - `test_partial_events`: Emitted every partial_frames during speech
  - `test_event_ordering`: Start → Partial* → Audio → End sequence
  - `test_flush_emits_pending`: Flush forces emission of buffered speech

- **Normalization Tests:**
  - `test_resampling_48k_to_16k`: Downsamples correctly
  - `test_stereo_to_mono`: Converts channels correctly
  - `test_format_conversion_s16_to_f32`: Handles format changes

- **Windowing Tests:**
  - `test_window_size_512_samples`: Windows at exact boundaries
  - `test_leftover_samples`: Handles partial windows correctly
  - `test_multiple_windows_per_frame`: Processes all available windows

#### 2. Implementation Tests (`plugins/silero/tests/test_silero_vad.py`)
These test real VAD implementations with both synthetic and real audio:

- **Basic Detection:**
  - `test_detects_real_speech`: Uses mia_audio_16khz fixture
  - `test_ignores_silence`: Pure silence produces no events
  - `test_detects_tone_as_speech`: 440Hz tone triggers detection

- **Threshold Calibration:**
  - `test_confidence_values`: is_speech() returns values in [0, 1]
  - `test_speech_vs_noise`: Higher confidence for speech than noise
  - `test_custom_thresholds`: Respects configured thresholds

- **Performance Tests:**
  - `test_processing_speed`: Processes faster than realtime
  - `test_memory_usage`: No memory leaks over long audio
  - `test_cpu_usage`: Reasonable CPU consumption

#### 3. Integration Tests (`tests/test_vad_integration.py`)
These test VAD with other components:

- **With STT:**
  - `test_vad_triggers_stt`: VADAudioEvent triggers STT processing
  - `test_partial_events_for_streaming`: Partials enable streaming STT

- **With Multiple Audio Sources:**
  - `test_different_sample_rates`: Handles 8/16/24/48kHz inputs
  - `test_different_formats`: Handles s16/f32 inputs
  - `test_chunked_vs_continuous`: Same results for chunked/continuous

#### 4. Regression Tests (`tests/test_vad_regression.py`)
Fixed test cases for known issues:

- `test_clicking_artifacts`: No audio artifacts from resampling
- `test_dropped_frames`: No frames lost during processing
- `test_state_persistence`: State survives multiple process_audio calls

### Test Patterns and Scenarios

#### Pattern 1: Simple Speech Detection
```python
async def test_simple_speech_detection():
    """Basic speech → silence → speech pattern."""
    vad = SileroVAD()
    session = VADTestSession(vad)

    await session.process_pattern([
        ("speech", 1000),  # 1 second speech
        ("silence", 500),  # 0.5 second gap
        ("speech", 1000),  # 1 second speech
    ])

    session.assert_turn_count(2)  # Two separate turns
    session.assert_total_speech_duration(1800, 2200)  # ~2 seconds total
```

#### Pattern 2: Speech Padding
```python
async def test_speech_padding_joins_utterances():
    """Brief pauses should be bridged."""
    thresholds = VADThresholds(speech_pad_ms=300)
    vad = SileroVAD(thresholds=thresholds)
    session = VADTestSession(vad)

    await session.process_pattern([
        ("speech", 500),
        ("silence", 200),  # Less than pad_ms
        ("speech", 500),
    ])

    session.assert_turn_count(1)  # Joined into single turn
    session.assert_total_speech_duration(1000, 1300)
```

#### Pattern 3: Maximum Duration Cutoff
```python
async def test_max_duration_enforcement():
    """Long speech should be cut at max_speech_ms."""
    thresholds = VADThresholds(max_speech_ms=2000)
    vad = SileroVAD(thresholds=thresholds)
    session = VADTestSession(vad)

    # 5 seconds of continuous speech
    await session.process_pattern([("speech", 5000)])

    audio_events = session.get_audio_events()
    assert len(audio_events) >= 2  # Split into multiple turns
    assert all(e.duration_ms <= 2000 for e in audio_events)
```

### Test Execution

#### Commands:
```bash
# Run all VAD tests
PYTHONPATH=stream-py:. uv run pytest tests/test_vad_base.py tests/test_vad_integration.py -v

# Run Silero-specific tests
PYTHONPATH=stream-py:. uv run pytest plugins/silero/tests/test_silero_vad.py -v

# Run with coverage
PYTHONPATH=stream-py:. uv run pytest tests/test_vad*.py --cov=vision_agents.core.vad --cov-report=html

# Run performance tests only
PYTHONPATH=stream-py:. uv run pytest tests/test_vad*.py -k "performance" -v
```

#### Test Organization:
```
tests/
├── test_vad_base.py           # Base class logic tests with MockVAD
├── test_vad_integration.py    # Integration with STT/TTS
└── test_vad_regression.py     # Specific bug regression tests

plugins/silero/tests/
└── test_silero_vad.py         # Silero-specific implementation tests

agents-core/vision_agents/core/vad/
└── testing.py                 # Shared test utilities
```

### Continuous Testing

#### GitHub Actions Workflow:
```yaml
name: VAD Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run VAD tests
        run: |
          PYTHONPATH=stream-py:. uv run pytest tests/test_vad*.py
          PYTHONPATH=stream-py:. uv run pytest plugins/*/tests/test_*vad*.py
```

## Migration Plan (Updated)
1) Create test infrastructure (VADTestAudioGenerator, MockVAD, VADTestSession)
2) Write base VAD tests using MockVAD to define expected behavior
3) Introduce model spec in base VAD constructor and shift normalization + windowing there
4) Update implementations (Silero) to only implement `is_speech` and pass model spec via super().__init__
5) Replace any implementation-level event emission with base emission
6) Run all tests with uv; fix regressions
7) Add integration and regression tests for complete coverage
8) Optional: extend `PcmData.resample` to support `target_format` (PCM_F32) for float models

## Acceptance Criteria (Updated)
- All VAD implementations rely on the base for buffering/windowing/events
- Base normalizes audio to the model's sample_rate/channels/format
- Events (start/partial/audio/end) originate from the base only
- **MockVAD tests pass, defining base class behavior**
- **Silero test suite passes with `uv run`**
- **Integration tests verify VAD works with STT/TTS components**
- **Test coverage > 80% for core VAD logic**
- **Performance tests confirm realtime processing**
- Plan documented and discoverable in `docs/`

## Test Coverage Metrics

### Minimum Required Coverage:
- Base VAD class: 85% line coverage
- State machine logic: 100% branch coverage
- Event emission: 100% coverage
- Audio normalization: 90% coverage
- Implementation is_speech(): 80% coverage

### How to Measure:
```bash
PYTHONPATH=stream-py:. uv run pytest tests/test_vad*.py \
  --cov=vision_agents.core.vad \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=80
```