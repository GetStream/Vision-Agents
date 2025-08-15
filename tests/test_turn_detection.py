"""Tests for turn detection module."""

import time
from typing import Dict, Any
import pytest
from getstream.models import User
from getstream.video.rtc.track_util import PcmData
from pyee import EventEmitter

from turn_detection.turn_detection import (
    TurnEvent,
    TurnEventData,
    BaseTurnDetector,
    TurnDetection,
)


# Example concrete implementation using the simplified protocol
class StreamTurnDetector(BaseTurnDetector):
    """Stream.io-specific turn detection implementation."""

    def __init__(self, mini_pause_duration: float, max_pause_duration: float) -> None:
        super().__init__(mini_pause_duration, max_pause_duration)
        self._participants: Dict[str, User] = {}  # Track participants automatically
        self._audio_levels: Dict[str, float] = {}

    async def process_audio(
        self, audio_data: PcmData, user_id: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Process audio and automatically track participants."""
        # Automatically track participant when audio comes in
        if user_id not in self._participants:
            # Create a basic user object for testing
            user = User(
                id=user_id,
                role="participant",
                banned=False,
                online=True,
                custom={"name": f"User {user_id}"},
                teams_role={},
            )
            self._participants[user_id] = user
            self._audio_levels[user_id] = 0.0
            print(f"Auto-added participant: {user.custom.get('name')} ({user_id})")

        # Simulate processing audio (in real implementation, this would analyze audio)
        # For testing, we just update audio level
        if metadata and "audio_level" in metadata:
            self._audio_levels[user_id] = metadata["audio_level"]

    def start(self) -> None:
        """Start detection using the unified interface."""
        self.start_detection()
        print("Started Stream turn detection")

    def stop(self) -> None:
        """Stop detection using the unified interface."""
        self.stop_detection()
        print("Stopped Stream turn detection")

    def simulate_speech_event(
        self, user_id: str, event_type: TurnEvent, **kwargs
    ) -> None:
        """Simulate a speech event for testing purposes."""
        if user_id not in self._participants:
            print(f"User {user_id} not found in participants")
            return

        user = self._participants[user_id]

        event_data = TurnEventData(
            timestamp=time.time(),
            speaker=user,
            confidence=kwargs.get("confidence", 0.9),
            audio_level=kwargs.get("audio_level", 0.7),
            duration=kwargs.get("duration"),
            custom=kwargs.get("custom"),
        )

        self._emit_turn_event(event_type, event_data)


# Example of a completely different implementation that still satisfies the protocol
class SimpleTurnDetector(EventEmitter):
    """Alternative implementation that uses the protocol without inheritance."""

    def __init__(self, mini_pause_duration: float, max_pause_duration: float) -> None:
        super().__init__()
        # Different validation approach
        assert mini_pause_duration > 0, "Mini pause must be positive"
        assert max_pause_duration > mini_pause_duration, (
            "Max pause must be greater than mini pause"
        )

        self._mini_pause = mini_pause_duration
        self._max_pause = max_pause_duration
        self._detecting = False
        self._participants = {}

    @property
    def mini_pause_duration(self) -> float:
        return self._mini_pause

    @property
    def max_pause_duration(self) -> float:
        return self._max_pause

    def is_detecting(self) -> bool:
        return self._detecting

    async def process_audio(
        self, audio_data: PcmData, user_id: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Process audio - required by protocol."""
        # Track participants automatically
        if user_id not in self._participants:
            self._participants[user_id] = {"added_time": time.time()}
            print(f"Simple detector: auto-added participant {user_id}")

    def start(self) -> None:
        """Start detection using unified interface."""
        self._detecting = True
        print("Simple detector started")

    def stop(self) -> None:
        """Stop detection using unified interface."""
        self._detecting = False
        print("Simple detector stopped")

    def start_detection(self) -> None:
        """Legacy method for backward compatibility."""
        self.start()

    def stop_detection(self) -> None:
        """Legacy method for backward compatibility."""
        self.stop()


def use_any_detector(detector: TurnDetection) -> None:
    """Function that works with any object implementing TurnDetection protocol."""
    print(f"Using detector with mini_pause: {detector.mini_pause_duration}s")

    # Set up event listener using decorator syntax
    @detector.on("speech_started")
    def on_speech(event_data: TurnEventData) -> None:
        if event_data.speaker and event_data.speaker.custom:
            speaker_name = event_data.speaker.custom.get("name", "Unknown")
        else:
            speaker_name = "Unknown"
        print(f"Detected speech from {speaker_name}")

    detector.start()
    print(f"Detection active: {detector.is_detecting()}")


class TestBaseTurnDetector:
    """Tests for BaseTurnDetector class."""

    def test_init_valid_durations(self):
        """Test initialization with valid durations."""
        detector = BaseTurnDetector(0.5, 2.0)
        assert detector.mini_pause_duration == 0.5
        assert detector.max_pause_duration == 2.0
        assert not detector.is_detecting()

    def test_init_invalid_mini_pause(self):
        """Test initialization with invalid mini pause duration."""
        with pytest.raises(ValueError, match="mini_pause_duration must be positive"):
            BaseTurnDetector(-0.5, 2.0)

        with pytest.raises(ValueError, match="mini_pause_duration must be positive"):
            BaseTurnDetector(0, 2.0)

    def test_init_invalid_max_pause(self):
        """Test initialization with invalid max pause duration."""
        with pytest.raises(ValueError, match="max_pause_duration must be positive"):
            BaseTurnDetector(0.5, -2.0)

        with pytest.raises(ValueError, match="max_pause_duration must be positive"):
            BaseTurnDetector(0.5, 0)

    def test_init_mini_greater_than_max(self):
        """Test initialization with mini pause greater than max pause."""
        with pytest.raises(
            ValueError, match="mini_pause_duration must be less than max_pause_duration"
        ):
            BaseTurnDetector(2.0, 0.5)

        with pytest.raises(
            ValueError, match="mini_pause_duration must be less than max_pause_duration"
        ):
            BaseTurnDetector(2.0, 2.0)

    def test_start_stop_detection(self):
        """Test starting and stopping detection."""
        detector = BaseTurnDetector(0.5, 2.0)
        assert not detector.is_detecting()

        # Test unified interface
        detector.start()
        assert detector.is_detecting()

        detector.stop()
        assert not detector.is_detecting()

        # Test legacy methods still work
        detector.start_detection()
        assert detector.is_detecting()

        detector.stop_detection()
        assert not detector.is_detecting()

    def test_emit_turn_event(self):
        """Test emitting turn events."""
        detector = BaseTurnDetector(0.5, 2.0)
        event_received = []

        @detector.on(TurnEvent.SPEECH_STARTED.value)
        def on_speech(sppech):
            event_received.append(sppech)

        event_data = TurnEventData(timestamp=time.time(), confidence=0.9)
        detector._emit_turn_event(TurnEvent.SPEECH_STARTED, event_data)

        assert len(event_received) == 1
        assert event_received[0] == event_data


class TestStreamTurnDetector:
    """Tests for StreamTurnDetector class."""

    def test_initialization(self):
        """Test StreamTurnDetector initialization."""
        detector = StreamTurnDetector(0.5, 2.0)
        assert detector.mini_pause_duration == 0.5
        assert detector.max_pause_duration == 2.0
        assert not detector.is_detecting()
        assert len(detector._participants) == 0
        assert len(detector._audio_levels) == 0

    @pytest.mark.asyncio
    async def test_process_audio_auto_adds_participant(self, capsys):
        """Test that process_audio automatically adds participants."""
        detector = StreamTurnDetector(0.5, 2.0)

        # Simulate audio from a new user
        await detector.process_audio(b"audio_data", "test-user", {"audio_level": 0.5})

        # Participant should be automatically added
        assert "test-user" in detector._participants
        assert "test-user" in detector._audio_levels
        assert detector._audio_levels["test-user"] == 0.5

        captured = capsys.readouterr()
        assert "Auto-added participant: User test-user (test-user)" in captured.out

    @pytest.mark.asyncio
    async def test_process_audio_updates_existing_participant(self):
        """Test that process_audio updates existing participants."""
        detector = StreamTurnDetector(0.5, 2.0)

        # First audio from user
        await detector.process_audio(b"audio_data1", "test-user", {"audio_level": 0.3})
        assert detector._audio_levels["test-user"] == 0.3

        # Second audio from same user - should update, not add duplicate
        await detector.process_audio(b"audio_data2", "test-user", {"audio_level": 0.8})
        assert detector._audio_levels["test-user"] == 0.8
        assert len(detector._participants) == 1  # Still only one participant

    @pytest.mark.asyncio
    async def test_process_audio_no_metadata(self):
        """Test process_audio works without metadata."""
        detector = StreamTurnDetector(0.5, 2.0)

        # Should not raise an error
        await detector.process_audio(b"audio_data", "test-user")

        # Participant should still be added
        assert "test-user" in detector._participants

    def test_unified_start_stop_interface(self, capsys):
        """Test the unified start/stop interface."""
        detector = StreamTurnDetector(0.5, 2.0)

        detector.start()
        assert detector.is_detecting()

        captured = capsys.readouterr()
        assert "Started Stream turn detection" in captured.out

        detector.stop()
        assert not detector.is_detecting()

        captured = capsys.readouterr()
        assert "Stopped Stream turn detection" in captured.out

    @pytest.mark.asyncio
    async def test_simulate_speech_event_with_auto_participant(self, capsys):
        """Test simulating speech events with automatically added participants."""
        detector = StreamTurnDetector(0.5, 2.0)

        # First, add participant via process_audio
        await detector.process_audio(b"audio_data", "test-user")

        events_received = []

        @detector.on(TurnEvent.SPEECH_STARTED.value)
        def on_speech(event_data):
            events_received.append(event_data)

        detector.simulate_speech_event(
            "test-user",
            TurnEvent.SPEECH_STARTED,
            confidence=0.95,
            audio_level=0.8,
            duration=1.5,
            custom={"extra": "data"},
        )

        assert len(events_received) == 1
        event = events_received[0]
        assert event.speaker.id == "test-user"
        assert event.confidence == 0.95
        assert event.audio_level == 0.8
        assert event.duration == 1.5
        assert event.custom == {"extra": "data"}

    def test_simulate_speech_event_nonexistent_user(self, capsys):
        """Test simulating speech event for nonexistent user."""
        detector = StreamTurnDetector(0.5, 2.0)

        detector.simulate_speech_event("nonexistent", TurnEvent.SPEECH_STARTED)

        captured = capsys.readouterr()
        assert "User nonexistent not found in participants" in captured.out


class TestSimpleTurnDetector:
    """Tests for SimpleTurnDetector class."""

    def test_initialization(self):
        """Test SimpleTurnDetector initialization."""
        detector = SimpleTurnDetector(0.3, 1.5)
        assert detector.mini_pause_duration == 0.3
        assert detector.max_pause_duration == 1.5
        assert not detector.is_detecting()

    def test_init_invalid_durations(self):
        """Test initialization with invalid durations."""
        with pytest.raises(AssertionError, match="Mini pause must be positive"):
            SimpleTurnDetector(-0.3, 1.5)

        with pytest.raises(AssertionError, match="Mini pause must be positive"):
            SimpleTurnDetector(0, 1.5)

        with pytest.raises(
            AssertionError, match="Max pause must be greater than mini pause"
        ):
            SimpleTurnDetector(1.5, 0.3)

        with pytest.raises(
            AssertionError, match="Max pause must be greater than mini pause"
        ):
            SimpleTurnDetector(1.5, 1.5)

    def test_unified_start_stop_detection(self, capsys):
        """Test unified start/stop interface."""
        detector = SimpleTurnDetector(0.3, 1.5)

        # Test unified interface
        detector.start()
        assert detector.is_detecting()
        captured = capsys.readouterr()
        assert "Simple detector started" in captured.out

        detector.stop()
        assert not detector.is_detecting()
        captured = capsys.readouterr()
        assert "Simple detector stopped" in captured.out

        # Test legacy methods still work
        detector.start_detection()
        assert detector.is_detecting()
        detector.stop_detection()
        assert not detector.is_detecting()

    @pytest.mark.asyncio
    async def test_process_audio(self, capsys):
        """Test process_audio method."""
        detector = SimpleTurnDetector(0.3, 1.5)

        await detector.process_audio(b"audio_data", "user1")
        assert "user1" in detector._participants

        captured = capsys.readouterr()
        assert "Simple detector: auto-added participant user1" in captured.out

        # Second call with same user should not add again
        await detector.process_audio(b"more_audio", "user1")
        assert len(detector._participants) == 1


class TestProtocolCompatibility:
    """Tests for protocol compatibility between different implementations."""

    @pytest.mark.asyncio
    async def test_use_any_detector_with_stream_detector(self, capsys):
        """Test that StreamTurnDetector works with use_any_detector function."""
        stream_detector = StreamTurnDetector(0.5, 2.0)

        # Add participant through process_audio (new way)
        await stream_detector.process_audio(b"audio", "test-user")

        # Capture output to verify function works
        use_any_detector(stream_detector)

        captured = capsys.readouterr()
        assert "Using detector with mini_pause: 0.5s" in captured.out
        assert "Started Stream turn detection" in captured.out
        assert "Detection active: True" in captured.out

        # Simulate an event to test the listener
        stream_detector.simulate_speech_event("test-user", TurnEvent.SPEECH_STARTED)
        captured = capsys.readouterr()
        assert "Detected speech from User test-user" in captured.out

    def test_use_any_detector_with_simple_detector(self, capsys):
        """Test that SimpleTurnDetector works with use_any_detector function."""
        simple_detector = SimpleTurnDetector(0.3, 1.5)

        use_any_detector(simple_detector)

        captured = capsys.readouterr()
        assert "Using detector with mini_pause: 0.3s" in captured.out
        assert "Simple detector started" in captured.out
        assert "Detection active: True" in captured.out

    def test_both_satisfy_protocol(self):
        """Test that both implementations satisfy the TurnDetectionProtocol."""
        stream_detector = StreamTurnDetector(0.5, 2.0)
        simple_detector = SimpleTurnDetector(0.3, 1.5)

        # Both should have required properties and methods
        for detector in [stream_detector, simple_detector]:
            # Required properties
            assert hasattr(detector, "mini_pause_duration")
            assert hasattr(detector, "max_pause_duration")
            assert hasattr(detector, "is_detecting")

            # Required methods (unified interface)
            assert hasattr(detector, "start")
            assert hasattr(detector, "stop")
            assert hasattr(detector, "process_audio")
            assert hasattr(detector, "on")
            assert hasattr(detector, "emit")

            # All methods should be callable
            assert callable(detector.is_detecting)
            assert callable(detector.start)
            assert callable(detector.stop)
            assert callable(detector.process_audio)
            assert callable(detector.on)
            assert callable(detector.emit)


class TestTurnEventData:
    """Tests for TurnEventData dataclass."""

    def test_minimal_initialization(self):
        """Test TurnEventData with minimal required fields."""
        event_data = TurnEventData(timestamp=1234567890.0)
        assert event_data.timestamp == 1234567890.0
        assert event_data.speaker is None
        assert event_data.duration is None
        assert event_data.confidence is None
        assert event_data.audio_level is None
        assert event_data.custom is None

    def test_full_initialization(self):
        """Test TurnEventData with all fields."""
        user = User(
            id="test",
            role="participant",
            banned=False,
            online=True,
            custom={"name": "Test User"},
            teams_role={},
        )
        event_data = TurnEventData(
            timestamp=1234567890.0,
            speaker=user,
            duration=2.5,
            confidence=0.95,
            audio_level=0.7,
            custom={"key": "value"},
        )
        assert event_data.timestamp == 1234567890.0
        assert event_data.speaker == user
        assert event_data.duration == 2.5
        assert event_data.confidence == 0.95
        assert event_data.audio_level == 0.7
        assert event_data.custom == {"key": "value"}


class TestTurnEvent:
    """Tests for TurnEvent enum."""

    def test_enum_values(self):
        """Test that TurnEvent enum has expected values."""
        assert TurnEvent.SPEECH_STARTED.value == "speech_started"
        assert TurnEvent.SPEECH_ENDED.value == "speech_ended"
        assert TurnEvent.TURN_STARTED.value == "turn_started"
        assert TurnEvent.TURN_ENDED.value == "turn_ended"
        assert TurnEvent.MINI_PAUSE_DETECTED.value == "mini_pause_detected"
        assert TurnEvent.MAX_PAUSE_REACHED.value == "max_pause_reached"
