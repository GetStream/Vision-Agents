"""Shared test helpers for the local plugin."""

import asyncio
import threading

import numpy as np
from vision_agents.core.edge.types import User
from vision_agents.plugins.local.devices import AudioInputDevice, AudioOutputDevice
from vision_agents.plugins.local.edge import LocalEdge


class _FakeAudioInput(AudioInputDevice):
    """Fake audio input device for testing."""

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        super().__init__(
            index=0, name="fake-input", sample_rate=sample_rate, channels=channels
        )
        self.started = False
        self.stopped = False
        self._data: list[np.ndarray] = []
        self._stop_event = threading.Event()

    def enqueue(self, data: np.ndarray) -> None:
        self._data.append(data)

    def start(self) -> None:
        self.started = True

    def read(self) -> np.ndarray | None:
        if self._data:
            return self._data.pop(0)
        self._stop_event.wait(timeout=0.1)
        return self._data.pop(0) if self._data else None

    def stop(self) -> None:
        self.stopped = True
        self._stop_event.set()


class _FakeAudioOutput(AudioOutputDevice):
    """Fake audio output device for testing."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        super().__init__(
            index=0, name="fake-output", sample_rate=sample_rate, channels=channels
        )
        self.started = False
        self.stopped = False
        self.written: list[np.ndarray] = []
        self._write_barrier = threading.Event()
        self._write_barrier.set()
        self._consumed = threading.Event()

    def start(self) -> None:
        self.started = True

    def write(self, samples: np.ndarray) -> None:
        self._write_barrier.wait()
        self.written.append(samples.copy())
        self._consumed.set()

    async def wait_consumed(self) -> None:
        await asyncio.to_thread(self._consumed.wait)
        self._consumed.clear()

    def flush(self) -> None:
        self.written.clear()

    def stop(self) -> None:
        self.stopped = True


class _FakeAgent:
    """Minimal Agent stub for transport.join() calls."""

    def __init__(self) -> None:
        self.agent_user = User(id="test-agent", name="Test Agent")


def _make_transport(
    audio_input: _FakeAudioInput | None = None,
    audio_output: _FakeAudioOutput | None = None,
    **kwargs: object,
) -> LocalEdge:
    """Create a LocalEdge with fake audio devices."""
    return LocalEdge(
        audio_input=audio_input or _FakeAudioInput(),
        audio_output=audio_output or _FakeAudioOutput(),
        **kwargs,
    )
