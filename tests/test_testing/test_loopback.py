"""Unit tests for the loopback edge: an Agent joins it with no network, hears a
microphone at real time and is heard back through its audio track."""

import asyncio
import time
from contextlib import AsyncExitStack

import numpy as np
import pytest
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import Agent, User
from vision_agents.core.edge.events import AudioReceivedEvent
from vision_agents.core.edge.types import Participant
from vision_agents.testing import LoopbackEdge

from .fakes import CodecSTT, CodecTTS, ScriptedLLM, decode_speech, encode_speech

CALLER = Participant(original=None, user_id="caller", id="caller")


@pytest.fixture
def edge() -> LoopbackEdge:
    return LoopbackEdge()


@pytest.fixture
def agent(edge: LoopbackEdge) -> Agent:
    return Agent(
        edge=edge,
        llm=ScriptedLLM(["Hello there"]),
        stt=CodecSTT(),
        tts=CodecTTS(),
        agent_user=User(id="agent", name="Agent"),
        instructions="Be brief.",
    )


async def _heard(frames: list[PcmData], text: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        heard = decode_speech(np.concatenate([f.samples for f in frames]))
        if text in heard:
            return heard
        await asyncio.sleep(0.05)
    return decode_speech(np.concatenate([f.samples for f in frames]))


class TestLoopbackEdge:
    async def test_agent_joins_and_leaves_without_network(self, edge, agent):
        call = await agent.create_call("default", "loopback-call")
        assert call.id == "loopback-call"
        edge.microphone(CALLER)

        async with agent.join(call):
            assert not agent.closed
            assert agent.idle_for() == 0.0
            await agent.send_custom_event({"type": "ping"})

        assert agent.closed
        assert edge.custom_events == [{"type": "ping"}]

    async def test_wait_for_participant_resolves_when_microphone_opens(
        self, edge, agent
    ):
        connection = await edge.join(agent, await edge.create_call("c"))
        assert edge.custom_events == []
        assert connection.idle_since() > 0.0

        with pytest.raises(asyncio.TimeoutError):
            await connection.wait_for_participant(timeout=0.05)

        waiter = asyncio.create_task(connection.wait_for_participant(timeout=2.0))
        await asyncio.sleep(0.05)
        edge.microphone(CALLER)
        await waiter
        assert connection.idle_since() == 0.0
        await edge.close()

    async def test_microphone_speech_is_transcribed_by_the_agent(self, edge, agent):
        frames: list[PcmData] = []

        async def hear(pcm: PcmData) -> None:
            frames.append(pcm)

        edge.on_agent_audio(hear)
        mic = edge.microphone(CALLER)
        call = await agent.create_call("default", "c")
        async with agent.join(call):
            await mic.play(encode_speech("hi"))
            heard = await _heard(frames, "Hello there")

        assert heard == "Hello there"
        assert all(f.sample_rate == 16000 and f.channels == 1 for f in frames)
        assert all(len(f.samples) == 320 for f in frames)

    async def test_hang_up_ends_the_call_for_the_agent(self, edge, agent):
        edge.microphone(CALLER)
        call = await agent.create_call("default", "c")
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(agent.join(call))
            finished = asyncio.create_task(agent.finish())

            await asyncio.sleep(0.05)
            assert not finished.done()
            await edge.hang_up()

            await asyncio.wait_for(finished, timeout=5.0)
            deadline = time.monotonic() + 5.0
            while not agent.closed and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert agent.closed

    async def test_close_is_idempotent(self, edge, agent):
        edge.microphone(CALLER)
        await edge.join(agent, await edge.create_call("c"))
        await edge.close()
        await edge.close()


class TestLoopbackMicrophone:
    async def test_play_returns_when_the_speaker_stops_talking(self, edge):
        mic = edge.microphone(CALLER)
        heard: list[PcmData] = []

        @edge.events.subscribe
        async def on_audio(event: AudioReceivedEvent) -> None:
            heard.append(event.pcm_data)

        pcm = encode_speech("x" * 10)  # 200 ms
        started = time.monotonic()
        stopped_at = await mic.play(pcm)
        elapsed = time.monotonic() - started

        assert 0.18 <= elapsed <= 1.0
        assert started < stopped_at <= time.monotonic()
        await asyncio.sleep(0.1)
        spoken = np.concatenate([p.samples for p in heard])
        assert decode_speech(spoken) == "x" * 10
        await mic.stop()

    async def test_sends_silence_between_utterances(self, edge):
        mic = edge.microphone(CALLER)
        heard: list[PcmData] = []

        @edge.events.subscribe
        async def on_audio(event: AudioReceivedEvent) -> None:
            heard.append(event.pcm_data)

        await asyncio.sleep(0.1)
        await mic.stop()

        assert len(heard) >= 3
        assert all(not np.any(p.samples) for p in heard)
        assert all(p.participant is CALLER for p in heard)

    async def test_rejects_overlapping_utterances(self, edge):
        mic = edge.microphone(CALLER)
        playing = asyncio.create_task(mic.play(encode_speech("x" * 10)))
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError, match="already playing"):
            await mic.play(encode_speech("y"))
        await playing
        await mic.stop()

    async def test_stop_releases_a_pending_play(self, edge):
        mic = edge.microphone(CALLER)
        playing = asyncio.create_task(mic.play(encode_speech("x" * 50)))
        await asyncio.sleep(0.05)
        await mic.stop()
        await asyncio.wait_for(playing, timeout=1.0)
        with pytest.raises(RuntimeError, match="stopped"):
            await mic.play(encode_speech("y"))
