"""Scripted LLM, judge, TTS and STT doubles for simulation unit tests (no real model)."""

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import AsyncIterator

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.stt.stt import STT, TranscriptResponse
from vision_agents.core.tts.tts import TTS
from vision_agents.testing import (
    Criterion,
    CriterionVerdict,
    JudgeVerdict,
    RunEvent,
    render_transcript,
)


def user_says(message: str) -> str:
    return json.dumps({"message": message, "done": False})


def user_done() -> str:
    return json.dumps({"message": "", "done": True})


class ScriptedLLM(LLM):
    """LLM that replays scripted replies, repeating the last one when exhausted.

    Raises ``error`` on the first call instead when it is given.
    """

    def __init__(
        self, replies: list[str], delay: float = 0.0, error: Exception | None = None
    ) -> None:
        super().__init__()
        self.replies = list(replies)
        self.delay = delay
        self.error = error

    @property
    def history(self) -> list[tuple[str, str]]:
        """(role, content) pairs recorded in the conversation given to this LLM."""
        if self._conversation is None:
            return []
        return [(m.role, m.content) for m in self._conversation.messages]

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        if self.error is not None:
            raise self.error
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        yield LLMResponseFinal(text=reply)


class BookingLLM(ScriptedLLM):
    """Agent double that calls its ``book_slot`` tool before every reply."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)

        @self.register_function(description="Book a slot")
        async def book_slot(day: str, time: str) -> dict[str, str]:
            return {"day": day, "time": time, "status": "booked"}

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        await self._dedup_and_execute(
            [
                {
                    "type": "tool_call",
                    "name": "book_slot",
                    "arguments_json": {"day": "Friday", "time": "10am"},
                    "id": f"call_{len(self.replies)}",
                }
            ]
        )
        async for item in super().simple_response(text, participant):
            yield item


JudgeOutcome = bool | Exception | Callable[[str, Criterion], bool]


class ScriptedJudge:
    """Judge that replays one outcome per criterion: a verdict, an exception to
    raise, or a predicate over (rendered transcript, criterion). Passes once
    the script is exhausted."""

    def __init__(self, outcomes: list[JudgeOutcome] | None = None) -> None:
        self.outcomes = list(outcomes or [])

    async def evaluate_conversation(
        self,
        events: Sequence[RunEvent],
        criteria: Sequence[Criterion | str],
        *,
        instructions: str | None = None,
    ) -> JudgeVerdict:
        transcript = render_transcript(list(events))
        verdicts: list[CriterionVerdict] = []
        for item in criteria:
            criterion = item if isinstance(item, Criterion) else Criterion(item, item)
            outcome: JudgeOutcome = self.outcomes.pop(0) if self.outcomes else True
            if isinstance(outcome, Exception):
                raise outcome
            if callable(outcome):
                outcome = outcome(transcript, criterion)
            verdicts.append(
                CriterionVerdict(
                    name=criterion.name,
                    success=outcome,
                    score=1.0 if outcome else 0.0,
                    reason="scripted",
                )
            )
        return JudgeVerdict(
            success=all(v.success for v in verdicts),
            reason="scripted",
            score=sum(v.score for v in verdicts) / len(verdicts),
            criteria=verdicts,
        )


BLOCK = 320  # samples per character: 20 ms at 16 kHz
LEVEL = 100  # int16 amplitude per code point


def encode_speech(text: str) -> PcmData:
    """Encode text as 16 kHz PCM: one 20 ms block of constant level per character."""
    codes = np.array([min(ord(c), 255) for c in text], dtype=np.int16) * LEVEL
    return PcmData(
        samples=np.repeat(codes, BLOCK),
        sample_rate=16000,
        format=AudioFormat.S16,
        channels=1,
    )


def decode_speech(samples: np.ndarray) -> str:
    """Inverse of ``encode_speech``: runs of a constant non-zero level become characters."""
    if samples.size == 0:
        return ""
    codes = np.rint(samples.astype(np.float64) / LEVEL).astype(int)
    changes = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(codes)]))
    text = []
    for start, end in zip(starts, ends):
        code = int(codes[start])
        count = round((end - start) / BLOCK)
        if code > 0 and count:
            text.append(chr(code) * count)
    return "".join(text)


class CodecTTS(TTS):
    """TTS whose audio a ``CodecSTT`` can read back verbatim."""

    model = "codec"

    def __init__(self) -> None:
        super().__init__(provider_name="codec")

    async def stream_audio(self, text: str, *args: object, **kwargs: object) -> PcmData:
        return encode_speech(text)

    async def stop_audio(self) -> None:
        return None


class CodecSTT(STT):
    """STT that decodes ``CodecTTS`` audio; a final transcript follows 40 ms of silence."""

    turn_detection = False
    model = "codec"

    def __init__(self) -> None:
        super().__init__(provider_name="codec")
        self._buffer: list[np.ndarray] = []
        self._silence = 0

    async def process_audio(self, pcm_data: PcmData, participant: Participant) -> None:
        samples = pcm_data.resample(16000, 1).to_int16().samples.reshape(-1)
        if np.any(samples):
            self._buffer.append(samples)
            self._silence = 0
            return
        if not self._buffer:
            return
        self._silence += len(samples)
        if self._silence < 2 * BLOCK:
            return
        text = decode_speech(np.concatenate(self._buffer))
        self._buffer.clear()
        self._silence = 0
        if text:
            self._emit_transcript_event(text, participant, TranscriptResponse())
