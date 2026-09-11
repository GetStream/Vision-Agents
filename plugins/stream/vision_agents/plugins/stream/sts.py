import asyncio
import json
import logging
import struct
from typing import Any, AsyncIterator, Optional

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm import realtime
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.utils import cancel_and_wait

from ._backend import Backend
from ._routerconfig import ensure_router
from ._socket import Socket

logger = logging.getLogger(__name__)

# SAMPLE_RATE is what the router hears at, and what every model behind it is fed.
SAMPLE_RATE = 16_000

# HEADER opens every audio frame the router sends: the sample rate, the channel count, the
# header version, the reply's generation and the chunk's index, all little-endian. The
# generation is what lets the tail of a reply the caller cut off be dropped: the model
# learns of a barge-in a round trip after the caller, and the chunks in that gap arrive
# after the router has said the reply was interrupted.
HEADER = struct.Struct("<IHHII")
HEADER_VERSION = 1

# READY is how long connect waits for the router to say the model took its configuration.
READY = 30.0

# TOOL_TIMEOUT bounds one tool call, the same as the direct providers allow.
TOOL_TIMEOUT = 30.0


class STS(realtime.Realtime):
    """A speech-to-speech model routed through the acceleration backend.

    For a pipeline that stays in Python and owns its media: the call, the microphone and
    the speaker are here, and only the model is somewhere else. It is a `Realtime` like the
    direct OpenAI and Gemini ones, so an `Agent` uses it the same way, and it gains what
    the router gives every session: a choice of vendor, failover at start, a data policy
    and one bill.

    ```python
    llm = stream.STS(target="sts-fast", voice="Kore")
    agent = Agent(edge=edge, llm=llm, instructions="Be brief.")
    ```

    Tools registered on it before `connect()` are handed to the model as the session
    opens, because some models take them then and never again.
    """

    provider_name = "stream"

    def __init__(
        self,
        target: str = "",
        voice: str = "",
        language: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
        config_id: str = "",
        options: Optional[dict[str, Any]] = None,
        fps: int = 1,
    ):
        """Route a conversation to `target`.

        Args:
            target: A `provider/model` name or a capability shortcut such as `sts-fast`.
                It may instead be held in the named config.
            voice: The vendor's own name for a voice, such as `marin` or `Kore`.
            language: A language hint, which narrows the candidates.
            tags: Cost labels carried onto every request.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
            config_id: A stored router config to take the options from, by name or by id.
            options: Per-call overrides of that config's sts block. Usually built by
                `Router.sts.realtime`.
            fps: Kept for the `Realtime` contract; the router takes no video yet.
        """
        super().__init__(fps=fps)
        self.model = target
        self.voice = voice
        self.language = language
        self.tags = tags or {}
        self.config_id = config_id
        self.options = options or {}
        self.backend = Backend(url=url, customer_id=customer_id)

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task[None]] = None
        self._ready: Optional[asyncio.Future[dict[str, Any]]] = None
        # replies maps a generation to the router's id for it, stale is the highest
        # generation whose audio is no longer wanted, and speaking the one in flight.
        self._replies: dict[int, str] = {}
        self._stale = 0
        self._speaking = 0

    async def connect(self) -> None:
        """Open the socket and wait for the router to say the model is ready."""
        await ensure_router(self.config_id, self.backend)
        self._socket = Socket(
            self.backend.socket("/v1/sts/stream"), self.backend.headers
        )
        await self._socket.connect()
        self._ready = asyncio.get_running_loop().create_future()
        self._reader = asyncio.create_task(self._read())
        await self._socket.send(self._start())
        # The started frame is sent only once the model has taken its configuration, so
        # waiting for it is what keeps audio from being sent to a session that is not there.
        started = await asyncio.wait_for(asyncio.shield(self._ready), READY)
        self._on_connected(
            session_config=started,
            capabilities=["audio", "text", "function_calling"],
        )

    def _start(self) -> dict[str, Any]:
        """The frame that says what to route to and how the model should behave."""
        block = dict(self.options)
        if self._instructions:
            block["instructions"] = self._instructions
        if self.voice:
            block["voice"] = self.voice
        # Both transcripts, unless the caller said otherwise: the conversation store and
        # the agent's events key on what was heard and said.
        block.setdefault("input_transcript", True)
        block.setdefault("output_transcript", True)

        frame: dict[str, Any] = {
            "type": "start",
            "config_id": self.config_id,
            "target": self.model,
            "languages": [self.language] if self.language else [],
            "tags": self.tags,
            "sample_rate": SAMPLE_RATE,
            "sts": block,
        }
        tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters_schema", {}),
            }
            for tool in self.get_available_functions()
        ]
        if tools:
            frame["tools"] = tools
        return frame

    def set_instructions(self, instructions: Instructions | str) -> None:
        """Change the system prompt, on the models that allow it once the session is open.

        Before `connect()` this only decides what the session opens with. Afterwards it is
        sent, and a model that took its instructions at setup answers with an error frame
        rather than pretending.
        """
        super().set_instructions(instructions)
        if self._socket is not None and self._socket.open:
            asyncio.ensure_future(
                self._socket.send(
                    {"type": "instructions", "instructions": self._instructions}
                )
            )

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Inject a typed turn, which the model answers as it would a spoken one.

        The reply is spoken and reported as events rather than yielded here. A model that
        takes no typed turns refuses with an error event.
        """
        if self._socket is not None and self._socket.open:
            await self._socket.send({"type": "text", "text": text})
        yield LLMResponseFinal()

    async def simple_audio_response(self, pcm: PcmData, participant: Participant):
        """Send the caller's audio, resampled to what the router expects."""
        if self._socket is None or not self._socket.open:
            return
        self._current_participant = participant
        resampled = pcm.resample(SAMPLE_RATE, 1)
        await self._socket.send_audio(resampled.samples.tobytes())

    async def interrupt(self) -> None:
        """Stop the reply in flight, here and at the model."""
        await super().interrupt()
        if self._speaking > self._stale:
            self._stale = self._speaking
        if self._socket is not None and self._socket.open and self._speaking:
            await self._socket.send({"type": "interrupt"})

    async def watch_video_track(self, track, shared_forwarder=None) -> None:
        """The routed socket takes frames, but this client does not send them yet."""
        logger.warning("stream.STS does not forward video frames yet")

    async def close(self) -> None:
        """Close the socket and stop reading it."""
        await self._close_audio_input()
        await self._await_pending_tools()
        if self._reader is not None:
            await cancel_and_wait(self._reader)
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
            self._on_disconnected()

    async def _read(self) -> None:
        """Turn what the router says into the events a Realtime emits."""
        if self._socket is None:
            return

        async for frame in self._socket.frames():
            if isinstance(frame, bytes):
                self._heard_audio(frame)
            else:
                await self._received(frame)

    def _heard_audio(self, payload: bytes) -> None:
        """Play a piece of the model's voice, unless it belongs to a reply that was cut off."""
        if len(payload) < HEADER.size:
            return
        rate, channels, version, generation, _ = HEADER.unpack_from(payload)
        if version != HEADER_VERSION or generation <= self._stale:
            return
        samples = np.frombuffer(payload[HEADER.size :], dtype="<i2").astype(np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=rate,
            channels=channels,
            format=AudioFormat.S16,
        )
        self._emit_audio_output_event(pcm, response_id=self._replies.get(generation))

    async def _received(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type", "")

        if kind == "started":
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(frame)
        elif kind == "speech_started":
            self._emit_user_speech_started()
        elif kind == "speech_stopped":
            self._emit_user_speech_ended()
        elif kind == "input_transcript":
            self._emit_user_speech_transcription(
                frame.get("text", ""), mode=frame.get("mode", "delta")
            )
        elif kind == "output_transcript":
            self._emit_agent_speech_transcription(
                frame.get("text", ""), mode=frame.get("mode", "delta")
            )
        elif kind == "response_started":
            generation = int(frame.get("generation", 0))
            self._replies[generation] = frame.get("id", "")
            self._speaking = generation
            self._emit_agent_speech_started(response_id=frame.get("id"))
        elif kind == "response_complete":
            await self._completed(frame)
        elif kind == "tool_call":
            self._run_tool_in_background(self._call_tool(frame))
        elif kind == "session_expiring":
            logger.warning(
                "the model's session is about to be cut off", extra={"frame": frame}
            )
        elif kind == "error":
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(RuntimeError(frame.get("error", "")))
                return
            self._emit_error_event(
                RuntimeError(frame.get("error", "")), context=frame.get("context", "")
            )
        elif kind == "closed":
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(RuntimeError("the router closed the socket"))
            self._on_disconnected(reason="closed")

    async def _completed(self, frame: dict[str, Any]) -> None:
        """Settle a reply. Interrupted is the one signal that the caller cut in."""
        generation = int(frame.get("generation", 0))
        response_id = frame.get("id")
        interrupted = bool(frame.get("interrupted"))
        if interrupted:
            if generation > self._stale:
                self._stale = generation
            await super().interrupt()
        if self._speaking == generation:
            self._speaking = 0
        self._replies.pop(generation, None)
        self._emit_agent_speech_ended(response_id=response_id, interrupted=interrupted)
        self._emit_audio_output_done_event(
            response_id=response_id, interrupted=interrupted
        )
        self._emit_response_event("", response_id=response_id)

    async def _call_tool(self, frame: dict[str, Any]) -> None:
        """Run the tool the model asked for and send back what it produced."""
        call_id = frame.get("id", "")
        name = frame.get("name", "")
        try:
            arguments = json.loads(frame.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        tool_call = {
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "arguments_json": arguments,
        }

        _, result, error = await self._run_one_tool(tool_call, timeout_s=TOOL_TIMEOUT)
        if self._socket is None or not self._socket.open:
            return
        if error:
            self._emit_error_event(
                error if isinstance(error, BaseException) else Exception(str(error)),
                context=f"tool_call:{name}",
            )
            await self._socket.send(
                {"type": "tool_result", "tool_call_id": call_id, "error": str(error)}
            )
            return
        output = result if isinstance(result, str) else json.dumps(result)
        await self._socket.send(
            {"type": "tool_result", "tool_call_id": call_id, "output": output}
        )
