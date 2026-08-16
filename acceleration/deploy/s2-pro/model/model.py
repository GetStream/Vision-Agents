"""Streaming Fish Audio S2 Pro text-to-speech over a Baseten WebSocket.

Wire protocol, deliberately the mirror image of the Parakeet deployment next door:
    1. Client sends a JSON text frame: {"sample_rate": 24000, "encoding": "linear16"}.
       The server replies {"type": "ready", "sample_rate": 24000}.
    2. Client sends {"type": "text", "id": "s1", "text": "..."} one or more times, then
       {"type": "flush", "id": "s1"} to say it now.
    3. Server streams binary frames of little-endian mono PCM16, then
       {"type": "final", "id": "s1", "audio_duration_ms": ..., "processing_time_ms": ...}.
    4. Client may send {"type": "cancel", "id": "s1"} for barge-in, which stops generation
       and is answered with a final marked {"cancelled": true}.

The model itself is served by sglang-omni's OpenAI-compatible endpoint, started once per
replica and reached over localhost. Wrapping it rather than reimplementing it means the
engine's batching, paged KV cache and prefix caching all still apply.
"""

import asyncio
import json
import logging
import os
import subprocess
import time

import fastapi
import httpx

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = "/app/model_cache/s2-pro"


class ProtocolError(Exception):
    """The client sent something the server cannot honour."""


class Model:
    def __init__(self, lazy_data_resolver, **kwargs) -> None:
        self._lazy_data_resolver = lazy_data_resolver
        self._port = int(os.getenv("ENGINE_PORT", "8000"))
        self._config = os.getenv("ENGINE_CONFIG", "")
        self._startup_s = float(os.getenv("ENGINE_STARTUP_S", "600"))
        self._sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))

        self._engine: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None

    def load(self) -> None:
        self._lazy_data_resolver.block_until_download_complete()

        started = time.time()
        self._engine = self._start_engine()
        self._await_engine()
        logger.info("engine ready in %.1fs", time.time() - started)

        # No timeout: a long utterance streams for as long as it takes, and the client's
        # own deadline is what bounds a request.
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self._port}", timeout=None
        )

    def _start_engine(self) -> subprocess.Popen:
        # --model-path overrides the model_path in the pipeline config, so the weights come
        # from the volume the cache already downloaded rather than being fetched again.
        command = [
            "sgl-omni",
            "serve",
            "--model-path",
            MODEL_CACHE_DIR,
            "--port",
            str(self._port),
        ]
        if self._config:
            command += ["--config", self._config]

        logger.info("starting inference engine: %s", " ".join(command))
        return subprocess.Popen(command)

    def _await_engine(self) -> None:
        """Blocks until the engine answers, so no request arrives before it can serve."""
        deadline = time.time() + self._startup_s

        while time.time() < deadline:
            if self._engine is not None and self._engine.poll() is not None:
                raise RuntimeError(
                    f"inference engine exited with code {self._engine.returncode}"
                )
            try:
                response = httpx.get(f"http://127.0.0.1:{self._port}/health", timeout=5)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)

        raise RuntimeError(f"inference engine was not ready within {self._startup_s}s")

    async def websocket(self, websocket: fastapi.WebSocket) -> None:
        try:
            await self._negotiate(websocket)
        except ProtocolError as exc:
            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}))
            await websocket.close(code=1008)
            return

        # Text arrives in deltas but the engine takes a whole utterance, so each id
        # accumulates until it is flushed.
        pending: dict[str, list[str]] = {}
        speaking: dict[str, asyncio.Task] = {}

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

                text = message.get("text")
                if text is None:
                    continue

                try:
                    frame = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if await self._handle_frame(websocket, frame, pending, speaking):
                    return
        except fastapi.WebSocketDisconnect:
            pass
        finally:
            for task in speaking.values():
                task.cancel()

    async def _negotiate(self, websocket: fastapi.WebSocket) -> None:
        """Reads and validates the opening metadata frame."""
        raw = await websocket.receive_text()
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"first frame must be JSON metadata: {exc}") from exc

        encoding = config.get("encoding", "linear16")
        if encoding != "linear16":
            raise ProtocolError(f"encoding must be linear16, got {encoding}")

        sample_rate = config.get("sample_rate", self._sample_rate)
        if sample_rate != self._sample_rate:
            raise ProtocolError(
                f"sample_rate must be {self._sample_rate}, got {sample_rate}"
            )

        await websocket.send_text(
            json.dumps({"type": "ready", "sample_rate": self._sample_rate})
        )

    async def _handle_frame(
        self,
        websocket: fastapi.WebSocket,
        frame: dict,
        pending: dict[str, list[str]],
        speaking: dict[str, asyncio.Task],
    ) -> bool:
        """Handles one control frame. Returns True when the session should end."""
        kind = frame.get("type")
        synthesis_id = str(frame.get("id", ""))

        if kind == "text":
            pending.setdefault(synthesis_id, []).append(frame.get("text", ""))
            return False

        if kind == "cancel":
            task = speaking.pop(synthesis_id, None)
            if task is not None:
                task.cancel()
            pending.pop(synthesis_id, None)
            return False

        if kind == "close":
            return True

        if kind != "flush":
            logger.debug("unhandled frame type %s", kind)
            return False

        text = "".join(pending.pop(synthesis_id, [])).strip()
        if not text:
            await websocket.send_text(
                json.dumps({"type": "final", "id": synthesis_id, "empty": True})
            )
            return False

        task = asyncio.create_task(
            self._speak(websocket, synthesis_id, text, frame)
        )
        speaking[synthesis_id] = task
        task.add_done_callback(lambda _: speaking.pop(synthesis_id, None))
        return False

    async def _speak(
        self,
        websocket: fastapi.WebSocket,
        synthesis_id: str,
        text: str,
        frame: dict,
    ) -> None:
        """Streams one utterance from the engine to the client."""
        payload = {
            "model": MODEL_CACHE_DIR,
            "input": text,
            "stream": True,
            "response_format": "pcm",
            "voice": frame.get("voice") or "default",
        }
        if frame.get("reference_audio"):
            payload["references"] = [
                {
                    "audio_path": frame["reference_audio"],
                    "text": frame.get("reference_text", ""),
                }
            ]

        started = time.monotonic()
        total_bytes = 0
        cancelled = False

        try:
            async with self._client.stream(
                "POST", "/v1/audio/speech", json=payload
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "id": synthesis_id,
                                "error": f"engine returned {response.status_code}: {detail[:500]}",
                            }
                        )
                    )
                    return

                # The engine reports the rate the codec actually reconstructed at. If it
                # disagrees with what the handshake promised, the client would play the
                # audio at the wrong speed, so the utterance fails instead.
                engine_rate = response.headers.get("x-sample-rate")
                if engine_rate is not None and int(engine_rate) != self._sample_rate:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "id": synthesis_id,
                                "error": (
                                    f"engine produced {engine_rate} Hz but the session "
                                    f"was opened at {self._sample_rate} Hz"
                                ),
                            }
                        )
                    )
                    return

                async for chunk in response.aiter_bytes():
                    if chunk:
                        total_bytes += len(chunk)
                        await websocket.send_bytes(chunk)
        except asyncio.CancelledError:
            cancelled = True
        except httpx.HTTPError as exc:
            await websocket.send_text(
                json.dumps({"type": "error", "id": synthesis_id, "error": str(exc)})
            )
            return

        # Two bytes per mono sample.
        duration_ms = total_bytes / 2 / self._sample_rate * 1000.0
        await websocket.send_text(
            json.dumps(
                {
                    "type": "final",
                    "id": synthesis_id,
                    "audio_duration_ms": duration_ms,
                    "processing_time_ms": (time.monotonic() - started) * 1000.0,
                    "cancelled": cancelled,
                }
            )
        )
