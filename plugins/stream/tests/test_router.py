import asyncio
import base64
import json
from typing import Any, AsyncIterator

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.plugins import stream
from vision_agents.plugins.stream import router as router_module

SETTLE = 2.0

# A job the fake finishes on the first ask, so a test polling it takes one turn of the
# loop rather than a real interval.
QUICK_POLL = 0.01


class Router:
    """A stand-in for the acceleration router, serving what a `Router` uses.

    It is a real server rather than a stub object, so what the client sends is what the
    backend would receive: JSON over HTTP for the jobs and the configs, and frames over a
    socket for the live modalities.
    """

    def __init__(self):
        self.started: dict[str, dict[str, Any]] = {}
        self.spoken: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.transcriptions: list[dict[str, Any]] = []
        self.speeches: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any]] = []
        self.url = ""
        # failing makes the next job fail, which is the other half of what a job does.
        self.failing = False
        self._opened: dict[str, asyncio.Event] = {
            modality: asyncio.Event() for modality in ("stt", "tts", "llm")
        }

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/{modality}/stream", self._stream)
        app.router.add_post("/v1/stt/recordings", self._transcribe)
        app.router.add_get("/v1/stt/recordings/{id}", self._transcription)
        app.router.add_post("/v1/tts/recordings", self._speak)
        app.router.add_get("/v1/tts/recordings/{id}", self._speech)
        app.router.add_post("/v1/search", self._search)
        app.router.add_get("/v1/router/configs", self._list_configs)
        app.router.add_post("/v1/router/configs", self._create_config)
        app.router.add_put("/v1/router/configs/{id}", self._update_config)
        return app

    async def opening(self, modality: str) -> dict[str, Any]:
        """The start frame a session opened with."""
        await asyncio.wait_for(self._opened[modality].wait(), SETTLE)
        return self.started[modality]

    async def _stream(self, request: web.Request) -> web.WebSocketResponse:
        modality = request.match_info["modality"]
        socket = web.WebSocketResponse()
        await socket.prepare(request)

        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            frame = json.loads(message.data)
            if frame.get("type") == "start":
                self.started[modality] = frame
                self._opened[modality].set()
            elif frame.get("type") == "speak":
                await self.spoken.put(frame)
                await socket.send_json(
                    {"type": "synthesis_complete", "id": frame.get("id")}
                )
        return socket

    async def _transcribe(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.transcriptions.append(body)
        return web.json_response(status=202, data=self._job("queued"))

    async def _transcription(self, request: web.Request) -> web.Response:
        if self.failing:
            failed = self._job("failed")
            failed["error"] = "no provider took the recording"
            return web.json_response(failed)

        done = self._job("completed")
        done.update(
            {
                "language": "en",
                "text": "a call costs a penny",
                "speakers": ["speaker_0"],
                "audio_duration_ms": 4000,
                "provider": "deepgram",
                "model": "nova-3",
            }
        )
        return web.json_response(done)

    async def _speak(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.speeches.append(body)
        return web.json_response(status=202, data=self._job("queued"))

    async def _speech(self, request: web.Request) -> web.Response:
        done = self._job("completed")
        done.update(
            {
                "format": "mp3_44100_128",
                "audio": base64.b64encode(b"\xff\xfb\x90").decode(),
                "characters": 21,
            }
        )
        return web.json_response(done)

    async def _search(self, request: web.Request) -> web.Response:
        self.searches.append(await request.json())
        return web.json_response(
            {
                "provider": "exa",
                "model": "fast",
                "answer": "within an hour of the incision",
                "results": [
                    {"title": "Guidance", "url": "https://nice.org.uk/1", "score": 0.9}
                ],
            }
        )

    async def _list_configs(self, request: web.Request) -> web.Response:
        return web.json_response(self.configs)

    async def _create_config(self, request: web.Request) -> web.Response:
        body = await request.json()
        stored = dict(body, id=f"config-{len(self.configs) + 1}", **self._stamps())
        self.configs.append(stored)
        return web.json_response(status=201, data=stored)

    async def _update_config(self, request: web.Request) -> web.Response:
        body = await request.json()
        held = request.match_info["id"]
        stored = dict(body, id=held, **self._stamps())
        self.configs = [stored if one["id"] == held else one for one in self.configs]
        return web.json_response(stored)

    def _job(self, status: str) -> dict[str, Any]:
        return dict(id="recording-1", status=status, **self._stamps())

    def _stamps(self) -> dict[str, str]:
        return {
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }


class TestRouter:
    @pytest.fixture
    async def backend(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    def router(self, backend: Router, monkeypatch: pytest.MonkeyPatch) -> stream.Router:
        monkeypatch.setattr(router_module, "POLL", QUICK_POLL)
        return stream.Router(
            "healthcare",
            tags={"project": "clinic"},
            url=backend.url,
            customer_id="acme",
        )

    async def test_a_transcription_socket_opens_from_the_named_config(
        self, router: stream.Router, backend: Router
    ):
        async with router.stt.realtime():
            opening = await backend.opening("stt")

        assert opening["config_id"] == "healthcare"
        assert opening["tags"] == {"project": "clinic"}
        assert opening["stt"] == {}, "a call that overrides nothing sends nothing"

    async def test_a_keyword_overrides_one_field_of_the_config(
        self, router: stream.Router, backend: Router
    ):
        async with router.stt.realtime(diarize=True, keyterms=["perioperative"]):
            opening = await backend.opening("stt")

        assert opening["config_id"] == "healthcare"
        assert opening["stt"] == {"diarize": True, "keyterms": ["perioperative"]}

    async def test_a_voice_is_configured_per_call_and_speaks_what_it_is_sent(
        self, router: stream.Router, backend: Router
    ):
        async with router.tts.realtime(voice="dc4e4a1f", speed=1.1) as tts:
            opening = await backend.opening("tts")
            async for _ in await tts.stream_audio("hello there"):
                pass

        assert opening["tts"] == {"voice": "dc4e4a1f", "speed": 1.1}
        spoken = await asyncio.wait_for(backend.spoken.get(), SETTLE)
        assert spoken["text"] == "hello there"

    async def test_the_model_is_told_what_the_config_holds(
        self, router: stream.Router, backend: Router
    ):
        async with router.llm.realtime(temperature=0.2, max_output_tokens=200):
            opening = await backend.opening("llm")

        assert opening["llm"] == {"temperature": 0.2, "max_output_tokens": 200}

    async def test_an_option_the_modality_does_not_have_is_refused(
        self, router: stream.Router
    ):
        # Refusing here is the same bargain the backend makes with a provider that cannot
        # express a term: better to be told than to be answered wrongly.
        with pytest.raises(ValueError, match="diarize"):
            router.tts.realtime(diarize=True)

    async def test_a_recording_is_waited_for_and_handed_back_whole(
        self, router: stream.Router, backend: Router
    ):
        transcript = await router.stt.recording(
            "https://example.test/call.mp3", diarize=True, words=True
        )

        assert transcript.text == "a call costs a penny"
        assert transcript.speakers == ["speaker_0"]
        assert transcript.provider == "deepgram"

        asked = backend.transcriptions[0]
        assert asked["source"] == {"url": "https://example.test/call.mp3"}
        assert asked["config_id"] == "healthcare"
        assert asked["options"] == {"diarize": True, "words": True}
        assert asked["tags"] == {"project": "clinic"}

    async def test_a_local_file_is_sent_as_the_audio_itself(
        self, router: stream.Router, backend: Router, tmp_path
    ):
        clip = tmp_path / "call.wav"
        clip.write_bytes(b"RIFFclip")

        await router.stt.recording(clip)

        assert backend.transcriptions[0]["source"] == {
            "audio": base64.b64encode(b"RIFFclip").decode()
        }

    async def test_a_source_that_is_neither_a_url_nor_a_file_says_which(
        self, router: stream.Router
    ):
        # Base64 audio handed over as text is the way this goes wrong in practice, and it
        # is long enough that the filesystem refuses the name rather than answering.
        with pytest.raises(ValueError, match="neither a URL nor a file"):
            await router.stt.recording(base64.b64encode(b"RIFF" * 200).decode())

    async def test_a_recording_with_a_callback_is_not_waited_for(
        self, router: stream.Router, backend: Router
    ):
        job = await router.stt.recording(
            "https://example.test/call.mp3", callback="https://example.test/done"
        )

        assert job.status.value == "queued", "a caller being called back does not poll"
        assert backend.transcriptions[0]["callback"] == "https://example.test/done"

    async def test_a_failed_recording_raises_what_went_wrong(
        self, router: stream.Router, backend: Router
    ):
        backend.failing = True

        with pytest.raises(RuntimeError, match="no provider took the recording"):
            await router.stt.recording("https://example.test/call.mp3")

    async def test_a_whole_text_is_spoken_into_one_file(
        self, router: stream.Router, backend: Router
    ):
        audiobook = await router.tts.recording(
            "Chapter one.", format="mp3_44100_128", voice="dc4e4a1f"
        )

        assert audiobook.format_ == "mp3_44100_128"
        assert base64.b64decode(audiobook.audio) == b"\xff\xfb\x90"
        assert backend.speeches[0]["options"] == {
            "format": "mp3_44100_128",
            "voice": "dc4e4a1f",
        }

    async def test_a_search_answers_out_of_what_is_true_now(
        self, router: stream.Router, backend: Router
    ):
        found = await router.search(
            "perioperative antibiotic guidance",
            results=5,
            include_domains=["nice.org.uk"],
        )

        assert found.answer == "within an hour of the incision"
        assert found.results[0].url == "https://nice.org.uk/1"
        assert backend.searches[0]["options"] == {
            "results": 5,
            "include_domains": ["nice.org.uk"],
        }
        assert backend.searches[0]["config_id"] == "healthcare"

    async def test_a_config_is_stored_under_its_name_and_edited_next_time(
        self, backend: Router
    ):
        stored = await stream.define_router(
            "healthcare",
            stt={"target": "en-recorded", "diarize": True},
            search={"depth": "standard"},
            url=backend.url,
            customer_id="acme",
        )
        assert stored.stt.target == "en-recorded"

        again = await stream.define_router(
            "healthcare",
            stt={"target": "multilingual-recorded"},
            url=backend.url,
            customer_id="acme",
        )

        assert again.id == stored.id, (
            "naming a config twice edits the one that is there"
        )
        assert len(backend.configs) == 1
        assert again.stt.target == "multilingual-recorded"

    async def test_a_config_option_that_does_not_exist_is_refused(
        self, backend: Router
    ):
        with pytest.raises(ValueError, match="diarise"):
            await stream.define_router(
                "healthcare",
                stt={"diarise": True},
                url=backend.url,
                customer_id="acme",
            )
