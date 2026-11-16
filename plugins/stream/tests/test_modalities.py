import asyncio
import json
import os
import urllib.request

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS, ImageContent
from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import CUSTOMER_ENV, URL_ENV
from vision_agents.testing import answer

STT_TARGET = os.getenv("STREAM_ACCELERATION_STT", "en-low-latency")
TTS_TARGET = os.getenv("STREAM_ACCELERATION_TTS", "en-low-latency")
LLM_TARGET = os.getenv("STREAM_ACCELERATION_LLM", "llm-fast")

# A royalty-free photo from Lorem Picsum (Unsplash licence). The fixed id keeps it the same
# picture every run: a black dog on wooden planks.
IMAGE_URL = "https://picsum.photos/id/237/640/480.jpg"

# The message-building tests send nothing, but a client with no URL now goes to the hosted
# router and wants a credential. A local router takes a customer id instead.
LOCAL_ROUTER = "http://localhost:8080"


def _require_router() -> None:
    if not os.getenv(URL_ENV) or not os.getenv(CUSTOMER_ENV):
        pytest.fail(
            f"These tests need a running acceleration router. Set {URL_ENV} and "
            f"{CUSTOMER_ENV} in the environment or in a .env file before running tests "
            "marked with @pytest.mark.integration.",
            pytrace=False,
        )


@pytest.mark.integration
class TestModalityStreams:
    @pytest.fixture(autouse=True)
    def router(self) -> None:
        _require_router()

    async def test_speech_comes_back_as_audio_that_can_be_played(self):
        tts = stream.TTS(TTS_TARGET)
        await tts.start()
        try:
            chunks = [
                chunk
                async for chunk in tts.send_iter("Hello from the acceleration router.")
                if chunk.data is not None
            ]
        finally:
            await tts.close()

        assert chunks
        first = chunks[0].data
        assert isinstance(first, PcmData)
        assert first.sample_rate > 0
        assert len(first.samples) > 0

    async def test_a_completion_comes_back_as_text(self):
        llm = stream.LLM(LLM_TARGET)
        await llm.start()
        try:
            answers = [answer async for answer in llm.simple_response("Say hello.")]
        finally:
            await llm.close()

        assert answers
        assert answers[-1].text

    async def test_an_attached_image_is_described(self):
        image = await asyncio.to_thread(
            lambda: urllib.request.urlopen(IMAGE_URL, timeout=30).read()
        )
        llm = stream.LLM(os.getenv("STREAM_ACCELERATION_VLM", "vlm"))
        await llm.start()
        try:
            answers = [
                answer
                async for answer in llm.responses.create(
                    "Describe this image in one short sentence.",
                    images=[ImageContent(data=image, mime="image/jpeg")],
                )
            ]
        finally:
            await llm.close()

        assert answers
        described = answers[-1].text.lower()
        assert described
        assert any(
            word in described
            for word in (
                "dog",
                "puppy",
                "canine",
                "labrador",
                "animal",
                "pet",
                "black",
            )
        ), described

    async def test_a_name_routes_to_the_modality_that_serves_it(self):
        # resolve asks the router which modality serves the name and blocks on the answer,
        # since it is what builds an agent rather than something a call waits on.
        routed = await asyncio.to_thread(stream.Router().resolve, TTS_TARGET)

        assert isinstance(routed, stream.TTS)


class TestLLM:
    @pytest.fixture
    def conversation(self) -> InMemoryConversation:
        return InMemoryConversation(instructions="", messages=[])

    @pytest.fixture
    def model(self, conversation: InMemoryConversation) -> stream.LLM:
        model = stream.LLM(target="vlm", url=LOCAL_ROUTER, customer_id="test")
        model.set_conversation(conversation)
        return model

    @pytest.mark.parametrize("ordered", [False, True])
    async def test_followup_keeps_images_on_their_original_turn(
        self, model: stream.LLM, conversation: InMemoryConversation, ordered: bool
    ) -> None:
        image = ImageContent(url="https://example.com/first.png")
        await conversation.send_message("user", "user", "Describe this")
        first = model._messages(
            "Describe this",
            images=None if ordered else [image],
            ordered=["Describe this", image, "Look at the cap"] if ordered else None,
        )
        await conversation.send_message("assistant", "agent", "A bottle")
        await conversation.send_message("user", "user", "What color is the cap?")
        followup = model._messages("What color is the cap?")
        assert followup[0] == first[0]
        assert followup[-1]["content"] == "What color is the cap?"

    async def test_repeated_text_keeps_distinct_attachments(
        self, model: stream.LLM, conversation: InMemoryConversation
    ) -> None:
        for name in ("first", "second"):
            await conversation.send_message("user", "user", "Describe this")
            model._messages(
                "Describe this", [ImageContent(url=f"https://example.com/{name}.png")]
            )
            await conversation.send_message("assistant", "agent", name)
        await conversation.send_message("user", "user", "Compare them")
        messages = model._messages("Compare them")
        assert messages[0]["content"][1]["image_url"]["url"].endswith("first.png")
        assert messages[2]["content"][1]["image_url"]["url"].endswith("second.png")

    async def test_removed_or_replaced_history_does_not_replay_images(
        self, model: stream.LLM, conversation: InMemoryConversation
    ) -> None:
        message = await conversation.send_message("user", "user", "Describe this")
        model._messages(
            "Describe this", [ImageContent(url="https://example.com/image.png")]
        )
        conversation.messages.clear()
        assert model._messages("new question") == []
        conversation.messages.append(message)
        assert model._messages("Describe this")[0]["content"] == "Describe this"
        model._messages(
            "Describe this", [ImageContent(url="https://example.com/image.png")]
        )
        model.set_conversation(
            InMemoryConversation(instructions="", messages=[message])
        )
        assert model._messages("Describe this")[0]["content"] == "Describe this"

    @pytest.mark.parametrize("image", [False, True])
    async def test_the_router_is_told_who_spoke_once_two_people_have(
        self, model: stream.LLM, conversation: InMemoryConversation, image: bool
    ) -> None:
        ada = Participant(original=None, user_id="ada", id="ada-1", name="Ada")
        bob = Participant(original=None, user_id="bob", id="bob-1", name="Bob")
        await conversation.upsert_message(
            role="user", user_id="ada", content="Hi", participant=ada
        )
        await conversation.upsert_message(
            role="user", user_id="bob", content="Hello", participant=bob
        )

        messages = model._messages(
            "Bob: Hello",
            images=[ImageContent(url="https://example.com/a.png")] if image else None,
        )

        assert messages[0]["content"] == "Hi"
        if image:
            assert messages[1]["content"][0] == {"type": "text", "text": "Bob: Hello"}
        else:
            assert messages[1]["content"] == "Bob: Hello"

    async def test_a_turn_already_written_down_is_not_written_again_when_answered(
        self, conversation: InMemoryConversation
    ) -> None:
        responded: list[dict[str, object]] = []

        async def llm_stream(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                frame = json.loads(message.data)
                if frame["type"] == "respond":
                    responded.append(frame)
                    await socket.send_json({"type": "complete", "text": "ok"})
            return socket

        app = web.Application()
        app.router.add_get("/v1/llm/stream", llm_stream)
        ada = Participant(original=None, user_id="ada", id="ada-1", name="Ada")
        bob = Participant(original=None, user_id="bob", id="bob-1", name="Bob")
        await conversation.upsert_message(
            role="user", user_id="ada", content="Hi", participant=ada
        )
        await conversation.upsert_message(
            role="user", user_id="bob", content="Hello", participant=bob
        )
        async with TestServer(app) as server:
            model = stream.LLM(
                target="vlm",
                url=str(server.make_url("")).rstrip("/"),
                customer_id="test",
            )
            model.set_conversation(conversation)
            async with model:
                async for _ in model.simple_response(
                    "Bob: Hello",
                    participant=bob,
                    images=[ImageContent(url="https://example.com/a.png")],
                ):
                    pass

        assert len(conversation.messages) == 2
        assert responded[0]["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bob: Hello"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                ],
            },
        ]

    @pytest.mark.parametrize(
        ("instructions", "told"),
        [
            ("be brief", "be brief\n\n" + SEVERAL_SPEAKERS),
            # Empty asks the router for the config's instructions, so it stays empty.
            ("", ""),
        ],
    )
    async def test_the_router_is_told_several_people_are_talking(
        self, conversation: InMemoryConversation, instructions: str, told: str
    ) -> None:
        responded: list[dict[str, object]] = []

        async def llm_stream(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                frame = json.loads(message.data)
                if frame["type"] == "respond":
                    responded.append(frame)
                    await socket.send_json({"type": "complete", "text": "ok"})
            return socket

        app = web.Application()
        app.router.add_get("/v1/llm/stream", llm_stream)
        ada = Participant(original=None, user_id="ada", id="ada-1", name="Ada")
        bob = Participant(original=None, user_id="bob", id="bob-1", name="Bob")
        async with TestServer(app) as server:
            model = stream.LLM(
                target="llm-fast",
                url=str(server.make_url("")).rstrip("/"),
                customer_id="test",
            )
            model.set_conversation(conversation)
            model.set_instructions(instructions)
            async with model:
                for participant, text in ((ada, "Hi"), (ada, "Hello?"), (bob, "Hey")):
                    await answer(model, conversation, participant, text)

        # One person: what the base sha (9653cb696) sent.
        assert [frame["instructions"] for frame in responded] == [
            instructions,
            instructions,
            told,
        ]

    def test_ordered_content_preserves_interleaving(self):
        model = stream.LLM(target="vlm", url=LOCAL_ROUTER, customer_id="test")
        first = ImageContent(url="https://example.com/first.png")
        second = ImageContent(data=b"second")
        messages = model._messages(
            "compare", ordered=["first", first, "second", second]
        )
        parts = messages[0]["content"]
        assert [part["type"] for part in parts] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert parts[1]["image_url"]["url"] == first.url
        assert parts[3]["image_url"]["url"] == second.data_uri()
