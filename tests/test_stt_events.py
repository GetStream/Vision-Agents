from getstream.video.rtc.track_util import PcmData

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import STT
from vision_agents.core.stt.events import (
    STTConnectedEvent,
    STTDisconnectedEvent,
    STTErrorEvent,
)


class DummySTT(STT):
    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ) -> None:
        return None


async def test_stt_plugin_events_are_registered():
    stt = DummySTT(provider_name="dummy")
    seen: list[object] = []

    @stt.events.subscribe
    async def on_connected(event: STTConnectedEvent) -> None:
        seen.append(event)

    @stt.events.subscribe
    async def on_disconnected(event: STTDisconnectedEvent) -> None:
        seen.append(event)

    @stt.events.subscribe
    async def on_error(event: STTErrorEvent) -> None:
        seen.append(event)

    stt._on_connected()
    stt._on_disconnected(reason="done", clean=True)
    stt._emit_error_event(RuntimeError("boom"), context="test")
    await stt.events.wait()

    assert [event.plugin_name for event in seen] == ["dummy", "dummy", "dummy"]
    assert isinstance(seen[0], STTConnectedEvent)
    assert isinstance(seen[1], STTDisconnectedEvent)
    assert isinstance(seen[2], STTErrorEvent)
