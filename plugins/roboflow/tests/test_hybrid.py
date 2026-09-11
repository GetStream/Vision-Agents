import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest
from conftest import skip_blockbuster
from vision_agents.core import Agent, User
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.plugins import roboflow
from vision_agents.plugins import stream as acceleration
from vision_agents.plugins.roboflow.roboflow_streaming_processor import ManualSource


def _cat_video(assets_dir: Path, dest: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(assets_dir / "cat.jpg"),
            "-t",
            "30",
            "-r",
            "5",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


@skip_blockbuster
@pytest.mark.integration
@pytest.mark.timeout(180)
@pytest.mark.skipif(
    ManualSource is None, reason="inference-sdk[webrtc] is not installed"
)
@pytest.mark.skipif(
    not os.getenv("ROBOFLOW_API_KEY"), reason="ROBOFLOW_API_KEY not set"
)
@pytest.mark.skipif(not os.getenv("STREAM_API_KEY"), reason="STREAM_API_KEY not set")
class TestHybridVideoWorkerLive:
    async def test_get_video_state_round_trip(self, assets_dir, tmp_path):
        await acceleration.sync_agent("flower_spotter")
        processor = roboflow.RoboflowStreamingProcessor(model_id="rfdetr-nano", fps=1)
        agent = Agent(
            config="flower_spotter",
            processors=[processor],
            agent_user=User(name="hybrid-worker", id="hybrid-worker"),
        )
        agent.set_video_track_override_path(
            str(_cat_video(Path(assets_dir), tmp_path / "cat.mp4"))
        )

        replies: list[str] = []

        @agent.events.subscribe
        async def on_reply(event: LLMResponseFinalEvent):
            if event.text:
                replies.append(event.text)

        call = await agent.create_call("agent", f"hybrid-rfdetr-{int(time.time())}")
        async with agent.join(call, wait_for_end=False, participant_wait_timeout=0):
            deadline = time.monotonic() + 70
            snapshot: dict[str, object] = {}
            while time.monotonic() < deadline:
                snapshot = processor.state()
                objects = snapshot.get("objects")
                if isinstance(objects, list) and objects:
                    break
                await asyncio.sleep(0.5)
            objects = snapshot.get("objects")
            assert isinstance(objects, list) and objects, snapshot
            labels = {
                str(item["label"])
                for item in objects
                if isinstance(item, dict) and "label" in item
            }
            assert labels

            names = [schema["name"] for schema in agent.llm.get_available_functions()]
            assert "get_video_state" in names
            assert "get_video_frame" not in names

            replies.clear()
            await agent.responses.create(
                "Call get_video_state now and name every label it returns. "
                "Do not guess. Do not add objects that are not in the tool result."
            )
            reply_deadline = time.monotonic() + 45
            while time.monotonic() < reply_deadline:
                spoken = " ".join(replies).lower()
                if any(label.lower() in spoken for label in labels):
                    break
                await asyncio.sleep(0.2)

        spoken = " ".join(replies).lower()
        assert spoken, "the model did not reply"
        assert any(label.lower() in spoken for label in labels), (
            f"reply {spoken!r} did not use labels {labels}"
        )

    async def test_delegated_vision_round_trip(self, assets_dir, tmp_path):
        await acceleration.sync_agent("flower_spotter")
        processor = roboflow.RoboflowStreamingProcessor(model_id="rfdetr-nano", fps=1)
        agent = Agent(
            config="flower_spotter",
            processors=[processor],
            agent_user=User(name="hybrid-frame", id="hybrid-frame"),
        )
        agent.set_video_track_override_path(
            str(_cat_video(Path(assets_dir), tmp_path / "cat-frame.mp4"))
        )

        replies: list[str] = []

        @agent.events.subscribe
        async def on_reply(event: LLMResponseFinalEvent):
            if event.text:
                replies.append(event.text)

        call = await agent.create_call("agent", f"hybrid-frame-{int(time.time())}")
        async with agent.join(call, wait_for_end=False, participant_wait_timeout=0):
            deadline = time.monotonic() + 70
            while time.monotonic() < deadline:
                if processor.latest_frame() is not None:
                    break
                await asyncio.sleep(0.5)
            assert processor.latest_frame() is not None

            replies.clear()
            await agent.responses.create(
                "Delegate to the vision skill and describe the current camera image after its findings arrive. "
                "Do not guess from memory."
            )
            reply_deadline = time.monotonic() + 45
            while time.monotonic() < reply_deadline:
                if replies:
                    break
                await asyncio.sleep(0.2)

        spoken = " ".join(replies).lower()
        assert spoken, "the model did not reply after looking at the frame"
