import av
import pytest
from PIL import Image

from vision_agents.core.llm import ImageContent
from vision_agents.core.processors.observations import ObservationBuffer


@pytest.fixture
def frame() -> av.VideoFrame:
    return av.VideoFrame.from_image(Image.new("RGB", (4, 4), "red"))


class TestObservationBuffer:
    def test_selection_pins_older_frame_when_camera_changes(self, frame):
        buffer = ObservationBuffer(max_frames=2)
        buffer.append("camera", frame, captured_at_ms=1000)
        pinned = buffer.select("camera", 1000)
        buffer.append("camera", frame, captured_at_ms=1100)
        buffer.append("camera", frame, captured_at_ms=1200)
        assert pinned[0].frame_id == "camera:1"
        assert pinned[0].captured_at_ms == 1000
        assert [item.frame_id for item in buffer.select("camera", 1200, 2)] == [
            "camera:2",
            "camera:3",
        ]
        with pytest.raises(ValueError, match="no recent frame"):
            buffer.select("camera", 1000)

    def test_missing_history_is_not_replaced_with_latest(self, frame):
        buffer = ObservationBuffer()
        buffer.append("camera", frame, captured_at_ms=1000)
        with pytest.raises(ValueError, match="requested 2"):
            buffer.select("camera", 1000, 2)
        with pytest.raises(ValueError, match="no recent frame"):
            buffer.select("camera", 7000)

    def test_source_and_memory_bounds(self, frame):
        size = sum(plane.buffer_size for plane in frame.planes)
        buffer = ObservationBuffer(max_bytes=size)
        buffer.append("alice", frame, captured_at_ms=1000)
        buffer.append("bob", frame, captured_at_ms=1001)
        with pytest.raises(ValueError, match="no recent frame"):
            buffer.select("alice", 1001)
        assert buffer.select("bob", 1001)[0].source == "bob"
        buffer.clear()
        with pytest.raises(ValueError, match="no recent frame"):
            buffer.select("bob", 1001)

    def test_predictions_keep_separate_time_and_alignment(self, frame):
        buffer = ObservationBuffer()
        buffer.append(
            "detector",
            frame,
            captured_at_ms=1000,
            processor_result='{"roses":2}',
            processed_at_ms=950,
        )
        selected = buffer.select("detector", 1000)[0]
        assert selected.processed_at_ms == 950
        assert not selected.aligned
        assert selected.captured_at_ms == 1000


class TestImageContent:
    def test_bytes_and_url_use_same_wire_shape(self):
        image = ImageContent(url="https://example.com/image.png", detail="low")
        assert image.as_content_part() == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "low"},
        }
        assert (
            ImageContent(data=b"image")
            .as_image_dict()["url"]
            .startswith("data:image/jpeg;base64,")
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"data": b"x", "url": "https://example.com/x"},
            {"url": "file:///etc/passwd"},
            {"data": b"x", "detail": "invalid"},
        ],
    )
    def test_invalid_image_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ImageContent(**kwargs)
