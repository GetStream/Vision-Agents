import os

import av
import numpy as np
import pytest
from PIL import Image
from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import CUSTOMER_ENV, URL_ENV
from vision_agents.plugins.stream._generated.models import DetectedObject, DetectionBox
from vision_agents.plugins.stream._generated.types import UNSET, Unset


def _found(
    label: str,
    box: tuple[float, float, float, float],
    embedding: list[float] | Unset = UNSET,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=0.9,
        box=DetectionBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        embedding=embedding,
    )


@pytest.fixture
def grey() -> np.ndarray:
    return np.full((360, 640, 3), 120, dtype=np.uint8)


@pytest.fixture
def cat(assets_dir: str) -> av.VideoFrame:
    return av.VideoFrame.from_image(
        Image.open(os.path.join(assets_dir, "cat.jpg")).convert("RGB")
    )


class TestTracks:
    def test_a_box_seen_again_is_the_same_track(self):
        tracks = stream.Tracks()
        appeared, _ = tracks.update([_found("package", (0.1, 0.1, 0.3, 0.3))], 1000)
        again, gone = tracks.update([_found("package", (0.12, 0.1, 0.32, 0.3))], 2000)

        assert [track.id for track in appeared] == [1]
        assert again == [] and gone == []
        assert tracks.live[0].box == (0.12, 0.1, 0.32, 0.3)
        assert tracks.live[0].last_seen_ms == 2000

    def test_a_box_of_another_label_is_a_new_track(self):
        tracks = stream.Tracks()
        tracks.update([_found("package", (0.1, 0.1, 0.3, 0.3))], 1000)
        appeared, gone = tracks.update([_found("person", (0.1, 0.1, 0.3, 0.3))], 2000)

        assert [track.label for track in appeared] == ["person"]
        assert gone == []

    def test_a_missed_detection_is_forgiven_until_it_is_not(self):
        tracks = stream.Tracks(misses=1)
        tracks.update([_found("package", (0.1, 0.1, 0.3, 0.3))], 1000)

        _, gone_once = tracks.update([], 2000)
        _, gone_twice = tracks.update([], 3000)

        assert gone_once == []
        assert [track.label for track in gone_twice] == ["package"]
        assert tracks.live == []

    def test_two_boxes_are_each_paired_with_the_track_they_overlap_most(self):
        tracks = stream.Tracks()
        tracks.update(
            [
                _found("person", (0.0, 0.0, 0.2, 0.5)),
                _found("person", (0.5, 0.0, 0.7, 0.5)),
            ],
            1000,
        )
        appeared, gone = tracks.update(
            [
                _found("person", (0.52, 0.0, 0.72, 0.5)),
                _found("person", (0.02, 0.0, 0.22, 0.5)),
            ],
            2000,
        )

        assert appeared == [] and gone == []
        assert [track.box[0] for track in tracks.live] == [0.02, 0.52]

    def test_the_latest_embedding_is_kept(self):
        tracks = stream.Tracks()
        tracks.update(
            [_found("face", (0.1, 0.1, 0.3, 0.3), embedding=[1.0, 0.0])], 1000
        )
        tracks.update(
            [_found("face", (0.1, 0.1, 0.3, 0.3), embedding=[0.0, 1.0])], 2000
        )

        assert tracks.live[0].embedding == [0.0, 1.0]

    def test_the_model_reads_a_track_without_its_embedding(self):
        track = stream.Track(
            id=3,
            label="face",
            box=(0.1, 0.2, 0.3, 0.4),
            confidence=0.876,
            first_seen_ms=1000,
            last_seen_ms=3500,
            embedding=[0.5],
            identity="Thierry",
        )

        assert track.as_dict() == {
            "id": 3,
            "label": "face",
            "confidence": 0.88,
            "box": [0.1, 0.2, 0.3, 0.4],
            "seen_for_s": 2.5,
            "identity": "Thierry",
        }


class TestChangeGate:
    def test_the_first_frame_is_let_through(self, grey: np.ndarray):
        assert stream.ChangeGate().changed(grey, 0.0)

    def test_the_same_frame_again_is_not(self, grey: np.ndarray):
        gate = stream.ChangeGate()
        gate.changed(grey, 0.0)

        assert not gate.changed(grey, 1.0)

    def test_noise_everywhere_is_not_a_change(self, grey: np.ndarray):
        gate = stream.ChangeGate()
        gate.changed(grey, 0.0)
        noisy = grey + np.random.default_rng(1).integers(
            0, 8, grey.shape, dtype=np.uint8
        )

        assert not gate.changed(noisy, 1.0)

    def test_something_put_down_is_a_change(self, grey: np.ndarray):
        gate = stream.ChangeGate()
        gate.changed(grey, 0.0)
        placed = grey.copy()
        placed[200:300, 300:420] = 20

        assert gate.changed(placed, 1.0)

    def test_an_unchanged_scene_is_looked_at_again_after_the_refresh_interval(
        self, grey: np.ndarray
    ):
        gate = stream.ChangeGate(refresh_interval=5.0)
        gate.changed(grey, 0.0)

        assert gate.changed(grey, 5.0)


class TestDetectionProcessor:
    def test_a_processor_needs_a_name(self):
        with pytest.raises(ValueError):
            stream.DetectionProcessor(name="")

    @pytest.mark.integration
    async def test_the_cat_is_found_and_followed(self, cat: av.VideoFrame):
        if not os.getenv(URL_ENV) or not os.getenv(CUSTOMER_ENV):
            pytest.fail(f"set {URL_ENV} and {CUSTOMER_ENV} to reach a router")
        processor = stream.DetectionProcessor(target="detection-fast", classes=["cat"])
        try:
            first = await processor.detect(cat, captured_at_ms=1000)
            second = await processor.detect(cat, captured_at_ms=2000)
        finally:
            await processor.close()

        assert [track.label for track in first] == ["cat"]
        assert [track.id for track in second] == [first[0].id]
        assert processor.state()["counts"] == {"cat": 1}
        box = first[0].box
        assert box[0] < 0.5 < box[2] and box[1] < 0.5 < box[3]

    @pytest.mark.integration
    async def test_a_box_smaller_than_min_area_is_not_followed(
        self, cat: av.VideoFrame
    ):
        if not os.getenv(URL_ENV) or not os.getenv(CUSTOMER_ENV):
            pytest.fail(f"set {URL_ENV} and {CUSTOMER_ENV} to reach a router")
        processor = stream.DetectionProcessor(
            target="detection-fast", classes=["cat"], min_area=0.99
        )
        try:
            found = await processor.detect(cat)
        finally:
            await processor.close()

        assert found == []
        assert processor.state()["counts"] == {}
