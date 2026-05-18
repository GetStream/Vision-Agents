"""Unit tests for `vision_agents.plugins.tencent.scene.resolve_room_scene`."""

import pytest

from vision_agents.plugins.tencent.bindings import (
    TRTC_SCENE_CALL,
    TRTC_SCENE_RECORD,
    TRTC_SCENE_VIDEOCALL,
)
from vision_agents.plugins.tencent.scene import resolve_room_scene


_VIDEOCALL_AVAILABLE = TRTC_SCENE_VIDEOCALL is not None
_CALL_AVAILABLE = TRTC_SCENE_CALL is not None


class TestResolveRoomScene:
    def test_auto_picks_videocall_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _VIDEOCALL_AVAILABLE:
            pytest.skip("liteav binding does not expose TRTC_SCENE_VIDEOCALL")
        monkeypatch.delenv("TENCENT_TRTC_SCENE", raising=False)
        scene, name = resolve_room_scene()
        assert scene == TRTC_SCENE_VIDEOCALL
        assert name == "videocall"

    def test_auto_falls_back_to_record_when_only_record_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if _VIDEOCALL_AVAILABLE or _CALL_AVAILABLE:
            pytest.skip("videocall/call scenes are available on this binding")
        monkeypatch.delenv("TENCENT_TRTC_SCENE", raising=False)
        scene, name = resolve_room_scene()
        assert scene == TRTC_SCENE_RECORD
        assert name == "record"

    def test_explicit_record_returns_record_constant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if TRTC_SCENE_RECORD is None:
            pytest.skip("liteav not loaded — record scene constant is unavailable")
        monkeypatch.setenv("TENCENT_TRTC_SCENE", "record")
        scene, name = resolve_room_scene()
        assert scene == TRTC_SCENE_RECORD
        assert name == "record"

    def test_unknown_scene_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENCENT_TRTC_SCENE", "broadcast")
        with pytest.raises(ValueError, match="TENCENT_TRTC_SCENE"):
            resolve_room_scene()

    def test_explicit_unsupported_scene_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if _VIDEOCALL_AVAILABLE:
            pytest.skip("videocall is available, can't test the 'unsupported' branch")
        monkeypatch.setenv("TENCENT_TRTC_SCENE", "videocall")
        with pytest.raises(ValueError, match="not supported by this liteav binding"):
            resolve_room_scene()
