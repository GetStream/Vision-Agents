"""TRTC scene selection driven by the `TENCENT_TRTC_SCENE` env var."""

import os

from vision_agents.plugins.tencent.bindings import (
    TRTC_SCENE_CALL,
    TRTC_SCENE_RECORD,
    TRTC_SCENE_VIDEOCALL,
)


def resolve_room_scene() -> tuple[int, str]:
    """Pick a TRTC scene constant from the env var.

    Reads ``TENCENT_TRTC_SCENE`` (default ``auto``). Returns the scene's
    int value plus its human-readable name. ``auto`` falls through
    videocall → call → record, picking the first one this liteav
    binding actually exposes.
    """
    configured_scene = os.getenv("TENCENT_TRTC_SCENE", "auto").strip().lower()
    scene_map = {
        "record": TRTC_SCENE_RECORD,
        "videocall": TRTC_SCENE_VIDEOCALL,
        "call": TRTC_SCENE_CALL,
    }
    if configured_scene == "auto":
        for candidate in ("videocall", "call"):
            scene = scene_map[candidate]
            if scene is not None:
                return scene, candidate
        return TRTC_SCENE_RECORD, "record"

    if configured_scene in scene_map:
        scene = scene_map[configured_scene]
        if scene is None:
            raise ValueError(
                f"TENCENT_TRTC_SCENE={configured_scene} is not supported by this liteav binding."
            )
        return scene, configured_scene

    raise ValueError(
        "TENCENT_TRTC_SCENE must be one of: auto, videocall, call, record."
    )
