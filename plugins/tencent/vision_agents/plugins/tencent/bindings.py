"""Conditional `liteav` imports for the Tencent plugin.

`liteav` ships only manylinux wheels on PyPI, so on macOS/Windows the
import will fail. This module centralises the conditional import noise:
all other modules in the plugin re-import the names they need from
here. ``require_liteav()`` raises a friendly RuntimeError at call time
if liteav couldn't be loaded.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIO_CODEC_TYPE_PCM",
    "AUDIO_OBTAIN_METHOD_CALLBACK",
    "AudioEncodeParams",
    "AudioFrame",
    "CreateTRTCCloud",
    "DestroyTRTCCloud",
    "EnterRoomParams",
    "LITEAV_IMPORT_ERROR",
    "PixelFrame",
    "STREAM_TYPE_VIDEO_HIGH",
    "TRTCCloudDelegate",
    "TRTC_ROLE_ANCHOR",
    "TRTC_SCENE_CALL",
    "TRTC_SCENE_RECORD",
    "TRTC_SCENE_VIDEOCALL",
    "TrtcString",
    "VIDEO_PIXEL_FORMAT_YUV420p",
    "VIDEO_ROTATION_0",
    "VideoEncodeParams",
    "require_liteav",
]

LITEAV_IMPORT_ERROR: Optional[ImportError] = None

try:
    # liteav's typed stubs don't expose these top-level module constants
    # and convenience constructors even though they exist at runtime, so
    # we silence the attr-defined / call-arg complaints in this block.
    from liteav import (  # type: ignore[attr-defined]
        AUDIO_CODEC_TYPE_PCM,
        AUDIO_OBTAIN_METHOD_CALLBACK,
        STREAM_TYPE_VIDEO_HIGH,
        VIDEO_PIXEL_FORMAT_YUV420p,
        VIDEO_ROTATION_0,
        AudioEncodeParams,
        AudioFrame,
        CreateTRTCCloud,
        DestroyTRTCCloud,
        EnterRoomParams,
        PixelFrame,
        TrtcString,
        TRTCCloudDelegate,
        TRTC_ROLE_ANCHOR,
        TRTC_SCENE_RECORD,
        VideoEncodeParams,
    )
except ImportError as e:
    LITEAV_IMPORT_ERROR = e
    # All names need module-level fallbacks so downstream `from bindings
    # import ...` succeeds on non-linux. TencentEdge.__init__ calls
    # require_liteav() which raises before any of these get touched.
    AUDIO_CODEC_TYPE_PCM = None  # type: ignore[assignment]
    AUDIO_OBTAIN_METHOD_CALLBACK = None  # type: ignore[assignment]
    STREAM_TYPE_VIDEO_HIGH = None  # type: ignore[assignment]
    VIDEO_PIXEL_FORMAT_YUV420p = None  # type: ignore[assignment]
    VIDEO_ROTATION_0 = None  # type: ignore[assignment]
    AudioEncodeParams = None  # type: ignore[misc, assignment]
    AudioFrame = None  # type: ignore[misc, assignment]
    CreateTRTCCloud = None
    DestroyTRTCCloud = None
    EnterRoomParams = None  # type: ignore[misc, assignment]
    PixelFrame = None  # type: ignore[misc, assignment]
    TrtcString = None  # type: ignore[misc, assignment]
    TRTCCloudDelegate = None  # type: ignore[misc, assignment]
    TRTC_ROLE_ANCHOR = None  # type: ignore[assignment]
    TRTC_SCENE_RECORD = None  # type: ignore[assignment]
    VideoEncodeParams = None  # type: ignore[misc, assignment]

try:
    from liteav import TRTC_SCENE_CALL, TRTC_SCENE_VIDEOCALL  # type: ignore[attr-defined]
except ImportError:
    TRTC_SCENE_VIDEOCALL = None  # type: ignore[assignment]
    TRTC_SCENE_CALL = None  # type: ignore[assignment]


# liteav's C++ side writes verbose `[I][...]` / `[E][...]` lines straight
# to stdout. Calling DisableConsoleLog + SetLogLevel(kNone) silences the
# periodic noise (thread_watchdog, audio_track_health_monitor, etc.) but
# the SDK still writes a chunk of connection-setup logs from
# CreateTRTCCloud → EnterRoom that we can't suppress through its public
# API. Set TENCENT_LITEAV_DEBUG=1 to bring those periodic logs back for
# diagnostics. Note on the level enum (non-standard): 0=kAll, 1=kInfo,
# 2=kWarning, 3=kError, 4=kFatal, 5=kNone (>=6 is UN_DEF, no-op).
_TRUTHY = {"1", "true", "yes", "on"}
_LITEAV_LOG_LEVEL_NONE = 5

if (
    LITEAV_IMPORT_ERROR is None
    and os.environ.get("TENCENT_LITEAV_DEBUG", "").strip().lower() not in _TRUTHY
):
    try:
        import liteav as _liteav

        _liteav.DisableConsoleLog()  # type: ignore[attr-defined]
        _liteav.SetLogLevel(_LITEAV_LOG_LEVEL_NONE)  # type: ignore[attr-defined]
    except AttributeError as e:
        # Older liteav builds may not expose these helpers.
        logger.debug("Could not silence liteav console log: %s", e)


def require_liteav() -> None:
    """Raise a friendly error if liteav couldn't be imported on this platform."""
    if LITEAV_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tencent TRTC edge requires the `liteav` package, which ships only "
            "manylinux wheels and cannot be imported on this platform. Install "
            "on Linux (x86_64 or aarch64) with `pip install liteav` or run the "
            "agent inside a Linux container (see plugins/tencent/README.md)."
        ) from LITEAV_IMPORT_ERROR
