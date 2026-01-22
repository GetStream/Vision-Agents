"""Type adapters for converting between GetStream types and vision-agents core types.

This module provides adapter functions following the Adapter pattern to translate
between GetStream-specific types (from the getstream SDK) and the framework-native
core types defined in vision_agents.core.types.

The adapter pattern allows the framework to remain decoupled from transport-specific
implementations while still supporting multiple edge network providers.

Example:
    Converting GetStream PCM data to core PCM data::

        from getstream.video.rtc.track_util import PcmData as StreamPcmData
        from vision_agents.plugins.getstream.adapters import adapt_pcm_data

        stream_pcm = StreamPcmData(data=b"...", sample_rate=48000, channels=2, bit_depth=16)
        core_pcm = adapt_pcm_data(stream_pcm)

    Converting GetStream track types to core track types::

        from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType as StreamTrackType
        from vision_agents.plugins.getstream.adapters import adapt_track_type

        # Handle different track types
        track_type = adapt_track_type(StreamTrackType.TRACK_TYPE_AUDIO)
        # Returns: TrackType.AUDIO
"""

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    TrackType as StreamTrackType,
)
from getstream.video.rtc.track_util import PcmData as StreamPcmData
from vision_agents.core.types import PcmData, TrackType


def adapt_pcm_data(stream_pcm: StreamPcmData) -> PcmData:
    """Convert GetStream PcmData to vision-agents core PcmData.

    This adapter function translates PCM audio data from the GetStream SDK format
    to the framework-native PcmData type. The conversion is straightforward as both
    types represent the same PCM audio data structure.

    Args:
        stream_pcm: GetStream PcmData instance containing raw audio data with
            sample rate, channels, and bit depth information.

    Returns:
        Core PcmData instance with the same audio data and metadata. The timestamp
        attribute is preserved if present in the GetStream PCM data, otherwise None.

    Example:
        >>> from getstream.video.rtc.track_util import PcmData as StreamPcmData
        >>> stream_pcm = StreamPcmData(
        ...     data=b"\\x00\\x01\\x02\\x03",
        ...     sample_rate=48000,
        ...     channels=2,
        ...     bit_depth=16
        ... )
        >>> core_pcm = adapt_pcm_data(stream_pcm)
        >>> core_pcm.sample_rate
        48000
    """
    return PcmData(
        data=stream_pcm.data,
        sample_rate=stream_pcm.sample_rate,
        channels=stream_pcm.channels,
        bit_depth=stream_pcm.bit_depth,
        timestamp=getattr(stream_pcm, "timestamp", None),
    )


def adapt_track_type(stream_track_type: int) -> TrackType:
    """Convert GetStream TrackType integer to vision-agents core TrackType enum.

    This adapter function translates track type identifiers from the GetStream SDK
    (integer enum values from protobuf) to the framework-native TrackType enum.
    It handles all standard GetStream track types including audio, video, and
    screen sharing variants.

    The GetStream SDK uses integer constants from protobuf definitions:
    - TRACK_TYPE_AUDIO (0): Standard audio track
    - TRACK_TYPE_VIDEO (1): Standard video track
    - TRACK_TYPE_SCREEN_SHARE (2): Screen share video track
    - TRACK_TYPE_SCREEN_SHARE_AUDIO (3): Screen share audio track

    Args:
        stream_track_type: Integer value from GetStream's StreamTrackType enum,
            typically from the SFU (Selective Forwarding Unit) events.

    Returns:
        Core TrackType enum value corresponding to the GetStream track type.
        Unknown track types default to TrackType.VIDEO for safety.

    Example:
        >>> from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType as StreamTrackType
        >>> adapt_track_type(StreamTrackType.TRACK_TYPE_AUDIO)
        <TrackType.AUDIO: 'audio'>
        >>> adapt_track_type(StreamTrackType.TRACK_TYPE_SCREEN_SHARE)
        <TrackType.SCREENSHARE: 'screenshare'>

    Note:
        Both TRACK_TYPE_AUDIO and TRACK_TYPE_SCREEN_SHARE_AUDIO map to TrackType.AUDIO
        since the core type system doesn't distinguish between regular and screen share audio.
    """
    if stream_track_type in (
        StreamTrackType.TRACK_TYPE_AUDIO,
        StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
    ):
        return TrackType.AUDIO
    elif stream_track_type == StreamTrackType.TRACK_TYPE_SCREEN_SHARE:
        return TrackType.SCREENSHARE
    elif stream_track_type == StreamTrackType.TRACK_TYPE_VIDEO:
        return TrackType.VIDEO
    else:
        # Default to video for unknown types
        return TrackType.VIDEO
