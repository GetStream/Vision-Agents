"""Configuration management for LocalEdge RTC implementation."""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio configuration for LocalEdge.

    Attributes:
        input_sample_rate: Audio input sampling rate in Hz.
        output_sample_rate: Default output sample rate in Hz when not negotiated.
        input_channels: Number of audio input channels.
        output_channels: Number of audio output channels.
        bit_depth: Audio bit depth in bits.
        input_buffer_duration: Input buffer duration in seconds.
        output_buffer_size_ms: Output buffer size in milliseconds.
        capture_chunk_duration: Audio capture chunk duration in seconds.
        playback_chunk_duration: Audio playback chunk duration in seconds.
        loop_sleep_interval: Sleep interval in audio loops to avoid busy-waiting.
        flush_poll_interval: Poll interval when flushing audio in seconds.
        error_retry_delay: Delay before retrying on error in seconds.
        thread_join_timeout: Thread join timeout in seconds.
        max_int16_value: Maximum value for int16 audio conversion.
        eos_wait_time: Wait time for GStreamer end-of-stream in seconds.
    """

    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    input_channels: int = 1
    output_channels: int = 1
    bit_depth: int = 16

    input_buffer_duration: float = 2.0
    output_buffer_size_ms: int = 10000
    output_prebuffer_ms: int = 200

    capture_chunk_duration: float = 0.1
    playback_chunk_duration: float = 0.1
    loop_sleep_interval: float = 0.01
    flush_poll_interval: float = 0.05
    error_retry_delay: float = 0.1
    thread_join_timeout: float = 2.0

    max_int16_value: int = 32767

    eos_wait_time: float = 0.1


@dataclass
class VideoConfig:
    """Video configuration for LocalEdge.

    Attributes:
        default_width: Default video frame width in pixels.
        default_height: Default video frame height in pixels.
        default_fps: Default frames per second.
        format: Video format for GStreamer.
        max_buffers: Maximum buffers for GStreamer appsink.
    """

    default_width: int = 640
    default_height: int = 480
    default_fps: int = 30
    format: str = "BGR"
    max_buffers: int = 1


@dataclass
class GStreamerConfig:
    """GStreamer pipeline configuration for LocalEdge.

    Attributes:
        appsink_name: Name for GStreamer appsink element.
        appsrc_name: Name for GStreamer appsrc element.
        audio_layout: Audio layout for GStreamer caps.
    """

    appsink_name: str = "sink"
    appsrc_name: str = "src"
    audio_layout: str = "interleaved"


@dataclass
class LocalEdgeConfig:
    """Complete configuration for LocalEdge RTC implementation.

    Example:
        >>> config = LocalEdgeConfig(
        ...     audio=AudioConfig(input_sample_rate=48000),
        ...     video=VideoConfig(default_width=1920)
        ... )

    Attributes:
        audio: Audio configuration settings.
        video: Video configuration settings.
        gstreamer: GStreamer pipeline configuration.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    gstreamer: GStreamerConfig = field(default_factory=GStreamerConfig)
