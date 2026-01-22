"""Configuration management for LocalEdge RTC implementation.

This module provides configuration classes and environment variable support for
customizing LocalEdge behavior without code changes.
"""

import os
from dataclasses import dataclass, field


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable with fallback to default."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable with fallback to default."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@dataclass
class AudioConfig:
    """Audio configuration for LocalEdge.

    All values can be overridden via environment variables with the VA_ prefix.

    Attributes:
        input_sample_rate: Audio input sampling rate in Hz (env: VA_AUDIO_INPUT_SAMPLE_RATE)
        output_sample_rate: Default output sample rate in Hz when not negotiated (env: VA_AUDIO_OUTPUT_SAMPLE_RATE)
        input_channels: Number of audio input channels (env: VA_AUDIO_INPUT_CHANNELS)
        output_channels: Number of audio output channels (env: VA_AUDIO_OUTPUT_CHANNELS)
        bit_depth: Audio bit depth in bits (env: VA_AUDIO_BIT_DEPTH)
        input_buffer_duration: Input buffer duration in seconds (env: VA_AUDIO_INPUT_BUFFER_DURATION)
        output_buffer_size_ms: Output buffer size in milliseconds (env: VA_AUDIO_OUTPUT_BUFFER_SIZE_MS)
        capture_chunk_duration: Audio capture chunk duration in seconds (env: VA_AUDIO_CAPTURE_CHUNK_DURATION)
        playback_chunk_duration: Audio playback chunk duration in seconds (env: VA_AUDIO_PLAYBACK_CHUNK_DURATION)
        loop_sleep_interval: Sleep interval in audio loops to avoid busy-waiting (env: VA_AUDIO_LOOP_SLEEP_INTERVAL)
        flush_poll_interval: Poll interval when flushing audio in seconds (env: VA_AUDIO_FLUSH_POLL_INTERVAL)
        error_retry_delay: Delay before retrying on error in seconds (env: VA_AUDIO_ERROR_RETRY_DELAY)
        thread_join_timeout: Thread join timeout in seconds (env: VA_AUDIO_THREAD_JOIN_TIMEOUT)
        max_int16_value: Maximum value for int16 audio conversion (env: VA_AUDIO_MAX_INT16_VALUE)
        eos_wait_time: Wait time for GStreamer end-of-stream in seconds (env: VA_AUDIO_EOS_WAIT_TIME)
    """

    # Audio format configuration
    input_sample_rate: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_INPUT_SAMPLE_RATE", 16000)
    )
    output_sample_rate: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_OUTPUT_SAMPLE_RATE", 24000)
    )
    input_channels: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_INPUT_CHANNELS", 1)
    )
    output_channels: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_OUTPUT_CHANNELS", 1)
    )
    bit_depth: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_BIT_DEPTH", 16)
    )

    # Buffer configuration
    input_buffer_duration: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_INPUT_BUFFER_DURATION", 2.0)
    )
    output_buffer_size_ms: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_OUTPUT_BUFFER_SIZE_MS", 10000)
    )

    # Timing configuration
    capture_chunk_duration: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_CAPTURE_CHUNK_DURATION", 0.1)
    )
    playback_chunk_duration: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_PLAYBACK_CHUNK_DURATION", 0.05)
    )
    loop_sleep_interval: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_LOOP_SLEEP_INTERVAL", 0.01)
    )
    flush_poll_interval: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_FLUSH_POLL_INTERVAL", 0.05)
    )
    error_retry_delay: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_ERROR_RETRY_DELAY", 0.1)
    )
    thread_join_timeout: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_THREAD_JOIN_TIMEOUT", 2.0)
    )

    # Audio conversion constants
    max_int16_value: int = field(
        default_factory=lambda: _get_env_int("VA_AUDIO_MAX_INT16_VALUE", 32767)
    )

    # GStreamer timing
    eos_wait_time: float = field(
        default_factory=lambda: _get_env_float("VA_AUDIO_EOS_WAIT_TIME", 0.1)
    )


@dataclass
class VideoConfig:
    """Video configuration for LocalEdge.

    All values can be overridden via environment variables with the VA_ prefix.

    Attributes:
        default_width: Default video frame width in pixels (env: VA_VIDEO_DEFAULT_WIDTH)
        default_height: Default video frame height in pixels (env: VA_VIDEO_DEFAULT_HEIGHT)
        default_fps: Default frames per second (env: VA_VIDEO_DEFAULT_FPS)
        format: Video format for GStreamer (env: VA_VIDEO_FORMAT)
        max_buffers: Maximum buffers for GStreamer appsink (env: VA_VIDEO_MAX_BUFFERS)
    """

    default_width: int = field(
        default_factory=lambda: _get_env_int("VA_VIDEO_DEFAULT_WIDTH", 640)
    )
    default_height: int = field(
        default_factory=lambda: _get_env_int("VA_VIDEO_DEFAULT_HEIGHT", 480)
    )
    default_fps: int = field(
        default_factory=lambda: _get_env_int("VA_VIDEO_DEFAULT_FPS", 30)
    )
    format: str = field(
        default_factory=lambda: os.environ.get("VA_VIDEO_FORMAT", "BGR")
    )
    max_buffers: int = field(
        default_factory=lambda: _get_env_int("VA_VIDEO_MAX_BUFFERS", 1)
    )


@dataclass
class GStreamerConfig:
    """GStreamer pipeline configuration for LocalEdge.

    Attributes:
        appsink_name: Name for GStreamer appsink element (env: VA_GSTREAMER_APPSINK_NAME)
        appsrc_name: Name for GStreamer appsrc element (env: VA_GSTREAMER_APPSRC_NAME)
        audio_layout: Audio layout for GStreamer caps (env: VA_GSTREAMER_AUDIO_LAYOUT)
    """

    appsink_name: str = field(
        default_factory=lambda: os.environ.get("VA_GSTREAMER_APPSINK_NAME", "sink")
    )
    appsrc_name: str = field(
        default_factory=lambda: os.environ.get("VA_GSTREAMER_APPSRC_NAME", "src")
    )
    audio_layout: str = field(
        default_factory=lambda: os.environ.get("VA_GSTREAMER_AUDIO_LAYOUT", "interleaved")
    )


@dataclass
class LocalEdgeConfig:
    """Complete configuration for LocalEdge RTC implementation.

    This configuration class consolidates all settings for audio, video, and GStreamer
    pipelines. Values can be customized via:

    1. Direct instantiation:
        >>> config = LocalEdgeConfig(
        ...     audio=AudioConfig(input_sample_rate=48000),
        ...     video=VideoConfig(default_width=1920)
        ... )

    2. Environment variables (with VA_ prefix):
        >>> # Set via environment
        >>> os.environ["VA_AUDIO_INPUT_SAMPLE_RATE"] = "48000"
        >>> config = LocalEdgeConfig()  # Uses env value

    3. Mix of both (env vars take precedence for sub-configs):
        >>> config = LocalEdgeConfig()  # Sub-configs use env vars
        >>> config.audio.input_sample_rate = 48000  # Override after creation

    Attributes:
        audio: Audio configuration settings
        video: Video configuration settings
        gstreamer: GStreamer pipeline configuration
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    gstreamer: GStreamerConfig = field(default_factory=GStreamerConfig)

    @classmethod
    def from_defaults(cls) -> "LocalEdgeConfig":
        """Create configuration with default values (ignoring environment variables).

        Returns:
            LocalEdgeConfig instance with hard-coded defaults
        """
        return cls(
            audio=AudioConfig(
                input_sample_rate=16000,
                output_sample_rate=24000,
                input_channels=1,
                output_channels=1,
                bit_depth=16,
                input_buffer_duration=2.0,
                output_buffer_size_ms=10000,
                capture_chunk_duration=0.1,
                playback_chunk_duration=0.05,
                loop_sleep_interval=0.01,
                flush_poll_interval=0.05,
                error_retry_delay=0.1,
                thread_join_timeout=2.0,
                max_int16_value=32767,
                eos_wait_time=0.1,
            ),
            video=VideoConfig(
                default_width=640,
                default_height=480,
                default_fps=30,
                format="BGR",
                max_buffers=1,
            ),
            gstreamer=GStreamerConfig(
                appsink_name="sink",
                appsrc_name="src",
                audio_layout="interleaved",
            ),
        )
