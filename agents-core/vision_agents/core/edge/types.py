from dataclasses import dataclass
from typing import Any, Optional, NamedTuple, Union, Iterator, AsyncIterator
import logging

import numpy as np
from numpy._typing import NDArray
from pyee.asyncio import AsyncIOEventEmitter
import av

logger = logging.getLogger(__name__)


@dataclass
class User:
    id: Optional[str] = ""
    name: Optional[str] = ""
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any
    user_id: str


class Connection(AsyncIOEventEmitter):
    """
    To standardize we need to have a method to close
    and a way to receive a callback when the call is ended
    In the future we might want to forward more events
    """

    async def close(self):
        pass


class PcmData(NamedTuple):
    """
    A named tuple representing PCM audio data.

    Attributes:
        format: The format of the audio data.
        sample_rate: The sample rate of the audio data.
        samples: The audio samples as a numpy array.
        pts: The presentation timestamp of the audio data.
        dts: The decode timestamp of the audio data.
        time_base: The time base for converting timestamps to seconds.
    """

    format: str
    sample_rate: int
    samples: NDArray
    pts: Optional[int] = None  # Presentation timestamp
    dts: Optional[int] = None  # Decode timestamp
    time_base: Optional[float] = None  # Time base for converting timestamps to seconds
    channels: int = 1  # Number of channels (1=mono, 2=stereo)

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the audio data in seconds.

        Returns:
            float: Duration in seconds.
        """
        # The samples field contains a numpy array of audio samples
        # For s16 format, each element in the array is one sample (int16)
        # For f32 format, each element in the array is one sample (float32)

        if isinstance(self.samples, np.ndarray):
            # If array has shape (channels, samples), duration uses the samples dimension
            if self.samples.ndim == 2:
                num_samples = self.samples.shape[-1]
            else:
                num_samples = len(self.samples)
        elif isinstance(self.samples, bytes):
            # If samples is bytes, calculate based on format
            if self.format == "s16":
                # For s16 format, each sample is 2 bytes (16 bits)
                num_samples = len(self.samples) // 2
            elif self.format == "f32":
                # For f32 format, each sample is 4 bytes (32 bits)
                num_samples = len(self.samples) // 4
            else:
                # Default assumption for other formats (treat as raw bytes)
                num_samples = len(self.samples)
        else:
            # Fallback: try to get length
            try:
                num_samples = len(self.samples)
            except TypeError:
                logger.warning(
                    f"Cannot determine sample count for type {type(self.samples)}"
                )
                return 0.0

        # Calculate duration based on sample rate
        return num_samples / self.sample_rate

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds computed from samples and sample rate."""
        return self.duration * 1000.0

    @property
    def pts_seconds(self) -> Optional[float]:
        if self.pts is not None and self.time_base is not None:
            return self.pts * self.time_base
        return None

    @property
    def dts_seconds(self) -> Optional[float]:
        if self.dts is not None and self.time_base is not None:
            return self.dts * self.time_base
        return None

    @classmethod
    def from_bytes(
        cls,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        format: str = "s16",
        channels: int = 1,
    ) -> "PcmData":
        """Create PcmData from raw PCM bytes (interleaved for multi-channel).

        Args:
            audio_bytes: Raw PCM data as bytes.
            sample_rate: Sample rate in Hz.
            format: Audio sample format, e.g. "s16" or "f32".
            channels: Number of channels (1=mono, 2=stereo).

        Returns:
            PcmData object with numpy samples (mono: 1D, multi-channel: 2D [channels, samples]).
        """
        # Determine dtype and bytes per sample
        dtype: Any
        width: int
        if format == "s16":
            dtype = np.int16
            width = 2
        elif format == "f32":
            dtype = np.float32
            width = 4
        else:
            dtype = np.int16
            width = 2

        # Ensure buffer aligns to whole samples
        if len(audio_bytes) % width != 0:
            trimmed = len(audio_bytes) - (len(audio_bytes) % width)
            if trimmed <= 0:
                return cls(
                    samples=np.array([], dtype=dtype),
                    sample_rate=sample_rate,
                    format=format,
                    channels=channels,
                )
            logger.debug(
                "Trimming non-aligned PCM buffer: %d -> %d bytes",
                len(audio_bytes),
                trimmed,
            )
            audio_bytes = audio_bytes[:trimmed]

        arr = np.frombuffer(audio_bytes, dtype=dtype)
        if channels > 1 and arr.size > 0:
            # Convert interleaved [L,R,L,R,...] to shape (channels, samples)
            total_frames = (arr.size // channels) * channels
            if total_frames != arr.size:
                logger.debug(
                    "Trimming interleaved frames to channel multiple: %d -> %d elements",
                    arr.size,
                    total_frames,
                )
                arr = arr[:total_frames]
            try:
                frames = arr.reshape(-1, channels)
                arr = frames.T
            except Exception:
                logger.warning(
                    f"Unable to reshape audio buffer to {channels} channels; falling back to 1D"
                )
        return cls(
            samples=arr, sample_rate=sample_rate, format=format, channels=channels
        )

    @classmethod
    def from_data(
        cls,
        data: Union[bytes, bytearray, memoryview, NDArray],
        sample_rate: int = 16000,
        format: str = "s16",
        channels: int = 1,
    ) -> "PcmData":
        """Create PcmData from bytes or numpy arrays.

        - bytes-like: interpreted as interleaved PCM per channel.
        - numpy arrays: accepts 1D [samples], 2D [channels, samples] or [samples, channels].
        """
        if isinstance(data, (bytes, bytearray, memoryview)):
            return cls.from_bytes(
                bytes(data), sample_rate=sample_rate, format=format, channels=channels
            )

        if isinstance(data, np.ndarray):
            arr = data
            # Ensure dtype aligns with format
            if format == "s16" and arr.dtype != np.int16:
                arr = arr.astype(np.int16)
            elif format == "f32" and arr.dtype != np.float32:
                arr = arr.astype(np.float32)

            # Normalize shape to (channels, samples) for multi-channel
            if arr.ndim == 2:
                if arr.shape[0] == channels:
                    samples_arr = arr
                elif arr.shape[1] == channels:
                    samples_arr = arr.T
                else:
                    # Assume first dimension is channels if ambiguous
                    samples_arr = arr
            elif arr.ndim == 1:
                if channels > 1:
                    try:
                        frames = arr.reshape(-1, channels)
                        samples_arr = frames.T
                    except Exception:
                        logger.warning(
                            f"Could not reshape 1D array to {channels} channels; keeping mono"
                        )
                        channels = 1
                        samples_arr = arr
                else:
                    samples_arr = arr
            else:
                # Fallback
                samples_arr = arr.reshape(-1)
                channels = 1

            return cls(
                samples=samples_arr,
                sample_rate=sample_rate,
                format=format,
                channels=channels,
            )

        # Unsupported type
        raise TypeError(f"Unsupported data type for PcmData: {type(data)}")

    def resample(
        self, target_sample_rate: int, target_channels: Optional[int] = None
    ) -> "PcmData":
        """
        Resample PcmData to a different sample rate and/or channels using AV library.

        Args:
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels (defaults to current)

        Returns:
            New PcmData object with resampled audio
        """
        if target_channels is None:
            target_channels = self.channels
        if self.sample_rate == target_sample_rate and target_channels == self.channels:
            return self

        # Prepare ndarray shape for AV.
        # Our convention: (channels, samples) for multi-channel, (samples,) for mono.
        samples = self.samples
        if samples.ndim == 1:
            # Mono: reshape to (1, samples) for AV
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2:
            # Already (channels, samples)
            pass

        # Create AV audio frame from the samples
        in_layout = "mono" if self.channels == 1 else "stereo"
        # For multi-channel, use planar format to avoid packed shape errors
        in_format = "s16" if self.channels == 1 else "s16p"
        samples = np.ascontiguousarray(samples)
        frame = av.AudioFrame.from_ndarray(samples, format=in_format, layout=in_layout)
        frame.sample_rate = self.sample_rate

        # Create resampler
        out_layout = "mono" if target_channels == 1 else "stereo"
        resampler = av.AudioResampler(
            format="s16", layout=out_layout, rate=target_sample_rate
        )

        # Resample the frame
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            resampled_frame = resampled_frames[0]
            resampled_samples = resampled_frame.to_ndarray()

            # AV returns (channels, samples), so for mono we want the first (and only) channel
            if len(resampled_samples.shape) > 1:
                if target_channels == 1:
                    resampled_samples = resampled_samples[0]

            # Convert to int16
            resampled_samples = resampled_samples.astype(np.int16)

            return PcmData(
                samples=resampled_samples,
                sample_rate=target_sample_rate,
                format=self.format,
                pts=self.pts,
                dts=self.dts,
                time_base=self.time_base,
                channels=target_channels,
            )
        else:
            # If resampling failed, return original data
            return self

    def to_bytes(self) -> bytes:
        """Return interleaved PCM bytes (s16 or f32 depending on format)."""
        arr = self.samples
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                # (channels, samples) -> interleaved (samples, channels)
                interleaved = arr.T.reshape(-1)
                return interleaved.tobytes()
            return arr.tobytes()
        # Fallback
        if isinstance(arr, (bytes, bytearray)):
            return bytes(arr)
        try:
            return bytes(arr)
        except Exception:
            logger.warning("Cannot convert samples to bytes; returning empty")
            return b""

    def to_wav_bytes(self) -> bytes:
        """Return a complete WAV file (header + frames) as bytes.

        Notes:
        - If the data format is not s16, it will be converted to s16.
        - Channels and sample rate are taken from the PcmData instance.
        """
        import io
        import wave

        # Ensure s16 frames
        if self.format != "s16":
            arr = self.samples
            if isinstance(arr, np.ndarray):
                if arr.dtype != np.int16:
                    # Convert floats to int16 range
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                frames = PcmData(
                    samples=arr,
                    sample_rate=self.sample_rate,
                    format="s16",
                    pts=self.pts,
                    dts=self.dts,
                    time_base=self.time_base,
                    channels=self.channels,
                ).to_bytes()
            else:
                frames = self.to_bytes()
            width = 2
        else:
            frames = self.to_bytes()
            width = 2

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels or 1)
            wf.setsampwidth(width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(frames)
        return buf.getvalue()

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "s16",
    ) -> Union["PcmData", Iterator["PcmData"], AsyncIterator["PcmData"]]:
        """Create PcmData stream(s) from a provider response.

        Supported inputs:
        - bytes/bytearray/memoryview -> returns PcmData
        - async iterator of bytes or objects with .data -> returns async iterator of PcmData
        - iterator of bytes or objects with .data -> returns iterator of PcmData
        - already PcmData -> returns PcmData
        - single object with .data -> returns PcmData from its data
        """

        # bytes-like returns a single PcmData
        if isinstance(response, (bytes, bytearray, memoryview)):
            return cls.from_bytes(
                bytes(response),
                sample_rate=sample_rate,
                channels=channels,
                format=format,
            )

        # Already a PcmData
        if isinstance(response, PcmData):
            return response

        # Async iterator
        if hasattr(response, "__aiter__"):

            async def _agen():
                width = 2 if format == "s16" else 4 if format == "f32" else 2
                frame_width = width * max(1, channels)
                buf = bytearray()
                async for item in response:
                    if isinstance(item, PcmData):
                        yield item
                        continue
                    data = getattr(item, "data", item)
                    if not isinstance(data, (bytes, bytearray, memoryview)):
                        raise TypeError("Async iterator yielded unsupported item type")
                    buf.extend(bytes(data))
                    aligned = (len(buf) // frame_width) * frame_width
                    if aligned:
                        chunk = bytes(buf[:aligned])
                        del buf[:aligned]
                        yield cls.from_bytes(
                            chunk,
                            sample_rate=sample_rate,
                            channels=channels,
                            format=format,
                        )
                # pad remainder, if any
                if buf:
                    pad_len = (-len(buf)) % frame_width
                    if pad_len:
                        buf.extend(b"\x00" * pad_len)
                    yield cls.from_bytes(
                        bytes(buf),
                        sample_rate=sample_rate,
                        channels=channels,
                        format=format,
                    )

            return _agen()

        # Sync iterator (but skip treating bytes as iterable of ints)
        if hasattr(response, "__iter__") and not isinstance(
            response, (str, bytes, bytearray, memoryview)
        ):

            def _gen():
                width = 2 if format == "s16" else 4 if format == "f32" else 2
                frame_width = width * max(1, channels)
                buf = bytearray()
                for item in response:
                    if isinstance(item, PcmData):
                        yield item
                        continue
                    data = getattr(item, "data", item)
                    if not isinstance(data, (bytes, bytearray, memoryview)):
                        raise TypeError("Iterator yielded unsupported item type")
                    buf.extend(bytes(data))
                    aligned = (len(buf) // frame_width) * frame_width
                    if aligned:
                        chunk = bytes(buf[:aligned])
                        del buf[:aligned]
                        yield cls.from_bytes(
                            chunk,
                            sample_rate=sample_rate,
                            channels=channels,
                            format=format,
                        )
                if buf:
                    pad_len = (-len(buf)) % frame_width
                    if pad_len:
                        buf.extend(b"\x00" * pad_len)
                    yield cls.from_bytes(
                        bytes(buf),
                        sample_rate=sample_rate,
                        channels=channels,
                        format=format,
                    )

            return _gen()

        # Single object with .data
        if hasattr(response, "data"):
            data = getattr(response, "data")
            if isinstance(data, (bytes, bytearray, memoryview)):
                return cls.from_bytes(
                    bytes(data),
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                )

        raise TypeError(
            f"Unsupported response type for PcmData.from_response: {type(response)}"
        )
