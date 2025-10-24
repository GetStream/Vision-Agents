from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    NamedTuple,
    Union,
    Iterator,
    AsyncIterator,
    Protocol,
    runtime_checkable,
)
import logging

import numpy as np
from numpy._typing import NDArray
from pyee.asyncio import AsyncIOEventEmitter
import av
import asyncio
import os
import shutil
import tempfile
import time

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


@runtime_checkable
class OutputAudioTrack(Protocol):
    """
    A protocol describing an output audio track, the actual implementation depends on the edge transported used
    eg. getstream.video.rtc.audio_track.AudioStreamTrack
    """

    async def write(self, data: bytes) -> None: ...

    def stop(self) -> None: ...


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
    def stereo(self) -> bool:
        return self.channels == 2

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
            # If array has shape (channels, samples) or (samples, channels), duration uses the samples dimension
            if self.samples.ndim == 2:
                # Determine which dimension is samples vs channels
                # Standard format is (channels, samples), but we need to handle both
                ch = self.channels if self.channels else 1
                if self.samples.shape[0] == ch:
                    # Shape is (channels, samples) - correct format
                    num_samples = self.samples.shape[1]
                elif self.samples.shape[1] == ch:
                    # Shape is (samples, channels) - transposed format
                    num_samples = self.samples.shape[0]
                else:
                    # Ambiguous or unknown - assume (channels, samples) and pick larger dimension
                    # This handles edge cases like (2, 2) arrays
                    num_samples = max(self.samples.shape[0], self.samples.shape[1])
            else:
                num_samples = len(self.samples)
        elif isinstance(self.samples, bytes):
            # If samples is bytes, calculate based on format
            if self.format == "s16":
                # For s16 format, each sample is 2 bytes (16 bits)
                # For multi-channel, divide by channels to get sample count
                num_samples = len(self.samples) // (
                    2 * (self.channels if self.channels else 1)
                )
            elif self.format == "f32":
                # For f32 format, each sample is 4 bytes (32 bits)
                num_samples = len(self.samples) // (
                    4 * (self.channels if self.channels else 1)
                )
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
        self,
        target_sample_rate: int,
        target_channels: Optional[int] = None,
        resampler: Optional[Any] = None,
    ) -> "PcmData":
        """
        Resample PcmData to a different sample rate and/or channels using AV library.

        Args:
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels (defaults to current)
            resampler: Optional persistent AudioResampler instance to use. If None,
                      creates a new resampler (for one-off use). Pass a persistent
                      resampler to avoid discontinuities when resampling streaming chunks.

        Returns:
            New PcmData object with resampled audio
        """
        if target_channels is None:
            target_channels = self.channels
        if self.sample_rate == target_sample_rate and target_channels == self.channels:
            return self

        # Prepare ndarray shape for AV input frame.
        # Use planar input (s16p) with shape (channels, samples).
        in_layout = "mono" if self.channels == 1 else "stereo"
        cmaj = self.samples
        if isinstance(cmaj, np.ndarray):
            if cmaj.ndim == 1:
                # (samples,) -> (channels, samples)
                if self.channels > 1:
                    cmaj = np.tile(cmaj, (self.channels, 1))
                else:
                    cmaj = cmaj.reshape(1, -1)
            elif cmaj.ndim == 2:
                # Normalize to (channels, samples)
                ch = self.channels if self.channels else 1
                if cmaj.shape[0] == ch:
                    # Already (channels, samples)
                    pass
                elif cmaj.shape[1] == ch:
                    # (samples, channels) -> transpose
                    cmaj = cmaj.T
                else:
                    # Ambiguous - assume larger dim is samples
                    if cmaj.shape[1] > cmaj.shape[0]:
                        # Likely (channels, samples)
                        pass
                    else:
                        # Likely (samples, channels)
                        cmaj = cmaj.T
            cmaj = np.ascontiguousarray(cmaj)
        frame = av.AudioFrame.from_ndarray(cmaj, format="s16p", layout=in_layout)
        frame.sample_rate = self.sample_rate

        # Use provided resampler or create a new one
        if resampler is None:
            # Create new resampler for one-off use
            out_layout = "mono" if target_channels == 1 else "stereo"
            resampler = av.AudioResampler(
                format="s16", layout=out_layout, rate=target_sample_rate
            )

        # Resample the frame
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            resampled_frame = resampled_frames[0]
            # PyAV's to_ndarray() for packed format returns flattened interleaved data
            # For stereo s16 (packed), it returns shape (1, num_values) where num_values = samples * channels
            raw_array = resampled_frame.to_ndarray()
            num_frames = resampled_frame.samples  # Actual number of sample frames

            # Normalize output to (channels, samples) format
            ch = int(target_channels)

            # Handle PyAV's packed format quirk: returns (1, num_values) for stereo
            if raw_array.ndim == 2 and raw_array.shape[0] == 1 and ch > 1:
                # Flatten and deinterleave packed stereo data
                # Shape (1, 32000) -> (32000,) -> deinterleave to (2, 16000)
                flat = raw_array.reshape(-1)
                if len(flat) == num_frames * ch:
                    # Deinterleave: [L0,R0,L1,R1,...] -> [[L0,L1,...], [R0,R1,...]]
                    resampled_samples = flat.reshape(-1, ch).T
                else:
                    logger.warning(
                        "Unexpected array size %d for %d frames x %d channels",
                        len(flat),
                        num_frames,
                        ch,
                    )
                    resampled_samples = flat.reshape(ch, -1)
            elif raw_array.ndim == 2:
                # Standard case: (samples, channels) or already (channels, samples)
                if raw_array.shape[1] == ch:
                    # (samples, channels) -> transpose to (channels, samples)
                    resampled_samples = raw_array.T
                elif raw_array.shape[0] == ch:
                    # Already (channels, samples)
                    resampled_samples = raw_array
                else:
                    # Ambiguous - assume time-major
                    resampled_samples = raw_array.T
            elif raw_array.ndim == 1:
                # 1D output (mono)
                if ch == 1:
                    # Keep as 1D for mono
                    resampled_samples = raw_array
                elif ch > 1:
                    # Shouldn't happen if we requested stereo, but handle it
                    logger.warning(
                        "Got 1D array but requested %d channels, duplicating", ch
                    )
                    resampled_samples = np.tile(raw_array, (ch, 1))
                else:
                    resampled_samples = raw_array
            else:
                # Unexpected dimensionality
                logger.warning(
                    "Unexpected ndim %d from PyAV, reshaping", raw_array.ndim
                )
                resampled_samples = raw_array.reshape(ch, -1)

            # Flatten mono arrays to 1D for consistency
            if (
                ch == 1
                and isinstance(resampled_samples, np.ndarray)
                and resampled_samples.ndim > 1
            ):
                resampled_samples = resampled_samples.flatten()

            # Ensure int16 dtype for s16
            if (
                isinstance(resampled_samples, np.ndarray)
                and resampled_samples.dtype != np.int16
            ):
                resampled_samples = resampled_samples.astype(np.int16)

            return PcmData(
                samples=resampled_samples,
                sample_rate=target_sample_rate,
                format="s16",
                pts=self.pts,
                dts=self.dts,
                time_base=self.time_base,
                channels=target_channels,
            )
        else:
            # If resampling failed, return original data
            return self

    def to_bytes(self) -> bytes:
        """Return interleaved PCM bytes (s16 or f32 depending on format).

        For multi-channel audio, this returns packed/interleaved bytes in the order
        [L0, R0, L1, R1, ...]. The internal convention is (channels, samples).
        If the stored ndarray is (samples, channels), we transpose it.
        """
        arr = self.samples
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                channels = int(self.channels or arr.shape[0])
                # Normalize to (channels, samples)
                if arr.shape[0] == channels:
                    cmaj = arr
                elif arr.shape[1] == channels:
                    cmaj = arr.T
                else:
                    logger.warning(
                        "to_bytes: ambiguous array shape %s for channels=%d; assuming time-major",
                        arr.shape,
                        channels,
                    )
                    cmaj = arr.T
                samples_count = cmaj.shape[1]
                # Interleave channels explicitly to avoid any stride-related surprises
                out = np.empty(samples_count * channels, dtype=cmaj.dtype)
                for i in range(channels):
                    out[i::channels] = cmaj[i]
                return out.tobytes()
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


async def play_pcm_with_ffplay(
    pcm: PcmData,
    outfile_path: Optional[str] = None,
    timeout_s: float = 30.0,
) -> str:
    """Write PcmData to a WAV file and optionally play it with ffplay.

    This is a utility function for testing and debugging audio output.

    Args:
        pcm: PcmData object to play
        outfile_path: Optional path for the WAV file. If None, creates a temp file.
        timeout_s: Timeout in seconds for ffplay playback (default: 30.0)

    Returns:
        Path to the written WAV file

    Example:
        pcm = PcmData.from_bytes(audio_bytes, sample_rate=48000, channels=2)
        wav_path = await play_pcm_with_ffplay(pcm)
    """

    # Generate output path if not provided
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(tmpdir, f"pcm_playback_{timestamp}.wav")

    # Write WAV file
    with open(outfile_path, "wb") as f:
        f.write(pcm.to_wav_bytes())

    logger.info(f"Wrote WAV file: {outfile_path}")

    # Optional playback with ffplay
    if shutil.which("ffplay"):
        logger.info("Playing audio with ffplay...")
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-autoexit",
            "-nodisp",
            "-hide_banner",
            "-loglevel",
            "error",
            outfile_path,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(f"ffplay timed out after {timeout_s}s, killing process")
            proc.kill()
    else:
        logger.warning("ffplay not found in PATH, skipping playback")

    return outfile_path
