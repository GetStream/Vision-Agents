import asyncio
import logging
import os
from collections import deque
from typing import Optional

import numpy as np
from getstream.video.rtc.track_util import PcmData

logger = logging.getLogger(__name__)

# Check if audio debugging is enabled via environment variable
_DEBUG_AUDIO = os.environ.get("VISION_AGENTS_DEBUG_AUDIO", "").lower() in ("true", "1", "yes")


class AudioQueue:
    """
    Queue for audio.
    - allows you to read a specific number of samples or duration of audio (needed for Silero)
    - supports a limit defined in seconds

    If the queue is full when adding to it, log a warning
    When using queue.get allow specifying either the number of samples or the duration

    """

    def __init__(self, buffer_limit_ms: int):
        """
        Initialize AudioQueue.

        Args:
            buffer_limit_ms: Maximum buffer duration in milliseconds
        """
        self.buffer_limit_ms = buffer_limit_ms
        self._buffer: deque[PcmData] = deque()
        self._total_samples = 0
        self._sample_rate: Optional[int] = None
        self._not_empty = asyncio.Event()
        self._lock = asyncio.Lock()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._buffer) == 0

    def qsize(self) -> int:
        """Get the number of items in the queue."""
        return len(self._buffer)

    def _current_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        if self._sample_rate is None or self._total_samples == 0:
            return 0.0
        return (self._total_samples / self._sample_rate) * 1000

    async def put(self, item: PcmData) -> None:
        """
        Add PcmData to the queue.

        Args:
            item: PcmData to add to the queue

        Logs a warning if adding the item would exceed the buffer limit.
        """
        if not isinstance(item, PcmData):
            raise TypeError(f"AudioQueue only accepts PcmData, got {type(item)}")

        async with self._lock:
            # Track sample rate from first item
            if self._sample_rate is None:
                self._sample_rate = item.sample_rate
                if _DEBUG_AUDIO:
                    logger.debug(
                        f"[AUDIO DEBUG] AudioQueue initialized - "
                        f"sample_rate={self._sample_rate}Hz, channels={item.channels}, "
                        f"format={item.format}"
                    )
            elif self._sample_rate != item.sample_rate:
                logger.warning(
                    f"[AUDIO FORMAT MISMATCH] Sample rate mismatch in AudioQueue: "
                    f"expected {self._sample_rate}Hz, got {item.sample_rate}Hz. "
                    f"This may cause audio playback issues. "
                    f"Ensure all audio sources use consistent sample rates."
                )

            # Check if adding this would exceed the buffer limit
            new_samples = len(item.samples)
            new_total_samples = self._total_samples + new_samples
            new_duration_ms = (new_total_samples / item.sample_rate) * 1000

            if _DEBUG_AUDIO:
                logger.debug(
                    f"[AUDIO DEBUG] Adding to AudioQueue - "
                    f"samples={new_samples}, sample_rate={item.sample_rate}Hz, "
                    f"channels={item.channels}, format={item.format}, "
                    f"new_total_samples={new_total_samples}, new_duration_ms={new_duration_ms:.1f}ms"
                )

            if new_duration_ms > self.buffer_limit_ms:
                logger.warning(
                    f"[AUDIO DEBUG] AudioQueue buffer limit exceeded: "
                    f"{new_duration_ms:.1f}ms > {self.buffer_limit_ms}ms. "
                    f"This may indicate audio processing is slower than real-time."
                )

            self._buffer.append(item)
            self._total_samples += new_samples
            self._not_empty.set()

    def put_nowait(self, item: PcmData) -> None:
        """
        Add PcmData to the queue without waiting.

        Args:
            item: PcmData to add to the queue

        Logs a warning if adding the item would exceed the buffer limit.
        """
        if not isinstance(item, PcmData):
            raise TypeError(f"AudioQueue only accepts PcmData, got {type(item)}")

        # Track sample rate from first item
        if self._sample_rate is None:
            self._sample_rate = item.sample_rate
            if _DEBUG_AUDIO:
                logger.debug(
                    f"[AUDIO DEBUG] AudioQueue initialized (nowait) - "
                    f"sample_rate={self._sample_rate}Hz, channels={item.channels}, "
                    f"format={item.format}"
                )
        elif self._sample_rate != item.sample_rate:
            logger.warning(
                f"[AUDIO FORMAT MISMATCH] Sample rate mismatch in AudioQueue: "
                f"expected {self._sample_rate}Hz, got {item.sample_rate}Hz. "
                f"This may cause audio playback issues. "
                f"Ensure all audio sources use consistent sample rates."
            )

        # Check if adding this would exceed the buffer limit
        new_samples = len(item.samples)
        new_total_samples = self._total_samples + new_samples
        new_duration_ms = (new_total_samples / item.sample_rate) * 1000

        if _DEBUG_AUDIO:
            logger.debug(
                f"[AUDIO DEBUG] Adding to AudioQueue (nowait) - "
                f"samples={new_samples}, sample_rate={item.sample_rate}Hz, "
                f"channels={item.channels}, format={item.format}, "
                f"new_total_samples={new_total_samples}, new_duration_ms={new_duration_ms:.1f}ms"
            )

        if new_duration_ms > self.buffer_limit_ms:
            logger.warning(
                f"[AUDIO DEBUG] AudioQueue buffer limit exceeded: "
                f"{new_duration_ms:.1f}ms > {self.buffer_limit_ms}ms. "
                f"This may indicate audio processing is slower than real-time."
            )

        self._buffer.append(item)
        self._total_samples += new_samples
        self._not_empty.set()

    async def get(self) -> PcmData:
        """
        Get the next PcmData from the queue.

        Returns:
            PcmData object from the queue
        """
        while True:
            async with self._lock:
                if self._buffer:
                    item = self._buffer.popleft()
                    self._total_samples -= len(item.samples)
                    if not self._buffer:
                        self._not_empty.clear()
                    return item

            # Wait for items to be added
            await self._not_empty.wait()

    def get_nowait(self) -> PcmData:
        """
        Get the next PcmData from the queue without waiting.

        Returns:
            PcmData object from the queue
        """
        if not self._buffer:
            raise asyncio.QueueEmpty("Queue is empty")

        item = self._buffer.popleft()
        self._total_samples -= len(item.samples)
        if not self._buffer:
            self._not_empty.clear()
        return item

    async def get_samples(self, num_samples: int, timeout: float = 0.1) -> PcmData:
        """
        Get a specific number of audio samples from the queue.

        Args:
            num_samples: Number of samples to retrieve
            timeout: Max time to wait for more samples when queue is empty (seconds)

        Returns:
            PcmData containing exactly num_samples (or less if queue empties)
        """
        collected_samples: list[np.ndarray] = []
        collected_count = 0
        last_item_format = None
        last_item_channels = 1
        last_participant = None

        while collected_count < num_samples:
            # Wait for items if queue is empty, with timeout
            if self.empty():
                # If we've already collected some samples, return what we have
                if collected_samples:
                    break
                # Otherwise wait for items
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    # No items arrived in time
                    if collected_samples:
                        break
                    raise asyncio.QueueEmpty("Queue is empty")

            async with self._lock:
                if not self._buffer:
                    continue

                item = self._buffer.popleft()
                self._total_samples -= len(item.samples)
                last_participant = item.participant

                last_item_format = item.format
                last_item_channels = item.channels

                samples_needed = num_samples - collected_count

                if len(item.samples) <= samples_needed:
                    # Use the entire item
                    collected_samples.append(item.samples)
                    collected_count += len(item.samples)
                else:
                    # Split the item - take what we need, put rest back
                    collected_samples.append(item.samples[:samples_needed])
                    collected_count += samples_needed

                    # Put the remainder back at the front
                    remainder = PcmData(
                        samples=item.samples[samples_needed:],
                        sample_rate=item.sample_rate,
                        format=item.format,
                        channels=item.channels,
                    )
                    self._buffer.appendleft(remainder)
                    self._total_samples += len(remainder.samples)
                    break

                # Clear event if buffer is now empty
                if not self._buffer:
                    self._not_empty.clear()

        if not collected_samples:
            raise asyncio.QueueEmpty("Queue is empty")

        # Concatenate all collected samples
        all_samples = np.concatenate(collected_samples)

        # Use properties from the last item we got
        pcm = PcmData(
            samples=all_samples,
            sample_rate=self._sample_rate,
            format=last_item_format,
            channels=last_item_channels,
        )
        pcm.participant = last_participant
        return pcm

    async def get_duration(self, duration_ms: float) -> PcmData:
        """
        Get audio data for a specific duration from the queue.

        Args:
            duration_ms: Duration in milliseconds to retrieve

        Returns:
            PcmData containing audio for the requested duration (or less if queue empties)
        """
        # Wait for first item if needed to determine sample rate
        if self._sample_rate is None:
            await self._not_empty.wait()

        if self._sample_rate is None:
            raise asyncio.QueueEmpty("Queue is empty")

        # Convert duration to number of samples
        num_samples = int((duration_ms / 1000) * self._sample_rate)
        return await self.get_samples(num_samples)

    def get_buffer_info(self) -> dict:
        """
        Get information about the current buffer state.

        Returns:
            Dict with buffer statistics
        """
        return {
            "buffer_limit_ms": self.buffer_limit_ms,
            "current_duration_ms": self._current_duration_ms(),
            "total_samples": self._total_samples,
            "sample_rate": self._sample_rate,
            "num_chunks": len(self._buffer),
            "queue_size": self.qsize(),
        }
