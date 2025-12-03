"""Gradium Speech-to-Text implementation using the official Gradium SDK."""

import asyncio
import logging
import os
from typing import Optional, Any

import numpy as np
import gradium
from gradium.speech import STTStream

from getstream.video.rtc.track_util import PcmData

from vision_agents.core import stt
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.edge.types import Participant

logger = logging.getLogger(__name__)

# Gradium STT expects 24kHz PCM 16-bit mono
GRADIUM_SAMPLE_RATE = 24000
GRADIUM_CHANNELS = 1

# Gradium recommends 1920 samples (80ms) per chunk at 24kHz
GRADIUM_CHUNK_SAMPLES = 1920


class STT(stt.STT):
    """
    Gradium Speech-to-Text implementation using the official Gradium SDK.

    Gradium provides low-latency, high-quality speech-to-text with support for
    multiple languages (English, French, German, Spanish, Portuguese) and
    built-in Voice Activity Detection (VAD) for turn detection.

    Reference: https://gradium.ai/api_docs.html#tag/Documentation/Speech-to-Text-(STT)
    """

    turn_detection: bool = True  # Gradium supports VAD-based turn detection

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        vad_threshold: float = 0.7,
        base_url: Optional[str] = None,
        client: Optional[gradium.client.GradiumClient] = None,
    ):
        """
        Initialize Gradium STT.

        Args:
            api_key: Gradium API key. If not provided, uses GRADIUM_API_KEY env var.
            language: Language code (en, fr, de, es, pt). Defaults to "en".
            vad_threshold: VAD inactivity probability threshold for turn detection.
                Higher values require more confidence that speech has ended.
                Defaults to 0.7.
            base_url: Optional API base URL (e.g., "https://us.api.gradium.ai/api/").
            client: Optional pre-configured GradiumClient instance.
        """
        super().__init__(provider_name="gradium")

        self.language = language
        self.vad_threshold = vad_threshold

        logger.debug("Gradium STT __init__: creating client")
        if client is not None:
            self.client = client
            logger.debug("Gradium STT: using provided client")
        else:
            api_key = api_key or os.environ.get("GRADIUM_API_KEY")
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            
            logger.debug(f"Gradium STT: creating client with kwargs: {list(client_kwargs.keys())}")
            self.client = gradium.client.GradiumClient(**client_kwargs)

        # Audio queue for streaming to the SDK - stores numpy arrays
        self._audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self._stt_stream: Optional[gradium.speech.STTStream] = None
        self._stream_task: Optional[asyncio.Task[Any]] = None
        self._current_participant: Optional[Participant] = None
        self._started = False
        self._audio_chunk_count = 0
        self._audio_ended = False
        logger.debug("Gradium STT __init__: complete")

    async def _audio_generator(self):
        """Async generator that yields audio chunks from the queue."""
        logger.debug("Gradium STT _audio_generator: started, waiting for audio chunks")
        chunk_count = 0
        while True:
            logger.debug(f"Gradium STT _audio_generator: waiting for chunk {chunk_count}")
            audio_chunk = await self._audio_queue.get()
            if audio_chunk is None:
                logger.debug("Gradium STT _audio_generator: received None, ending stream")
                break
            chunk_count += 1
            logger.debug(f"Gradium STT _audio_generator: yielding chunk {chunk_count}, shape={audio_chunk.shape}, dtype={audio_chunk.dtype}")
            yield audio_chunk
        logger.debug(f"Gradium STT _audio_generator: finished, yielded {chunk_count} chunks")

    async def _run_stream(self):
        """Run the STT stream and process transcribed text."""
        logger.debug("Gradium STT _run_stream: starting")
        setup = gradium.speech.STTSetup(
            language=self.language,
            sample_rate=GRADIUM_SAMPLE_RATE,
        )
        logger.debug(f"Gradium STT _run_stream: setup created, language={self.language}, sample_rate={GRADIUM_SAMPLE_RATE}")

        try:
            logger.debug("Gradium STT _run_stream: calling gradium.speech.stt_stream()")
            self._stt_stream : STTStream = await gradium.speech.stt_stream(
                self.client,
                setup=setup,
                audio=self._audio_generator(),
            )

            logger.debug("Gradium STT _run_stream: stt_stream() returned, starting to iterate")

            # Iterate over transcribed text segments
            logger.debug("Gradium STT _run_stream: calling iter_text()")
            segment_count = 0
            async for text_segment in self._stt_stream.iter_text():
                segment_count += 1
                logger.debug(f"Gradium STT _run_stream: received text segment {segment_count}: {text_segment}")
                await self._handle_text_segment(text_segment)
            logger.debug(f"Gradium STT _run_stream: iter_text() finished, received {segment_count} segments")
        except asyncio.CancelledError:
            logger.debug("Gradium STT _run_stream: cancelled")
            raise
        except Exception as e:
            logger.error(f"Gradium STT _run_stream: error: {e}", exc_info=True)
            raise

    async def _handle_text_segment(self, text_segment):
        """Handle incoming transcribed text segment."""
        logger.debug(f"Gradium STT _handle_text_segment: received {text_segment}")
        if not self._current_participant:
            logger.warning("Gradium STT: Received transcript but no participant set")
            return

        text = text_segment.text if hasattr(text_segment, "text") else ""
        start_s = text_segment.start_s if hasattr(text_segment, "start_s") else 0.0
        stop_s = text_segment.stop_s if hasattr(text_segment, "stop_s") else 0.0

        logger.debug(f"Gradium STT _handle_text_segment: text='{text}', start={start_s}, stop={stop_s}")

        if not text:
            logger.debug("Gradium STT _handle_text_segment: empty text, skipping")
            return

        response = TranscriptResponse(
            language=self.language,
            model_name="gradium-stt",
            audio_duration_ms=int(stop_s * 1000),
        )

        logger.debug(f"Gradium STT _handle_text_segment: emitting transcript event for '{text}'")
        self._emit_transcript_event(text, self._current_participant, response)

        logger.debug("Gradium STT _handle_text_segment: emitting turn ended event")
        self._emit_turn_ended_event(
            participant=self._current_participant,
            eager_end_of_turn=False,
            confidence=0.9,
        )

    async def start(self):
        """Start the Gradium STT stream."""
        logger.debug("Gradium STT start: called")
        await super().start()

        if self._started:
            logger.warning("Gradium STT start: stream already started")
            return

        self._started = True
        self._audio_ended = False
        logger.debug("Gradium STT start: creating stream task")
        self._stream_task = asyncio.create_task(self._run_stream())
        logger.debug("Gradium STT start: stream task created")

    async def end_audio(self):
        """
        Signal that no more audio will be sent.
        
        Call this after sending all audio to allow the STT to finish processing
        and return final transcripts.
        """
        if self._audio_ended:
            logger.debug("Gradium STT end_audio: already ended")
            return
            
        logger.debug("Gradium STT end_audio: signaling end of audio stream")
        self._audio_ended = True
        await self._audio_queue.put(None)

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through Gradium for transcription.

        Audio is resampled to 24kHz mono and chunked into 80ms pieces
        as required by Gradium.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata.
        """
        if self.closed:
            logger.warning("Gradium STT process_audio: closed, ignoring audio")
            return

        if not self._started:
            logger.warning("Gradium STT process_audio: not started, call start() first")
            return

        if self._audio_ended:
            logger.warning("Gradium STT process_audio: audio already ended, ignoring")
            return

        self._current_participant = participant
        self._audio_chunk_count += 1

        logger.debug(f"Gradium STT process_audio: incoming chunk {self._audio_chunk_count}, duration={pcm_data.duration_ms}ms, sample_rate={pcm_data.sample_rate}")

        # Resample to 24kHz mono (required by Gradium)
        resampled = pcm_data.resample(GRADIUM_SAMPLE_RATE, GRADIUM_CHANNELS)
        samples = resampled.samples  # numpy int16 array
        
        logger.debug(f"Gradium STT process_audio: resampled to {GRADIUM_SAMPLE_RATE}Hz, total samples={len(samples)}")

        # Chunk the audio into 1920 samples (80ms) as recommended by Gradium
        # SDK expects numpy int16 arrays, all chunks should be exactly 1920 samples
        chunks_sent = 0
        for i in range(0, len(samples), GRADIUM_CHUNK_SAMPLES):
            chunk = samples[i:i + GRADIUM_CHUNK_SAMPLES]
            # Ensure it's a numpy int16 array
            if not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=np.int16)
            elif chunk.dtype != np.int16:
                chunk = chunk.astype(np.int16)
            
            # Pad short chunks with zeros to exactly 1920 samples
            if len(chunk) < GRADIUM_CHUNK_SAMPLES:
                padded = np.zeros(GRADIUM_CHUNK_SAMPLES, dtype=np.int16)
                padded[:len(chunk)] = chunk
                chunk = padded
                
            await self._audio_queue.put(chunk)
            chunks_sent += 1
        
        logger.debug(f"Gradium STT process_audio: sent {chunks_sent} chunks of ~{GRADIUM_CHUNK_SAMPLES} samples each")

    async def close(self):
        """Close the Gradium STT stream and clean up resources."""
        logger.debug("Gradium STT close: called")
        await super().close()

        # Signal end of audio stream if not already done
        if not self._audio_ended:
            logger.debug("Gradium STT close: putting None into queue to signal end")
            await self._audio_queue.put(None)
            self._audio_ended = True

        if self._stream_task and not self._stream_task.done():
            logger.debug("Gradium STT close: cancelling stream task")
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                logger.debug("Gradium STT close: stream task cancelled")

        self._stt_stream = None
        self._started = False
        logger.debug("Gradium STT close: complete")
