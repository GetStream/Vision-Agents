import logging
import os
import time
from abc import ABC
from typing import Optional, Dict, Any
import asyncio

import krisp_audio
import numpy as np
from getstream.audio.utils import resample_audio
from getstream.video.rtc.track_util import PcmData
from stream_agents.turn_detection import BaseTurnDetector, TurnEvent, TurnEventData
from stream_agents.utils.utils import to_mono


def _int_to_frame_duration(frame_dur: int):
    """Convert integer frame duration to Krisp enum."""
    durations = {
        10: krisp_audio.FrameDuration.Fd10ms,
        15: krisp_audio.FrameDuration.Fd15ms,
        20: krisp_audio.FrameDuration.Fd20ms,
        30: krisp_audio.FrameDuration.Fd30ms,
        32: krisp_audio.FrameDuration.Fd32ms,
    }
    return durations[frame_dur]


def _resample(samples: np.ndarray) -> np.ndarray:
    """Resample audio from 48 kHz to 16 kHz."""
    return resample_audio(samples, 48000, 16000).astype(np.int16)


def log_callback(log_message, log_level):
    print(f"[{log_level}] {log_message}", flush=True)


class KrispTurnDetection(BaseTurnDetector, ABC):
    def __init__(
        self,
        model_path: Optional[str] = os.getcwd() + "/krisp-viva-tt-v1.kef",
        frame_duration_ms: int = 15,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.logger = logging.getLogger("KrispTurnDetection")
        self.model_path = model_path
        self.frame_duration_ms = frame_duration_ms
        self.turn_start_threshold = (
            1 - confidence_threshold
        )  # For Krisp Model, 0 is speaking, 1 is non-speaking
        self.turn_end_threshold = 0.75
        self._krisp_instance = None
        self._buffer: Optional[bytearray] = None
        self._initialize_krisp()

    def _initialize_krisp(self):
        try:
            krisp_audio.globalInit("", log_callback, krisp_audio.LogLevel.Debug)
            model_info = krisp_audio.ModelInfo()
            model_info.path = self.model_path
            tt_cfg = krisp_audio.TtSessionConfig()
            tt_cfg.inputSampleRate = (
                krisp_audio.SamplingRate.Sr16000Hz
            )  # We resample from 48khz
            tt_cfg.inputFrameDuration = _int_to_frame_duration(self.frame_duration_ms)
            tt_cfg.modelInfo = model_info
            self._krisp_instance = krisp_audio.TtInt16.create(tt_cfg)
        except Exception as e:
            self.logger.error(f"Failed to initialize Krisp: {e}")
            raise RuntimeError(f"Krisp initialization failed: {e}")

    async def process_audio(
        self,
        audio_data: PcmData,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._krisp_instance is None:
            self.logger.error("Krisp instance is not initialized")
            return

        # Validate sample format
        valid_formats = ["int16", "s16", "pcm_s16le"]
        if audio_data.format not in valid_formats:
            self.logger.error(
                f"Invalid sample format: {audio_data.format}. Expected one of {valid_formats}."
            )
            return
        if (
            not isinstance(audio_data.samples, np.ndarray)
            or audio_data.samples.dtype != np.int16
        ):
            self.logger.error(
                f"Invalid sample dtype: {audio_data.samples.dtype}. Expected int16."
            )
            return

        # Resample from 48 kHz to 16 kHz
        try:
            samples = _resample(audio_data.samples)
        except Exception as e:
            self.logger.error(f"Failed to resample audio: {e}")
            return

        # Infer number of channels (default to mono)
        num_channels = (
            metadata.get("channels", self._infer_channels(audio_data.format))
            if metadata
            else self._infer_channels(audio_data.format)
        )
        if num_channels != 1:
            self.logger.debug(f"Converting {num_channels}-channel audio to mono")
            try:
                samples = to_mono(samples, num_channels)
            except ValueError as e:
                self.logger.error(f"Failed to convert to mono: {e}")
                return

        # Create a new PcmData object with resampled data
        resampled_pcm = PcmData(
            format=audio_data.format,
            sample_rate=16000,
            samples=samples,
            pts=audio_data.pts,
            dts=audio_data.dts,
            time_base=audio_data.time_base,
        )

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.process_pcm_turn_taking, resampled_pcm, user_id, metadata
            )
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")

    def _infer_channels(self, format_str: str) -> int:
        """Infer number of channels from PcmData format string."""
        format_str = format_str.lower()
        if "stereo" in format_str:
            return 2
        elif any(f in format_str for f in ["mono", "s16", "int16", "pcm_s16le"]):
            return 1
        else:
            self.logger.warning(f"Unknown format string: {format_str}. Assuming mono.")
            return 1

    def process_pcm_turn_taking(
        self, pcm: PcmData, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ):
        if self._buffer is None:
            self.logger.error("Buffer not initialized. Call start() first.")
            return
        sr = pcm.sample_rate  # Now 16000 Hz after resampling
        frame_ms = self.frame_duration_ms
        samples_per_frame = sr * frame_ms // 1000
        frame_bytes = samples_per_frame * 2  # 2 bytes per int16 sample

        def process_frame(incoming_frame: np.ndarray) -> bool:
            if self._krisp_instance is None:
                self.logger.error("Krisp instance not initialized")
                return False
            score = self._krisp_instance.process(incoming_frame)
            # Ignore frames that are -1 since they are processing frames
            # Frames closer to 0 indicate an ongoing turn
            # Frames closer to 1 indicate an ending turn
            if score > 0.1:
                if not self._is_detecting and score <= self.turn_start_threshold:
                    self._is_detecting = True
                    event_data = TurnEventData(
                        timestamp=time.time(),
                        speaker_id=user_id,
                        confidence=score,
                        custom=metadata or {},
                    )
                    self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)
                elif self._is_detecting and score > self.turn_end_threshold:
                    self._is_detecting = False
                    event_data = TurnEventData(
                        timestamp=time.time(),
                        speaker_id=user_id,
                        confidence=score,
                        custom=metadata or {},
                    )
                    self._emit_turn_event(TurnEvent.TURN_ENDED, event_data)
                return self._is_detecting
            return self._is_detecting

        self._buffer.extend(pcm.samples.tobytes())
        while len(self._buffer) >= frame_bytes:
            frame_b = bytes(self._buffer[:frame_bytes])
            del self._buffer[:frame_bytes]
            frame = np.frombuffer(frame_b, dtype=np.int16)
            self._is_detecting = process_frame(frame)

        if self._buffer:
            self.logger.info(f"Accumulating {len(self._buffer)} bytes for next frame")

    def start(self) -> None:
        self._buffer = bytearray()
        self.logger.info("KrispTurnDetection started")

    def stop(self) -> None:
        if self._krisp_instance:
            self._krisp_instance = None
            krisp_audio.globalDestroy()
            self.logger.info("Krisp resources released")
        if self._buffer is not None:
            self._buffer.clear()
