"""Audio format negotiation with LLM providers.

This module handles negotiation of audio formats between local audio devices
and LLM provider requirements (e.g., Gemini Realtime API).
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from vision_agents.core.types import AudioCapabilities

if TYPE_CHECKING:
    from .config import LocalEdgeConfig

logger = logging.getLogger(__name__)


class AudioFormatNegotiator:
    """Handles audio format negotiation with LLM providers.

    This class manages the negotiation of audio formats between local device
    capabilities and LLM provider requirements. It ensures that audio streams
    are properly configured to match provider expectations while supporting
    fallback to sensible defaults.

    Attributes:
        output_sample_rate: Negotiated output sample rate in Hz
        output_channels: Negotiated output channel count
        input_sample_rate: Input device sample rate in Hz
        input_channels: Input device channel count
    """

    def __init__(
        self,
        input_sample_rate: int = 16000,
        input_channels: int = 1,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the audio format negotiator.

        Args:
            input_sample_rate: Input device sample rate in Hz (default: 16000)
            input_channels: Input device channel count (default: 1)
            config: Optional LocalEdgeConfig for default output format
        """
        from .config import LocalEdgeConfig

        self.config = config if config is not None else LocalEdgeConfig()
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.output_sample_rate: Optional[int] = None
        self.output_channels: Optional[int] = None

    def negotiate_format(self, agent: Any) -> Tuple[int, int]:
        """Negotiate audio format with LLM provider.

        Queries the LLM provider's audio requirements and configures output
        audio format accordingly. If the provider doesn't specify requirements,
        uses default format (24kHz mono for Gemini compatibility).

        Args:
            agent: Agent instance containing the LLM to query for requirements

        Returns:
            Tuple of (sample_rate, channels) for the negotiated output format
        """
        # Try to get audio requirements from LLM provider
        audio_requirements: Optional[AudioCapabilities] = None
        if hasattr(agent, "llm") and hasattr(agent.llm, "get_audio_requirements"):
            audio_requirements = agent.llm.get_audio_requirements()

        if audio_requirements is not None:
            # Negotiate output format based on provider requirements
            self.output_sample_rate = audio_requirements.sample_rate
            self.output_channels = audio_requirements.channels

            logger.info(
                f"[AUDIO NEGOTIATION] Provider requires: "
                f"{audio_requirements.sample_rate}Hz, "
                f"{audio_requirements.channels}ch, "
                f"{audio_requirements.bit_depth}-bit {audio_requirements.encoding}"
            )
            logger.info(
                f"[AUDIO NEGOTIATION] Supported formats: "
                f"sample_rates={audio_requirements.supported_sample_rates}, "
                f"channels={audio_requirements.supported_channels}"
            )

            # Check if our input format is supported
            if not audio_requirements.supports_format(
                self.input_sample_rate, self.input_channels
            ):
                closest_rate = audio_requirements.get_closest_sample_rate(
                    self.input_sample_rate
                )
                closest_channels = audio_requirements.get_closest_channels(
                    self.input_channels
                )
                logger.warning(
                    f"[AUDIO FORMAT MISMATCH] Input format {self.input_sample_rate}Hz, "
                    f"{self.input_channels}ch is not directly supported by provider. "
                    f"Will use closest supported format: {closest_rate}Hz, {closest_channels}ch"
                )
        else:
            # Use configured defaults
            self.output_sample_rate = self.config.audio.output_sample_rate
            self.output_channels = self.config.audio.output_channels
            logger.info(
                f"[AUDIO NEGOTIATION] No provider requirements found, "
                f"using default output format: {self.output_sample_rate}Hz, {self.output_channels}ch"
            )

        logger.info(
            f"[AUDIO NEGOTIATION] Configured output format: "
            f"{self.output_sample_rate}Hz, "
            f"{self.output_channels}ch"
        )

        return self.output_sample_rate, self.output_channels

    def get_output_format(self) -> Tuple[int, int]:
        """Get the negotiated output format.

        Returns:
            Tuple of (sample_rate, channels) for output format.
            If not negotiated yet, returns configured defaults.
        """
        sample_rate = self.output_sample_rate or self.config.audio.output_sample_rate
        channels = self.output_channels or self.config.audio.output_channels
        return sample_rate, channels
