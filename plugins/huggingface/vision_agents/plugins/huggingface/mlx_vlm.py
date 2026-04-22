"""
MlxVLM - Local vision-language model inference via Apple MLX.

Runs multimodal models on Apple Silicon using the mlx-vlm library.

Example:
    from vision_agents.plugins.huggingface import MlxVLM

    vlm = MlxVLM(model="mlx-community/gemma-4-e4b-it-8bit")
"""

import logging
from typing import Any, Optional

import av
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from ._local_vlm import LocalVLM, _extract_last_user_text

logger = logging.getLogger(__name__)


class MlxVLMResources:
    """Container for a loaded MLX VLM model and processor."""

    def __init__(self, model: Any, processor: Any, config: Any):
        self.model = model
        self.processor = processor
        self.config = config


class MlxVLM(LocalVLM[MlxVLMResources]):
    """Local VLM inference using Apple MLX.

    Runs vision-language models on Apple Silicon via the ``mlx-vlm`` library.

    Args:
        model: HuggingFace model ID (e.g. ``"mlx-community/gemma-4-e4b-it-8bit"``).
        fps: Frames per second to capture from video stream.
        frame_buffer_seconds: Seconds of frames to keep in the buffer.
        max_frames: Maximum frames to send per inference.
        max_new_tokens: Default maximum tokens to generate per response.
        max_tool_rounds: Maximum tool-call rounds per response (default 3).
    """

    _plugin_name = "mlx_vlm"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._multi_turn_warned = False
        self._tools_unsupported_warned = False

    def _load_model_sync(self) -> MlxVLMResources:
        model, processor = load(self.model_id)
        config = load_config(self.model_id)
        return MlxVLMResources(model=model, processor=processor, config=config)

    def _generate_with_frames(
        self,
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        tools_param: Optional[list[dict[str, Any]]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        assert self._resources is not None
        model = self._resources.model
        processor = self._resources.processor
        config = self._resources.config

        sampled = frames
        if len(frames) > self._max_frames:
            step = len(frames) / self._max_frames
            sampled = [frames[int(i * step)] for i in range(self._max_frames)]

        # mlx-vlm's apply_chat_template takes a single prompt string + image
        # count rather than a full message list, so multi-turn history is dropped.
        if len(messages) > 1 and not self._multi_turn_warned:
            logger.warning(
                "mlx-vlm only supports single-turn prompts; "
                "prior conversation context (%d messages) will be dropped",
                len(messages) - 1,
            )
            self._multi_turn_warned = True

        if tools_param and not self._tools_unsupported_warned:
            logger.warning(
                "mlx-vlm's apply_chat_template does not accept tools; registered functions (%d) will not be surfaced to the model",
                len(tools_param),
            )
            self._tools_unsupported_warned = True

        last_user_text = _extract_last_user_text(messages)
        images = [frame.to_image() for frame in sampled]

        prompt = apply_chat_template(
            processor,
            config,
            last_user_text,
            num_images=len(images),
        )

        result = generate(
            model,
            processor,
            prompt=prompt,
            image=images if images else None,
            max_tokens=max_tokens,
            temp=temperature,
        )

        return result.text
