"""
MlxVLM - Local vision-language model inference via Apple MLX.

Runs multimodal models on Apple Silicon using the mlx-vlm library.

Example:
    from vision_agents.plugins.huggingface import MlxVLM

    vlm = MlxVLM(model="mlx-community/gemma-4-e4b-it-8bit")
"""

import logging
import os
import tempfile
from typing import Any, Optional

import av
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from ._local_vlm import LocalVLM

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

        all_frames = list(frames)
        if len(all_frames) > self._max_frames:
            step = len(all_frames) / self._max_frames
            all_frames = [all_frames[int(i * step)] for i in range(self._max_frames)]

        # mlx-vlm's apply_chat_template accepts a single prompt string + image
        # count rather than a full message list, so only the last user text is
        # passed. Multi-turn context is not supported by this API.
        if len(messages) > 1 and not self._multi_turn_warned:
            logger.warning(
                "mlx-vlm only supports single-turn prompts; "
                "prior conversation context (%d messages) will be dropped",
                len(messages) - 1,
            )
            self._multi_turn_warned = True

        last_user_text = "Describe what you see."
        if messages:
            last_content = messages[-1].get("content", "")
            if isinstance(last_content, str):
                last_user_text = last_content
            elif isinstance(last_content, list):
                for item in last_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        last_user_text = item.get("text", last_user_text)
                        break

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths: list[str] = []
            for i, frame in enumerate(all_frames):
                img = frame.to_image()
                path = os.path.join(tmpdir, f"frame_{i}.png")
                img.save(path, format="PNG")
                image_paths.append(path)

            prompt = apply_chat_template(
                processor,
                config,
                last_user_text,
                num_images=len(image_paths),
            )

            result = generate(
                model,
                processor,
                prompt=prompt,
                image=image_paths if image_paths else None,
                max_tokens=max_tokens,
                temp=temperature,
            )

        return result.text if hasattr(result, "text") else str(result)
