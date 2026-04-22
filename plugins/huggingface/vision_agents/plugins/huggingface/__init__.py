import warnings

from .events import DetectionCompletedEvent
from .huggingface_llm import HuggingFaceLLM as LLM
from .huggingface_vlm import HuggingFaceVLM as VLM

__all__ = ["DetectionCompletedEvent", "LLM", "VLM"]

try:
    from .transformers_detection import TransformersDetectionProcessor
    from .transformers_llm import TransformersLLM
    from .transformers_vlm import TransformersVLM

    __all__ += ["TransformersDetectionProcessor", "TransformersLLM", "TransformersVLM"]
except ImportError as e:
    optional = {"torch", "transformers", "av", "aiortc", "jinja2", "supervision", "cv2"}
    if e.name in optional:
        warnings.warn(
            f"Optional dependency '{e.name}' is not installed. "
            "Install the [transformers] extra to enable Transformers plugins.",
            stacklevel=2,
        )
    else:
        raise


def _is_mlx_import_error(exc: ImportError) -> bool:
    if exc.name in {"mlx", "mlx_lm", "mlx_vlm", "mlx.core"}:
        return True
    return exc.name is None and "mlx" in str(exc).lower()


try:
    from .mlx_llm import MlxLLM

    __all__ += ["MlxLLM"]
except ImportError as e:
    if _is_mlx_import_error(e):
        warnings.warn(
            "MLX is not available on this platform. "
            "Install the [mlx] extra on Apple Silicon to enable MLX plugins.",
            stacklevel=2,
        )
    else:
        raise

try:
    from .mlx_vlm import MlxVLM

    __all__ += ["MlxVLM"]
except ImportError as e:
    if _is_mlx_import_error(e) or e.name in {"av", "aiortc"}:
        warnings.warn(
            "MLX-VLM is not available on this platform. "
            "Install the [mlx-vlm] extra on Apple Silicon to enable MLX VLM plugins.",
            stacklevel=2,
        )
    else:
        raise
