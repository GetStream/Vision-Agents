from .huggingface_llm import HuggingFaceLLM as LLM
from .huggingface_vlm import HuggingFaceVLM as VLM

__all__ = ["LLM", "VLM"]

try:
    from .transformers_llm import TransformersLLM
    from .transformers_vlm import TransformersVLM

    __all__ += ["TransformersLLM", "TransformersVLM"]
except ImportError as e:
    if e.name not in ("torch", "transformers", "av", "aiortc", "jinja2"):
        import warnings

        warnings.warn(
            f"Failed to import Transformers plugins: {e}. "
            "If you installed the [transformers] extra, this may indicate a broken installation.",
            stacklevel=2,
        )
