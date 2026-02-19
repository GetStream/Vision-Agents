from .huggingface_llm import HuggingFaceLLM as LLM
from .huggingface_vlm import HuggingFaceVLM as VLM

__all__ = ["LLM", "VLM"]

try:
    from .transformers_llm import TransformersLLM
    from .transformers_vlm import TransformersVLM

    __all__ += ["TransformersLLM", "TransformersVLM"]
except ImportError as e:
    import warnings

    optional = {"torch", "transformers", "av", "aiortc", "jinja2"}
    if e.name in optional:
        warnings.warn(
            f"Optional dependency '{e.name}' is not installed. "
            "Install the [transformers] extra to enable Transformers plugins.",
            stacklevel=2,
        )
    else:
        raise
