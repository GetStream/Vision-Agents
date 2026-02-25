from .events import DetectionCompletedEvent
from .huggingface_llm import HuggingFaceLLM as LLM
from .huggingface_vlm import HuggingFaceVLM as VLM

__all__ = ["DetectionCompletedEvent", "LLM", "VLM"]

try:
    from .transformers_llm import TransformersLLM
    from .transformers_vlm import TransformersVLM

    __all__ += ["TransformersLLM", "TransformersVLM"]

    try:
        from .transformers_detection import TransformersDetectionProcessor

        __all__ += ["TransformersDetectionProcessor"]
    except ImportError:
        pass
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
