import logging
from typing import Optional, Union

from ._backend import Backend
from ._generated.api.default import resolve_target
from ._generated.models import Modality
from .llm import LLM
from .stt import STT
from .tts import TTS

logger = logging.getLogger(__name__)

# ORDER is which modality a name is tried against first. Speech models are the ones named
# by hand most often, and the model that answers is usually asked for by capability.
ORDER = (Modality.TTS, Modality.STT, Modality.LLM)


def Router(  # noqa: N802 - it is spelled like the plugins it returns
    target: str,
    url: Optional[str] = None,
    customer_id: Optional[str] = None,
    **kwargs,
) -> Union[STT, TTS, LLM]:
    """Whatever the backend routes a name to.

    A convenience for a name you know and a modality you would rather not repeat:
    `tts=stream.Router("sonic_36")` asks the router which kind of model that is and hands
    back the plugin for it. It costs a request at startup, so a pipeline that knows what it
    wants is better off saying `stream.TTS("sonic_36")`.

    Args:
        target: A `provider/model` name or a capability shortcut.
        url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
        customer_id: Who the work is billed to. Defaults to
            `STREAM_ACCELERATION_CUSTOMER_ID`.
        **kwargs: Passed on to the plugin that is chosen.

    Raises:
        ValueError: if nothing routes that name.
    """
    backend = Backend(url=url, customer_id=customer_id)
    modality = _modality_of(backend, target)

    if modality is Modality.STT:
        return STT(target, url=url, customer_id=customer_id, **kwargs)
    if modality is Modality.TTS:
        return TTS(target, url=url, customer_id=customer_id, **kwargs)
    return LLM(target, url=url, customer_id=customer_id, **kwargs)


def _modality_of(backend: Backend, target: str) -> Modality:
    """Ask the router what kind of model a name is."""
    client = backend.client()

    for modality in ORDER:
        response = resolve_target.sync_detailed(modality, target, client=client)
        candidates = response.parsed
        if isinstance(candidates, list) and candidates:
            logger.debug("%s routes as %s", target, modality)
            return modality

    raise ValueError(f"nothing routes {target!r}")
