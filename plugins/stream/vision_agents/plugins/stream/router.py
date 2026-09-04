import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from attrs import fields

from ._backend import Backend
from ._generated import AuthenticatedClient
from ._generated.api.default import (
    create_router_config,
    get_speech,
    get_transcription,
    list_router_configs,
    record_speech,
    resolve_target,
    search as search_request,
    transcribe_recording,
    update_router_config,
)
from ._generated.models import (
    Error,
    LlmOptions,
    Modality,
    RecordingSource,
    RecordingStatus,
    RouterConfig,
    RouterConfigRequest,
    RouterConfigRequestTags,
    SearchOptions,
    SearchRequest,
    SearchRequestTags,
    SearchResponse,
    Speech,
    SpeechRequest,
    SpeechRequestTags,
    SttOptions,
    Transcription,
    TranscriptionRequest,
    TranscriptionRequestTags,
    TtsOptions,
)
from ._generated.types import UNSET, Unset
from .llm import LLM
from .stt import STT
from .tts import TTS

logger = logging.getLogger(__name__)

# Block is one modality's option block, which are the same shape for a stored config, a
# start frame and a recording job.
Block = TypeVar("Block", SttOptions, TtsOptions, LlmOptions, SearchOptions)

# ORDER is which modality a name is tried against first, for `resolve`. Speech models are
# the ones named by hand most often, and the model that answers is usually asked for by
# capability.
ORDER = (Modality.TTS, Modality.STT, Modality.LLM)

# POLL is how often a recording job is asked whether it is done. Transcription runs faster
# than real time, so a feature-length recording is minutes rather than hours, and asking
# every second costs nothing next to that.
POLL = 1.0


class Router:
    """Everything the acceleration backend routes, configured once.

    A router is a config plus four namespaces. Each of the three streaming modalities has
    a `realtime()` session and a `recording()` job, and search has neither because a
    question and its answer are one round trip.

    ```python
    router = Router("healthcare")

    async with router.stt.realtime() as stt:
        ...

    transcript = await router.stt.recording("movie.mp4", diarize=True)
    hits = await router.search("perioperative antibiotic guidance", results=5)
    ```

    Everything in the named config is a default, and every keyword on a call overrides one
    field of it.
    """

    def __init__(
        self,
        config: str = "",
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Route through `config`.

        Args:
            config: A stored router config, by name or by id. Without one every call says
                what it wants for itself.
            tags: Cost labels carried onto everything routed here, on top of the config's
                own.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        self.config = config
        self.tags = tags or {}
        self.backend = Backend(url=url, customer_id=customer_id)

        self.stt = SpeechToText(self)
        self.tts = TextToSpeech(self)
        self.llm = Completions(self)

    async def search(self, query: str, **options) -> SearchResponse:
        """Answer `query` out of what is true now.

        Args:
            query: The question, in your own words.
            **options: Any field of the config's search block - `depth`, `results`,
                `include_domains`, `category`, `max_age_hours`, `location`, `contents`.

        Returns:
            What was found, and the provider's own answer where it wrote one.

        Raises:
            ValueError: if an option is not one search takes.
            RuntimeError: if no provider could answer.
        """
        body = SearchRequest(query=query, options=_block(SearchOptions, options))
        if self.config:
            body.config_id = self.config
        if self.tags:
            body.tags = SearchRequestTags.from_dict(self.tags)

        return _answer(await search_request.asyncio(client=self.client(), body=body))

    def resolve(self, target: str, **kwargs) -> Union[STT, TTS, LLM]:
        """Whatever the backend routes a name to.

        For a name you know and a modality you would rather not repeat:
        `tts=router.resolve("sonic_36")` asks which kind of model that is and hands back
        the session for it. It costs a request, so a pipeline that knows what it wants is
        better off saying `router.tts.realtime(target="sonic_36")`.

        Args:
            target: A `provider/model` name or a capability shortcut.
            **kwargs: Passed on to the session that is chosen.

        Raises:
            ValueError: if nothing routes that name.
        """
        modality = self._modality_of(target)
        if modality is Modality.STT:
            return self.stt.realtime(target=target, **kwargs)
        if modality is Modality.TTS:
            return self.tts.realtime(target=target, **kwargs)
        return self.llm.realtime(target=target, **kwargs)

    def client(self) -> AuthenticatedClient:
        """An HTTP client for the router this is configured against."""
        return self.backend.client()

    def _modality_of(self, target: str) -> Modality:
        """Ask the router what kind of model a name is."""
        client = self.client()
        for modality in ORDER:
            response = resolve_target.sync_detailed(modality, target, client=client)
            candidates = response.parsed
            if isinstance(candidates, list) and candidates:
                logger.debug("%s routes as %s", target, modality)
                return modality
        raise ValueError(f"nothing routes {target!r}")


async def define_router(
    name: str,
    stt: Optional[dict[str, Any]] = None,
    tts: Optional[dict[str, Any]] = None,
    llm: Optional[dict[str, Any]] = None,
    search: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
    url: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> RouterConfig:
    """Store the routing options a `Router` can then be named after.

    A config is what makes "this is the healthcare setup" a thing you say once rather than
    a set of keywords repeated at every call site. It is found by name first, so calling
    this twice edits what is stored rather than storing another copy of it.

    Args:
        name: What the config is called, which is also how `Router(name)` finds it.
        stt: How it transcribes.
        tts: How it speaks.
        llm: How it answers.
        search: How it looks things up.
        tags: Cost labels carried onto everything routed under it.
        url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
        customer_id: Who the work is billed to. Defaults to
            `STREAM_ACCELERATION_CUSTOMER_ID`.

    Returns:
        The stored config.

    Raises:
        ValueError: if an option is not one that modality takes.
    """
    client = Backend(url=url, customer_id=customer_id).client()

    wanted = RouterConfigRequest(name=name)
    if stt:
        wanted.stt = _block(SttOptions, stt)
    if tts:
        wanted.tts = _block(TtsOptions, tts)
    if llm:
        wanted.llm = _block(LlmOptions, llm)
    if search:
        wanted.search = _block(SearchOptions, search)
    if tags:
        wanted.tags = RouterConfigRequestTags.from_dict(tags)

    for stored in _answer(await list_router_configs.asyncio(client=client)):
        if stored.name == name:
            logger.info("updating router config %s", stored.id)
            return _answer(
                await update_router_config.asyncio(
                    stored.id, client=client, body=wanted
                )
            )
    return _answer(await create_router_config.asyncio(client=client, body=wanted))


class SpeechToText:
    """Transcription, live or from a recording."""

    def __init__(self, router: Router):
        self._router = router

    def realtime(self, **options) -> STT:
        """A transcription session, configured and not yet started.

        `async with` starts and closes it; handing the same object to an `Agent` lets the
        agent own its lifecycle instead.

        Args:
            **options: Any field of the config's stt block - `target`, `languages`,
                `interim`, `endpointing`, `diarize`, `keyterms`, `format`, `redact`.

        Raises:
            ValueError: if an option is not one transcription takes.
        """
        return STT(
            config_id=self._router.config,
            options=_block(SttOptions, options).to_dict(),
            tags=self._router.tags,
            url=self._router.backend.url,
            customer_id=self._router.backend.customer_id,
        )

    async def recording(
        self,
        source: Union[str, Path, bytes],
        callback: str = "",
        **options,
    ) -> Transcription:
        """Transcribe a whole recording.

        This is the non-realtime form: a whole recording in, a whole transcript out, done
        by the batch half of a vendor rather than the streaming one, which is both cheaper
        and more accurate. It waits for the job unless a `callback` is given, in which case
        it returns as soon as the job is accepted and the router calls back.

        Args:
            source: A URL, a path to a file, or the audio itself.
            callback: A URL the finished job is POSTed to.
            **options: Any field of the config's stt block - `languages`, `diarize`,
                `max_speakers`, `words`, `output` for subtitles, `redact`, `summary`,
                `entities`, `keyterms`.

        Returns:
            The transcript, or the accepted job when a callback was given.

        Raises:
            ValueError: if an option is not one transcription takes.
            RuntimeError: if the job failed.
        """
        body = TranscriptionRequest(
            source=await _source(source), options=_block(SttOptions, options)
        )
        if self._router.config:
            body.config_id = self._router.config
        if callback:
            body.callback = callback
        if self._router.tags:
            body.tags = TranscriptionRequestTags.from_dict(self._router.tags)

        client = self._router.client()
        job = _answer(await transcribe_recording.asyncio(client=client, body=body))
        if callback:
            return job
        return await _until_done(
            job, lambda: get_transcription.asyncio(job.id, client=client)
        )


class TextToSpeech:
    """A voice, live or recorded."""

    def __init__(self, router: Router):
        self._router = router

    def realtime(self, **options) -> TTS:
        """A speaking session, configured and not yet started.

        Args:
            **options: Any field of the config's tts block - `target`, `voice`,
                `languages`, `speed`, `emotion`, `stability`, `format`.

        Raises:
            ValueError: if an option is not one a voice takes.
        """
        return TTS(
            config_id=self._router.config,
            options=_block(TtsOptions, options).to_dict(),
            tags=self._router.tags,
            url=self._router.backend.url,
            customer_id=self._router.backend.customer_id,
        )

    async def recording(
        self,
        text: str,
        callback: str = "",
        **options,
    ) -> Speech:
        """Speak a whole text into one file.

        Nothing is listening to an audiobook while it is being made, so this asks for the
        file rather than the stream, which is what lets a codec and a bitrate be chosen.

        Args:
            text: What to say, in whole paragraphs.
            callback: A URL the finished job is POSTed to.
            **options: Any field of the config's tts block - `voice`, `format`, `speed`,
                `stability`.

        Returns:
            The audio, or the accepted job when a callback was given.

        Raises:
            ValueError: if an option is not one a voice takes.
            RuntimeError: if the job failed.
        """
        body = SpeechRequest(text=text, options=_block(TtsOptions, options))
        if self._router.config:
            body.config_id = self._router.config
        if callback:
            body.callback = callback
        if self._router.tags:
            body.tags = SpeechRequestTags.from_dict(self._router.tags)

        client = self._router.client()
        job = _answer(await record_speech.asyncio(client=client, body=body))
        if callback:
            return job
        return await _until_done(job, lambda: get_speech.asyncio(job.id, client=client))


class Completions:
    """The model that answers.

    There is no `recording()` here. A completion is already whole by the time it is
    returned, and what the socket buys is the answer arriving as it is written.
    """

    def __init__(self, router: Router):
        self._router = router

    def realtime(self, **options) -> LLM:
        """An answering session, configured and not yet started.

        Args:
            **options: Any field of the config's llm block - `target`, `instructions`,
                `max_output_tokens`, `temperature`, `reasoning_effort`, `format`,
                `verbosity`, `tool_choice`.

        Raises:
            ValueError: if an option is not one the model takes.
        """
        return LLM(
            config_id=self._router.config,
            options=_block(LlmOptions, options).to_dict(),
            tags=self._router.tags,
            url=self._router.backend.url,
            customer_id=self._router.backend.customer_id,
        )


def _block(model: type[Block], given: dict[str, Any]) -> Block:
    """Turn keywords into one modality's option block.

    An option the modality does not have is refused here rather than sent and ignored,
    which is the same bargain the backend makes with a provider that cannot express a
    term: better to be told than to be answered wrongly.
    """
    allowed = {
        field.name.rstrip("_")
        for field in fields(model)
        if field.name != "additional_properties"
    }
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} is not something {model.__name__} takes; "
            f"it has {', '.join(sorted(allowed))}"
        )
    named = {name: value for name, value in given.items() if value is not None}
    return model.from_dict(named)


async def _source(source: Union[str, Path, bytes]) -> RecordingSource:
    """What to transcribe, however it was handed over.

    A URL is passed on for the provider to fetch, since that is what makes a long
    recording somebody else's bandwidth. A path is read and sent inline, and so are bytes.
    """
    if isinstance(source, bytes):
        return RecordingSource(audio=base64.b64encode(source).decode())
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return RecordingSource(url=source)

    path = Path(source)
    if not await asyncio.to_thread(_is_file, path):
        raise ValueError(
            f"{str(source)[:80]!r} is neither a URL nor a file that exists"
        )
    audio = await asyncio.to_thread(path.read_bytes)
    return RecordingSource(audio=base64.b64encode(audio).decode())


def _is_file(path: Path) -> bool:
    """Whether this names a file, for anything a caller might have handed over.

    `Path.is_file` raises rather than answers for a name the filesystem will not even
    consider, which is what base64 audio passed as a string looks like. That is still just
    "not a file", and the caller is better told what they gave than told its errno.
    """
    try:
        return path.is_file()
    except OSError:
        return False


async def _until_done(job, ask):
    """Ask about a job until it has stopped being one.

    Polling rather than waiting on the response is what the endpoint is: a job outlives
    the request that created it, which is the whole reason it is a job.
    """
    while job.status in (RecordingStatus.QUEUED, RecordingStatus.RUNNING):
        await asyncio.sleep(POLL)
        job = _answer(await ask())

    if job.status is RecordingStatus.FAILED:
        raise RuntimeError(_value(job.error) or "the recording failed")
    return job


def _answer(answer):
    """Return what the router sent, raising what it said went wrong instead."""
    if isinstance(answer, Error):
        raise RuntimeError(answer.error)
    if answer is None:
        raise RuntimeError("the router did not answer")
    return answer


def _value(held: Union[str, Unset, None]) -> str:
    """A field that may not have been set."""
    if held is UNSET or held is None:
        return ""
    return str(held)
