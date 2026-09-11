import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Optional, Union

from ._backend import Backend
from ._generated import AuthenticatedClient
from ._generated.api.default import (
    get_speech,
    get_transcription,
    record_speech,
    resolve_target,
    search as search_request,
    transcribe_recording,
)
from ._generated.models import (
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
    SearchAnswer,
    Speech,
    SpeechRequest,
    SpeechRequestTags,
    StsOptions,
    SttOptions,
    Transcription,
    TranscriptionRequest,
    TranscriptionRequestTags,
    TtsOptions,
)
from ._generated.types import UNSET, Unset
from ._routerconfig import (
    ROUTER_FILE,
    ROUTER_STAMP,
    answer,
    block,
    ensure_router,
    find,
    fingerprint,
    read_router,
    router_folders,
    store,
    wanted,
)
from .folder import write_stamp
from .llm import LLM
from .sts import STS
from .stt import STT
from .tts import TTS

logger = logging.getLogger(__name__)

# ORDER is which modality a name is tried against first, for `resolve`. Speech models are
# the ones named by hand most often, and the model that answers is usually asked for by
# capability.
ORDER = (Modality.TTS, Modality.STT, Modality.STS, Modality.LLM)

# POLL is how often a recording job is asked whether it is done. Transcription runs faster
# than real time, so a feature-length recording is minutes rather than hours, and asking
# every second costs nothing next to that.
POLL = 1.0


class Router:
    """Everything the acceleration backend routes, configured once.

    A router is a config plus five namespaces. Each of the three streaming modalities has
    a `realtime()` session and a `recording()` job, a speech-to-speech conversation has
    only a `realtime()` session, and search has neither because a question and its answer
    are one round trip.

    ```python
    router = Router("healthcare")

    async with router.stt.realtime() as stt:
        ...

    transcript = await router.stt.recording("movie.mp4", diarize=True)
    hits = await router.search("perioperative antibiotic guidance", results=5)
    ```

    Everything in the named config is a default, and every keyword on a call overrides one
    field of it.

    A config that lives on disk as `routers/{name}/router.yaml` is stored on first use, so
    naming one here is all it takes to route through it.
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
        self.sts = SpeechToSpeech(self)

    async def configure_stt(self, **options) -> RouterConfig:
        """Store how this router transcribes.

        `Router("healthcare")` reads a config; this writes one, so the setup and the use
        of it are the same object rather than two names for it. The other three modalities
        are left as they were stored, since configuring how something is heard is not a
        statement about how it speaks.

        ```python
        await Router("healthcare").configure_stt(
            providers=["deepgram", "parakeet"],
            data_policy={"allow_training": False, "retention": "none"},
            profanity_filter=True,
        )
        ```

        Args:
            **options: Any field of the stt block - `providers`, `target`, `languages`,
                `mode`, `profanity_filter`, `data_policy`, `overwrites`, `diarize`,
                `keyterms`, `endpointing`, `redact`.

        Returns:
            The stored config.

        Raises:
            ValueError: if this router was not named, or an option is not one
                transcription takes.
            RuntimeError: if the router refuses the config, which is what it does with a
                provider it does not have or a data policy nothing it offers can meet.
        """
        if not self.config:
            raise ValueError(
                "configure_stt writes a named config, so the router needs a name: "
                'Router("healthcare").configure_stt(...)'
            )

        client = self.client()
        stored = await find(client, self.config)

        wanted = RouterConfigRequest(name=self.config, stt=block(SttOptions, options))
        # Carried forward rather than restated: a config is one row, and writing the
        # speech half of it should not silently drop the voice half.
        if stored is not None:
            wanted.tts, wanted.llm, wanted.search, wanted.sts = (
                stored.tts,
                stored.llm,
                stored.search,
                stored.sts,
            )
            if not isinstance(stored.tags, Unset):
                wanted.tags = RouterConfigRequestTags.from_dict(stored.tags.to_dict())
        if self.tags:
            wanted.tags = RouterConfigRequestTags.from_dict(self.tags)

        return await store(client, wanted, stored)

    async def configure_sts(self, **options) -> RouterConfig:
        """Store how this router holds a conversation with one native audio model.

        ```python
        await Router("healthcare").configure_sts(
            target="sts-fast",
            voice="Kore",
            data_policy={"allow_training": False},
        )
        ```

        Args:
            **options: Any field of the sts block - `target`, `providers`, `instructions`,
                `voice`, `languages`, `turn_detection`, `silence_ms`, `interrupt_response`,
                `input_transcript`, `output_transcript`, `tools`, `text`, `images`,
                `data_policy`, `overwrites`.

        Returns:
            The stored config.

        Raises:
            ValueError: if this router was not named, or an option is not one a
                speech-to-speech model takes.
            RuntimeError: if the router refuses the config, which is what it does with a
                provider it does not have or a term nothing it offers can serve.
        """
        if not self.config:
            raise ValueError(
                "configure_sts writes a named config, so the router needs a name: "
                'Router("healthcare").configure_sts(...)'
            )

        client = self.client()
        stored = await find(client, self.config)

        wanted = RouterConfigRequest(name=self.config, sts=block(StsOptions, options))
        if stored is not None:
            wanted.stt, wanted.tts, wanted.llm, wanted.search = (
                stored.stt,
                stored.tts,
                stored.llm,
                stored.search,
            )
            if not isinstance(stored.tags, Unset):
                wanted.tags = RouterConfigRequestTags.from_dict(stored.tags.to_dict())
        if self.tags:
            wanted.tags = RouterConfigRequestTags.from_dict(self.tags)

        return await store(client, wanted, stored)

    async def configure_tts(self, **options) -> RouterConfig:
        """Store how this router speaks.

        The voice half of `configure_stt`, on the same terms: the other three modalities
        are left as they were stored.

        ```python
        await Router("healthcare").configure_tts(
            providers=["elevenlabs", "en-low-latency"],
            voice="custom:receptionist",
            data_policy={"allow_training": False, "retention": "none"},
        )
        ```

        Args:
            **options: Any field of the tts block - `providers`, `target`, `voice`,
                `languages`, `speed`, `emotion`, `format`, `data_policy`, `overwrites`,
                `pronunciations`.

        Returns:
            The stored config.

        Raises:
            ValueError: if this router was not named, or an option is not one speech
                takes.
            RuntimeError: if the router refuses the config, which is what it does with a
                provider it does not have or a data policy nothing it offers can meet.
        """
        if not self.config:
            raise ValueError(
                "configure_tts writes a named config, so the router needs a name: "
                'Router("healthcare").configure_tts(...)'
            )

        client = self.client()
        stored = await find(client, self.config)

        wanted = RouterConfigRequest(name=self.config, tts=block(TtsOptions, options))
        if stored is not None:
            wanted.stt, wanted.llm, wanted.search, wanted.sts = (
                stored.stt,
                stored.llm,
                stored.search,
                stored.sts,
            )
            if not isinstance(stored.tags, Unset):
                wanted.tags = RouterConfigRequestTags.from_dict(stored.tags.to_dict())
        if self.tags:
            wanted.tags = RouterConfigRequestTags.from_dict(self.tags)

        return await store(client, wanted, stored)

    async def search(self, query: str, **options) -> SearchAnswer:
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
        await ensure_router(self.config, self.backend)
        body = SearchRequest(query=query, options=block(SearchOptions, options))
        if self.config:
            body.config_id = self.config
        if self.tags:
            body.tags = SearchRequestTags.from_dict(self.tags)

        return answer(await search_request.asyncio(client=self.client(), body=body))

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
    sts: Optional[dict[str, Any]] = None,
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
        sts: How it holds a conversation with one native audio model.
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
    request = wanted(
        name,
        {
            "stt": stt,
            "tts": tts,
            "llm": llm,
            "sts": sts,
            "search": search,
            "tags": tags,
        },
    )
    return await store(client, request, await find(client, name))


async def sync_routers(
    directory: Union[str, Path],
    url: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> list[RouterConfig]:
    """Store every router config a directory of router folders describes.

    The same bargain as `sync_agent`, for routing: a config that lives in the repository
    is one that can be reviewed, and one written by hand at a call site is not. Each
    `{name}/router.yaml` is one config, named by its `name` key or by the folder it is in,
    and each is written by name, so running this twice edits rather than duplicates.

    `Router("healthcare")` stores its own folder on first use, so this is for storing a
    whole directory of them at once, ahead of anything routing through them.

    ```yaml
    # routers/healthcare/router.yaml
    tags:
      team: clinical
    stt:
      providers: [deepgram, parakeet]
      data_policy:
        allow_training: false
        retention: none
    ```

    Args:
        directory: Where the router folders are.
        url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
        customer_id: Who the work is billed to. Defaults to
            `STREAM_ACCELERATION_CUSTOMER_ID`.

    Returns:
        The stored configs, in the order the folders were read.

    Raises:
        ValueError: if the directory holds no router folders, or one names an option a
            modality does not take.
        RuntimeError: if the router refuses one of them.
    """
    folder = Path(directory)
    found = await asyncio.to_thread(router_folders, folder)
    if not found:
        raise ValueError(f"{folder} holds no {{name}}/{ROUTER_FILE} router configs")

    stored = []
    for path in found:
        file = path / ROUTER_FILE
        described = await asyncio.to_thread(read_router, file)
        described.pop("description", None)
        stored.append(
            await define_router(
                described.pop("name", path.name),
                url=url,
                customer_id=customer_id,
                **described,
            )
        )
        md5 = await asyncio.to_thread(fingerprint, file)
        await asyncio.to_thread(write_stamp, path, ROUTER_STAMP, md5)
    logger.info("synced %d router configs from %s", len(stored), folder)
    return stored


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
            options=block(SttOptions, options).to_dict(),
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
        await ensure_router(self._router.config, self._router.backend)
        body = TranscriptionRequest(
            source=await _source(source), options=block(SttOptions, options)
        )
        if self._router.config:
            body.config_id = self._router.config
        if callback:
            body.callback = callback
        if self._router.tags:
            body.tags = TranscriptionRequestTags.from_dict(self._router.tags)

        client = self._router.client()
        job = answer(await transcribe_recording.asyncio(client=client, body=body))
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
            **options: Any field of the config's tts block - `target`, `providers`,
                `voice`, `languages`, `speed`, `emotion`, `stability`, `format`,
                `data_policy`, `overwrites`. A voice named `custom:receptionist` is one of
                your own and nothing else; a bare name is looked for among yours and
                passed to the provider's library when it is not there.

        Raises:
            ValueError: if an option is not one a voice takes.
        """
        return TTS(
            config_id=self._router.config,
            options=block(TtsOptions, options).to_dict(),
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
            **options: Any field of the config's tts block - `providers`, `voice`,
                `format`, `speed`, `stability`, `data_policy`, `overwrites`. A voice named
                `custom:reader` is one of your own and nothing else.

        Returns:
            The audio, or the accepted job when a callback was given.

        Raises:
            ValueError: if an option is not one a voice takes.
            RuntimeError: if the job failed.
        """
        await ensure_router(self._router.config, self._router.backend)
        body = SpeechRequest(text=text, options=block(TtsOptions, options))
        if self._router.config:
            body.config_id = self._router.config
        if callback:
            body.callback = callback
        if self._router.tags:
            body.tags = SpeechRequestTags.from_dict(self._router.tags)

        client = self._router.client()
        job = answer(await record_speech.asyncio(client=client, body=body))
        if callback:
            return job
        return await _until_done(job, lambda: get_speech.asyncio(job.id, client=client))


class SpeechToSpeech:
    """A conversation with one native audio model.

    There is no `recording()` here: a conversation is live or it is not one.
    """

    def __init__(self, router: Router):
        self._router = router

    def realtime(self, **options) -> STS:
        """A speech-to-speech session, configured and not yet started.

        Hand it to an `Agent` as its `llm`: it is a `Realtime`, so the agent runs no
        transcriber, turn detector or voice of its own, and the model on the other end of
        the socket does all three.

        Args:
            **options: Any field of the config's sts block - `target`, `instructions`,
                `voice`, `languages`, `turn_detection`, `silence_ms`, `interrupt_response`,
                `input_transcript`, `output_transcript`, `tools`, `text`, `images`.

        Raises:
            ValueError: if an option is not one a speech-to-speech model takes.
        """
        return STS(
            config_id=self._router.config,
            options=block(StsOptions, options).to_dict(),
            tags=self._router.tags,
            url=self._router.backend.url,
            customer_id=self._router.backend.customer_id,
        )


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
            options=block(LlmOptions, options).to_dict(),
            tags=self._router.tags,
            url=self._router.backend.url,
            customer_id=self._router.backend.customer_id,
        )


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
        job = answer(await ask())

    if job.status is RecordingStatus.FAILED:
        raise RuntimeError(_value(job.error) or "the recording failed")
    return job


def _value(held: Union[str, Unset, None]) -> str:
    """A field that may not have been set."""
    if held is UNSET or held is None:
        return ""
    return str(held)
