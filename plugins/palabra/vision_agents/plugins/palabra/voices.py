"""Palabra AI voice cloning.

Docs: https://platform.palabra.ai/docs/assets/voices

Cloning is a three step dance: ``POST /saas/voice/clone`` reserves a voice and
returns a presigned upload target, the sample is uploaded there, and the voice
then moves through ``created -> pending -> ready``. ``Voices.clone`` runs all
three and hands back a ``voice_id`` usable as ``TTS(voice_id=...)``.
"""

import asyncio
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.palabra.ai"

# Palabra's documented sample limits: at least 30 seconds of clean, single
# speaker audio, at most 10 MB.
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
SUPPORTED_SAMPLE_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".webm",
    ".mp4",
    ".mpeg",
    ".mpg",
}

READY = "ready"
FAILED = "failed"


class PalabraVoiceError(Exception):
    """Raised when the Palabra voice API rejects a request or cloning fails."""


@dataclass
class ClonedVoice:
    """A cloned voice. ``voice_id`` is what ``TTS(voice_id=...)`` expects."""

    voice_id: str
    name: str
    processing_status: str
    errors: list[str]
    warnings: list[str]

    @property
    def ready(self) -> bool:
        """True once the voice can be used for synthesis."""
        return self.processing_status == READY


@dataclass
class VoiceLimits:
    """Cloned voice quota for the account."""

    total: int
    limit: int
    remaining: int
    ready: int
    pending: int
    failed: int


def _read_sample(path: Path) -> tuple[bytes, str]:
    """Validate a sample and read it.

    Runs in a worker thread: samples are up to 10 MB and the event loop is on a
    realtime audio path.

    Returns:
        The file contents and its MIME type.
    """
    if not path.is_file():
        raise ValueError(f"Voice sample not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SAMPLE_SUFFIXES:
        raise ValueError(
            f"Unsupported voice sample format '{suffix}'. "
            f"Expected one of: {sorted(SUPPORTED_SAMPLE_SUFFIXES)}"
        )

    size = path.stat().st_size
    if size > MAX_SAMPLE_BYTES:
        raise ValueError(
            f"Voice sample is {size} bytes; Palabra accepts at most {MAX_SAMPLE_BYTES}"
        )

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.read_bytes(), mime_type


def _parse_voice(data: dict) -> ClonedVoice:
    result = data.get("processing_result") or {}
    return ClonedVoice(
        voice_id=data["voice_id"],
        name=data.get("name", ""),
        processing_status=data.get("processing_status", ""),
        errors=list(result.get("errors") or []),
        warnings=list(result.get("warnings") or []),
    )


class Voices:
    """Client for Palabra's cloned voice API.

    Example:
        >>> async with palabra.Voices() as voices:
        ...     voice = await voices.clone("Narrator", "sample.wav")
        >>> tts = palabra.TTS(voice_id=voice.voice_id)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = API_BASE_URL,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the voice API client.

        Args:
            api_key: Palabra API key. Falls back to ``PALABRA_API_KEY`` env var.
            base_url: Palabra REST API base URL.
            client: Optional pre-configured ``httpx.AsyncClient``.
        """
        api_key = api_key or os.getenv("PALABRA_API_KEY")
        if not api_key:
            raise ValueError(
                "PALABRA_API_KEY env var or api_key parameter required for Palabra voices"
            )

        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None

    async def clone(
        self,
        name: str,
        sample: str | Path,
        *,
        lang_code: str = "en",
        description: Optional[str] = None,
        denoise: bool = False,
        speech_normalization: bool = True,
        labels: Optional[dict[str, str]] = None,
        wait: bool = True,
        timeout: float = 300.0,
        poll_interval: float = 3.0,
    ) -> ClonedVoice:
        """Clone a voice from an audio sample.

        Args:
            name: Name to file the voice under.
            sample: Path to the audio or video sample. Palabra wants at least
                30 seconds of clean, single speaker audio, at most 10 MB.
            lang_code: Language spoken in the sample (e.g. ``en``, ``uk``).
            description: Optional free-form description.
            denoise: Ask Palabra to denoise the sample before training.
            speech_normalization: Normalize loudness of the sample.
            labels: Optional metadata, e.g. ``{"gender": "Female"}``. Used for
                filtering in ``list``.
            wait: Poll until the voice is ready (or failed) before returning.
            timeout: Seconds to wait for processing when ``wait`` is True.
            poll_interval: Seconds between status polls.

        Returns:
            The cloned voice. When ``wait`` is False it is still processing and
            ``ready`` will be False.

        Raises:
            PalabraVoiceError: If the API rejects the request, processing fails,
                or processing does not finish within ``timeout``.
        """
        path = Path(sample)
        data, mime_type = await asyncio.to_thread(_read_sample, path)

        payload: dict[str, object] = {
            "name": name,
            "samples": [
                {
                    "filename": path.name,
                    "mime_type": mime_type,
                    "lang_code": lang_code,
                    "speech_normalization": speech_normalization,
                    "denoise": denoise,
                }
            ],
        }
        if description is not None:
            payload["description"] = description
        if labels is not None:
            payload["labels"] = labels

        # The API wraps request bodies in a `data` envelope, mirroring responses.
        created = await self._request(
            "POST", "/saas/voice/clone", json={"data": payload}
        )
        voice = _parse_voice(created)

        samples = created.get("samples") or []
        if not samples:
            raise PalabraVoiceError(
                f"Palabra returned no upload target for voice {voice.voice_id}"
            )
        await self._upload_sample(samples[0], path.name, data, mime_type)
        logger.debug("Uploaded sample for Palabra voice %s", voice.voice_id)

        if not wait:
            return voice
        return await self._wait_until_processed(voice.voice_id, timeout, poll_interval)

    async def get(self, voice_id: str) -> ClonedVoice:
        """Fetch a single voice, including its processing status."""
        return _parse_voice(await self._request("GET", f"/saas/voice/m/{voice_id}"))

    async def list(
        self,
        *,
        search: Optional[str] = None,
        lang: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> list[ClonedVoice]:
        """List cloned voices.

        Args:
            search: Case-insensitive name search (at least 2 characters).
            lang: Filter by sample language code.
            page_size: Voices per page (1–100). Only the first page is returned.
        """
        params: dict[str, str | int] = {}
        if search is not None:
            params["search"] = search
        if lang is not None:
            params["lang"] = lang
        if page_size is not None:
            params["page_size"] = page_size

        data = await self._request("GET", "/saas/voice", params=params)
        return [_parse_voice(item) for item in data.get("items") or []]

    async def delete(self, voice_id: str) -> None:
        """Permanently delete a cloned voice."""
        await self._request("DELETE", f"/saas/voice/m/{voice_id}")

    async def limits(self) -> VoiceLimits:
        """Return the account's cloned voice quota."""
        data = await self._request("GET", "/saas/voice/limits")
        return VoiceLimits(
            total=data["total"],
            limit=data["limit"],
            remaining=data["remaining"],
            ready=data["ready"],
            pending=data["pending"],
            failed=data["failed"],
        )

    async def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    async def _upload_sample(
        self, sample: dict, filename: str, data: bytes, mime_type: str
    ) -> None:
        """POST the sample to the presigned target Palabra handed back."""
        url = sample.get("url")
        if not url:
            raise PalabraVoiceError("Palabra upload target is missing its URL")

        response = await self._client.post(
            url,
            data=sample.get("form_data") or {},
            files={"file": (filename, data, mime_type)},
        )
        if response.is_error:
            raise PalabraVoiceError(
                f"Uploading the voice sample failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

    async def _wait_until_processed(
        self, voice_id: str, timeout: float, poll_interval: float
    ) -> ClonedVoice:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            voice = await self.get(voice_id)
            if voice.ready:
                logger.debug("Palabra voice %s is ready", voice_id)
                return voice
            if voice.processing_status == FAILED:
                raise PalabraVoiceError(
                    f"Palabra failed to clone voice {voice_id}: "
                    f"{'; '.join(voice.errors) or 'no reason given'}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise PalabraVoiceError(
                    f"Palabra voice {voice_id} was still '{voice.processing_status}' "
                    f"after {timeout:.0f}s"
                )
            await asyncio.sleep(poll_interval)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Call the Palabra REST API and unwrap its ``{ok, data}`` envelope."""
        response = await self._client.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            params=params,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        if response.is_error:
            raise PalabraVoiceError(
                f"Palabra {method} {path} failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        body = response.json()
        if not body.get("ok", False):
            raise PalabraVoiceError(f"Palabra {method} {path} returned {body}")
        return body.get("data") or {}

    async def __aenter__(self) -> "Voices":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
