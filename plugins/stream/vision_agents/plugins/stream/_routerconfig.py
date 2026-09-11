"""What a router config is made of: option blocks, and the directory one can live in."""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import yaml
from attrs import fields

from ._backend import Backend
from ._generated import AuthenticatedClient
from ._generated.api.default import (
    create_router_config,
    list_router_configs,
    update_router_config,
)
from ._generated.models import (
    Error,
    LlmOptions,
    RouterConfig,
    RouterConfigRequest,
    RouterConfigRequestTags,
    SearchOptions,
    StsOptions,
    SttOptions,
    TtsOptions,
)
from .folder import read_stamp, write_stamp

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Block is one modality's option block, which are the same shape for a stored config, a
# start frame and a recording job.
Block = TypeVar("Block", SttOptions, TtsOptions, LlmOptions, StsOptions, SearchOptions)

# ROUTER_FILE is what makes a directory a router config, and ROUTER_STAMP is what it was
# last stored as, so a launch that changed nothing costs a file read.
ROUTER_FILE = "router.yaml"
ROUTER_STAMP = ".router_sync"

# ROUTER_KEYS are the top-level keys a router config file may hold.
ROUTER_KEYS = frozenset({"name", "description", "stt", "tts", "llm", "search", "tags"})


async def ensure_router(name: str, backend: Backend) -> Optional[RouterConfig]:
    """Store the router folder called `name`, if there is one and it has changed.

    This is what makes `Router("clinic")` enough. `.router_sync` next to `router.yaml`
    holds the fingerprint of what was last stored, so a run that changed nothing costs a
    file read rather than a request, and a name with no folder behind it is left to
    whatever is already on the server.

    Args:
        name: The stored config, which is also the folder's name.
        backend: The router to store it in.

    Returns:
        The stored config, or None when nothing needed storing.
    """
    if not name:
        return None

    path = await asyncio.to_thread(resolve_router, name)
    if path is None:
        return None

    file = path / ROUTER_FILE
    described = await asyncio.to_thread(read_router, file)
    md5 = await asyncio.to_thread(fingerprint, file)
    if await asyncio.to_thread(read_stamp, path, ROUTER_STAMP) == md5:
        logger.debug("%s is unchanged since it was last synced", path)
        return None

    described.pop("description", None)
    named = str(described.pop("name", path.name))
    client = backend.client()
    config = await store(client, wanted(named, described), await find(client, named))

    await asyncio.to_thread(write_stamp, path, ROUTER_STAMP, md5)
    logger.info("synced router config %s from %s", named, path)
    return config


def resolve_router(name: str, start: Optional[Path] = None) -> Optional[Path]:
    """The router folder called `name`, or None when there is none.

    Walks up from `start` (the current working directory by default) looking for
    `routers/{name}/router.yaml`, then for the same under an example, so a config is found
    whether it is run from its own directory or from the top of the repository.
    """
    here = (start or Path.cwd()).resolve()
    while True:
        for candidate in (
            here / "routers" / name,
            *sorted(here.glob("examples/*/*/routers/" + name)),
        ):
            if (candidate / ROUTER_FILE).is_file():
                return candidate.resolve()
        if here.parent == here:
            return None
        here = here.parent


def router_folders(folder: Path) -> list[Path]:
    """The router folders in a directory, in a fixed order so two runs do the same."""
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_dir() and (path / ROUTER_FILE).is_file()
    )


def read_router(path: Path) -> dict[str, Any]:
    """One config file, as the keywords `define_router` takes.

    A key that is not a modality is refused here rather than ignored, the same way an
    unknown option inside a block is: a misspelt `sst:` would otherwise store a config
    that transcribes at whatever the fallback happens to be.
    """
    described = yaml.safe_load(path.read_text()) or {}
    if not isinstance(described, dict):
        raise ValueError(f"{path} should describe one config, as a mapping")

    unknown = sorted(set(described) - ROUTER_KEYS)
    if unknown:
        raise ValueError(
            f"{path} names {', '.join(unknown)}, which is not "
            f"{', '.join(sorted(ROUTER_KEYS))}"
        )
    return described


def fingerprint(path: Path) -> str:
    """A fingerprint of one config file. The same file produces the same md5."""
    return hashlib.md5(path.read_text().strip().encode()).hexdigest()


def wanted(name: str, described: dict[str, Any]) -> RouterConfigRequest:
    """A config as the request that stores it, whether it came from YAML or keywords."""
    request = RouterConfigRequest(name=name)
    if described.get("stt"):
        request.stt = block(SttOptions, described["stt"])
    if described.get("tts"):
        request.tts = block(TtsOptions, described["tts"])
    if described.get("llm"):
        request.llm = block(LlmOptions, described["llm"])
    if described.get("sts"):
        request.sts = block(StsOptions, described["sts"])
    if described.get("search"):
        request.search = block(SearchOptions, described["search"])
    if described.get("tags"):
        request.tags = RouterConfigRequestTags.from_dict(described["tags"])
    return request


def block(model: type[Block], given: dict[str, Any]) -> Block:
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


async def find(client: AuthenticatedClient, name: str) -> Optional[RouterConfig]:
    """The stored config called `name`, if there is one."""
    for stored in answer(await list_router_configs.asyncio(client=client)):
        if stored.name == name:
            return stored
    return None


async def store(
    client: AuthenticatedClient,
    request: RouterConfigRequest,
    stored: Optional[RouterConfig],
) -> RouterConfig:
    """Write a config, editing the one of that name rather than adding a second."""
    if stored is not None:
        logger.info("updating router config %s", stored.id)
        return answer(
            await update_router_config.asyncio(stored.id, client=client, body=request)
        )
    return answer(await create_router_config.asyncio(client=client, body=request))


def answer(sent: Union[T, Error, None]) -> T:
    """Return what the router sent, raising what it said went wrong instead."""
    if isinstance(sent, Error):
        raise RuntimeError(sent.error)
    if sent is None:
        raise RuntimeError("the router did not answer")
    return sent
