import asyncio
import logging
from typing import Optional, TypeVar, Union

from vision_agents.core.harness import Skill

from ._backend import Backend
from ._generated import AuthenticatedClient
from ._generated.api.default import (
    create_agent_config,
    create_skill,
    list_agent_configs,
    list_skills,
    sync_agent as sync_agent_request,
    update_agent_config,
    update_skill,
)
from ._generated.models import (
    AgentConfig,
    AgentConfigRequest,
    Error,
    KnowledgeDocument,
    SkillRequest,
    SyncAgentRequest,
    SyncAgentResult,
)
from .folder import Folder, load, resolve

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def sync_agent(
    name: str,
    path: Optional[str] = None,
    url: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> SyncAgentResult:
    """Store an agent directory's instructions, skills and knowledge.

    Reads `examples/agents/{name}/` (or `path`) and writes what it holds to the
    acceleration server. A hash of the directory is sent with it: a second call with
    the same files does nothing.

    Args:
        name: What the agent is called, which is also its directory's name.
        path: The directory to read. Defaults to `examples/agents/{name}` walking up
            from the current working directory.
        url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
        customer_id: Who the work is billed to. Defaults to
            `STREAM_ACCELERATION_CUSTOMER_ID`.

    Returns:
        What the router stored, and whether it wrote anything.
    """
    folder = await asyncio.to_thread(_read_agent, name, path)
    client = Backend(url=url, customer_id=customer_id).client()

    skills = [
        SkillRequest(
            config_id="",
            name=skill.name,
            description=skill.description,
            instructions=skill.instructions,
            deadline_ms=int((skill.deadline_seconds or 0) * 1000),
        )
        for skill in folder.skills
    ]
    knowledge = [
        KnowledgeDocument(source=document.source, text=document.text)
        for document in folder.knowledge
    ]

    body = SyncAgentRequest(name=folder.name, hash_=folder.hash())
    if folder.instructions:
        body.instructions = folder.instructions
    if skills:
        body.skills = skills
    if knowledge:
        body.knowledge = knowledge

    result = _answer(await sync_agent_request.asyncio(client=client, body=body))
    if result.unchanged:
        logger.info("agent %s is already in sync", folder.name)
    else:
        logger.info("synced agent %s", folder.name)
    return result


async def define_agent(
    name: str,
    instructions: str = "",
    llm: str = "",
    subagent: str = "",
    stt: str = "",
    tts: str = "",
    voice: str = "",
    greeting: str = "",
    skills: Optional[list[Skill]] = None,
    knowledge: str = "",
    url: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> AgentConfig:
    """Store an agent configuration, along with the skills it names.

    A config is what a session can be created from by id, so the things worth deciding
    once are decided once: which models answer, what the agent is told to be, what it may
    hand to a slower model, and what it may look things up in.

    Both the config and its skills are found by name first, so calling this twice edits
    what is stored rather than storing another copy of it.

    Args:
        name: What the config is called, which is also how it is found again.
        instructions: The system prompt.
        llm: The model that answers.
        subagent: The slower model that runs delegated work. Without one the skills mean
            nothing, since there is nobody to hand the work to.
        stt: The model that transcribes, for a config a call will use.
        tts: The model that speaks, for a config a call will use.
        voice: A provider-specific voice id.
        greeting: Said on joining, without going through the model.
        skills: What the model may hand work to. A skill named here replaces a built-in
            of the same name.
        knowledge: The knowledge base the agent may look things up in.
        url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
        customer_id: Who the work is billed to. Defaults to
            `STREAM_ACCELERATION_CUSTOMER_ID`.

    Returns:
        The stored config, whose id is what a session names.
    """
    client = Backend(url=url, customer_id=customer_id).client()

    named = skills or []
    wanted = AgentConfigRequest(name=name)
    if instructions:
        wanted.instructions = instructions
    if llm:
        wanted.llm = llm
    if subagent:
        wanted.subagent = subagent
    if stt:
        wanted.stt = stt
    if tts:
        wanted.tts = tts
    if voice:
        wanted.voice = voice
    if greeting:
        wanted.greeting = greeting
    if named:
        wanted.skills = [skill.name for skill in named]
    if knowledge:
        wanted.knowledge_namespace = knowledge

    config: Optional[AgentConfig] = None
    for stored in _answer(await list_agent_configs.asyncio(client=client)):
        if stored.name == name:
            logger.info("updating agent config %s", stored.id)
            config = _answer(
                await update_agent_config.asyncio(stored.id, client=client, body=wanted)
            )
            break
    if config is None:
        config = _answer(await create_agent_config.asyncio(client=client, body=wanted))

    # The skills belong to the config, so they are written after it: a new agent has no id
    # to hang them off until it has been stored.
    if named:
        await define_skills(named, config.id, client)
    return config


async def define_skills(
    skills: list[Skill], config_id: str, client: AuthenticatedClient
) -> list[Skill]:
    """Store one config's skills, editing whichever is already under each name.

    Args:
        skills: What to store.
        config_id: The config they belong to, which is what a name is unique within.
        client: The router to store them in, from `Backend.client`.

    Returns:
        The skills as given, so this reads as a step in configuring an agent.
    """
    known = {
        stored.name: stored.id
        for stored in _answer(
            await list_skills.asyncio(client=client, config_id=config_id)
        )
    }

    for skill in skills:
        body = SkillRequest(
            config_id=config_id,
            name=skill.name,
            description=skill.description,
            instructions=skill.instructions,
            deadline_ms=int((skill.deadline_seconds or 0) * 1000),
        )
        if skill.name in known:
            _answer(
                await update_skill.asyncio(known[skill.name], client=client, body=body)
            )
        else:
            _answer(await create_skill.asyncio(client=client, body=body))
    logger.info("stored %d skills", len(skills))
    return skills


def _answer(answer: Union[T, Error, None]) -> T:
    """Return what the router sent, raising what it said went wrong instead."""
    if isinstance(answer, Error):
        raise RuntimeError(answer.error)
    if answer is None:
        raise RuntimeError("the router did not answer")
    return answer


def _read_agent(name: str, path: Optional[str]) -> Folder:
    """Load an agent directory, resolving `name` when no path was given."""
    return load(path or resolve(name))
