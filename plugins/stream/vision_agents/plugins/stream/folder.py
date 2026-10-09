"""An agent written down as a directory of instructions, skills and knowledge."""

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import yaml
from vision_agents.core.harness import Skill

AGENT_FILE = "agent.yaml"
AGENT_STAMP = ".agent_sync"
INSTRUCTIONS_FILE = "instructions.md"
GUARDRAIL_FILE = "guardrail.md"
SKILLS_DIR = "skills"
KNOWLEDGE_DIR = "knowledge"
KNOWLEDGE_URLS_FILE = "urls.yaml"
SIMULATIONS_DIR = "simulations"

logger = logging.getLogger(__name__)

_READABLE = {".md", ".mdx", ".txt", ".rst", ".yaml", ".yml"}
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)$")
_DURATION_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}
_GO_ESCAPES = (
    ("<", "\\u003c"),
    (">", "\\u003e"),
    ("&", "\\u0026"),
    ("\u2028", "\\u2028"),
    ("\u2029", "\\u2029"),
)


@dataclass
class Document:
    """One file from an agent's knowledge directory, as it will be ingested."""

    source: str
    text: str


@dataclass
class KnowledgeURL:
    """One page from `knowledge/urls.yaml`, which the knowledge base is kept filled from."""

    url: str
    title: str = ""
    description: str = ""
    refresh_hours: int = 0
    """How often the backend reads the page again on its own. Zero is never."""


@dataclass
class Simulation:
    """One conversation a `simulations/*.yaml` file declares, run against the agent.

    The fields are in the Go SDK's order, since the fingerprint is taken over them as JSON.
    """

    name: str
    scenario: str
    assertion: str
    mode: str = ""
    variations: int = 0
    max_turns: int = 0
    caller_target: str = ""
    judge_target: str = ""
    caller_stt: str = ""
    caller_tts: str = ""
    caller_voice: str = ""
    tags: dict[str, str] | None = None


@dataclass
class SandboxSettings:
    """How the agent's sandbox is built and how long code may run in it.

    Any of them builds an image on top of `image`, which the provider keeps, so only the
    first sandbox from a given setup waits for the build.
    """

    image: str = ""
    """The container image to start from, which must have Python. Empty is slim Python."""
    setup: list[str] = field(default_factory=list)
    """Shell commands run once on top of the image when it is built."""
    timeout_seconds: float = 0.0
    """How long one run of code may take, at most 30 minutes. Zero is 30 seconds."""
    cpu: int = 0
    memory_gb: int = 0
    disk_gb: int = 0


@dataclass
class MCPServerSettings:
    """An MCP server outside the plugin catalog, which the router opens by its URL."""

    name: str
    """What its tools are prefixed with, as `<name>__<tool>`."""
    url: str
    tools: list[str] = field(default_factory=list)
    """Offer only the tools matching these names or patterns such as `search_*`.
    Empty offers every tool."""
    scopes: list[str] = field(default_factory=list)
    """What its OAuth login asks for. Empty asks for what the server advertises.
    Without `user`, the app logs in once, on the dashboard."""
    user: bool = False
    """Each end user logs in with their own account, in the chat."""


@dataclass
class ChannelSettings:
    """One line the agent is reachable on."""

    number: str
    """The number people write to, in E.164. It must be a line the app connected."""


@dataclass
class ChannelsSettings:
    """The lines the agent answers on outside its Stream Chat channel."""

    whatsapp: ChannelSettings | None = None
    sms: ChannelSettings | None = None
    imessage: ChannelSettings | None = None
    identity: str = ""
    """How a sender becomes an end user: `phone`, the default, makes each number an end
    user of its own; `link` answers only a number somebody tied to an end user with a
    code."""


@dataclass
class PluginSettings:
    """One catalog plugin the agent names, and how the router reaches it.

    `agent.yaml` gives it as the plugin's id alone, or as a mapping naming it with the
    rest.
    """

    name: str
    user: bool = False
    """Each end user connects the plugin with their own account, in the chat, rather than
    the app once, on the dashboard, for every session."""
    readonly: bool = False
    """Reach the plugin's read-only endpoint, for a vendor that runs one."""
    scopes: list[str] = field(default_factory=list)
    """Asked for at consent in place of the catalog's. Empty keeps the catalog's."""
    toolsets: list[str] = field(default_factory=list)
    """Limit the server to these groups of tools, such as calcom's bookings. Empty
    offers every tool."""
    tools: list[str] = field(default_factory=list)
    """Offer only the tools matching these names or patterns such as `get_*`.
    Empty offers every tool."""


@dataclass
class ToolSettings:
    """How plugin, MCP server and connector tools are offered."""

    progressive: bool | None = None
    """Offer each tool by a summary, the first call to it returning its full description
    instead of running it. None when the file says nothing about it."""


@dataclass
class Settings:
    """What `agent.yaml` declares.

    The rest of the directory is what the agent is told; this is what it is run with. A
    field left out leaves whatever the config already has stored, so a model chosen in the
    dashboard survives a sync that says nothing about it.
    """

    name: str = ""
    description: str = ""
    mode: str = ""
    stt: str = ""
    tts: str = ""
    sts: str | None = None
    voice: str = ""
    speed: float = 0.0
    """The voice's rate of delivery, 1 being its own. Zero leaves it there."""
    llm: str = ""
    harness: str = ""
    subagent: str = ""
    """The model a voice agent hands its skills to. A text agent runs on its llm alone."""
    search: str = ""
    greeting: str = ""
    sandbox: str = ""
    sandbox_options: SandboxSettings | None = None
    """How the sandbox is built. None when the file says nothing about it."""
    plugins: list[PluginSettings] = field(default_factory=list)
    """Catalog MCP servers the app connects once, on the dashboard, for every session, or
    each end user with their own account, in the chat, for one with `user`."""
    mcp_servers: list[MCPServerSettings] = field(default_factory=list)
    """MCP servers outside the catalog, opened by the router by their URL, with a login
    when the server asks for one."""
    tools: ToolSettings | None = None
    """How plugin, MCP server and connector tools are offered. None when the file says
    nothing about it."""
    channels: ChannelsSettings | None = None
    """Lines the agent answers on besides Stream Chat. The provider's credentials live on
    the router, connected once for the app; this only names the numbers."""
    keyterms: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    video_source: str = ""
    video_max_frames: int = 0
    dispatch: dict[str, str] | None = None
    """What the agent leaves to this application's own dispatch worker: `incoming_call`
    and `text`, each "enabled" or "disabled". None when the file says nothing about it."""
    app: dict[str, object] = field(default_factory=dict)
    """The application's own section, which this SDK never reads and the backend is never
    sent. It is the one place an unknown key is not refused."""


@dataclass
class Folder:
    """An agent written down as a directory.

    ::

        agents/jean/
          agent.yaml
          instructions.md
          guardrail.md
          skills/think.md
          knowledge/pricing.md
          knowledge/urls.yaml
          simulations/lunch.yaml
    """

    path: Path
    name: str
    declaration: str = ""
    settings: Settings = field(default_factory=Settings)
    instructions: str = ""
    """guardrail.md, whole and unparsed. The backend parses it, so a policy this SDK has
    never heard of still reaches it. Empty means every turn is answered."""
    guardrail: str = ""
    skills: list[Skill] = field(default_factory=list)
    knowledge: list[Document] = field(default_factory=list)
    knowledge_urls: list[KnowledgeURL] = field(default_factory=list)
    simulations: list[Simulation] | None = None
    """What `simulations/*.yaml` declare. None when there is no `simulations/`, which leaves
    the stored ones alone; empty when it has none, which deletes them."""

    def knowledge_namespace(self) -> str:
        """Where the directory's knowledge is looked up, which is the agent's own name."""
        if not self.knowledge and not self.knowledge_urls:
            return ""
        return self.name

    def hash(self) -> str:
        """A fingerprint of the directory. The same files produce the same hash, and the
        Go SDK takes it the same way."""
        hasher = hashlib.md5()
        hasher.update(self.declaration.encode())
        hasher.update(b"\n")
        hasher.update(self.instructions.encode())
        hasher.update(b"\n")
        hasher.update(self.guardrail.encode())
        for skill in sorted(self.skills, key=lambda item: item.name):
            hasher.update(b"\nskill:")
            hasher.update(skill.name.encode())
            hasher.update(b"\n")
            hasher.update(skill.description.encode())
            hasher.update(b"\n")
            hasher.update(skill.instructions.encode())
            hasher.update(str(skill.capture_video).encode())
            hasher.update(b"\n")
            if skill.deadline_seconds:
                hasher.update(str(skill.deadline_seconds).encode())
        for document in sorted(self.knowledge, key=lambda item: item.source):
            hasher.update(b"\nknowledge:")
            hasher.update(document.source.encode())
            hasher.update(b"\n")
            hasher.update(document.text.encode())
        for page in self.knowledge_urls:
            hasher.update(b"\nurl:")
            hasher.update(page.url.encode())
            hasher.update(b"\n")
            hasher.update(page.title.encode())
            hasher.update(b"\n")
            hasher.update(page.description.encode())
            if page.refresh_hours:
                hasher.update(f"\nrefresh_hours:{page.refresh_hours}".encode())
        if self.simulations is not None:
            hasher.update(b"\nsimulations:")
            for simulation in self.simulations:
                hasher.update(_go_json(simulation).encode())
        return hasher.hexdigest()


def load(path: str | Path) -> Folder:
    """Read an agent directory.

    `agent.yaml` is what makes a directory an agent, so it is required. Everything else
    is optional: a directory with only instructions.md beside it is a valid agent, and
    so is one with only skills.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{root} is not an agent directory")
    declaration = root / AGENT_FILE
    if not declaration.is_file():
        raise ValueError(f"{root} has no {AGENT_FILE}, so it is not an agent directory")

    folder = Folder(path=root, name=root.name)
    folder.declaration = declaration.read_text().strip()
    folder.settings = _declare(declaration)
    folder.name = folder.settings.name or root.name
    instructions = root / INSTRUCTIONS_FILE
    if instructions.is_file():
        folder.instructions = instructions.read_text().strip()
    guardrail = root / GUARDRAIL_FILE
    if guardrail.is_file():
        folder.guardrail = guardrail.read_text().strip()
    folder.skills = _load_skills(root / SKILLS_DIR)
    folder.knowledge = _load_knowledge(root / KNOWLEDGE_DIR)
    folder.knowledge_urls = _load_knowledge_urls(
        root / KNOWLEDGE_DIR / KNOWLEDGE_URLS_FILE
    )
    folder.simulations = _load_simulations(root / SIMULATIONS_DIR)
    return folder


def resolve(name: str, start: Path | None = None) -> Path:
    """Find the agent directory called `name`.

    `name` may itself be a path. Otherwise this walks up from `start` (the current
    working directory by default) looking for it under `examples/`, then for
    `agents/{name}`, then for a directory of that name.
    """
    found = find(name, start)
    if found is None:
        raise FileNotFoundError(
            f"no agent directory called {name!r}; expected examples/voice_agents/{name}/"
            f"{AGENT_FILE} or a sibling of it"
        )
    return found


def find(name: str, start: Path | None = None) -> Path | None:
    """The agent directory called `name`, or None when there is none.

    What makes a directory an agent is `agent.yaml`: a name is looked up rather than
    guessed at, so a config that lives on the router and nowhere on disk is simply not
    found here rather than mistaken for a directory that happens to share its name.
    """
    given = Path(name)
    if given.is_dir() and _looks_like_agent(given):
        return given.resolve()

    here = (start or Path.cwd()).resolve()
    if here.name == name and _looks_like_agent(here):
        return here

    while True:
        # Every kind of example is looked under rather than only examples/voice_agents, since
        # which folder an agent was filed in says nothing about how it is loaded.
        for candidate in (
            *sorted((here / "examples").glob("*/" + name)),
            here / "agents" / name,
            here / name,
        ):
            if candidate.is_dir() and _looks_like_agent(candidate):
                return candidate.resolve()
        if here.parent == here:
            return None
        here = here.parent


def read_stamp(path: Path, filename: str) -> str:
    """The fingerprint a directory was last synced under, or empty when it never was."""
    stamp = path / filename
    if not stamp.is_file():
        return ""
    try:
        recorded = json.loads(stamp.read_text())
    except json.JSONDecodeError:
        logger.debug("%s is not readable, so the directory is synced again", stamp)
        return ""
    if not isinstance(recorded, dict):
        return ""
    return str(recorded.get("hash", ""))


def write_stamp(path: Path, filename: str, fingerprint: str) -> None:
    """Record what was synced and when, so a second launch can do nothing."""
    stamp = path / filename
    synced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stamp.write_text(json.dumps({"hash": fingerprint, "synced_at": synced_at}) + "\n")


def _declare(path: Path) -> Settings:
    """Read `agent.yaml`: what the agent is called, and what it is run with.

    A key nobody knows is refused rather than dropped, since a misspelled `llm` that goes
    quietly is a config running on a model the file does not name.
    """
    declared = yaml.safe_load(path.read_text()) or {}
    if not isinstance(declared, dict):
        raise ValueError(f"{path} should describe one agent, as a mapping")

    settings = Settings()
    for key, value in declared.items():
        field_name = str(key)
        if field_name == "name":
            settings.name = _word(value)
        elif field_name == "description":
            settings.description = _word(value)
        elif field_name == "mode":
            settings.mode = _word(value)
        elif field_name == "stt":
            settings.stt = _word(value)
        elif field_name == "tts":
            settings.tts = _word(value)
        elif field_name == "sts":
            settings.sts = _word(value)
        elif field_name == "voice":
            settings.voice = _word(value)
        elif field_name == "speed":
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"{path} should give speed as a number")
            settings.speed = float(value or 0)
        elif field_name == "llm":
            settings.llm = _word(value)
        elif field_name == "harness":
            settings.harness = _word(value)
        elif field_name == "subagent":
            settings.subagent = _word(value)
        elif field_name == "search":
            settings.search = _word(value)
        elif field_name == "greeting":
            settings.greeting = _word(value)
        elif field_name == "sandbox":
            settings.sandbox = _word(value)
        elif field_name == "sandbox_options":
            settings.sandbox_options = _sandbox_options(path, value)
        elif field_name == "plugins":
            settings.plugins = _plugins(path, field_name, value)
        elif field_name == "mcp_servers":
            settings.mcp_servers = _mcp_servers(path, value)
        elif field_name == "tools":
            settings.tools = _tools(path, value)
        elif field_name == "channels":
            settings.channels = _channels(path, value)
        elif field_name == "keyterms":
            settings.keyterms = _terms(path, field_name, value)
        elif field_name == "tags":
            settings.tags = _tags(path, value)
        elif field_name == "video":
            settings.video_source, settings.video_max_frames = _video(path, value)
        elif field_name == "dispatch":
            settings.dispatch = _dispatch(path, value)
        elif field_name == "app":
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"{path} should give app as a mapping")
            settings.app = dict(value or {})
        else:
            raise ValueError(
                f"{path} declares {field_name!r}, which is not something an agent has"
            )
    return settings


def _word(value: object) -> str:
    """One setting as text. A key with nothing after it reads as unset."""
    if value is None:
        return ""
    return str(value).strip()


def _text(value: object) -> str:
    """One setting as written rather than trimmed, the way Go reads it, for the ones a
    fingerprint is taken over field by field."""
    return "" if value is None else str(value)


def _terms(path: Path, field_name: str, value: object) -> list[str]:
    """One setting as a list of names, with the blanks left out."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path} should give {field_name} as a list")
    return [_word(item) for item in value if _word(item)]


def _tags(path: Path, value: object) -> dict[str, str]:
    """The cost labels a config carries onto every request a session makes."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give tags as a mapping of label to value")
    return {str(key): _word(item) for key, item in value.items()}


def _video(path: Path, value: object) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give video as a mapping")
    extra = set(value) - {"source", "max_frames"}
    if extra:
        raise ValueError(f"{path} unknown video setting: {sorted(extra)[0]}")
    limit = value.get("max_frames", 1)
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("video.max_frames must be an integer from 1 to 8")
    return _word(value.get("source")), limit


def _dispatch(path: Path, value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give dispatch as a mapping")
    extra = set(value) - {"incoming_call", "text"}
    if extra:
        raise ValueError(f"{path} unknown dispatch setting: {sorted(extra)[0]}")
    return {str(key): _word(item) for key, item in value.items() if _word(item)}


def _plugins(path: Path, field_name: str, value: object) -> list[PluginSettings]:
    """The catalog plugins one list names, each an id or a mapping naming it."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path} should give {field_name} as a list")
    named = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                named.append(PluginSettings(name=item.strip()))
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"{path} should give each of {field_name} as a plugin id or a mapping"
            )
        extra = set(item) - {"name", "user", "readonly", "scopes", "toolsets", "tools"}
        if extra:
            raise ValueError(f"{path} unknown {field_name} setting: {sorted(extra)[0]}")
        user = item.get("user", False)
        if not isinstance(user, bool):
            raise ValueError(f"{path} should give {field_name} user as true or false")
        readonly = item.get("readonly", False)
        if not isinstance(readonly, bool):
            raise ValueError(
                f"{path} should give {field_name} readonly as true or false"
            )
        named.append(
            PluginSettings(
                name=_word(item.get("name")),
                user=user,
                readonly=readonly,
                scopes=_terms(path, f"{field_name}.scopes", item.get("scopes")),
                toolsets=_terms(path, f"{field_name}.toolsets", item.get("toolsets")),
                tools=_terms(path, f"{field_name}.tools", item.get("tools")),
            )
        )
    return named


def _tools(path: Path, value: object) -> ToolSettings | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give tools as a mapping")
    extra = set(value) - {"progressive"}
    if extra:
        raise ValueError(f"{path} unknown tools setting: {sorted(extra)[0]}")
    progressive = value.get("progressive")
    if progressive is not None and not isinstance(progressive, bool):
        raise ValueError(f"{path} should give tools progressive as true or false")
    return ToolSettings(progressive=progressive)


def _mcp_servers(path: Path, value: object) -> list[MCPServerSettings]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path} should give mcp_servers as a list")
    servers = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{path} should give each of mcp_servers as a mapping")
        extra = set(item) - {"name", "url", "tools", "scopes", "user"}
        if extra:
            raise ValueError(f"{path} unknown mcp_servers setting: {sorted(extra)[0]}")
        user = item.get("user", False)
        if not isinstance(user, bool):
            raise ValueError(f"{path} should give mcp_servers user as true or false")
        servers.append(
            MCPServerSettings(
                name=_word(item.get("name")),
                url=_word(item.get("url")),
                tools=_terms(path, "mcp_servers.tools", item.get("tools")),
                scopes=_terms(path, "mcp_servers.scopes", item.get("scopes")),
                user=user,
            )
        )
    return servers


def _channels(path: Path, value: object) -> ChannelsSettings | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give channels as a mapping")
    extra = set(value) - {"whatsapp", "sms", "imessage", "identity"}
    if extra:
        raise ValueError(f"{path} unknown channels setting: {sorted(extra)[0]}")
    identity = _word(value.get("identity"))
    if identity and identity not in {"phone", "link"}:
        raise ValueError(f"{path} channels.identity is phone or link, not {identity}")
    return ChannelsSettings(
        whatsapp=_channel(path, "whatsapp", value.get("whatsapp")),
        sms=_channel(path, "sms", value.get("sms")),
        imessage=_channel(path, "imessage", value.get("imessage")),
        identity=identity,
    )


def _channel(path: Path, kind: str, value: object) -> ChannelSettings | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give channels.{kind} as a mapping")
    extra = set(value) - {"number"}
    if extra:
        raise ValueError(f"{path} unknown channels.{kind} setting: {sorted(extra)[0]}")
    number = _word(value.get("number"))
    if not number:
        raise ValueError(f"{path} channels.{kind} needs a number")
    return ChannelSettings(number=number)


def _sandbox_options(path: Path, value: object) -> SandboxSettings:
    if not isinstance(value, dict):
        raise ValueError(f"{path} should give sandbox_options as a mapping")
    extra = set(value) - {"image", "setup", "timeout", "cpu", "memory_gb", "disk_gb"}
    if extra:
        raise ValueError(f"{path} unknown sandbox_options setting: {sorted(extra)[0]}")
    options = SandboxSettings(
        image=_word(value.get("image")),
        setup=_terms(path, "sandbox_options.setup", value.get("setup")),
    )
    timeout = value.get("timeout")
    if timeout is not None:
        matched = _DURATION.fullmatch(str(timeout).strip())
        seconds = (
            float(matched.group(1)) * _DURATION_UNITS[matched.group(2)]
            if matched
            else 0.0
        )
        if not 0 < seconds <= 30 * 60:
            raise ValueError(
                f"sandbox_options.timeout must be a duration up to 30m, not {timeout!r}"
            )
        options.timeout_seconds = seconds
    for size in ("cpu", "memory_gb", "disk_gb"):
        given = value.get(size, 0)
        if type(given) is not int or given < 0:
            raise ValueError(f"sandbox_options.{size} must be a whole number")
    options.cpu = value.get("cpu", 0)
    options.memory_gb = value.get("memory_gb", 0)
    options.disk_gb = value.get("disk_gb", 0)
    return options


def _looks_like_agent(path: Path) -> bool:
    return (path / AGENT_FILE).is_file()


def _load_skills(path: Path) -> list[Skill]:
    if not path.is_dir():
        return []

    skills: list[Skill] = []
    for file in sorted(path.iterdir()):
        if not file.is_file() or file.suffix != ".md":
            continue
        skills.append(_parse_skill(file.stem, file.read_text()))
    return skills


def _parse_skill(name: str, content: str) -> Skill:
    skill = Skill(name=name, description="", instructions="")
    frontmatter, body, found = _cut_frontmatter(content)
    if found:
        for line in frontmatter.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"{line!r} is not a key and a value")
            value = value.strip().strip("\"'")
            if key.strip() == "name":
                skill.name = value
            elif key.strip() == "description":
                skill.description = value
            elif key.strip() == "capture_video":
                if value not in ("true", "false"):
                    raise ValueError("capture_video must be true or false")
                skill.capture_video = value == "true"
            elif key.strip() == "deadline":
                skill.deadline_seconds = _parse_deadline(value)

    skill.instructions = body.strip()
    if not skill.description:
        raise ValueError(
            "a skill needs a description, since it is all the fast model sees"
        )
    if not skill.instructions:
        raise ValueError(
            "a skill needs instructions, since they are what the subagent answers under"
        )
    return skill


def _parse_deadline(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    matched = _DURATION.fullmatch(value)
    if matched is None:
        raise ValueError(f"{value!r} is not a deadline")
    return float(matched.group(1)) * _DURATION_UNITS[matched.group(2)]


def _cut_frontmatter(content: str) -> tuple[str, str, bool]:
    trimmed = content.lstrip("\ufeff \t\r\n")
    if not trimmed.startswith("---"):
        return "", content, False

    rest = trimmed[3:].lstrip("\r\n")
    end = rest.find("\n---")
    if end < 0:
        return "", content, False
    return rest[:end], rest[end + 4 :].lstrip("-\r\n"), True


def _load_knowledge(path: Path) -> list[Document]:
    if not path.is_dir():
        return []

    documents: list[Document] = []
    for dirpath, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            file = Path(dirpath) / filename
            if file.suffix.lower() not in _READABLE:
                continue
            # The declaration of what pages to read is not itself something to look
            # things up in. Only the one at the root is; deeper, it is a document.
            if file == path / KNOWLEDGE_URLS_FILE:
                continue
            text = file.read_text()
            if not text.strip():
                continue
            source = file.relative_to(path).as_posix()
            documents.append(Document(source=source, text=text))
    documents.sort(key=lambda item: item.source)
    return documents


def _load_knowledge_urls(path: Path) -> list[KnowledgeURL]:
    """Read the pages a knowledge base is kept filled from, as urls or mappings.

    A bad url or a key nobody knows is refused here, before anything is written.
    """
    if not path.is_file():
        return []
    declared = yaml.safe_load(path.read_text()) or []
    if not isinstance(declared, list):
        raise ValueError(f"{path} should list pages")

    pages: list[KnowledgeURL] = []
    for item in declared:
        if isinstance(item, str):
            page = KnowledgeURL(url=item)
        elif isinstance(item, dict):
            extra = set(item) - {"url", "title", "description", "refresh_hours"}
            if extra:
                raise ValueError(
                    f"{path}: {sorted(extra)[0]!r} is not something a page says; "
                    "url, title, description and refresh_hours are"
                )
            hours = item.get("refresh_hours", 0)
            if "refresh_hours" in item and (type(hours) is not int or hours < 1):
                raise ValueError(
                    f"{path}: refresh_hours is how many hours between reads, so it is "
                    "at least 1; leave it out for never"
                )
            page = KnowledgeURL(
                url=_word(item.get("url")),
                title=_word(item.get("title")),
                description=_word(item.get("description")),
                refresh_hours=hours,
            )
        else:
            raise ValueError(f"{path}: a page is a url, or a mapping naming one")
        if not page.url.startswith(("http://", "https://")):
            raise ValueError(f"{path}: {page.url!r} is not an http or https url")
        pages.append(page)
    return pages


def _load_simulations(path: Path) -> list[Simulation] | None:
    """Read every .yaml and .yml file in `simulations/`, each a list of simulations.

    A key nobody knows is refused, as in `agent.yaml`, and so is a name two simulations
    share, since a sync finds a simulation by its name.
    """
    if not path.is_dir():
        return None

    known = {item.name for item in fields(Simulation)}
    simulations: list[Simulation] = []
    named: dict[str, str] = {}
    for file in sorted(path.iterdir(), key=lambda entry: entry.name):
        if not file.is_file() or file.suffix.lower() not in (".yaml", ".yml"):
            continue
        listed = yaml.safe_load(file.read_text()) or []
        if not isinstance(listed, list):
            raise ValueError(f"{file} should list simulations")

        for item in listed:
            if not isinstance(item, dict):
                raise ValueError(f"{file}: a simulation is a mapping")
            extra = set(item) - known
            if extra:
                raise ValueError(
                    f"{file}: {sorted(extra)[0]!r} is not something a simulation says"
                )
            for count in ("variations", "max_turns"):
                if item.get(count) is not None and type(item[count]) is not int:
                    raise ValueError(f"{file}: {count} should be a whole number")
            tags = item.get("tags")
            if tags is not None and not isinstance(tags, dict):
                raise ValueError(f"{file}: tags should be a mapping of label to value")

            simulation = Simulation(
                name=_text(item.get("name")),
                scenario=_text(item.get("scenario")),
                assertion=_text(item.get("assertion")),
                mode=_text(item.get("mode")),
                variations=item.get("variations") or 0,
                max_turns=item.get("max_turns") or 0,
                caller_target=_text(item.get("caller_target")),
                judge_target=_text(item.get("judge_target")),
                caller_stt=_text(item.get("caller_stt")),
                caller_tts=_text(item.get("caller_tts")),
                caller_voice=_text(item.get("caller_voice")),
                tags=None
                if tags is None
                else {str(key): _text(value) for key, value in tags.items()},
            )
            if not simulation.name:
                raise ValueError(f"{file}: a simulation needs a name")
            if not simulation.scenario:
                raise ValueError(
                    f"{file}: simulation {simulation.name!r} needs a scenario"
                )
            if not simulation.assertion:
                raise ValueError(
                    f"{file}: simulation {simulation.name!r} needs an assertion"
                )
            if simulation.mode not in ("", "text", "audio"):
                raise ValueError(
                    f"{file}: simulation {simulation.name!r} is text or audio, "
                    f"not {simulation.mode!r}"
                )
            if simulation.name in named:
                raise ValueError(
                    f"{file}: simulation {simulation.name!r} is also declared in "
                    f"{named[simulation.name]}"
                )
            named[simulation.name] = file.name
            simulations.append(simulation)
    return simulations


def _go_json(simulation: Simulation) -> str:
    """A simulation as Go's `json.Marshal` writes it, so both SDKs fingerprint it alike:
    fields in order, map keys sorted, and `<`, `>` and `&` escaped."""
    encoded = asdict(simulation)
    if simulation.tags is not None:
        encoded["tags"] = dict(sorted(simulation.tags.items()))
    text = json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))
    for raw, escaped in _GO_ESCAPES:
        text = text.replace(raw, escaped)
    return text
