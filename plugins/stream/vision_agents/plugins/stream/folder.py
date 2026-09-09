"""An agent written down as a directory of instructions, skills and knowledge."""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from vision_agents.core.harness import Skill

AGENT_FILE = "agent.yaml"
AGENT_STAMP = ".agent_sync"
INSTRUCTIONS_FILE = "instructions.md"
SKILLS_DIR = "skills"
KNOWLEDGE_DIR = "knowledge"

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


@dataclass
class Document:
    """One file from an agent's knowledge directory, as it will be ingested."""

    source: str
    text: str


@dataclass
class Folder:
    """An agent written down as a directory.

    ::

        agents/jean/
          agent.yaml
          instructions.md
          skills/think.md
          knowledge/pricing.md
    """

    path: Path
    name: str
    declaration: str = ""
    instructions: str = ""
    skills: list[Skill] = field(default_factory=list)
    knowledge: list[Document] = field(default_factory=list)

    def knowledge_namespace(self) -> str:
        """Where the directory's knowledge is looked up, which is the agent's own name."""
        if not self.knowledge:
            return ""
        return self.name

    def hash(self) -> str:
        """A fingerprint of the directory. The same files produce the same hash."""
        hasher = hashlib.md5()
        hasher.update(self.declaration.encode())
        hasher.update(b"\n")
        hasher.update(self.instructions.encode())
        for skill in sorted(self.skills, key=lambda item: item.name):
            hasher.update(b"\nskill:")
            hasher.update(skill.name.encode())
            hasher.update(b"\n")
            hasher.update(skill.description.encode())
            hasher.update(b"\n")
            hasher.update(skill.instructions.encode())
            hasher.update(b"\n")
            if skill.deadline_seconds:
                hasher.update(str(skill.deadline_seconds).encode())
        for document in sorted(self.knowledge, key=lambda item: item.source):
            hasher.update(b"\nknowledge:")
            hasher.update(document.source.encode())
            hasher.update(b"\n")
            hasher.update(document.text.encode())
        return hasher.hexdigest()


def load(path: str | Path) -> Folder:
    """Read an agent directory.

    Everything but the name is optional: a directory with only instructions.md is a
    valid agent, and so is one with only skills.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{root} is not an agent directory")

    folder = Folder(path=root, name=root.name)
    declaration = root / AGENT_FILE
    if declaration.is_file():
        folder.declaration = declaration.read_text().strip()
        folder.name = _declared_name(declaration) or root.name
    instructions = root / INSTRUCTIONS_FILE
    if instructions.is_file():
        folder.instructions = instructions.read_text().strip()
    folder.skills = _load_skills(root / SKILLS_DIR)
    folder.knowledge = _load_knowledge(root / KNOWLEDGE_DIR)
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
            f"no agent directory called {name!r}; expected examples/agents/{name}/"
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
        # Every kind of example is looked under rather than only examples/agents, since
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
    return str(recorded.get("md5", ""))


def write_stamp(path: Path, filename: str, md5: str) -> None:
    """Record what was synced and when, so a second launch can do nothing."""
    stamp = path / filename
    synced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stamp.write_text(json.dumps({"md5": md5, "synced_at": synced_at}) + "\n")


def _declared_name(path: Path) -> str:
    """The name `agent.yaml` gives, if it gives one."""
    declared = yaml.safe_load(path.read_text()) or {}
    if not isinstance(declared, dict):
        raise ValueError(f"{path} should describe one agent, as a mapping")
    return str(declared.get("name", "")).strip()


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
            text = file.read_text()
            if not text.strip():
                continue
            source = file.relative_to(path).as_posix()
            documents.append(Document(source=source, text=text))
    documents.sort(key=lambda item: item.source)
    return documents
