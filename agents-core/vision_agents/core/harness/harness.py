from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Sandbox:
    """Where code the agent writes gets run.

    Code execution never happens on the live speech path: a sandbox is offered to the
    slower model doing delegated work, not to the one holding the conversation.

    Attributes:
        provider: The sandbox provider's name, as the backend knows it.
    """

    provider: str = ""

    def __post_init__(self):
        if not self.provider:
            raise ValueError("a sandbox needs a provider")


@dataclass
class Daytona(Sandbox):
    """A Daytona sandbox.

    The backend needs ``DAYTONA_API_KEY`` for this to do anything.
    """

    provider: str = "daytona"


@dataclass
class Skill:
    """A kind of work worth handing to the slower model.

    There is nothing behind a skill but a better model and more time. What it declares is
    the description the fast model chooses by, and the instructions the slow one answers
    under.

    Attributes:
        name: How the fast model asks for it.
        description: The one line the fast model sees.
        instructions: The full prompt, which only the subagent sees.
        deadline_seconds: How long the work may run before it is abandoned.
    """

    name: str
    description: str
    instructions: str
    deadline_seconds: Optional[float] = None


class Harness:
    """What stands between what a caller said and the model that answers them.

    The loop itself runs in the backend, so this is configuration rather than behaviour:
    it is serialized into the session and the decisions are taken there. Reimplementing
    the loop here would mean two of them to keep in step.

    Attributes:
        use_skills: Offer the backend's built-in skills. Setting ``skills`` replaces them.
        subagents: Model targets for the work handed over, keyed by name. The entry under
            ``default``, or the only entry, is the model that runs skills. Empty means the
            fast model answers everything itself.
        vm: Where delegated code runs. Either a :class:`Sandbox` or the class itself, so
            ``vm=Daytona`` reads the way it is meant to.
        skills: Skills of your own, replacing the built-in set.
        tasks: How much delegated work may run at once. Zero leaves the backend's default.
    """

    def __init__(
        self,
        use_skills: bool = True,
        subagents: Optional[dict[str, str]] = None,
        vm: Optional[Union[Sandbox, type[Sandbox]]] = None,
        skills: Optional[list[Skill]] = None,
        tasks: int = 0,
    ):
        if tasks < 0:
            raise ValueError("tasks cannot be negative")

        self.use_skills = use_skills
        self.subagents = subagents or {}
        self.vm = vm() if isinstance(vm, type) else vm
        self.skills = skills or []
        self.tasks = tasks

    @property
    def subagent(self) -> Optional[str]:
        """The model that runs delegated work, or None when nothing is delegated."""
        if not self.subagents:
            return None
        if "default" in self.subagents:
            return self.subagents["default"]
        return next(iter(self.subagents.values()))

    def spec(self) -> dict[str, Any]:
        """Render the harness for a remote session.

        Returns:
            The fields a session spec understands. Skills are omitted when the built-in
            set is wanted, since an empty list and an absent one mean different things:
            one turns delegation off, the other leaves the defaults alone.
        """
        spec: dict[str, Any] = {"tasks": self.tasks}
        if self.subagent is not None:
            spec["subagent"] = self.subagent
        if self.vm is not None:
            spec["sandbox"] = self.vm.provider

        if self.skills:
            spec["skills"] = [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "instructions": skill.instructions,
                    "deadline_ms": int((skill.deadline_seconds or 0) * 1000),
                }
                for skill in self.skills
            ]
        elif not self.use_skills:
            spec["skills"] = []
        return spec


class DefaultHarness(Harness):
    """The harness most agents want: the built-in skills and nothing else changed."""


@dataclass
class CostTracking:
    """Cost labels carried onto every request a session makes.

    They are the backend's own tags, so spend can be attributed to whatever the labels
    mean to you rather than only to a model.

    Attributes:
        tags: Label names to values.
    """

    tags: dict[str, str] = field(default_factory=dict)
