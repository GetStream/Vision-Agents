import tempfile
from dataclasses import dataclass, asdict


@dataclass
class AgentOptions:
    model_dir: str

    def update(self, other: "AgentOptions") -> "AgentOptions":
        merged_dict = asdict(self)

        for key, value in asdict(other).items():
            if value is not None:
                merged_dict[key] = value

        return AgentOptions(**merged_dict)


# Cache tempdir at module load time to avoid blocking I/O during async operations
_DEFAULT_MODEL_DIR = tempfile.gettempdir()


def default_agent_options():
    return AgentOptions(model_dir=_DEFAULT_MODEL_DIR)