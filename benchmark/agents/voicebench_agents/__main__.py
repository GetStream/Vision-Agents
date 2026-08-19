"""CLI: uv run python -m voicebench_agents <pack>."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from voicebench_agents.healthcare import create_agent as create_healthcare
from voicebench_agents.restaurant import create_agent as create_restaurant
from voicebench_agents.serve_webrtc import serve_webrtc
from voicebench_agents.telecom import create_agent as create_telecom

FACTORIES = {
    "restaurant": create_restaurant,
    "healthcare": create_healthcare,
    "telecom": create_telecom,
}


def _load_env() -> None:
    for directory in Path(__file__).resolve().parents:
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            return


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="Run a vertical voicebench agent.")
    parser.add_argument("pack", choices=sorted(FACTORIES), help="scenario pack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    serve_webrtc(FACTORIES[args.pack], host=args.host, port=args.port)


if __name__ == "__main__":
    main()
