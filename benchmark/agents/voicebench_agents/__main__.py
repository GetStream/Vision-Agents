"""CLI: uv run python -m voicebench_agents <pack> [--transport telnyx|webrtc]."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from voicebench_agents.healthcare import create_agent as create_healthcare
from voicebench_agents.inbound import serve
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
    parser.add_argument("--transport", choices=["telnyx", "webrtc"], default="webrtc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--phone-number-id", default=None)
    parser.add_argument("--phone-number", default=None)
    parser.add_argument("--call-control-app-id", default=None)
    parser.add_argument("--ngrok-url", default=None)
    parser.add_argument("--setup-telnyx", action="store_true")
    args = parser.parse_args()
    create_agent = FACTORIES[args.pack]
    if args.transport == "webrtc":
        serve_webrtc(create_agent, host=args.host, port=args.port)
        return
    serve(
        create_agent,
        host=args.host,
        port=args.port,
        phone_number_id=args.phone_number_id,
        phone_number=args.phone_number,
        call_control_app_id=args.call_control_app_id,
        ngrok_url=args.ngrok_url,
        setup_telnyx=args.setup_telnyx,
    )


if __name__ == "__main__":
    main()
