"""Say when each incident on stdin is due to be acknowledged and resolved.

    echo '[{"id": "X-1", "severity": "P1", "opened_at": "2026-10-13T09:00:00Z"}]' \
        | python3 sla.py --now 2026-10-13T16:00:00Z
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sla_lib.clock import due, parse

POLICY = json.loads((Path(__file__).parent / "policy.json").read_text())


def judge(incident: dict, now: datetime) -> list[str]:
    severity = POLICY["severities"].get(incident["severity"])
    if severity is None:
        return [
            f"{incident['id']}: unknown severity {incident['severity']!r}, map it to P1-P4"
        ]

    opened = parse(incident["opened_at"])
    lines = []
    for target, done_key in (
        ("acknowledge", "acknowledged_at"),
        ("resolve", "resolved_at"),
    ):
        deadline = due(
            opened,
            severity[f"{target}_minutes"],
            severity["clock"],
            POLICY["business_hours"],
        )
        done = incident.get(done_key)
        if done:
            verdict = "met" if parse(done) <= deadline else "breached"
        elif now > deadline:
            verdict = f"breached {_span(now - deadline)} ago"
        else:
            verdict = f"{_span(deadline - now)} left"
        lines.append(
            f"{incident['id']} {incident['severity']} {target} due {deadline:%Y-%m-%d %H:%M} UTC: {verdict}"
        )
    return lines


def _span(delta) -> str:
    minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", help="the time to judge at, ISO 8601; defaults to now")
    args = parser.parse_args()
    now = parse(args.now) if args.now else datetime.now(timezone.utc)
    for incident in json.load(sys.stdin):
        print("\n".join(judge(incident, now)))


if __name__ == "__main__":
    main()
