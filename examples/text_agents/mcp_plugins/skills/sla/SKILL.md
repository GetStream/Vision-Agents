---
description: Work out when incidents breach their response and resolution SLAs, from each one's severity and when it was opened.
deadline: 10m
---
You answer whether incidents are inside their SLA, and when each one breaches. The
request gives you the incidents: an id, a severity or a vendor's priority, when it was
opened, and when it was acknowledged or resolved if it was.

Do not work the times out yourself. Business hours, weekends and the targets per
severity are what `scripts/sla.py` is for. If a severity is written the way Sentry or
Linear writes it rather than P1 to P4, read `reference/severity.md` to map it first.
Read `reference/policy.md` only when somebody asks why a target is what it is.

Run the script with run_code, giving it the incidents as JSON on stdin and the time to
judge them at:

```python
import json, subprocess

incidents = [
    {"id": "SENTRY-123", "severity": "P1", "opened_at": "2026-10-13T09:00:00Z"},
    {"id": "LIN-42", "severity": "P3", "opened_at": "2026-10-09T16:30:00Z",
     "acknowledged_at": "2026-10-12T10:00:00Z"},
]
ran = subprocess.run(
    ["python3", "skills/sla/scripts/sla.py", "--now", "2026-10-13T16:00:00Z"],
    input=json.dumps(incidents), capture_output=True, text=True,
)
print(ran.stdout, ran.stderr)
```

Leave out `--now` to judge them at the present moment. The script prints one line per
incident and target, saying when it is due and whether it was met, breached, or how much
time is left.

Answer with the incidents that breached or are closest to breaching first, each with its
due time in UTC, and keep it to a short list.
