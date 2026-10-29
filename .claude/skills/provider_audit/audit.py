# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print what is left on each vendor account the router spends from, read-only.

Run from the repo root with `uv run --no-project --env-file .env
.claude/skills/provider_audit/audit.py [--min UNIT=AMOUNT]`. Each check's docstring is
the vendor's doc for the endpoint it calls.

Keys come from the repo-root `.env` only, which uv loads into the environment, read as
the router reads them (keypool: NAME, comma separated, then NAME_1, NAME_2 ...), and one
row is printed per key. Every call is a GET to a billing or usage endpoint; nothing is
written. A vendor that fails (a 401, a timeout, an unexpected body) is an error row,
never a balance of 0. Core vendors are printed first, connector vendors after them. The
exit code is 1 only when a core vendor's balance is below the --min of its unit
(`--min USD=20 --min characters=100000`), else 0: a connector vendor is bring-your-own,
so its low balance is shown and marked but never fails the run.
"""

import argparse
import base64
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

type Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None

# What is left (None when unknown), its unit, and a note.
type Reading = tuple[float | None, str, str]

# vendor, the env vars that may hold its key (the first one set wins), the check.
type Check = tuple[str, tuple[str, ...], Callable[[str], list[Reading]]]

# One line of the table: vendor, left, unit, source, checked_at, note.
type Row = dict[str, str | float | None]

CHECKS: list[Check] = []

# Vendors only connectors use: each customer brings their own account, so the platform
# does not depend on ours. Every other vendor is core.
CONNECTORS = frozenset({"Telnyx", "Twilio"})

# Seconds to wait for a vendor; Baseten took over 20 s on 2026-10-28.
TIMEOUT = 90.0

# Accounts the router spends from with no read-only balance API: where to look instead.
DASHBOARDS = {
    "Google AI Studio (Gemini)": "https://aistudio.google.com/usage",
    "Cartesia": "https://play.cartesia.ai/subscription",
    "Inworld": "https://platform.inworld.ai/",
    "Exa": "https://dashboard.exa.ai/billing",
    "Perplexity": "https://www.perplexity.ai/account/api/billing",
    "Meta (api.meta.ai)": "https://dev.meta.ai/",
    "Typesafe": "https://typesafe.ai/",
    "Cerebras": "https://cloud.cerebras.ai/",
    "Mem0": "https://app.mem0.ai/",
    "Turbopuffer": "https://turbopuffer.com/dashboard",
    "Daytona": "https://app.daytona.io/dashboard/billing",
    "Nebius (Gemma GPU, active; alternative Baseten)": "https://console.nebius.com/",
    "Railway (docs search)": "https://railway.com/account/usage",
}


class AuditError(Exception):
    """A vendor answered with something other than a balance."""


type CheckFn = Callable[[str], list[Reading]]


def check[F: CheckFn](vendor: str, *envs: str) -> Callable[[F], F]:
    """Register a check of vendor's account, run with the key in the first of envs set."""

    def register(fn: F) -> F:
        CHECKS.append((vendor, envs, fn))
        return fn

    return register


def keys(name: str) -> list[str]:
    """The keys in NAME, then NAME_<n> in numeric order, as internal/keypool reads them."""
    numbered = []
    for env, value in os.environ.items():
        suffix = env.removeprefix(name + "_")
        if suffix != env and suffix.isdigit() and str(int(suffix)) == suffix:
            numbered.append((int(suffix), value))
    found: list[str] = []
    for value in [os.environ.get(name, "")] + [value for _, value in sorted(numbered)]:
        for key in (k.strip() for k in value.split(",")):
            if key and key not in found:
                found.append(key)
    return found


def get(url: str, headers: dict[str, str], key: str) -> dict[str, Json]:
    """GET url and return its JSON object; any other answer raises AuditError."""
    headers = {"User-Agent": "provider-audit", "Accept": "application/json", **headers}
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read()
    except urllib.error.HTTPError as e:
        raise AuditError(f"HTTP {e.code}: {redact(e.read()[:300], key)}") from None
    except (OSError, http.client.HTTPException) as e:
        raise AuditError(f"unreachable: {e}") from None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise AuditError(f"not JSON: {redact(body[:120], key)}") from None
    if not isinstance(data, dict):
        raise AuditError(f"expected a JSON object, got {type(data).__name__}")
    return data


def redact(body: bytes, key: str) -> str:
    """A vendor's error text without the key or anything that looks like one."""
    text = " ".join(body.decode(errors="replace").split()).replace(key, "<key>")
    return hide(text)


def hide(text: str) -> str:
    """text without key-shaped tokens and email addresses."""
    text = re.sub(r"[A-Za-z0-9_\-]{32,}", "<redacted>", text)
    return re.sub(r"[\w.+\-]+@[\w\-]+(\.[\w\-]+)+", "<email>", text)


def field(data: Json, *path: str) -> Json:
    """data[path[0]][path[1]]...; a missing step raises AuditError."""
    for step in path:
        if not isinstance(data, dict) or step not in data:
            raise AuditError(f"no {'.'.join(path)} in the answer")
        data = data[step]
    return data


def num(data: Json, *path: str) -> float:
    """The number, or numeric string, at path; anything else raises AuditError."""
    value = field(data, *path)
    if isinstance(value, int | float | str) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError:
            pass
    raise AuditError(f"{'.'.join(path)} is {value!r}, not a number")


def items(data: Json, *path: str) -> list[Json]:
    """The list at path; anything else raises AuditError."""
    value = field(data, *path)
    if isinstance(value, list):
        return value
    raise AuditError(f"{'.'.join(path)} is {value!r}, not a list")


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def month_start() -> datetime:
    return datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@check("OpenRouter", "OPENROUTER_MANAGEMENT_KEY", "OPENROUTER_API_KEY")
def openrouter(key: str, base: str = "https://openrouter.ai") -> list[Reading]:
    """https://openrouter.ai/docs/api/api-reference/credits/get-remaining-credits"""
    data = get(f"{base}/api/v1/credits", bearer(key), key)
    left = num(data, "data", "total_credits") - num(data, "data", "total_usage")
    return [(left, "USD", "credits bought minus used, whole account")]


@check("Tavily", "TAVILY_API_KEY")
def tavily(key: str, base: str = "https://api.tavily.com") -> list[Reading]:
    """https://docs.tavily.com/documentation/api-reference/endpoint/usage"""
    account = field(get(f"{base}/usage", bearer(key), key), "account")
    left = num(account, "plan_limit") - num(account, "plan_usage")
    return [(left, "credits", f"plan {field(account, 'current_plan')}, this cycle")]


@check("Telnyx", "TELNYX_API_KEY")
def telnyx(key: str, base: str = "https://api.telnyx.com") -> list[Reading]:
    """https://developers.telnyx.com/api-reference/billing/get-user-balance-details"""
    data = field(get(f"{base}/v2/balance", bearer(key), key), "data")
    currency = str(field(data, "currency"))
    return [(num(data, "available_credit"), currency, "balance + credit limit")]


@check("Twilio", "TWILIO_AUTH_TOKEN")
def twilio(key: str, base: str = "https://api.twilio.com") -> list[Reading]:
    """https://www.twilio.com/docs/usage/api/account"""
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    if not sid:
        raise AuditError("TWILIO_ACCOUNT_SID is not set")
    auth = base64.b64encode(f"{sid}:{key}".encode()).decode()
    url = f"{base}/2010-04-01/Accounts/{sid}/Balance.json"
    data = get(url, {"Authorization": f"Basic {auth}"}, key)
    return [(num(data, "balance"), str(field(data, "currency")), "")]


@check("ElevenLabs", "ELEVENLABS_API_KEY")
def elevenlabs(key: str, base: str = "https://api.elevenlabs.io") -> list[Reading]:
    """https://elevenlabs.io/docs/api-reference/user/subscription/get"""
    data = get(f"{base}/v1/user/subscription", {"xi-api-key": key}, key)
    left = num(data, "character_limit") - num(data, "character_count")
    resets = data.get("next_character_count_reset_unix")
    return [(left, "characters", f"tier {data.get('tier')}, resets at unix {resets}")]


@check("Deepgram", "DEEPGRAM_ADMIN_KEY", "DEEPGRAM_API_KEY")
def deepgram(key: str, base: str = "https://api.deepgram.com") -> list[Reading]:
    """https://developers.deepgram.com/reference/manage/billing/list"""
    auth = {"Authorization": f"Token {key}"}
    readings: list[Reading] = []
    for project in items(get(f"{base}/v1/projects", auth, key), "projects"):
        url = f"{base}/v1/projects/{field(project, 'project_id')}/balances"
        for balance in items(get(url, auth, key), "balances"):
            note = f"project {hide(str(field(project, 'name')))}"
            readings.append(
                (num(balance, "amount"), str(field(balance, "units")), note)
            )
    return readings


@check("Fish Audio", "FISH_API_KEY")
def fish(key: str, base: str = "https://api.fish.audio") -> list[Reading]:
    """https://docs.fish.audio/api-reference/endpoint/wallet/get-api-credit"""
    data = get(f"{base}/wallet/self/api-credit", bearer(key), key)
    return [(num(data, "credit"), "credits", "API credit")]


@check("fal", "FAL_KEY")
def fal(key: str, base: str = "https://api.fal.ai") -> list[Reading]:
    """https://fal.ai/docs/platform-apis/v1/account/billing"""
    url = f"{base}/v1/account/billing?expand=credits"
    credits = field(get(url, {"Authorization": f"Key {key}"}, key), "credits")
    return [(num(credits, "current_balance"), str(field(credits, "currency")), "")]


@check("Ollama Cloud", "OLLAMA_API_KEY")
def ollama(key: str, base: str = "https://ollama.com") -> list[Reading]:
    """https://docs.ollama.com/api/balance"""
    data = get(f"{base}/api/balance", bearer(key), key)
    left = sum(num(data, part, "balance_usd") for part in ("included", "purchased"))
    return [(left, "USD", "included + purchased credits")]


@check("xAI", "XAI_MANAGEMENT_KEY")
def xai(
    key: str, base: str = "https://management-api.x.ai", api: str = "https://api.x.ai"
) -> list[Reading]:
    """https://docs.x.ai/developers/rest-api-reference/management/billing"""
    api_key = (keys("XAI_API_KEY") or [""])[0]
    if not api_key:
        raise AuditError("XAI_API_KEY is not set: it names the team")
    team = field(get(f"{api}/v1/api-key", bearer(api_key), api_key), "team_id")
    data = get(f"{base}/v1/billing/teams/{team}/prepaid/balance", bearer(key), key)
    # total is USD cents, and a purchase is negative: -total is what is left.
    return [(-num(data, "total", "val") / 100, "USD", "prepaid credit")]


@check("Baseten", "BASETEN_API_KEY")
def baseten(key: str, base: str = "https://api.baseten.co") -> list[Reading]:
    """https://docs.baseten.co/reference/management-api/billing/gets-billing-usage-summary-for-a-date-range"""
    now = datetime.now(UTC)
    query = urllib.parse.urlencode(
        {"start_date": month_start().isoformat(), "end_date": now.isoformat()}
    )
    data = get(f"{base}/v1/billing/usage_summary?{query}", bearer(key), key)
    spent = sum((num(data, part, "total") for part, v in data.items() if v), 0.0)
    return [(spent, "USD spent this month", "postpaid: dedicated + training + APIs")]


@check("OpenAI", "OPENAI_ADMIN_KEY")
def openai(key: str, base: str = "https://api.openai.com") -> list[Reading]:
    """https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/costs"""
    query = urllib.parse.urlencode(
        {"start_time": int(month_start().timestamp()), "limit": 31}
    )
    data = get(f"{base}/v1/organization/costs?{query}", bearer(key), key)
    spent = sum(
        (
            num(result, "amount", "value")
            for bucket in items(data, "data")
            for result in items(bucket, "results")
        ),
        0.0,
    )
    return [(spent, "USD spent this month", "credit balance: dashboard only")]


@check("Anthropic", "ANTHROPIC_ADMIN_KEY")
def anthropic(key: str, base: str = "https://api.anthropic.com") -> list[Reading]:
    """https://platform.claude.com/docs/en/api/admin-api/usage-cost/get-cost-report"""
    start = month_start().strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.urlencode({"starting_at": start, "limit": 31})
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    data = get(f"{base}/v1/organizations/cost_report?{query}", headers, key)
    cents = sum(  # the amounts are cents, as decimal strings
        num(result, "amount")
        for bucket in items(data, "data")
        for result in items(bucket, "results")
    )
    return [(cents / 100, "USD spent this month", "credit balance: dashboard only")]


def audit(
    now: str,
    checks: list[Check],
    dashboards: dict[str, str],
    connectors: frozenset[str] = CONNECTORS,
) -> list[Row]:
    """One row per key of each check, then one per dashboard-only account.

    Each row's type is "connector" for a vendor in connectors, else "core".
    """
    rows: list[Row] = []
    for vendor, envs, fn in checks:
        kind = "connector" if vendor in connectors else "core"
        base: Row = {
            "vendor": vendor,
            "type": kind,
            "source": fn.__doc__,
            "checked_at": now,
        }
        env = next((e for e in envs if keys(e)), "")
        if not env:
            note = f"skipped: {' or '.join(envs)} not set"
            rows.append(base | {"left": None, "unit": "", "note": note})
            continue
        pool = keys(env)
        for i, key in enumerate(pool):
            name = vendor if len(pool) == 1 else f"{vendor} ({env} #{i + 1})"
            try:
                readings = fn(key) or [(None, "", "no balances in the answer")]
            except AuditError as e:
                readings = [(None, "", f"error ({env}): {e}")]
            for left, unit, note in readings:
                row = {"vendor": name, "left": left, "unit": unit, "note": note}
                rows.append(base | row)
    for vendor, url in dashboards.items():
        note = "unknown (dashboard only)"
        kind = "connector" if vendor in connectors else "core"
        row = {"vendor": vendor, "type": kind, "left": None, "unit": "", "note": note}
        rows.append(row | {"source": url, "checked_at": now})
    return rows


def below(rows: list[Row], mins: dict[str, float]) -> list[Row]:
    """The rows whose balance is under the --min of their unit."""
    return [
        row
        for row in rows
        if isinstance(row["left"], float)
        and str(row["unit"]).lower() in mins
        and row["left"] < mins[str(row["unit"]).lower()]
    ]


def parse_min(text: str) -> tuple[str, float]:
    unit, _, amount = text.partition("=")
    try:
        return unit.strip().lower(), float(amount)
    except ValueError:
        raise argparse.ArgumentTypeError(f"want UNIT=AMOUNT, not {text!r}") from None


def main(
    argv: list[str],
    checks: list[Check] = CHECKS,
    dashboards: dict[str, str] = DASHBOARDS,
    connectors: frozenset[str] = CONNECTORS,
) -> int:
    """Print the table; return 1 when a core balance is under its --min, else 0."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--min", type=parse_min, action="append", default=[])
    args = parser.parse_args(argv)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = audit(now, checks, dashboards, connectors)
    core = [row for row in rows if row["type"] == "core"]
    optional = [row for row in rows if row["type"] != "core"]
    print("vendor | left | unit | source | checked_at | note")
    print_rows(core)
    if optional:
        print("\nConnector vendors (bring your own; never fail the run)")
        print_rows(optional)
    low = below(core, dict(args.min))
    for row in low:
        print(f"BELOW: {row['vendor']} has {row['left']:.2f} {row['unit']}")
    for row in below(optional, dict(args.min)):
        print(f"BELOW (connector): {row['vendor']} has {row['left']:.2f} {row['unit']}")
    return 1 if low else 0


def print_rows(rows: list[Row]) -> None:
    for row in rows:
        left = f"{row['left']:.2f}" if isinstance(row["left"], float) else "-"
        if str(row["note"]).startswith("error"):
            left = "error"
        cells = (row["vendor"], left, row["unit"], row["source"])
        print(" | ".join(map(str, cells + (row["checked_at"], row["note"]))))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
