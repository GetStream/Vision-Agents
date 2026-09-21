"""``simulate`` subcommand: judge resolution, terminal table and report files."""

import asyncio
import dataclasses
import importlib
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

import vision_agents.testing as testing
from vision_agents.core.llm.llm import LLM
from vision_agents.core.utils.logging import configure_sdk_logger

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent
    from vision_agents.testing import (
        Scenario,
        SimulationCase,
        SimulationReport,
        SimulationRun,
        TranscriptLine,
        Trial,
    )

logger = logging.getLogger(__name__)

DEFAULT_JUDGE = "gemini/gemini-2.5-flash"
DEFAULT_REPORT_DIR = "simulation-report"
DEFAULT_MAX_TURNS = 10
REPORT_JSON = "report.json"
REPORT_MD = "report.md"

_TABLE_HEADERS = (
    "Scenario",
    "Variations",
    "Passed",
    "pass@k",
    "Turns",
    "P50 latency",
    "Failed criteria",
)
_ISSUES_MAX_LEN = 60
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

AgentFactory = Callable[[], Awaitable["Agent"]]
CaseCallback = Callable[["Scenario", "SimulationCase"], None]


class SimulateError(click.ClickException):
    """Setup or provider failure. Exits 2 so CI can tell it from a failed scenario."""

    exit_code = 2


def run_simulation(
    create_agent: AgentFactory,
    target: str,
    *,
    repeat: Optional[int],
    variations: Optional[int],
    max_turns: int,
    judge: str,
    report_dir: str,
    name_filter: Optional[str],
    log_level: str,
) -> int:
    """Run the scenarios at ``target``, print the table, write the reports.

    Returns:
        The process exit code: 0 all passed, 1 a scenario failed, 2 an error.

    Raises:
        ValueError: if ``log_level`` is not a standard logging level name.
    """
    try:
        level = _LOG_LEVELS[log_level.upper()]
    except KeyError as err:
        raise ValueError(f"unsupported log level: {log_level!r}") from err
    configure_sdk_logger(level=level)
    scenarios = load_scenarios(
        Path(target), name_filter, repeat=repeat, variations=variations
    )
    llm_factory = resolve_llm_factory(judge)

    async def _run() -> "SimulationReport":
        await probe_llm_factory(llm_factory, judge)
        return await simulate_scenarios(
            create_agent,
            scenarios,
            llm_factory,
            judge_target=judge,
            max_turns=max_turns,
            on_case=_echo_case,
        )

    report = asyncio.run(_run())

    click.echo()
    click.echo(render_table(report))
    click.echo()
    click.echo(_summary(report))
    json_path, md_path = write_reports(report, Path(report_dir))
    click.echo(click.style(f"Report written to {json_path} and {md_path}", dim=True))
    return report.exit_code


def load_scenarios(
    target: Path,
    name_filter: Optional[str],
    *,
    repeat: Optional[int] = None,
    variations: Optional[int] = None,
) -> list["Scenario"]:
    """Load every text scenario at ``target`` whose name contains ``name_filter``.

    Audio scenarios are skipped with a notice. ``repeat`` and ``variations``
    override the scenario files when given.
    """
    try:
        paths = testing.find_scenarios(target)
        scenarios = [testing.load_scenario(path) for path in paths]
    except (OSError, ValueError) as err:
        raise SimulateError(str(err)) from err
    if not scenarios:
        raise SimulateError(f"no scenario files (*.yaml) found in {target}")
    if name_filter:
        scenarios = [s for s in scenarios if name_filter.lower() in s.name.lower()]
        if not scenarios:
            raise SimulateError(f"no scenarios match --filter {name_filter!r}")
    spoken = [s.name for s in scenarios if s.mode == "audio"]
    if spoken:
        click.echo(
            "Skipping audio scenario(s) "
            + ", ".join(spoken)
            + ": agent simulate runs in text mode, run them with pytest",
            file=sys.stderr,
        )
        scenarios = [s for s in scenarios if s.mode == "text"]
        if not scenarios:
            raise SimulateError(f"no text scenarios found in {target}")
    if repeat is not None or variations is not None:
        scenarios = [
            dataclasses.replace(
                s,
                repeat=s.repeat if repeat is None else repeat,
                variations=s.variations if variations is None else variations,
            )
            for s in scenarios
        ]
    return scenarios


async def simulate_scenarios(
    create_agent: AgentFactory,
    scenarios: list["Scenario"],
    llm_factory: Callable[[], LLM],
    *,
    judge_target: str,
    max_turns: int,
    on_case: Optional[CaseCallback] = None,
) -> "SimulationReport":
    """Play every scenario against a fresh agent per conversation and build the report.

    ``llm_factory`` supplies the LLMs that play the user, reword the brief for
    variations and judge the transcript; a fresh judge is built per scenario.
    ``on_case`` is called after every conversation, for progress output.
    """
    started = _now()
    simulation = testing.Simulation(user_llm=llm_factory, max_turns=max_turns)
    runs = [
        await _run_scenario(
            simulation,
            create_agent,
            scenario,
            llm_factory,
            judge_target=judge_target,
            max_turns=max_turns,
            on_case=on_case,
        )
        for scenario in scenarios
    ]
    return testing.SimulationReport(
        runs=runs, judge_target=judge_target, started_at=started, finished_at=_now()
    )


async def _run_scenario(
    simulation: "testing.Simulation",
    create_agent: AgentFactory,
    scenario: "Scenario",
    llm_factory: Callable[[], LLM],
    *,
    judge_target: str,
    max_turns: int,
    on_case: Optional[CaseCallback],
) -> "SimulationRun":
    started = _now()
    cases: list["SimulationCase"] = []
    case_started = started

    def on_trial(trial: "Trial") -> None:
        nonlocal case_started
        finished = _now()
        case = testing.SimulationCase.from_trial(
            trial, max_turns=max_turns, started_at=case_started, finished_at=finished
        )
        case_started = finished
        cases.append(case)
        if on_case is not None:
            on_case(scenario, case)

    error: Optional[str] = None
    pass_at_k: Optional[float] = None
    try:
        result = await simulation.run(
            create_agent,
            scenario,
            lambda: testing.LLMJudge(llm_factory()),
            on_trial=on_trial,
        )
        pass_at_k = result.pass_at_k
    except Exception as exc:
        logger.exception("Scenario %s could not run", scenario.name)
        error = str(exc)
    return testing.SimulationRun.from_scenario(
        scenario,
        conversations=cases,
        max_turns=max_turns,
        judge_target=judge_target,
        started_at=started,
        finished_at=_now(),
        error=error,
        pass_at_k=pass_at_k,
    )


def resolve_llm_factory(spec: str) -> Callable[[], LLM]:
    """Turn a ``--judge`` value into a factory that returns a fresh LLM per call.

    ``provider/model`` instantiates ``vision_agents.plugins.<provider>.LLM(model=...)``.
    ``module:attribute`` imports a zero-argument callable that returns an LLM.
    """
    if ":" in spec:
        return _import_factory(spec)
    return _plugin_factory(spec)


async def probe_llm_factory(factory: Callable[[], LLM], spec: str) -> None:
    """Create and close one LLM so a bad ``--judge`` fails before any scenario runs."""
    try:
        llm = factory()
    except Exception as err:
        raise SimulateError(
            f"--judge {spec!r}: failed to create the LLM: {err}"
        ) from err
    if not isinstance(llm, LLM):
        raise SimulateError(
            f"--judge {spec!r}: expected an LLM instance, got {type(llm).__name__}"
        )
    try:
        await llm.close()
    except Exception as err:
        raise SimulateError(
            f"--judge {spec!r}: failed to close the probe LLM: {err}"
        ) from err


def _import_factory(spec: str) -> Callable[[], LLM]:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute or ":" in attribute:
        raise SimulateError(
            f"--judge {spec!r}: expected 'module:attribute' naming a callable "
            "that returns an LLM"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise SimulateError(
            f"--judge {spec!r}: failed to import module '{module_name}': {err}"
        ) from err
    try:
        factory = module.__dict__[attribute]
    except KeyError as err:
        raise SimulateError(
            f"--judge {spec!r}: '{module_name}' has no callable '{attribute}'"
        ) from err
    if not callable(factory):
        raise SimulateError(
            f"--judge {spec!r}: '{module_name}' has no callable '{attribute}'"
        )
    return factory


def _plugin_factory(spec: str) -> Callable[[], LLM]:
    provider, _, model = spec.partition("/")
    if not provider or not model:
        raise SimulateError(
            f"--judge {spec!r}: expected 'provider/model' "
            f"(e.g. {DEFAULT_JUDGE}) or 'module:attribute'"
        )
    try:
        plugin = importlib.import_module(f"vision_agents.plugins.{provider}")
    except ImportError as err:
        raise SimulateError(
            f"--judge {spec!r}: plugin '{provider}' is not installed "
            f"(try: uv add 'vision-agents[{provider}]'): {err}"
        ) from err
    try:
        llm_cls = plugin.LLM
    except AttributeError as err:
        raise SimulateError(
            f"--judge {spec!r}: plugin '{provider}' has no text LLM"
        ) from err
    return lambda: llm_cls(model=model)


def render_table(report: "SimulationReport") -> str:
    """Plain-text table with one row per scenario."""
    rows = [
        (
            run.name,
            str(run.variations),
            f"{run.passed}/{run.cases}",
            _format_pass_at_k(run),
            _format_turns(run),
            _format_latency(run),
            _format_issues(run),
        )
        for run in report.runs
    ]
    widths = [
        max([len(header), *(len(row[i]) for row in rows)])
        for i, header in enumerate(_TABLE_HEADERS)
    ]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths)).rstrip()

    return "\n".join(
        [
            line(_TABLE_HEADERS),
            line(tuple("-" * w for w in widths)),
            *(line(row) for row in rows),
        ]
    )


def render_markdown(report: "SimulationReport") -> str:
    """Markdown report: summary table, then every transcript and verdict."""
    lines = [
        "# Simulation report",
        "",
        f"- State: **{report.state}**",
        f"- Judge: `{report.judge_target}`",
        f"- Cases: {report.cases} ({report.passed} passed, "
        f"{report.failed} failed, {report.errored} errored)",
        f"- Started: {report.started_at.isoformat()}",
        f"- Finished: {report.finished_at.isoformat()}",
        "",
        "| " + " | ".join(_TABLE_HEADERS) + " |",
        "|" + "---|" * len(_TABLE_HEADERS),
    ]
    for run in report.runs:
        cells = (
            run.name,
            str(run.variations),
            f"{run.passed}/{run.cases}",
            _format_pass_at_k(run),
            _format_turns(run),
            _format_latency(run),
            _format_issues(run, limit=None),
        )
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    lines.append("")

    for run in report.runs:
        lines += [
            f"## {run.name}: {run.state}",
            "",
            f"- Mode: {run.mode}",
            f"- Variations: {run.variations}, repeat: {run.repeat}, "
            f"max turns: {run.max_turns}, pass@{run.repeat}: {_format_pass_at_k(run)}",
            "",
            "**Brief:**",
            "",
            "```text",
            run.scenario,
            "```",
            "",
            "**Criteria:**",
            *(f"- {criterion}" for criterion in run.criteria),
            "",
        ]
        if run.error:
            lines += [f"**Error:** {run.error}", ""]
        for case in run.conversations:
            lines += _md_case(case)
    return "\n".join(lines)


def write_reports(report: "SimulationReport", report_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``report_dir``."""
    json_path = report_dir / REPORT_JSON
    md_path = report_dir / REPORT_MD
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    except OSError as err:
        raise SimulateError(
            f"failed to write the report to {report_dir}: {err}"
        ) from err
    return json_path, md_path


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _echo_case(scenario: "Scenario", case: "SimulationCase") -> None:
    labels = {
        "passed": click.style("PASS ", fg="green"),
        "failed": click.style("FAIL ", fg="red"),
        "errored": click.style("ERROR", fg="red", bold=True),
    }
    if case.error:
        detail = case.error
    else:
        detail = f"{case.turns} turns"
        if case.failed_criteria:
            detail += ", failed: " + "; ".join(case.failed_criteria)
    click.echo(
        f"{labels[case.state]} {scenario.name} v{case.variation} #{case.attempt}: {detail}",
        file=sys.stderr,
    )


def _summary(report: "SimulationReport") -> str:
    text = (
        f"{len(report.runs)} scenarios, {report.cases} cases: "
        f"{report.passed} passed, {report.failed} failed, {report.errored} errored"
    )
    color = {"passed": "green", "failed": "red", "errored": "red"}[report.state]
    return click.style(text, fg=color, bold=True)


def _format_pass_at_k(run: "SimulationRun") -> str:
    return "-" if run.pass_at_k is None else f"{run.pass_at_k:.2f}"


def _format_turns(run: "SimulationRun") -> str:
    mean = run.mean_turns
    if mean is None:
        return "-"
    return str(int(mean)) if mean == int(mean) else f"{mean:.1f}"


def _format_latency(run: "SimulationRun") -> str:
    p50 = run.p50_latency_ms
    return "-" if p50 is None else f"{p50:.0f} ms"


def _format_issues(run: "SimulationRun", limit: Optional[int] = _ISSUES_MAX_LEN) -> str:
    parts = list(run.failed_criteria)
    if run.error:
        parts.append(f"error: {run.error}")
    parts += [f"error: {case.error}" for case in run.conversations if case.error]
    text = "; ".join(dict.fromkeys(parts)) if parts else "-"
    if limit is not None and len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_case(case: "SimulationCase") -> list[str]:
    lines = [
        f"### Variation {case.variation}, attempt {case.attempt}: {case.state}",
        "",
        f"- Brief: {_md_cell(case.scenario)}",
        f"- Turns: {case.turns}",
        f"- Ended: {case.ended}",
        "",
    ]
    if case.error:
        lines += [f"**Error:** {case.error}", ""]
    if case.transcript:
        lines += ["**Transcript**", ""]
        for line in case.transcript:
            lines += _md_transcript_line(line)
        lines.append("")
    if case.criteria:
        lines += ["**Verdicts**", ""]
        lines += [
            f"- {'PASS' if v.passed else 'FAIL'}: {v.criterion} ({v.reason})"
            for v in case.criteria
        ]
        lines.append("")
    return lines


def _md_transcript_line(line: "TranscriptLine") -> list[str]:
    text = line.text.replace("\n", " ")
    if line.caller:
        return [f"- **User:** {text}"]
    latency = f" ({line.latency_ms:.0f} ms)" if line.latency_ms is not None else ""
    out = [f"- **Agent**{latency}: {text}"]
    for call in line.tool_calls:
        status = "error" if call.is_error else "ok"
        out.append(
            f"  - tool `{call.name}({json.dumps(call.arguments, default=str)})` "
            f"-> {status}: {json.dumps(call.output, default=str)}"
        )
    return out
