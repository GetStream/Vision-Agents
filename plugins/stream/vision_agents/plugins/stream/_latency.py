"""A console view of the measured path to the first audio packet."""

from typing import Any

from vision_agents.core.llm.remote import JoinStep


def render_turn(frame: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    """Show dependencies without adding overlapping model work to the critical path."""
    spoken = bool(frame.get("stt_latency_ms"))
    total = frame.get("speech_end_to_audio_ms") if spoken else frame.get("roundtrip_ms")
    origin = "speech end" if spoken else "turn start"
    heading = f"voice latency DAG turn={frame.get('turn_id', '')}"
    if total:
        heading += f" | {origin} -> first audio ~{total:,.0f} ms"
    lines = [heading, f"  [{origin}]"]

    stages = (
        ("stt_latency_ms", "STT"),
        ("cadence_ms", "cadence"),
        ("decision_ms", "decision"),
        ("model_to_first_text_ms", "first speakable text"),
        ("text_to_tts_ms", "TTS handoff"),
        ("tts_to_audio_ms", "first audio"),
    )
    for key, label in stages:
        duration = frame.get(key)
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        lines.extend(("       |", "       v", f"  [{label} {duration:,.0f} ms]"))
        if key == "decision_ms":
            lines.extend(_calls(calls, {"flow"}))
        elif key == "model_to_first_text_ms":
            lines.extend(_calls(calls, {"reply"}))
        elif key == "tts_to_audio_ms" and frame.get("tts_ttfb_ms"):
            lines.append(
                f"       +-- TTS provider first byte {frame['tts_ttfb_ms']:,.0f} ms"
            )

    other = [call for call in calls if call.get("purpose") not in {"flow", "reply"}]
    if other:
        lines.append("  other model calls (may overlap the path):")
        lines.extend(_calls(other, None))
    if frame.get("interrupted"):
        lines.append("  [interrupted]")
    return "\n".join(lines)


def _calls(calls: list[dict[str, Any]], purposes: set[str] | None) -> list[str]:
    lines = []
    for call in calls:
        if purposes is not None and call.get("purpose") not in purposes:
            continue
        name = "/".join(filter(None, (call.get("provider"), call.get("model"))))
        purpose = call.get("purpose") or "model"
        timing = f"TTFT {call.get('ttft_ms', 0):,.0f} ms, full {call.get('duration_ms', 0):,.0f} ms"
        status = "" if call.get("success", True) else " FAILED"
        lines.append(f"       +-- {purpose} {name}: {timing}{status}")
    return lines


def render_join(call_id: str, steps: list[JoinStep]) -> str:
    """Show the agent's own path to the call, one step after another."""
    total = sum(step.ms for step in steps)
    lines = [
        f"join DAG call={call_id} | join start -> in the call ~{total:,.0f} ms",
        "  [join start]",
    ]
    for step in steps:
        lines.extend(("       |", "       v", f"  [{step.name} {step.ms:,.0f} ms]"))
        if step.name == "router session":
            lines.append(
                "       +-- the router joins the call here; its WebRTC connection DAG follows"
            )
    return "\n".join(lines)


_TIMELINE_WIDTH = 40


def render_connection(frame: dict[str, Any]) -> str:
    """Draw the router's join as the SDK recorded it: a DAG of steps.

    One row per step in start order, with what it waited for, when it started, how long it
    took in ms and in round trips of its peer, and a timeline that shows which steps ran in
    parallel. Critical-path steps are marked with "*" and drawn with "="; detail spans (TCP,
    TLS, first byte) sit under their step. The heading names the flow the join took, "fast"
    or "legacy", when the router says.
    """
    trace = frame.get("trace") or {}
    spans = trace.get("spans") or []
    rtt = trace.get("rtt_ms") or {}
    flow = frame.get("flow")
    heading = f"join DAG (webrtc, {flow} join)" if flow else "join DAG (webrtc)"
    if rtt:
        heading += " " + ", ".join(
            f"RTT {peer} {ms:,.1f} ms" for peer, ms in sorted(rtt.items())
        )
    if not spans:
        return heading + "\n  no steps recorded"

    critical = set(trace.get("critical_path") or [])
    end = max(float(span.get("end_ms") or 0) for span in spans)
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        if span.get("parent"):
            children.setdefault(span["parent"], []).append(span)

    lines = [
        heading,
        f"  {'step':<22} {'after':<36} {'start':>7} {'ms':>8} {'RTT':>6}  timeline",
    ]
    for span in spans:
        if span.get("parent"):
            continue
        name = span.get("name", "")
        on_path = name in critical
        after = ",".join(span.get("after") or []) or "-"
        lines.append(
            f"{'*' if on_path else ' '} {name:<22} {after:<36} {_row(span, end, on_path)}"
        )
        for child in sorted(children.get(name, []), key=lambda c: c.get("start_ms", 0)):
            detail = "  " + str(child.get("name", "")).removeprefix(name)
            lines.append(f"  {detail:<22} {'':<36} {_row(child, end, False)}")

    path = trace.get("critical_path") or []
    if path:
        lines.append("critical path: " + " > ".join(path))
        lines.append(
            f"  {trace.get('critical_ms', 0):,.1f} ms = {trace.get('critical_rtts', 0):.2f} RTT"
            f" ({trace.get('critical_net_ms', 0):,.1f} ms network)"
            f" + {trace.get('critical_timer_ms', 0):,.1f} ms timers"
            f" + {trace.get('critical_local_ms', 0):,.1f} ms local;"
            f" {trace.get('critical_wait_ms', 0):,.1f} ms unaccounted waiting"
        )
    if trace.get("publish_to_media_ms") is not None:
        lines.append(
            f"publish to media: {trace['publish_to_media_ms']:,.1f} ms from Join"
            " (first RTP sent + RTT/2)"
        )
    if trace.get("subscribe_to_media_ms") is not None:
        lines.append(
            f"subscribe to media: {trace['subscribe_to_media_ms']:,.1f} ms from Join"
        )
    return "\n".join(lines)


def _row(span: dict[str, Any], end: float, on_path: bool) -> str:
    start = float(span.get("start_ms") or 0)
    stop = float(span.get("end_ms") or 0)
    kind = span.get("kind", "")
    rtts = f"{span.get('rtts', 0):.2f}" if kind == "net" and span.get("rtts") else kind
    if kind == "net" and not span.get("rtts"):
        rtts = "?"
    cells = [" "] * _TIMELINE_WIDTH
    if end > 0:
        first = min(int(start / end * _TIMELINE_WIDTH), _TIMELINE_WIDTH - 1)
        last = min(
            max(int(stop / end * _TIMELINE_WIDTH + 0.5), first + 1), _TIMELINE_WIDTH
        )
        for i in range(first, last):
            cells[i] = "=" if on_path else "-"
    note = f" {span['note']}" if span.get("note") else ""
    return f"{start:>7.1f} {float(span.get('ms') or 0):>8.1f} {rtts:>6}  |{''.join(cells)}|{note}"
