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


def render_connection(frame: dict[str, Any]) -> str:
    """Show how one media path connected, from the router starting to join to DTLS done."""
    peer = frame.get("peer", "")
    total = frame.get("total_ms") or 0
    lines = [
        f"webrtc connection DAG peer={peer} | edge join start -> DTLS connected ~{total:,.0f} ms",
        "  [edge join start]",
    ]
    for step in frame.get("steps") or []:
        duration = step.get("ms") or 0
        lines.extend(
            ("       |", "       v", f"  [{step.get('name', '')} {duration:,.0f} ms]")
        )
    first_media = frame.get("first_media_ms") or 0
    if first_media:
        lines.append(f"       +-- first RTP at ~{first_media:,.0f} ms")
    return "\n".join(lines)
