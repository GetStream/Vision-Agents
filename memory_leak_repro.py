"""
Memory leak reproduction script for Vision-Agents v0.4.0.

Key insight: Python's cyclic GC CAN collect simple reference cycles. The real
leak happens when asyncio Tasks are still pending — they hold strong refs to
coroutines/closures that prevent GC. In production, handler tasks often involve
real async I/O (STT websockets, TTS streaming, LLM calls) that take time to
complete. If close() doesn't cancel these tasks, they keep the entire Agent
graph alive.

Usage:
    uv run python memory_leak_repro.py          # Shows leak
    uv run python memory_leak_repro.py --fix    # Shows fix
"""

import argparse
import asyncio
import gc
import os
import tracemalloc
import weakref
from dataclasses import dataclass

os.environ.setdefault("STREAM_API_KEY", "dummy")
os.environ.setdefault("STREAM_API_SECRET", "dummy")

from vision_agents.core.events.base import BaseEvent
from vision_agents.core.events.manager import EventManager


@dataclass
class FakeAudioEvent(BaseEvent):
    type: str = "test.audio"
    data: bytes = b""


@dataclass
class FakeTranscriptEvent(BaseEvent):
    type: str = "test.transcript"
    text: str = ""


class FakePlugin:
    def __init__(self, name: str):
        self.name = name
        self.events = EventManager()
        self.events.register(FakeAudioEvent, FakeTranscriptEvent)
        self._buffer = bytearray(256 * 1024)


class FakeAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.events = EventManager()
        self.events.register(FakeAudioEvent, FakeTranscriptEvent)

        self.stt = FakePlugin("stt")
        self.tts = FakePlugin("tts")
        self.llm = FakePlugin("llm")

        self.events.merge(self.stt.events)
        self.events.merge(self.tts.events)
        self.events.merge(self.llm.events)

        # ~1MB agent state
        self._state = bytearray(1 * 1024 * 1024)
        self._setup_handlers()

    def _setup_handlers(self):
        @self.events.subscribe
        async def on_audio(event: FakeAudioEvent):
            # Simulate real STT processing — this is the key difference!
            # In production, Deepgram/ElevenLabs handlers await websocket I/O.
            # These pending tasks hold refs to `self` via the closure.
            await asyncio.sleep(0.5)
            _ = self.session_id

        @self.events.subscribe
        async def on_transcript(event: FakeTranscriptEvent):
            # Simulate LLM processing delay
            await asyncio.sleep(1.0)
            _ = self.session_id

    async def simulate_work(self, n_events: int = 20):
        """Send fewer events but with slow handlers (like real I/O)."""
        for i in range(n_events):
            self.events.send(FakeAudioEvent(data=b"\x00" * 960))
            if i % 10 == 0:
                self.events.send(FakeTranscriptEvent(text=f"word {i}"))
        # DON'T wait for all tasks to finish — simulates abrupt session end
        await asyncio.sleep(0.05)

    async def close(self, apply_fix: bool = False):
        if apply_fix:
            # FIX 1: Cancel all pending handler tasks
            for task in self.events._handler_tasks.values():
                if not task.done():
                    task.cancel()
            self.events._handler_tasks.clear()

            # FIX 2: Clear handlers to release closures
            self.events._handlers.clear()

            # FIX 3: Clear queue
            self.events._queue.clear()

        self.events.stop()

        if apply_fix:
            self.stt = None  # type: ignore[assignment]
            self.tts = None  # type: ignore[assignment]
            self.llm = None  # type: ignore[assignment]
            self._state = None  # type: ignore[assignment]


async def run_simulation(n_sessions: int, apply_fix: bool):
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    weak_refs: list[weakref.ref] = []

    print(f"\n{'=' * 60}")
    print(f"  Mode: {'WITH FIX' if apply_fix else 'WITHOUT FIX (leaky)'}")
    print(f"  Sessions: {n_sessions}")
    print("  Each agent: ~1.75MB state + slow async handlers")
    print(f"{'=' * 60}\n")

    for i in range(n_sessions):
        agent = FakeAgent(session_id=f"session-{i}")
        weak_refs.append(weakref.ref(agent))

        await agent.simulate_work(n_events=20)
        await agent.close(apply_fix=apply_fix)
        del agent

        if (i + 1) % 3 == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            alive = sum(1 for ref in weak_refs if ref() is not None)
            print(
                f"  After {i + 1:3d} sessions: "
                f"mem={_fmt(current - baseline):>8s}  "
                f"peak={_fmt(peak - baseline):>8s}  "
                f"alive_agents={alive}/{i + 1}"
            )

    # Wait a bit then GC
    await asyncio.sleep(0.5)
    gc.collect()
    gc.collect()

    current, peak = tracemalloc.get_traced_memory()
    alive = sum(1 for ref in weak_refs if ref() is not None)

    print(f"\n{'─' * 60}")
    print(
        f"  FINAL: mem={_fmt(current - baseline):>8s}  peak={_fmt(peak - baseline):>8s}"
    )
    print(f"  Agents still in memory: {alive}/{n_sessions}")
    if alive > 0 and not apply_fix:
        print(f"  ⚠️  LEAK: {alive} agents NOT garbage collected!")
        print(f"  Leaked ~{_fmt(alive * 1_800_000)} of agent state")
    elif alive == 0:
        print("  ✓  All agents garbage collected.")
    print(f"{'─' * 60}\n")

    tracemalloc.stop()
    return alive


def _fmt(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply memory leak fix")
    parser.add_argument("-n", type=int, default=15, help="Number of sessions")
    args = parser.parse_args()

    leaked = await run_simulation(args.n, apply_fix=args.fix)

    if not args.fix and leaked > 0:
        print("Run with --fix to see the fix in action.\n")


if __name__ == "__main__":
    asyncio.run(main())
