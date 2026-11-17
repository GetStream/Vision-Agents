import asyncio
import json
import logging
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.core.llm import ImageContent
from vision_agents.core.llm.llm import jpeg_bytes
from vision_agents.core.llm.remote import RemotePipelineError
from vision_agents.plugins import stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A voice agent watching a door camera.

Voice runs in the Go acceleration backend. This process joins the same call as the video
worker and runs two detectors on the camera through the router's /v1/detection: one asks
Moondream for people and packages, the other runs Roboflow's face Workflow, which returns
an embedding for every face. Frames that barely changed are never sent.

What comes and goes is written to an activity log and, a few seconds later, told to the
agent as a [camera] message so it speaks up unprompted. A package that vanishes while
someone is at the door is logged as picked up by them. remember_me stores whoever is
facing the camera in faces.json, so they are named from then on.
"""

FACES_FILE = Path(__file__).parent / "faces.json"
# Cosine similarity at which a face is the person enrolled. The face Workflow embeds with
# CLIP, which tells people apart well enough for a demo and is no face recognition model:
# two photos of one person scored 0.92 and two people 0.76. Lower this if enrolled people
# go unnamed, raise it if strangers are named.
MATCH_THRESHOLD = 0.85
# How long camera events are gathered before the agent is told, so a person and the
# package they carry are one message rather than two.
ANNOUNCE_AFTER_S = 3.0
# How long a package has to stay gone before it counts as picked up rather than missed.
PICKUP_CONFIRM_S = 3.0
# How long after a person was last seen they still count as having been at the door.
PRESENT_WITHIN_MS = 10_000


class Faces:
    """The people enrolled with remember_me, kept in faces.json."""

    def __init__(self, path: Path):
        self._path = path
        self._known: dict[str, list[list[float]]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._known)

    def identify(self, embedding: list[float]) -> str:
        """The enrolled person this face is most like, or an empty string."""
        best, best_score = "", MATCH_THRESHOLD
        for name, embeddings in self._known.items():
            for known in embeddings:
                score = _cosine(embedding, known)
                if score >= best_score:
                    best, best_score = name, score
        return best

    def remember(self, name: str, embedding: list[float]) -> None:
        self._known.setdefault(name, []).append(embedding)
        self._path.write_text(json.dumps(self._known))


class Camera:
    """What the two detectors saw, as a log and as messages for the agent."""

    def __init__(
        self,
        agent: Agent,
        objects: stream.DetectionProcessor,
        faces: stream.DetectionProcessor,
    ):
        self._agent = agent
        self._objects = objects
        self._faces = faces
        self.log: deque[dict[str, object]] = deque(maxlen=200)
        self._pending: list[str] = []
        self._announcing: asyncio.Task[None] | None = None
        # Who was at the door and when they were last seen, in Unix milliseconds.
        self._seen: dict[str, int] = {}

    async def on_appeared(self, event: stream.ObjectAppearedEvent) -> None:
        track = event.track
        if track is None:
            return
        if event.plugin_name == "faces":
            if track.identity:
                self._seen[track.identity] = track.last_seen_ms
                self.record(f"{track.identity} is at the door", track.first_seen_ms)
            return
        if track.label == "person":
            self._seen["unknown"] = track.last_seen_ms
            self.record("a person came into view", track.first_seen_ms)
        elif track.label == "package":
            self.record("a package appeared", track.first_seen_ms)

    async def on_disappeared(self, event: stream.ObjectDisappearedEvent) -> None:
        track = event.track
        if track is None:
            return
        if event.plugin_name == "faces":
            if track.identity:
                self._seen[track.identity] = track.last_seen_ms
            return
        if track.label == "person":
            self._seen["unknown"] = track.last_seen_ms
            self.record("the person left", track.last_seen_ms)
        elif track.label == "package":
            asyncio.create_task(self._confirm_pickup(track))

    def record(self, what: str, at_ms: int) -> None:
        """Log an event and tell the agent about it once the moment has passed."""
        at = datetime.fromtimestamp(at_ms / 1000).strftime("%H:%M:%S")
        self.log.append({"at": at, "at_ms": at_ms, "event": what})
        logger.info("camera: %s at %s", what, at)
        self._pending.append(f"{what} at {at}")
        if self._announcing is None:
            self._announcing = asyncio.create_task(self._announce())

    def present(self, at_ms: int) -> list[str]:
        """Who was at the door around ``at_ms``, names first."""
        for track in self._faces.tracks:
            if track.identity:
                self._seen[track.identity] = track.last_seen_ms
        if any(track.label == "person" for track in self._objects.tracks):
            self._seen["unknown"] = at_ms
        names = [
            name
            for name, seen in self._seen.items()
            if at_ms - seen <= PRESENT_WITHIN_MS and name != "unknown"
        ]
        if names:
            return sorted(names)
        unknown = self._seen.get("unknown")
        if unknown is not None and at_ms - unknown <= PRESENT_WITHIN_MS:
            return ["an unknown person"]
        return []

    async def _confirm_pickup(self, package: stream.Track) -> None:
        await asyncio.sleep(PICKUP_CONFIRM_S)
        if any(
            track.label == "package" and stream.iou(track.box, package.box) > 0
            for track in self._objects.tracks
        ):
            return
        who = self.present(package.last_seen_ms)
        if who:
            self.record(
                f"the package was picked up by {' and '.join(who)}",
                package.last_seen_ms,
            )
        else:
            self.record(
                "the package is gone, with nobody seen taking it", package.last_seen_ms
            )

    async def _announce(self) -> None:
        await asyncio.sleep(ANNOUNCE_AFTER_S)
        lines, self._pending = self._pending, []
        self._announcing = None
        if not isinstance(self._agent.llm, stream.Accelerated):
            return
        # The frame goes with the words so the agent can say what it sees, and because a
        # message with an image is answered after whatever the agent is saying, where
        # one without is refused while another reply is running.
        frame = self._objects.latest_frame(annotated=False)
        images = [ImageContent(data=jpeg_bytes(frame))] if frame is not None else None
        try:
            await self._agent.llm.respond_remote(
                "[camera] " + "; ".join(lines), interrupt=False, images=images
            )
        except RemotePipelineError:
            logger.info("not on a call, so nobody was told: %s", lines)


async def create_agent(**kwargs) -> Agent:
    known = Faces(FACES_FILE)
    objects = stream.DetectionProcessor(
        target="detection-open",
        classes=["person", "package"],
        min_area=0.005,
        name="objects",
        refresh_interval=10.0,
    )
    faces = stream.DetectionProcessor(
        target="detection-faces",
        embeddings=True,
        min_confidence=0.5,
        identify=known.identify,
        name="faces",
        refresh_interval=10.0,
    )
    agent = Agent(config="security_camera", processors=[objects, faces])
    camera = Camera(agent, objects, faces)
    agent.events.subscribe(camera.on_appeared)
    agent.events.subscribe(camera.on_disappeared)

    @agent.llm.register_function(
        description=(
            "Remember the person facing the camera by name, so they are recognised "
            "from now on. Use when someone says 'remember me as NAME' or 'I'm NAME'."
        )
    )
    async def remember_me(name: str) -> dict[str, object]:
        in_view = [
            (track, track.embedding) for track in faces.tracks if track.embedding
        ]
        if not in_view:
            return {"error": "no face in view; ask them to look at the camera"}
        # The largest face is the one nearest the camera, who is the one asking.
        face, embedding = max(
            in_view,
            key=lambda pair: (
                (pair[0].box[2] - pair[0].box[0]) * (pair[0].box[3] - pair[0].box[1])
            ),
        )
        known.remember(name, embedding)
        face.identity = name
        camera.record(f"{name} was enrolled", int(time.time() * 1000))
        return {"remembered": name, "known_people": known.names}

    @agent.llm.register_function(
        description=(
            "What the camera saw, oldest first: arrivals, departures, packages appearing "
            "and being picked up. Each entry has a time; at_ms is what to pass to "
            "the vision skill to look at that moment."
        )
    )
    async def get_activity_log(limit: int = 20) -> dict[str, object]:
        entries = list(camera.log)[-limit:]
        return {"events": entries, "known_people": known.names}

    return agent


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create(
            "greet the user in one short sentence and say you are watching the door"
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
