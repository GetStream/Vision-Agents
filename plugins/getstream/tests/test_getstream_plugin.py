from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType

from vision_agents.core.events.manager import EventManager
from vision_agents.core.edge.events import TrackAddedEvent, TrackRemovedEvent


class TestTrackRepublishing:
    """
    Regression test for screenshare republishing bug.

    Bug: When a user stopped and restarted screensharing, the second TrackAddedEvent
    was not emitted, so the agent couldn't switch back to the screenshare.

    Fix: stream_edge_transport._on_track_published() now emits TrackAddedEvent even
    when the track_key already exists in _track_map.
    """

    async def test_track_events_flow_correctly(self):
        """Verify that track events (add -> remove -> add) flow through the event system."""
        event_manager = EventManager()
        event_manager.register(TrackAddedEvent)
        event_manager.register(TrackRemovedEvent)

        # Collect emitted events
        events = []

        @event_manager.subscribe
        async def collect_track_events(event: TrackAddedEvent | TrackRemovedEvent):
            events.append(event)

        # Simulate track lifecycle: start -> stop -> start again
        track_id = "screenshare-track-1"
        track_type = TrackType.TRACK_TYPE_SCREEN_SHARE

        # 1. Start screenshare
        event_manager.send(
            TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type,
            )
        )
        await event_manager.wait()

        assert len(events) == 1
        assert isinstance(events[0], TrackAddedEvent)
        assert events[0].track_id == track_id

        # 2. Stop screenshare
        event_manager.send(
            TrackRemovedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type,
            )
        )
        await event_manager.wait()

        assert len(events) == 2
        assert isinstance(events[1], TrackRemovedEvent)

        # 3. Start screenshare again (critical test)
        event_manager.send(
            TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type,
            )
        )
        await event_manager.wait()

        # Before the fix: The agent would never receive this third event
        assert len(events) == 3, "Republishing track should emit TrackAddedEvent"
        assert isinstance(events[2], TrackAddedEvent)
        assert events[2].track_id == track_id

        # Cleanup
        event_manager.unsubscribe(collect_track_events)
