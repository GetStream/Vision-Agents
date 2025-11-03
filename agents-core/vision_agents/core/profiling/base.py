import pyinstrument
import logging

from vision_agents.core.events import EventManager
from vision_agents.core.agents import events

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self, output_path='./profile.html'):
        self.output_path = output_path
        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.profiler = pyinstrument.Profiler()
        self.profiler.start()
        self.events.subscribe(self.on_finish)

    async def on_finish(self, event: events.AgentFinishEvent):
        self.profiler.stop()
        logger.info(f"Profiler stopped. Time file saved at: {self.output_path}")
        with open(self.output_path, 'w') as f:
            f.write(self.profiler.output_html())
