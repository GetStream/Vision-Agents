import logging

from vision_agents.core.events import EventManager
from vision_agents.core.agents import events

logger = logging.getLogger(__name__)


class Profiler:
    """Profiles agent execution using pyinstrument and generates an HTML report.

    The profiler automatically starts when instantiated and stops when the agent
    finishes (on AgentFinishEvent), saving an HTML performance report to disk.

    Example:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(name="Agent", id="agent"),
            llm=gemini.LLM("gemini-2.0-flash"),
            profiler=Profiler(output_path='./profile.html'),
        )
    """

    def __init__(self, output_path="./profile.html"):
        """Initialize the profiler.

        Args:
            output_path: Path where the HTML profile report will be saved.
                Defaults to './profile.html'.
        """
        import pyinstrument

        self.output_path = output_path
        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.profiler = pyinstrument.Profiler()
        self.profiler.start()
        self.events.subscribe(self.on_finish)

    async def on_finish(self, event: events.AgentFinishEvent):
        """Handle agent finish event by stopping profiler and saving report.

        Args:
            event: The AgentFinishEvent emitted when the agent finishes.
        """
        self.profiler.stop()
        logger.info(f"Profiler stopped. Time file saved at: {self.output_path}")
        with open(self.output_path, "w") as f:
            f.write(self.profiler.output_html())
