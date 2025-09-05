from .agents import Agent as Agent
from getstream import Stream as Stream
from stream_agents.edge import StreamEdge as StreamEdge
from stream_agents.cli import start_dispatcher as start_dispatcher
from stream_agents.utils import open_demo as open_demo


from importlib import metadata

try:
    __version__ = metadata.version("stream_agents")
except metadata.PackageNotFoundError:  # editable/checkout scenarios
    __version__ = "0.0.0"

# Extend namespace to allow for plugins
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
