"""Edge Transport Package.

This package provides edge transport abstractions for Stream Agents.
"""

from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge import sfu_events

__all__ = ["EdgeTransport", "sfu_events"]

# LocalTransport is imported lazily to avoid errors when sounddevice is not installed
# Users should import directly: from vision_agents.core.edge.local_transport import LocalTransport
