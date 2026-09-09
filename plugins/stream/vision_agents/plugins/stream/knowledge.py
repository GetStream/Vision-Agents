import logging

from vision_agents.core.llm.remote import KnowledgePage

from ._backend import Backend
from ._generated.api.default import add_knowledge_url
from ._generated.models import Error, KnowledgeUrlRequest

logger = logging.getLogger(__name__)


class Knowledge:
    """A knowledge base on the acceleration backend, as somewhere to put more of it.

    The namespace is the agent's own name, which is where the knowledge in a directory
    synced with `sync_agent` lands, so what is added here is found by the same lookup
    mid-answer rather than a second one.
    """

    def __init__(self, namespace: str, backend: Backend):
        """Name the knowledge base to fill.

        Args:
            namespace: The agent whose knowledge base this is.
            backend: The router holding it.
        """
        self.namespace = namespace
        self.backend = backend

    async def add_url(self, url: str) -> KnowledgePage:
        """Keep the knowledge base filled from a page published elsewhere.

        The page is read straight away and cut into passages the same way a document is,
        so what comes back already says whether it worked. It stays a subscription rather
        than a one-off: the passages are keyed by the url, and reading it again replaces
        them.

        Args:
            url: The http or https address to read.

        Returns:
            The page as stored, including how many passages it became and why it failed if
            it did.
        """
        if not self.namespace:
            raise ValueError(
                "a knowledge base is named by the agent it belongs to; build the agent "
                "from a stored config, e.g. Agent(config=...)"
            )

        added = await add_knowledge_url.asyncio(
            client=self.backend.client(),
            body=KnowledgeUrlRequest(namespace=self.namespace, url=url),
        )
        if isinstance(added, Error):
            raise RuntimeError(added.error)
        if added is None:
            raise RuntimeError("the router did not answer with a page")

        logger.info(
            "read %s into %s as %d passages", url, self.namespace, added.passages
        )
        return KnowledgePage(
            url=added.url,
            state=added.state.value,
            passages=added.passages,
            error=added.error if isinstance(added.error, str) else "",
        )
