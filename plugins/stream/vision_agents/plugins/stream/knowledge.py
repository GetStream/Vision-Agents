import asyncio
import logging
import time

from vision_agents.core.llm.remote import KnowledgePage

from ._backend import Backend
from ._generated.api.default import add_knowledge_url, get_knowledge_url
from ._generated.models import (
    Error,
    KnowledgeUrl,
    KnowledgeUrlRequest,
    KnowledgeUrlState,
)

logger = logging.getLogger(__name__)

# The router queues the read and retries one that fails, so a page can take a while to
# settle. Past this it is returned still pending rather than waited on forever.
READ_TIMEOUT = 180.0
POLL_INTERVAL = 0.25


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

    async def add_url(
        self, url: str, title: str = "", description: str = ""
    ) -> KnowledgePage:
        """Keep the knowledge base filled from a page published elsewhere.

        The router queues the read and cuts the page into passages the same way a
        document is; this waits for it, so what comes back already says whether it
        worked. It stays a subscription rather than a one-off: the passages are keyed by
        the url, and reading it again replaces them.

        Args:
            url: The http or https address to read.
            title: What to call the page. Defaults to what it calls itself.
            description: What the page is, in the caller's own words.

        Returns:
            The page as stored, including how many passages it became and why it failed if
            it did. Still pending if it was not read within READ_TIMEOUT seconds.
        """
        if not self.namespace:
            raise ValueError(
                "a knowledge base is named by the agent it belongs to; build the agent "
                "from a stored config, e.g. Agent(config=...)"
            )

        body = KnowledgeUrlRequest(namespace=self.namespace, url=url)
        if title:
            body.title = title
        if description:
            body.description = description

        added = await add_knowledge_url.asyncio(client=self.backend.client(), body=body)
        page = await self._settled(self._page(added))

        logger.info(
            "%s is %s in %s as %d passages",
            url,
            page.state.value,
            self.namespace,
            page.passages,
        )
        return KnowledgePage(
            url=page.url,
            state=page.state.value,
            passages=page.passages,
            error=page.error if isinstance(page.error, str) else "",
        )

    async def _settled(self, page: KnowledgeUrl) -> KnowledgeUrl:
        """Wait for the router to have read a page, or for READ_TIMEOUT to pass."""
        deadline = time.monotonic() + READ_TIMEOUT
        while page.state == KnowledgeUrlState.PENDING and time.monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            page = self._page(
                await get_knowledge_url.asyncio(page.id, client=self.backend.client())
            )
        return page

    @staticmethod
    def _page(answer: KnowledgeUrl | Error | None) -> KnowledgeUrl:
        if isinstance(answer, Error):
            raise RuntimeError(answer.error)
        if answer is None:
            raise RuntimeError("the router did not answer with a page")
        return answer
