from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from ._backend import Backend
from ._generated.api.default import create_response, list_response_items, list_responses
from ._generated.models import (
    AgentResponse as ResponseRow,
    AgentResponseItem,
    CreateResponseRequest,
    Error,
    ImageSource,
)

logger = logging.getLogger(__name__)

# How many items are read per request while unwinding, and as many as the router will hand
# over at once.
ITEM_PAGE = 200
ITEM_CEILING = 1000


class RouterError(RuntimeError):
    """What the router said it would not do, raised where it was asked."""


class Items:
    """The things one or more turns were made of, in the order they happened.

    Read rather than watched: this is what the backend wrote down, so it reads the same
    whether the conversation is still going or ended last week. Deltas are not here -- a
    hundred fragments of one sentence are the sentence -- so a caller who wants to watch words
    arrive reads ``session.events()`` and a caller who wants the shape of a turn reads this.
    """

    def __init__(self, backend: Backend, session_id: str, response_id: str = ""):
        self._backend = backend
        self._session_id = session_id
        # Empty for every turn in the session, set for one turn's own items.
        self._response_id = response_id

    async def unwind(self, limit: int = ITEM_PAGE) -> AsyncIterator[AgentResponseItem]:
        """Yield every item, oldest first, fetching a page at a time.

        Paging is inside rather than outside because a conversation's length is not something
        the caller chose: ``async for item in session.responses.items.unwind()`` reads a turn
        or a thousand the same way.
        """
        page = min(limit or ITEM_PAGE, ITEM_CEILING)
        offset = 0
        while True:
            read = await self.list(limit=page, offset=offset)
            for item in read:
                yield item
            # A short page is the last page. Asking again to see an empty one would double
            # the requests for every conversation that happens to be a multiple of the page
            # size, which is not worth avoiding one extra round trip in the rare exact fit.
            if len(read) < page:
                return
            offset += len(read)

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> list[AgentResponseItem]:
        """One page of items, for a caller doing its own paging."""
        listed = await list_response_items.asyncio(
            self._session_id,
            client=self._backend.client(),
            **_set(response_id=self._response_id, limit=limit, offset=offset),
        )
        return _unwrapped(listed, f"reading the items of {self._session_id}")

    async def all(self) -> list[AgentResponseItem]:
        """Everything in one list, for a conversation short enough to hold."""
        return [item async for item in self.unwind()]


class AgentResponse:
    """One turn, and a way to read what it was made of.

    ``Responses.create`` returns as soon as the agent has started answering rather than when
    it has finished, because a model takes seconds and a request that waited them out would
    time out on anything worth asking. So this is a handle on an answer in progress:
    ``items`` reads what has been written down so far, and ``session.events()`` is what
    watches it arrive.
    """

    def __init__(self, backend: Backend, created: ResponseRow):
        self.created = created
        self.items = Items(backend, created.session_id, created.id)

    @property
    def id(self) -> str:
        """The backend's id for this turn, empty for a session that records nothing."""
        return self.created.id

    @property
    def status(self) -> str:
        return str(self.created.status)


class Responses:
    """A session's turns.

    ``items`` here is the whole conversation flattened, which is how a conversation reads and
    how it gets rendered: the question, what the agent did about it, what it said, then the
    next question. A single turn's items come off the handle ``create`` returns.
    """

    def __init__(self, backend: Backend, session_id: str):
        self._backend = backend
        self._session_id = session_id
        self.items = Items(backend, session_id)

    async def create(
        self, text: str, images: Optional[list[ImageSource]] = None
    ) -> AgentResponse:
        """Ask the agent something and name the turn it answers as.

        An incognito session records nothing, so the turn it hands back has no id: there is
        nothing to read back afterwards, which is what incognito means.
        """
        request = CreateResponseRequest(text=text)
        if images:
            request.images = images

        created = await create_response.asyncio(
            self._session_id, client=self._backend.client(), body=request
        )
        return AgentResponse(
            self._backend, _unwrapped(created, f"asking {self._session_id}")
        )

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> list[ResponseRow]:
        """The turns so far, oldest first."""
        listed = await list_responses.asyncio(
            self._session_id,
            client=self._backend.client(),
            **_set(limit=limit, offset=offset),
        )
        return _unwrapped(listed, f"reading the turns of {self._session_id}")


def _unwrapped(answer: Any, what: str) -> Any:
    """What the router answered with, or what it said instead.

    Every refusal in the spec is the same shape, so raising is the same three lines
    everywhere and worth having once.
    """
    if isinstance(answer, Error):
        raise RouterError(f"{what}: {answer.error}")
    if answer is None:
        raise RouterError(f"{what}: the router answered with nothing")
    return answer


def _set(**values: Any) -> dict[str, Any]:
    """The arguments a caller actually set.

    Anything left out keeps the router's own default rather than being sent as an empty
    value, which would replace a default with nothing.
    """
    return {name: value for name, value in values.items() if value not in (None, "", 0)}
