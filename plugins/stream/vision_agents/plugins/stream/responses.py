from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any, AsyncIterator, Awaitable, List, Optional, TypeVar, Union

from ._backend import Backend
from ._errors import RouterError
from ._generated.types import Response
from ._generated.api.default import (
    create_response,
    list_response_items,
    list_responses,
    rewind_session,
)
from ._generated.models import (
    AgentResponse as ResponseRow,
    AgentResponseItem,
    AgentResponseItemPage,
    AgentResponsePage,
    CreateResponseRequest,
    ImageSource,
    RewindSessionRequest,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# How many items are read per request while unwinding, and as many as the router will hand
# over at once.
ITEM_PAGE = 200
ITEM_CEILING = 1000


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
        cursor = ""
        while True:
            read = await self.list(limit=page, cursor=cursor)
            for item in read.items:
                yield item
            if not read.has_more or not read.next_cursor:
                return
            cursor = read.next_cursor

    async def list(
        self, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> AgentResponseItemPage:
        """One page of items, for a caller doing its own paging. An empty cursor is the first
        page, and the page's ``next_cursor`` the next."""
        return await _unwrapped(
            list_response_items.asyncio(
                self._session_id,
                client=self._backend.client(),
                **_set(response_id=self._response_id, limit=limit, cursor=cursor),
            ),
            f"reading the items of {self._session_id}",
        )

    async def all(self) -> List[AgentResponseItem]:
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

    def __init__(self, backend: Backend, session_id: str, kept: bool = False):
        self._backend = backend
        self._session_id = session_id
        self._kept = kept
        self.items = Items(backend, session_id)

    async def create(
        self,
        text: str,
        images: Optional[list[ImageSource]] = None,
        command_id: str = "",
    ) -> AgentResponse:
        """Ask the agent something and name the turn it answers as.

        An incognito session records nothing, so the turn it hands back has no id: there is
        nothing to read back afterwards, which is what incognito means.

        Args:
            text: The question.
            images: Pictures to ask about alongside it.
            command_id: Names the question, so a retry is answered once rather than twice.
                Left empty, a session kept in Stream Chat is given a fresh one, because the
                router requires one there.
        """
        request = CreateResponseRequest(text=text)
        if images:
            request.images = images
        # A command carries text only, so a question with images goes without one.
        if not command_id and self._kept and not images:
            command_id = str(uuid.uuid4())
        if command_id:
            request.command_id = command_id

        created = await _unwrapped(
            create_response.asyncio(
                self._session_id, client=self._backend.client(), body=request
            ),
            f"asking {self._session_id}",
        )
        return AgentResponse(self._backend, created)

    async def list(
        self, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> AgentResponsePage:
        """A page of the turns so far, oldest first. Pass the page's ``next_cursor`` for the
        next one."""
        return await _unwrapped(
            list_responses.asyncio(
                self._session_id,
                client=self._backend.client(),
                **_set(limit=limit, cursor=cursor),
            ),
            f"reading the turns of {self._session_id}",
        )

    async def rewind(
        self, to: Union[AgentResponse, ResponseRow, AgentResponseItem, str]
    ) -> None:
        """Go back to a response and carry on from there.

        The reply being spoken is abandoned and the conversation continues as though nothing
        after that response had been said: later turns are no longer listed, and the next
        question is answered from that point. The response itself is kept. A text
        conversation kept in Chat cannot be rewound, because its transcript lives there; fork
        it at the response instead.

        Args:
            to: The response to carry on from, any item of it, or its id.
        """
        response_id = to.response_id if isinstance(to, AgentResponseItem) else to
        if not isinstance(response_id, str):
            response_id = response_id.id
        if not response_id:
            raise ValueError(
                "that response has no id, which is what a session that records nothing "
                "hands back; there is nothing to rewind to"
            )

        await _asked(
            rewind_session.asyncio(
                self._session_id,
                client=self._backend.client(),
                body=RewindSessionRequest(response_id=response_id),
            ),
            f"rewinding {self._session_id}",
        )


async def _asked(call: Awaitable[T], what: str) -> T:
    """Await one request, naming what it was for in the RouterError it raises."""
    try:
        return await call
    except RouterError as refused:
        raise RouterError(
            f"{what}: {refused}",
            status=refused.status,
            type=refused.type,
            code=refused.code,
            doc_url=refused.doc_url,
            request_id=refused.request_id,
        ) from None


async def _unwrapped(call: Awaitable[Any], what: str) -> Any:
    """What the router answered with, or what it said instead.

    Every refusal is raised the same way, so it is the same few lines everywhere and
    worth having once.
    """
    answer = await _asked(call, what)
    if answer is None:
        raise RouterError(f"{what}: the router answered with nothing")
    return answer


async def _deleted(call: Awaitable[Response[Any]], what: str) -> None:
    """Raise what the router said instead of the 204 a delete answers with."""
    answer = await _asked(call, what)
    if answer.status_code != HTTPStatus.NO_CONTENT:
        raise RouterError(f"{what}: the router answered {answer.status_code}")


def _set(**values: Any) -> dict[str, Any]:
    """The arguments a caller actually set.

    Anything left out keeps the router's own default rather than being sent as an empty
    value, which would replace a default with nothing.
    """
    return {name: value for name, value in values.items() if value not in (None, "", 0)}
