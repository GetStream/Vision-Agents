"""The accounts the app and its users hold at connectors."""

from typing import Any, List

from ._backend import Backend, acting_for
from ._generated import AuthenticatedClient
from ._generated.api.default import (
    create_authorization,
    create_connection,
    delete_connection,
    get_connection,
    list_connections,
)
from ._generated.models import (
    Authorization,
    AuthorizationRequest,
    Connection,
    ConnectionOwner,
    ConnectionOwnerType,
    ConnectionRequest,
)
from ._generated.types import UNSET
from .responses import _deleted, _set, _unwrapped

# The most connections the router answers in one page.
_PAGE = 200


class Connections:
    """The accounts the app and its users hold at connectors, such as a user's Linear: what an
    agent's tools reach that account through. Server side only.

    Every method takes the user as ``user_id``, named once per call. The user goes to the
    router as the owner and as ``X-Stream-User-Id`` alike, so the two never differ, whoever
    the client acts for otherwise. A user's connection is reached only naming that user; an
    app's is reached by every call, one naming a user included. No user is the app's own
    connection.

    Example:
        ```python
        connection = await api.connections.create("linear", user_id="alice")
        consent = await api.connections.authorize(
            connection.id, return_url="https://app.example.com/connected", user_id="alice"
        )
        # Open consent.launch_url in a popup and post it consent.handoff_token.
        ```
    """

    def __init__(self, backend: Backend):
        self._backend = backend

    async def create(
        self, connector: str, *, label: str = "", user_id: str = ""
    ) -> Connection:
        """Store a pending connection to ``connector``, owned by ``user_id`` or else by the app.

        Args:
            connector: A built-in, such as ``linear``, or one of the app's own.
            label: Tells the owner's connections to the same connector apart.
            user_id: The app's user the connection is for. Empty is the app's own.
        """
        owner = ConnectionOwner(type_=ConnectionOwnerType.CONNECTION_OWNER_TYPE_APP)
        if user_id:
            owner = ConnectionOwner(
                type_=ConnectionOwnerType.CONNECTION_OWNER_TYPE_USER, user_id=user_id
            )
        return await _unwrapped(
            create_connection.asyncio(
                client=self._client(user_id),
                body=ConnectionRequest(
                    connector_id=connector, owner=owner, **_set(label=label)
                ),
            ),
            f"creating a connection to {connector}",
        )

    async def authorize(
        self, id: str, *, return_url: str = "", user_id: str = ""
    ) -> Authorization:
        """Start the provider's consent for a connection: a first one for a pending
        connection, a reconnect for one connected before.

        Open ``launch_url`` in a popup from a page of ``return_url``'s origin and post it
        ``handoff_token``; the attempt ends at ``expires_at``.

        Args:
            id: The connection.
            return_url: The app's page the browser goes back to when the consent ends, with
                ``connection_id`` and ``status`` added to its query. Empty goes back to the
                dashboard.
            user_id: The user the connection belongs to. Empty is an app connection.
        """
        return await _unwrapped(
            create_authorization.asyncio(
                _id(id),
                client=self._client(user_id),
                body=AuthorizationRequest(**_set(return_url=return_url)),
            ),
            f"authorizing the connection {id}",
        )

    async def get(self, id: str, *, user_id: str = "") -> Connection:
        """One connection of ``user_id``'s, or of the app's when it is empty.

        Another user's is not found, the same as one that does not exist. An app's is
        reached with any user named too.
        """
        return await _unwrapped(
            get_connection.asyncio(_id(id), client=self._client(user_id)),
            f"reading the connection {id}",
        )

    async def list(self, *, connector: str = "", user_id: str = "") -> List[Connection]:
        """Every connection of ``user_id``'s, or of the app's when it is empty, newest
        first, read page by page. With a user it holds only that user's, not the app's.

        Args:
            connector: Keeps one connector's. Empty is every connector's.
            user_id: Whose connections to read. Empty reads the app's own.
        """
        owner = ConnectionOwnerType.CONNECTION_OWNER_TYPE_APP
        if user_id:
            owner = ConnectionOwnerType.CONNECTION_OWNER_TYPE_USER
        client = self._client(user_id)
        connections: List[Connection] = []
        cursor: Any = UNSET
        while True:
            page = await _unwrapped(
                list_connections.asyncio(
                    client=client,
                    owner_type=owner,
                    limit=_PAGE,
                    cursor=cursor,
                    **_set(connector_id=connector),
                ),
                "listing connections",
            )
            connections.extend(page.items or [])
            if not page.has_more or not page.next_cursor:
                return connections
            cursor = page.next_cursor

    async def delete(self, id: str, *, user_id: str = "") -> None:
        """Disconnect a connection of ``user_id``'s, or of the app's when it is empty, and drop
        its credentials. One an agent config binds as its fixed connection is refused with a
        409."""
        await _deleted(
            delete_connection.asyncio_detailed(_id(id), client=self._client(user_id)),
            f"deleting the connection {id}",
        )

    def _client(self, user_id: str) -> AuthenticatedClient:
        """The generated client acting for ``user_id``, or for nobody when it is empty,
        whoever this client acts for otherwise."""
        return acting_for(self._backend, user_id).client()


def _id(id: str) -> str:
    """The connection's id, refused before anything is sent when it is empty: it would ask
    for ``/v1/agents/connections/``, which names no connection."""
    if not id:
        raise ValueError("a connection needs its id")
    return id
