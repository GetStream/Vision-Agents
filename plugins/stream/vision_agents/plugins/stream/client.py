from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from vision_agents.core.llm.function_registry import FunctionRegistry

from ._backend import Backend
from ._generated.api.default import (
    claim_guest_user,
    create_guest_user,
    list_agent_configs,
)
from ._generated.models import (
    AgentConfig,
    ClaimGuestRequest,
    ClaimGuestResult,
    GuestUser,
    GuestUserRequest,
    GuestUserRequestCustom,
)
from .responses import RouterError, _unwrapped
from .sessions import Sessions

logger = logging.getLogger(__name__)


@dataclass
class GuestOptions:
    """Who a guest is, as far as anybody knows yet.

    Attributes:
        id: A guest to reuse, for somebody coming back. Empty mints a new one. Asking for an
            id that is already a guest returns that guest with a fresh token rather than
            failing, because coming back is the same person.
        name: What to call them, for a transcript a person reads later.
        custom: Anything of the caller's own to keep against them.
    """

    id: str = ""
    name: str = ""
    custom: Optional[dict[str, Any]] = None


class Client:
    """The router held once: where it is, and who is calling it.

    ``Accelerated`` spells an agent out here and holds one conversation at a time. This is the
    other way round: an agent configured once in the backend and addressed by the name a
    person knows it as, with its conversations kept so they can be found again.

    Example:
        ```python
        api = stream.Client()
        docs = api.agent("docs")

        session = await docs.sessions.create(
            stream.SessionOptions(title="Is Stream better?", project="docs", persist=True)
        )
        answer = await session.responses.create("Is Stream better than Sendbird?")
        async for item in answer.items.unwind():
            print(item.kind, item.text)
        await session.close()
        ```
    """

    def __init__(
        self,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        token: str = "",
        user_id: str = "",
        authenticate: Optional[bool] = None,
        backend: Optional[Backend] = None,
    ):
        """Point a client at a router.

        Every argument falls back to the environment, so a process deployed next to a router
        needs none of them. Pass ``backend`` to hand one in already built.

        Args:
            url: The router's base URL. Defaults to ``STREAM_ACCELERATION_URL``.
            customer_id: Who the work is billed to, for a router that trusts the header.
            api_key: The public half of a Stream credential.
            api_secret: The secret belonging to it, which is what makes this a backend.
            token: A token minted for ``user_id`` to hold, in place of the secret.
            user_id: The end user this client is acting for, if it is acting for one.
            authenticate: Whether the router is reached through Stream's proxy.
            backend: A backend to use as it is, instead of building one from the above.
        """
        self.backend = backend or Backend(
            url=url,
            customer_id=customer_id,
            api_key=api_key,
            api_secret=api_secret,
            token=token,
            user_id=user_id,
            authenticate=authenticate,
        )

    @property
    def server_side(self) -> bool:
        """Whether this speaks for a process the customer runs rather than for a device."""
        return self.backend.server_side

    def as_user(self, user: Union[str, dict[str, Any]], token: str) -> "Client":
        """A client acting for one end user, holding the token that proves it.

        A new client rather than a change to this one, because a process usually holds both:
        its own credential for the things only a backend may do, and one per user for the
        conversations that belong to them. Sharing one and switching the user on it would make
        which user a request was for depend on when it happened to run.
        """
        return Client(backend=self.backend.as_user(user, token))

    def as_guest(self, guest: GuestUser) -> "Client":
        """A client acting for a guest, which is what their conversations belong to.

        The same thing ``as_user`` does, said the way the guest paths say it: a guest is a user
        with a token, and everything after this point treats them as one.
        """
        return self.as_user(
            {"id": guest.id, "name": guest.name or "Guest"}, guest.token
        )

    def agent(self, name: str) -> "Agent":
        """An agent addressed by the name it is configured under.

        No request is made: this is the name in a wrapper, and a name that matches nothing
        configured is refused when a conversation is opened rather than here.
        """
        return Agent(self.backend, name)

    async def guest_user(self, options: Optional[GuestOptions] = None) -> GuestUser:
        """Mint a guest so somebody can talk to an agent before they sign up.

        Nothing is remembered here. A Python process is a backend, and one handling two
        visitors that remembered a guest would hand them each other's conversations;
        remembering which visitor is which is the caller's job, because only the caller knows
        what a visitor is. The browser SDK does keep one, in a cookie, because there a page is
        one person.
        """
        options = options or GuestOptions()
        request = GuestUserRequest()
        if options.id:
            request.id = options.id
        if options.name:
            request.name = options.name
        if options.custom:
            request.custom = GuestUserRequestCustom.from_dict(options.custom)

        minted = await create_guest_user.asyncio(
            client=self.backend.client(), body=request
        )
        return _unwrapped(minted, "minting a guest")

    async def claim_guest_user(self, guest_id: str, user_id: str) -> ClaimGuestResult:
        """Move a guest's conversations onto the account they turned out to be.

        Server side only, and the one thing here that most needs to be: only the app's own
        backend knows that a given guest is a given account, because it is the thing that just
        authenticated them. A page able to ask this could claim anybody's conversations by
        guessing a guest id, so the router refuses it from one -- this raises first, with a
        reason, rather than letting a 403 be the explanation.
        """
        if not self.server_side:
            raise RouterError(
                "claiming a guest is server side only: it is the backend that just "
                "authenticated the account that knows which guest it was"
            )
        if not guest_id or not user_id:
            raise RouterError("claiming a guest needs the guest and the account")

        claimed = await claim_guest_user.asyncio(
            client=self.backend.client(),
            body=ClaimGuestRequest(guest_id=guest_id, user_id=user_id),
        )
        return _unwrapped(claimed, f"claiming the guest {guest_id}")


class Agent:
    """One configured agent, and its conversations."""

    def __init__(self, backend: Backend, name: str):
        self.name = name
        # The functions are held on the agent rather than per session, because a function
        # registered once should not have to be registered again for the next conversation.
        self.functions = FunctionRegistry()
        self.sessions = Sessions(backend, name, self.functions)
        self._backend = backend

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters_schema: Optional[dict[str, Any]] = None,
    ):
        """Register one of the caller's functions, which every conversation offers the model.

        The decorator the core registry provides, so the schema comes from the type hints and
        the description from the docstring:

            @agent.register()
            async def get_weather(location: str) -> str:
                "Get current weather for a location"
                return await weather_at(location)
        """
        return self.functions.register(
            name=name, description=description, parameters_schema=parameters_schema
        )

    async def config(self) -> Optional[AgentConfig]:
        """How the agent is configured, or None for a name nothing is stored under.

        Server side only: how an agent is configured is not a device's to read.
        """
        listed = await list_agent_configs.asyncio(
            client=self._backend.client(), name=self.name
        )
        for stored in _unwrapped(listed, f"looking up the agent {self.name}"):
            if stored.name == self.name:
                return stored
        return None
