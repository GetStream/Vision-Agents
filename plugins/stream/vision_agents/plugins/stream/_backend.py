import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from ._generated import AuthenticatedClient

URL_ENV = "STREAM_ACCELERATION_URL"
CUSTOMER_ENV = "STREAM_ACCELERATION_CUSTOMER_ID"
API_KEY_ENV = "STREAM_API_KEY"
API_SECRET_ENV = "STREAM_API_SECRET"
AUTHENTICATE_ENV = "STREAM_ACCELERATION_AUTHENTICATE"

CUSTOMER_HEADER = "X-Customer-Id"
DEFAULT_URL = "http://localhost:8080"

# How long a token minted here lasts. Short, because it is minted per request and a stolen
# one should stop working sooner than the credential behind it.
TOKEN_VALIDITY_SECONDS = 60 * 60


@dataclass
class Backend:
    """Where the acceleration router is, and who is calling it.

    There are three ways to say who that is, and which one a deployment takes is a property
    of the deployment rather than a choice: a customer id for a router with nothing in front
    of it, a key and secret for a process the customer runs, and a key and a user token for
    anything acting on one person's behalf.

    Attributes:
        url: The router's base URL. Defaults to ``STREAM_ACCELERATION_URL``, then localhost.
        customer_id: Who the work is billed to, taken at face value. What a router running
            without keys in front of it reads. Defaults to
            ``STREAM_ACCELERATION_CUSTOMER_ID``.
        api_key: The public half of a Stream credential. Defaults to ``STREAM_API_KEY``.
        api_secret: The secret belonging to that key, which mints a server-side token.
            Server side only. Defaults to ``STREAM_API_SECRET``.
        token: A token somebody else minted for ``user_id`` to hold, used instead of signing
            one here. With it the secret is not needed at all, which is the point: a token is
            the whole credential, and the secret behind it could mint any other.
        user_id: The end user this client is acting for, if it is acting for one. Empty
            speaks for the app itself, which is what a backend does and what keeps the
            per-user daily limits out of it.
        authenticate: Whether the router is reached through Stream's authenticating proxy,
            which is what every hosted deployment sits behind. Opt in rather than inferred
            from holding a credential, because a Stream key and secret are in the environment
            for plenty of reasons that have nothing to do with this router. Defaults to
            ``STREAM_ACCELERATION_AUTHENTICATE``.
        user: The user this is acting for, as chat and video want them. The id is what the
            router reads; a name is what a transcript shows without a second lookup.
    """

    url: Optional[str] = None
    customer_id: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    token: str = ""
    user_id: str = ""
    authenticate: Optional[bool] = None
    user: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.url = (self.url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.customer_id = self.customer_id or os.environ.get(CUSTOMER_ENV, "")
        # Naming a customer is choosing how a router with nothing in front of it is reached,
        # and a key that happens to be in the environment does not overrule the choice.
        # Otherwise pointing a client at a local router from a shell that has a Stream key in
        # it sends the credential that router does not read, and it answers that the customer
        # header is missing.
        if self.api_key is None:
            self.api_key = "" if self.customer_id else os.environ.get(API_KEY_ENV, "")
        # A token that was handed in is the caller's answer to who they are, so an ambient
        # secret does not overrule it: without this, a client built the way a device builds
        # one turns into a backend as soon as it runs in a process that happens to have the
        # secret in its environment, which is most of them and every test.
        if self.api_secret is None:
            self.api_secret = "" if self.token else os.environ.get(API_SECRET_ENV, "")
        if self.authenticate is None:
            self.authenticate = _flag(os.environ.get(AUTHENTICATE_ENV))

        if self.user and not self.user_id:
            self.user_id = str(self.user.get("id", ""))

        # Falling back to the customer header would send a request the proxy refuses, and
        # report it as whatever the proxy says rather than as what it is. Checked here rather
        # than at first use because it is a contradiction in what was passed.
        if self.authenticate and not self.api_key:
            raise ValueError(
                "a router behind the proxy is reached with a credential; pass api_key or "
                f"set {API_KEY_ENV}"
            )
        if not self.api_key and not self.customer_id:
            raise ValueError(
                f"who is calling is not set; pass customer_id or set {CUSTOMER_ENV} for a "
                "router that trusts one, or api_key with either api_secret or token"
            )
        if self.api_key and not self.api_secret and not self.token:
            raise ValueError(
                "api_key needs the secret it belongs to, or a token minted with it"
            )

    @property
    def server_side(self) -> bool:
        """Whether this speaks for a process the customer runs rather than for a device.

        Only a server-side caller reaches the operations the spec does not mark client
        accessible, which is everything about how an agent is configured and the claiming of
        a guest. Worth asking before a call rather than reading a 403 afterwards.
        """
        return bool(self.api_secret) or (not self.api_key and bool(self.customer_id))

    def as_user(self, user: Union[str, dict[str, object]], token: str) -> "Backend":
        """A backend acting for one end user, holding the token that proves it.

        A new backend rather than a change to this one, because a process usually holds both:
        its own credential for the things only a backend may do, and one per user for the
        conversations that belong to them. Sharing one and switching the user on it would make
        which user a request was for depend on when it happened to run.
        """
        named = {"id": user} if isinstance(user, str) else dict(user)
        if not named.get("id"):
            raise ValueError("a user needs an id")
        if not token:
            raise ValueError(f"there is no token for {named['id']} to hold")

        return Backend(
            url=self.url,
            customer_id=self.customer_id,
            api_key=self.api_key,
            # A token is the whole credential, so holding one stops this being a backend: a
            # client told who it is acting for should not keep the ability to speak for the
            # app itself.
            api_secret="",
            token=token,
            user_id=str(named["id"]),
            authenticate=self.authenticate,
            user=named,
        )

    @property
    def headers(self) -> dict[str, str]:
        """What every request to the router carries.

        Minted per read, so a client left idle longer than a token lasts does not wake up
        holding an expired one.
        """
        if not self.api_key:
            return {CUSTOMER_HEADER: str(self.customer_id)}

        if self.authenticate:
            # ``jwt`` whoever the token is for. The proxy works out the caller from the token
            # it verified and rewrites who the router is told is calling, so saying ``server``
            # here would be claiming what the proxy is there to decide -- and it refuses it.
            return {
                "api_key": str(self.api_key),
                "stream-auth-type": "jwt",
                "Authorization": f"Bearer {self._proxy_token()}",
            }

        headers = {"X-Api-Key": str(self.api_key)}
        if self.api_secret:
            headers["Authorization"] = f"Bearer {self._server_token()}"
            headers["Stream-Auth-Type"] = "server"
            if self.user_id:
                headers["X-Stream-User-Id"] = self.user_id
            return headers

        headers["Authorization"] = f"Bearer {self.token}"
        headers["Stream-Auth-Type"] = "jwt"
        return headers

    def client(self) -> AuthenticatedClient:
        """An HTTP client for the generated API, already carrying the credentials.

        The generated client puts one credential in one header, so the rest go in as plain
        headers. Built per call rather than kept, which is also what keeps a minted token
        fresh.
        """
        if not self.api_key:
            return AuthenticatedClient(
                base_url=str(self.url),
                token=str(self.customer_id),
                auth_header_name=CUSTOMER_HEADER,
                prefix="",
            )

        credentials = dict(self.headers)
        authorization = credentials.pop("Authorization", "")
        return AuthenticatedClient(
            base_url=str(self.url),
            token=authorization.removeprefix("Bearer "),
            auth_header_name="Authorization",
            prefix="Bearer",
            headers=credentials,
        )

    def socket(self, path: str) -> str:
        """The WebSocket URL for a path on the router.

        The credentials are not in it: a socket opened from here carries the same headers a
        request does, which the query string only exists to work around for a browser
        WebSocket that cannot send any.
        """
        base = str(self.url)
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :] + path
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :] + path
        return base + path

    def stream_credentials(self) -> Optional[dict[str, object]]:
        """What Stream's own chat and video clients need to connect, or None.

        None for a backend reached by customer id: that is this router's own way of trusting
        a caller and means nothing to Stream, so there is no credential to pass on. A
        server-side backend gets a token minted for the user it is acting for rather than its
        own server token, because a chat client connects as somebody.
        """
        if not self.api_key or not self.user_id:
            return None

        user = self.user or {"id": self.user_id}
        if self.token:
            return {"api_key": self.api_key, "user": user, "token": self.token}
        if not self.api_secret:
            return None
        return {
            "api_key": self.api_key,
            "user": user,
            "token": _sign({"user_id": self.user_id}, str(self.api_secret)),
        }

    def _server_token(self) -> str:
        """A token that speaks for the app itself, which is Stream's ``server: true``.

        It names no user by definition: one that named a user would be a token minted for
        that user to hold. Which of its users a backend is acting for goes in a header.
        """
        return _sign({"server": True}, str(self.api_secret))

    def _proxy_token(self) -> str:
        """The token the proxy is given, which names a user where there is one.

        A token handed in is used as it is. Otherwise it is minted here, and ``user_id`` is
        what decides whose it is: the proxy has no header to read a backend's choice of user
        from, so with a user named the token has to be that user's.
        """
        if self.token:
            return self.token
        if self.user_id:
            return _sign({"user_id": self.user_id}, str(self.api_secret))
        return self._server_token()


def _sign(
    claims: dict[str, object],
    secret: str,
    validity_seconds: int = TOKEN_VALIDITY_SECONDS,
) -> str:
    """Sign a Stream token.

    HS256 out of the standard library rather than a JWT package: it is twenty lines, and the
    alternative is a dependency in every process that installs this plugin for the pipeline
    and never authenticates against a hosted router at all.
    """
    if not secret:
        raise ValueError("a token cannot be signed without a secret")

    issued = int(time.time())
    payload = {"iat": issued, "exp": issued + validity_seconds, **claims}
    signing = (
        _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        + "."
        + _b64(json.dumps(payload, separators=(",", ":")).encode())
    )
    mac = hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest()
    return signing + "." + _b64(mac)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _flag(value: Optional[str]) -> bool:
    """Read a variable written as a flag, the way the Go and JS clients read the same one."""
    return value is not None and value.lower() in ("1", "true", "yes", "on")
