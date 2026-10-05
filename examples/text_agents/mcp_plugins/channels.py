"""Carrying a conversation over to Slack, Teams, WhatsApp, RCS, text messages and iMessage."""

import abc
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import pathlib
import secrets
import time
from typing import Optional
from urllib.parse import quote

import aiohttp
from aiohttp import web
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from vision_agents.core import Agent
from vision_agents.core.llm import RemoteFile
from vision_agents.plugins import omni

logger = logging.getLogger(__name__)

GRAPH_URL = "https://graph.facebook.com/v23.0"
TELNYX_URL = "https://api.telnyx.com/v2"
LINQ_URL = "https://api.linqapp.com/api/partner/v3"
SLACK_URL = "https://slack.com/api"
RBM_URL = "https://rcsbusinessmessaging.googleapis.com/v1"
TEAMS_LOGIN_URL = "https://login.microsoftonline.com"
# How old a signed delivery may be, so a recorded one cannot be played back later.
SIGNATURE_TOLERANCE_SECONDS = 300


def _b64url(value: str) -> bytes:
    """A base64url field of a JWT, whose padding the encoding leaves off."""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class JsonWebKeys:
    """The public keys an identity provider signs its tokens with.

    Fetched once at startup, which is enough for an example: a provider that rotates a
    key mid-run is a restart rather than a refresh.
    """

    def __init__(self, url: str):
        self._url = url
        self._keys: dict[str, rsa.RSAPublicKey] = {}

    async def start(self, http: aiohttp.ClientSession) -> None:
        """Fetch the key set."""
        async with http.get(self._url) as response:
            if response.status >= 400:
                raise RuntimeError(f"no keys at {self._url}: {await response.text()}")
            document = await response.json()
        for key in document.get("keys", []):
            if key.get("kty") != "RSA" or not key.get("kid"):
                continue
            numbers = rsa.RSAPublicNumbers(
                int.from_bytes(_b64url(key["e"]), "big"),
                int.from_bytes(_b64url(key["n"]), "big"),
            )
            self._keys[key["kid"]] = numbers.public_key()

    def verified(self, token: str, issuer: str, audience: str) -> Optional[dict]:
        """The claims of an RS256 token this key set signed, or None.

        Args:
            token: The compact JWT.
            issuer: Who the token must say issued it.
            audience: Who it must say it is for.

        Returns:
            The claims, or None when the token is not one of theirs, is for somebody
            else, or has expired.
        """
        parts = token.split(".")
        if len(parts) != 3:
            return None
        try:
            header = json.loads(_b64url(parts[0]))
            claims = json.loads(_b64url(parts[1]))
            signature = _b64url(parts[2])
        except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
            return None
        key = self._keys.get(header.get("kid", ""))
        if key is None or header.get("alg") != "RS256":
            return None
        try:
            key.verify(
                signature,
                f"{parts[0]}.{parts[1]}".encode(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except InvalidSignature:
            return None
        if claims.get("iss") != issuer or claims.get("aud") != audience:
            return None
        expires = claims.get("exp")
        if not isinstance(expires, int) or time.time() > expires:
            return None
        return claims


class ServiceAccount:
    """A Google service account, which signs its own way to an access token."""

    def __init__(self, account: dict[str, object], scope: str):
        self._email = str(account["client_email"])
        self._token_uri = str(
            account.get("token_uri", "https://oauth2.googleapis.com/token")
        )
        key = serialization.load_pem_private_key(
            str(account["private_key"]).encode(), password=None
        )
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("a service account key is RSA")
        self._key = key
        self._scope = scope
        self._token = ""
        self._expires = 0.0

    async def token(self, http: aiohttp.ClientSession) -> str:
        """An access token for the scope, renewed before it expires."""
        if self._token and time.time() < self._expires:
            return self._token
        async with http.post(
            self._token_uri,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": self._assertion(),
            },
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"Google refused the key: {await response.text()}")
            granted = await response.json()
        self._token = str(granted["access_token"])
        self._expires = time.time() + int(granted.get("expires_in", 3600)) - 60
        return self._token

    def _assertion(self) -> str:
        """The signed claim that this is the account, which Google trades for a token."""
        now = int(time.time())
        segments = [
            {"alg": "RS256", "typ": "JWT"},
            {
                "iss": self._email,
                "scope": self._scope,
                "aud": self._token_uri,
                "iat": now,
                "exp": now + 3600,
            },
        ]
        signing_input = b".".join(
            base64.urlsafe_b64encode(json.dumps(segment).encode()).rstrip(b"=")
            for segment in segments
        )
        signature = self._key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        return (
            signing_input + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")
        ).decode()


class Channel(abc.ABC):
    """One way into a conversation the agent is already holding.

    The person sends the code in `invitation` from their own account, which ties it to
    the conversation. From then on what they write is asked in it, as the conversation's
    own end user, and the agent's answers are sent back to them. Everything is still
    recorded in the conversation, so the dashboard shows every channel at once.

    Attributes:
        name: What the channel is called, in what is printed.
        path: Where its provider delivers webhooks.
    """

    name: str
    path: str

    def __init__(self, agent: Agent, asking: asyncio.Lock, provider: omni.OmniProvider):
        self._agent = agent
        self._asking = asking
        self._provider = provider
        self._code = secrets.token_hex(3)
        self._linked = ""
        self._seen: set[str] = set()
        self._answering: set[asyncio.Task[None]] = set()
        self._http: Optional[aiohttp.ClientSession] = None

    async def start(self, http: aiohttp.ClientSession) -> None:
        """Get ready to send, over the inbox's HTTP session."""
        self._http = http

    async def stop(self) -> None:
        """Drop what is still being answered."""
        for task in self._answering:
            task.cancel()
        await asyncio.gather(*self._answering, return_exceptions=True)

    @property
    @abc.abstractmethod
    def invitation(self) -> str:
        """What to print so the person can carry the conversation over."""

    def routes(self) -> list[web.RouteDef]:
        """The webhook routes the inbox serves for this channel."""
        return [web.post(self.path, self._receive)]

    @abc.abstractmethod
    def _signed(self, request: web.Request, body: bytes) -> bool:
        """Whether a delivery really came from the provider."""

    @abc.abstractmethod
    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        """Send one body `render` made, in answer to `route`."""

    async def _ask_to_connect(
        self, route: omni.OmniMessage, text: str, url: str, logo: str
    ) -> None:
        """Ask the person to connect an account.

        The default is the link in the text, which every channel can carry. A channel
        with a button of its own overrides this.
        """
        await self._send(route, f"{text}: {url}")

    async def _receive(self, request: web.Request) -> web.Response:
        body = await request.read()
        if not self._signed(request, body):
            return web.Response(status=401)
        # Providers wait on this delivery and send it again if it takes too long, so the
        # answer is written once the model is done rather than in the response.
        for message in self._provider.parse(json.loads(body)):
            if message.id in self._seen:
                continue
            self._seen.add(message.id)
            task = asyncio.create_task(self._answer(message))
            self._answering.add(task)
            task.add_done_callback(self._answered)
        return web.Response()

    async def _answer(self, message: omni.OmniMessage) -> None:
        if message.sender_id != self._linked:
            await self._link(message)
            return
        async with self._asking:
            async for event in self._agent.ask(message.text):
                if event.type == "agent_speech" and event.text:
                    await self._send(message, event.text)
                elif event.type == "authorization_required":
                    await self._ask_to_connect(
                        message, event.text, event.url, event.image_url
                    )
                elif event.type == "task_settled" and event.files:
                    await self._send(message, "", event.files)
                elif event.type == "error":
                    await self._send(message, f"Something went wrong: {event.error}")

    async def _link(self, message: omni.OmniMessage) -> None:
        if self._code and self._code in message.text:
            self._linked = message.sender_id
            self._code = ""
            logger.info(
                "%s %s now holds the conversation",
                self.name,
                message.sender_name or message.sender_id,
            )
            await self._send(message, "You're through. Carry on where you left off.")
            return
        await self._send(
            message,
            "This conversation belongs to someone else. "
            "Send the code it printed to carry it over here.",
        )

    async def _send(
        self,
        route: omni.OmniMessage,
        text: str,
        files: Optional[list[RemoteFile]] = None,
    ) -> None:
        reply = omni.OmniMessage(
            channel=route.channel,
            provider=route.provider,
            conversation_id=route.conversation_id,
            account_id=route.account_id,
            text=text,
            attachments=[
                omni.OmniAttachment(
                    kind=omni.AttachmentKind.of(file.mime_type),
                    url=file.url,
                    mime_type=file.mime_type,
                    name=file.name,
                )
                for file in files or []
            ],
        )
        for body in self._provider.render(reply):
            await self._post(route, body)

    async def _post_json(self, url: str, token: str, body: dict[str, object]) -> None:
        assert self._http is not None
        async with self._http.post(
            url, json=body, headers={"Authorization": f"Bearer {token}"}
        ) as response:
            if response.status >= 400:
                logger.error(
                    "%s refused a message: %s", self.name, await response.text()
                )

    def _answered(self, task: asyncio.Task[None]) -> None:
        self._answering.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Could not answer on %s", self.name, exc_info=task.exception())


class WhatsApp(Channel):
    """WhatsApp, through Meta's Cloud API. A login arrives as a button."""

    name = "WhatsApp"
    path = "/whatsapp"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        token: str,
        phone_number_id: str,
        app_secret: str,
        verify_token: str,
    ):
        super().__init__(agent, asking, omni.WhatsAppProvider())
        self._token = token
        self._phone_number_id = phone_number_id
        self._app_secret = app_secret
        self._verify_token = verify_token
        self._number = ""

    async def start(self, http: aiohttp.ClientSession) -> None:
        """Look up the business number the wa.me link opens a chat with."""
        await super().start(http)
        async with http.get(
            f"{GRAPH_URL}/{self._phone_number_id}",
            params={"fields": "display_phone_number"},
            headers={"Authorization": f"Bearer {self._token}"},
        ) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"WhatsApp refused the phone number id: {await response.text()}"
                )
            number = (await response.json())["display_phone_number"]
        self._number = "".join(digit for digit in number if digit.isdigit())

    @property
    def invitation(self) -> str:
        """The wa.me link that opens a chat with the business, the code filled in."""
        link = f"https://wa.me/{self._number}?text={quote(f'link {self._code}')}"
        return f"carry on in WhatsApp: {link}"

    def routes(self) -> list[web.RouteDef]:
        """Meta's subscription check as well as its deliveries."""
        return [*super().routes(), web.get(self.path, self._verify)]

    async def _verify(self, request: web.Request) -> web.Response:
        query = request.query
        if query.get("hub.mode") == "subscribe" and hmac.compare_digest(
            query.get("hub.verify_token", ""), self._verify_token
        ):
            return web.Response(text=query.get("hub.challenge", ""))
        return web.Response(status=403)

    def _signed(self, request: web.Request, body: bytes) -> bool:
        signed = hmac.new(self._app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(
            request.headers.get("X-Hub-Signature-256", ""), f"sha256={signed}"
        )

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        await self._post_json(
            f"{GRAPH_URL}/{self._phone_number_id}/messages", self._token, body
        )

    async def _ask_to_connect(
        self, route: omni.OmniMessage, text: str, url: str, logo: str
    ) -> None:
        """The WhatsApp form of a `plugin_authorization` attachment: a button to the login."""
        await self._post(
            route,
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": route.conversation_id,
                "type": "interactive",
                "interactive": {
                    "type": "cta_url",
                    "body": {"text": text or "Connect your account to carry on."},
                    "action": {
                        "name": "cta_url",
                        "parameters": {"display_text": "Connect", "url": url},
                    },
                },
            },
        )


class Slack(Channel):
    """Slack, through the Events API. A login arrives as a Block Kit button."""

    name = "Slack"
    path = "/slack"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        bot_token: str,
        signing_secret: str,
    ):
        super().__init__(agent, asking, omni.SlackProvider())
        self._bot_token = bot_token
        self._signing_secret = signing_secret.encode()

    @property
    def invitation(self) -> str:
        """What to say to the app, from any channel or DM it can read."""
        return f"carry on in Slack: send 'link {self._code}' to the app"

    async def _receive(self, request: web.Request) -> web.Response:
        """Answer Slack's subscription check, which arrives signed on this same path."""
        body = await request.read()
        if not self._signed(request, body):
            return web.Response(status=401)
        payload = json.loads(body)
        if payload.get("type") == "url_verification":
            return web.json_response({"challenge": payload.get("challenge", "")})
        # aiohttp keeps the body it has read, so the base reads the same bytes again.
        return await super()._receive(request)

    def _signed(self, request: web.Request, body: bytes) -> bool:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        if not timestamp.isdigit():
            return False
        if abs(time.time() - int(timestamp)) > SIGNATURE_TOLERANCE_SECONDS:
            return False
        signed = hmac.new(
            self._signing_secret,
            b"v0:" + timestamp.encode() + b":" + body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(
            request.headers.get("X-Slack-Signature", ""), f"v0={signed}"
        )

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        await self._post_json(f"{SLACK_URL}/chat.postMessage", self._bot_token, body)

    async def _ask_to_connect(
        self, route: omni.OmniMessage, text: str, url: str, logo: str
    ) -> None:
        """The Slack form of the attachment: a section with a link button beside it."""
        section: dict[str, object] = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text or "Connect your account to carry on.",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Connect"},
                "url": url,
            },
        }
        await self._post(
            route,
            {
                "channel": route.conversation_id,
                "text": f"{text}: {url}",
                "blocks": [section],
            },
        )


class Teams(Channel):
    """Microsoft Teams, through the Bot Framework. A login arrives as a hero card."""

    name = "Teams"
    path = "/teams"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        app_id: str,
        app_password: str,
        tenant_id: str = "",
    ):
        super().__init__(agent, asking, omni.TeamsProvider())
        self._app_id = app_id
        self._app_password = app_password
        self._tenant_id = tenant_id or "botframework.com"
        self._keys = JsonWebKeys("https://login.botframework.com/v1/.well-known/keys")
        self._token = ""
        self._token_expires = 0.0
        # Where replies go, taken from the token rather than the body, so a delivery
        # cannot point the agent's answers at a host of its own.
        self._service_url = ""

    async def start(self, http: aiohttp.ClientSession) -> None:
        await super().start(http)
        await self._keys.start(http)

    @property
    def invitation(self) -> str:
        """What to say to the bot, in a chat or a channel it is in."""
        return f"carry on in Teams: send 'link {self._code}' to the bot"

    def _signed(self, request: web.Request, body: bytes) -> bool:
        """Check the Bot Framework's bearer token, and keep the service URL it names."""
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            return False
        claims = self._keys.verified(
            authorization.removeprefix("Bearer "),
            issuer="https://api.botframework.com",
            audience=self._app_id,
        )
        if claims is None:
            return False
        service_url = claims.get("serviceurl")
        if not isinstance(service_url, str) or not service_url:
            return False
        self._service_url = service_url.rstrip("/")
        return True

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        url = f"{self._service_url}/v3/conversations/{quote(route.conversation_id)}/activities"
        await self._post_json(url, await self._app_token(), body)

    async def _ask_to_connect(
        self, route: omni.OmniMessage, text: str, url: str, logo: str
    ) -> None:
        """The Teams form of the attachment: a hero card with an openUrl button."""
        card: dict[str, object] = {
            "title": text or "Connect your account to carry on.",
            "buttons": [{"type": "openUrl", "title": "Connect", "value": url}],
        }
        if logo:
            card["images"] = [{"url": logo}]
        await self._post(
            route,
            {
                "type": "message",
                "text": f"{text}: {url}",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.hero",
                        "content": card,
                    }
                ],
            },
        )

    async def _app_token(self) -> str:
        """The app's own token for the Bot Connector, renewed before it expires."""
        if self._token and time.time() < self._token_expires:
            return self._token
        assert self._http is not None
        async with self._http.post(
            f"{TEAMS_LOGIN_URL}/{self._tenant_id}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._app_id,
                "client_secret": self._app_password,
                "scope": "https://api.botframework.com/.default",
            },
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"Teams refused the app: {await response.text()}")
            token = await response.json()
        self._token = str(token["access_token"])
        self._token_expires = time.time() + int(token.get("expires_in", 3600)) - 60
        return self._token


class Rcs(Channel):
    """RCS, through Google RCS Business Messaging. A login arrives as a suggestion."""

    name = "RCS"
    path = "/rcs"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        service_account: dict[str, object],
        agent_id: str,
        client_token: str,
    ):
        super().__init__(agent, asking, omni.GoogleRBMProvider())
        self._account = ServiceAccount(
            service_account, "https://www.googleapis.com/auth/rcsbusinessmessaging"
        )
        self._agent_id = agent_id
        self._client_token = client_token

    @property
    def invitation(self) -> str:
        """What to send the RCS agent from an Android phone."""
        return f"carry on over RCS: send 'link {self._code}' to {self._agent_id}"

    def _signed(self, request: web.Request, body: bytes) -> bool:
        """Compare the client token RBM was registered with, which it sends on each push."""
        payload = json.loads(body)
        data = (
            payload.get("message", {}).get("data")
            if isinstance(payload, dict)
            else None
        )
        if isinstance(data, str):
            payload = json.loads(base64.b64decode(data, validate=True))
        sent = payload.get("clientToken") if isinstance(payload, dict) else None
        if not isinstance(sent, str):
            return False
        return hmac.compare_digest(sent, self._client_token)

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        # RBM wants a new message id on every send, in the query rather than the body.
        assert self._http is not None
        url = (
            f"{RBM_URL}/phones/{quote(route.conversation_id)}/agentMessages"
            f"?messageId={secrets.token_hex(16)}&agentId={quote(self._agent_id)}"
        )
        await self._post_json(url, await self._account.token(self._http), body)

    async def _ask_to_connect(
        self, route: omni.OmniMessage, text: str, url: str, logo: str
    ) -> None:
        """The RCS form of the attachment: a suggestion that opens the login."""
        await self._post(
            route,
            {
                "contentMessage": {
                    "text": text or "Connect your account to carry on.",
                    "suggestions": [
                        {
                            "action": {
                                "text": "Connect",
                                "postbackData": "connect",
                                "openUrlAction": {"url": url},
                            }
                        }
                    ],
                }
            },
        )


class Texting(Channel):
    """SMS, through Telnyx Messaging. A login arrives as a link in the text."""

    name = "SMS"
    path = "/sms"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        api_key: str,
        number: str,
        public_key: str,
    ):
        super().__init__(agent, asking, omni.TelnyxProvider())
        self._api_key = api_key
        self._number = number
        self._public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(public_key)
        )

    @property
    def invitation(self) -> str:
        """The number to text, and the code to text it."""
        return f"carry on by text: send 'link {self._code}' to {self._number}"

    def _signed(self, request: web.Request, body: bytes) -> bool:
        signature = request.headers.get("telnyx-signature-ed25519", "")
        timestamp = request.headers.get("telnyx-timestamp", "")
        if not timestamp.isdigit():
            return False
        if abs(time.time() - int(timestamp)) > SIGNATURE_TOLERANCE_SECONDS:
            return False
        try:
            self._public_key.verify(
                base64.b64decode(signature), f"{timestamp}|".encode() + body
            )
        except (InvalidSignature, binascii.Error):
            return False
        return True

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        await self._post_json(f"{TELNYX_URL}/messages", self._api_key, body)


class IMessage(Channel):
    """iMessage, through Linq, which falls back to RCS or SMS when Apple cannot deliver.

    A login arrives as a link, which iMessage previews.
    """

    name = "iMessage"
    path = "/imessage"

    def __init__(
        self,
        agent: Agent,
        asking: asyncio.Lock,
        api_key: str,
        number: str,
        signing_secret: str,
    ):
        super().__init__(agent, asking, omni.LinqProvider())
        self._api_key = api_key
        self._number = number
        self._signing_key = base64.b64decode(signing_secret.removeprefix("whsec_"))

    @property
    def invitation(self) -> str:
        """The line to message from an iPhone, and the code to send it."""
        return f"carry on in iMessage: send 'link {self._code}' to {self._number}"

    def _signed(self, request: web.Request, body: bytes) -> bool:
        """Standard Webhooks: an HMAC of the delivery's id, its time and the body."""
        delivery = request.headers.get("webhook-id", "")
        timestamp = request.headers.get("webhook-timestamp", "")
        if not delivery or not timestamp.isdigit():
            return False
        if abs(time.time() - int(timestamp)) > SIGNATURE_TOLERANCE_SECONDS:
            return False
        signed = base64.b64encode(
            hmac.new(
                self._signing_key,
                f"{delivery}.{timestamp}.".encode() + body,
                hashlib.sha256,
            ).digest()
        ).decode()
        return any(
            hmac.compare_digest(signature.removeprefix("v1,"), signed)
            for signature in request.headers.get("webhook-signature", "").split()
            if signature.startswith("v1,")
        )

    async def _post(self, route: omni.OmniMessage, body: dict[str, object]) -> None:
        await self._post_json(
            f"{LINQ_URL}/chats/{route.conversation_id}/messages", self._api_key, body
        )


class Inbox:
    """Serves the webhooks of every channel the environment configures, on one port."""

    def __init__(self, channels: list[Channel], port: int = 8090):
        self._channels = channels
        self._port = port
        self._http: Optional[aiohttp.ClientSession] = None
        self._runner: Optional[web.AppRunner] = None

    @classmethod
    def from_env(cls, agent: Agent) -> Optional["Inbox"]:
        """The channels the environment configures, or None without any.

        Slack needs SLACK_BOT_TOKEN, Teams TEAMS_APP_ID, RCS RBM_AGENT_ID, WhatsApp
        WHATSAPP_ACCESS_TOKEN, texting TELNYX_SMS_NUMBER and iMessage LINQ_API_KEY.
        """
        asking = asyncio.Lock()
        channels: list[Channel] = []
        if os.environ.get("SLACK_BOT_TOKEN"):
            channels.append(
                Slack(
                    agent,
                    asking,
                    bot_token=os.environ["SLACK_BOT_TOKEN"],
                    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
                )
            )
        if os.environ.get("TEAMS_APP_ID"):
            channels.append(
                Teams(
                    agent,
                    asking,
                    app_id=os.environ["TEAMS_APP_ID"],
                    app_password=os.environ["TEAMS_APP_PASSWORD"],
                    tenant_id=os.environ.get("TEAMS_TENANT_ID", ""),
                )
            )
        if os.environ.get("RBM_AGENT_ID"):
            channels.append(
                Rcs(
                    agent,
                    asking,
                    service_account=json.loads(
                        pathlib.Path(os.environ["RBM_SERVICE_ACCOUNT"]).read_text()
                    ),
                    agent_id=os.environ["RBM_AGENT_ID"],
                    client_token=os.environ["RBM_CLIENT_TOKEN"],
                )
            )
        if os.environ.get("WHATSAPP_ACCESS_TOKEN"):
            channels.append(
                WhatsApp(
                    agent,
                    asking,
                    token=os.environ["WHATSAPP_ACCESS_TOKEN"],
                    phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
                    app_secret=os.environ["WHATSAPP_APP_SECRET"],
                    verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
                )
            )
        if os.environ.get("TELNYX_SMS_NUMBER"):
            channels.append(
                Texting(
                    agent,
                    asking,
                    api_key=os.environ["TELNYX_API_KEY"],
                    number=os.environ["TELNYX_SMS_NUMBER"],
                    public_key=os.environ["TELNYX_PUBLIC_KEY"],
                )
            )
        if os.environ.get("LINQ_API_KEY"):
            channels.append(
                IMessage(
                    agent,
                    asking,
                    api_key=os.environ["LINQ_API_KEY"],
                    number=os.environ["LINQ_NUMBER"],
                    signing_secret=os.environ["LINQ_WEBHOOK_SECRET"],
                )
            )
        if not channels:
            return None
        return cls(channels, port=int(os.environ.get("INBOX_PORT", "8090")))

    async def start(self) -> None:
        """Get every channel ready and start taking their webhooks."""
        self._http = aiohttp.ClientSession()
        app = web.Application()
        for channel in self._channels:
            await channel.start(self._http)
            app.add_routes(channel.routes())
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, port=self._port).start()
        logger.info(
            "Taking webhooks on :%d at %s",
            self._port,
            ", ".join(channel.path for channel in self._channels),
        )

    async def stop(self) -> None:
        """Stop taking webhooks and drop what is still being answered."""
        for channel in self._channels:
            await channel.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        if self._http is not None:
            await self._http.close()

    @property
    def invitations(self) -> list[str]:
        """What to print, one line a channel."""
        return [channel.invitation for channel in self._channels]
