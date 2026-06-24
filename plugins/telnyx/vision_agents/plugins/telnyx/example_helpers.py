"""Helpers shared by the minimal Telnyx phone examples."""

import base64
import binascii
import json
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import HTTPException, Request


TELNYX_API_BASE_URL = "https://api.telnyx.com/v2"
TELNYX_MEDIA_PATH = "/telnyx/media"
TELNYX_EVENTS_PATH = "/telnyx/events"


class TelnyxSetupError(RuntimeError):
    """Raised when local or Telnyx account setup is not ready for the example."""


class TelnyxWebhookVerificationError(ValueError):
    """Raised when a Telnyx webhook signature fails verification."""


class TelnyxAPIError(RuntimeError):
    """Raised when a Telnyx API request fails."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Telnyx API failed with HTTP {status_code}: {detail}")


@dataclass(frozen=True)
class TelnyxConfig:
    api_key: str
    call_control_app_id: str
    phone_number: str
    ngrok_url: str


@dataclass(frozen=True)
class TelnyxExampleSetup:
    config: TelnyxConfig
    phone_number_id: str | None = None
    created_call_control_app_id: str | None = None
    original_connection_id: str | None = None


class TelnyxClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = TELNYX_API_BASE_URL,
        timeout: float = 20.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TelnyxAPIError(exc.code, detail) from exc

    def retrieve_call_control_app(self, app_id: str) -> dict[str, Any]:
        return self.request("GET", f"/call_control_applications/{quote_path(app_id)}")

    def retrieve_phone_number(self, phone_number_id: str) -> dict[str, Any]:
        return self.request("GET", f"/phone_numbers/{quote_path(phone_number_id)}")

    def find_phone_number(self, phone_number: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"filter[phone_number]": phone_number})
        response = self.request("GET", f"/phone_numbers?{query}")
        phone_numbers = response.get("data", [])
        if not phone_numbers:
            raise TelnyxSetupError(f"Telnyx phone number {phone_number} was not found.")
        return phone_numbers[0]

    def get_first_outbound_voice_profile_id(self) -> str:
        response = self.request("GET", "/outbound_voice_profiles")
        profiles = response.get("data", [])
        if not profiles:
            raise TelnyxSetupError("No Telnyx outbound voice profiles were found.")
        return profiles[0]["id"]

    def create_call_control_app(
        self,
        *,
        application_name: str,
        webhook_event_url: str,
        outbound_voice_profile_id: str,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/call_control_applications",
            {
                "application_name": application_name,
                "webhook_event_url": webhook_event_url,
                "active": True,
                "outbound": {
                    "outbound_voice_profile_id": outbound_voice_profile_id,
                },
            },
        )

    def delete_call_control_app(self, app_id: str) -> None:
        self.request("DELETE", f"/call_control_applications/{quote_path(app_id)}")

    def update_phone_number_connection(
        self,
        phone_number_id: str,
        connection_id: str,
    ) -> None:
        self.request(
            "PATCH",
            f"/phone_numbers/{quote_path(phone_number_id)}",
            {"connection_id": connection_id},
        )

    def get_verified_number(self, phone_number: str) -> dict[str, Any] | None:
        try:
            return self.request("GET", f"/verified_numbers/{quote_path(phone_number)}")
        except TelnyxAPIError as exc:
            if exc.status_code == 404:
                return None
            raise

    def dial_call(
        self,
        *,
        connection_id: str,
        from_number: str,
        to_number: str,
        stream_url: str,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/calls",
            {
                "connection_id": connection_id,
                "from": from_number,
                "to": to_number,
                "stream_url": stream_url,
                "stream_track": "inbound_track",
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": "PCMU",
            },
        )

    def answer_call(self, call_control_id: str, *, stream_url: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/calls/{quote_path(call_control_id)}/actions/answer",
            {
                "stream_url": stream_url,
                "stream_track": "inbound_track",
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": "PCMU",
            },
        )


def quote_path(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def normalize_public_host(value: str) -> str:
    value = value.strip()
    value = value.removeprefix("https://").removeprefix("http://")
    return value.strip("/")


def webhook_url(public_host: str) -> str:
    return f"https://{normalize_public_host(public_host)}{TELNYX_EVENTS_PATH}"


def media_stream_url(public_host: str, call_id: str, token: str) -> str:
    return (
        f"wss://{normalize_public_host(public_host)}"
        f"{TELNYX_MEDIA_PATH}/{call_id}/{token}"
    )


def detect_ngrok_url() -> str | None:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:4040/api/tunnels", timeout=2
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    for tunnel in data.get("tunnels", []):
        if tunnel.get("proto") == "https" and tunnel.get("public_url"):
            return normalize_public_host(tunnel["public_url"])
    return None


def require_env(
    names: list[str], env: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = env if env is not None else os.environ
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise TelnyxSetupError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return {name: env[name] for name in names}


def load_config(env: Mapping[str, str] | None = None) -> TelnyxConfig:
    values = require_env(
        [
            "TELNYX_API_KEY",
            "TELNYX_CALL_CONTROL_APP_ID",
            "TELNYX_PHONE_NUMBER",
            "NGROK_URL",
        ],
        env,
    )
    return TelnyxConfig(
        api_key=values["TELNYX_API_KEY"],
        call_control_app_id=values["TELNYX_CALL_CONTROL_APP_ID"],
        phone_number=values["TELNYX_PHONE_NUMBER"],
        ngrok_url=values["NGROK_URL"],
    )


def resolve_ngrok_url(value: str | None) -> str:
    if value:
        return normalize_public_host(value)

    detected = detect_ngrok_url()
    if detected:
        return detected

    raise TelnyxSetupError(
        "Missing NGROK_URL and no local ngrok tunnel was detected. "
        "Start ngrok with `ngrok http 8000` or set NGROK_URL."
    )


def prepare_telnyx_example_setup(
    client: TelnyxClient,
    *,
    api_key: str,
    phone_number: str | None,
    ngrok_url: str | None,
    call_control_app_id: str | None = None,
    phone_number_id: str | None = None,
    setup_telnyx: bool = False,
    route_phone_number: bool = False,
) -> TelnyxExampleSetup:
    if not phone_number:
        raise TelnyxSetupError(
            "Missing Telnyx phone number. Set TELNYX_PHONE_NUMBER or pass "
            "`--from`/`--phone-number`."
        )

    resolved_ngrok_url = resolve_ngrok_url(ngrok_url)
    resolved_phone_number_id = phone_number_id
    original_connection_id: str | None = None
    created_app_id: str | None = None
    resolved_app_id = call_control_app_id

    if setup_telnyx:
        phone_number_data = client.find_phone_number(phone_number)
        resolved_phone_number_id = resolved_phone_number_id or phone_number_data["id"]

        try:
            if not resolved_app_id:
                outbound_voice_profile_id = client.get_first_outbound_voice_profile_id()
                response = client.create_call_control_app(
                    application_name=f"vision-agents-example-{int(time.time())}",
                    webhook_event_url=webhook_url(resolved_ngrok_url),
                    outbound_voice_profile_id=outbound_voice_profile_id,
                )
                app = response.get("data", response)
                resolved_app_id = app["id"]
                created_app_id = resolved_app_id

            if route_phone_number:
                original_connection_id = phone_number_data.get("connection_id") or ""
                client.update_phone_number_connection(
                    resolved_phone_number_id,
                    resolved_app_id,
                )
        except (TelnyxAPIError, TelnyxSetupError, urllib.error.URLError, TimeoutError):
            if created_app_id:
                client.delete_call_control_app(created_app_id)
            raise
    elif not resolved_app_id:
        raise TelnyxSetupError(
            "Missing TELNYX_CALL_CONTROL_APP_ID. Pass `--setup-telnyx` to create "
            "a temporary Call Control App for this example."
        )

    return TelnyxExampleSetup(
        config=TelnyxConfig(
            api_key=api_key,
            call_control_app_id=resolved_app_id,
            phone_number=phone_number,
            ngrok_url=resolved_ngrok_url,
        ),
        phone_number_id=resolved_phone_number_id,
        created_call_control_app_id=created_app_id,
        original_connection_id=original_connection_id,
    )


def cleanup_telnyx_example_setup(
    client: TelnyxClient,
    setup: TelnyxExampleSetup,
) -> None:
    previous_sigint_handler = None
    try:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass

    errors: list[str] = []

    try:
        if setup.original_connection_id is not None and setup.phone_number_id:
            try:
                client.update_phone_number_connection(
                    setup.phone_number_id,
                    setup.original_connection_id,
                )
            except (TelnyxAPIError, urllib.error.URLError, TimeoutError) as exc:
                errors.append(f"restore phone number routing failed: {exc}")

        if setup.created_call_control_app_id:
            try:
                client.delete_call_control_app(setup.created_call_control_app_id)
            except (TelnyxAPIError, urllib.error.URLError, TimeoutError) as exc:
                errors.append(f"delete temporary Call Control App failed: {exc}")

        if errors:
            raise TelnyxSetupError("; ".join(errors))
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)


def validate_call_control_app(
    app: Mapping[str, Any],
    *,
    app_id: str,
    expected_webhook_url: str,
) -> None:
    data = app.get("data", app)
    if data.get("id") != app_id:
        raise TelnyxSetupError(
            f"Telnyx Call Control App {app_id} was not returned by the API."
        )
    if data.get("record_type") != "call_control_application":
        raise TelnyxSetupError(f"{app_id} is not a Telnyx Call Control App.")
    if data.get("active") is False:
        raise TelnyxSetupError(f"Telnyx Call Control App {app_id} is inactive.")
    actual_webhook_url = data.get("webhook_event_url")
    if actual_webhook_url != expected_webhook_url:
        raise TelnyxSetupError(
            "Telnyx Call Control App webhook URL mismatch. "
            f"Expected {expected_webhook_url}, got {actual_webhook_url!r}."
        )


def validate_phone_number_routing(
    phone_number: Mapping[str, Any],
    *,
    phone_number_id: str,
    expected_connection_id: str,
) -> None:
    data = phone_number.get("data", phone_number)
    if data.get("id") != phone_number_id:
        raise TelnyxSetupError(
            f"Telnyx phone number {phone_number_id} was not returned by the API."
        )
    actual_connection_id = data.get("connection_id")
    if actual_connection_id != expected_connection_id:
        raise TelnyxSetupError(
            "Telnyx phone number is not routed to the Call Control App. "
            f"Expected connection_id {expected_connection_id}, "
            f"got {actual_connection_id!r}."
        )


def validate_verified_destination(
    verified_number: Mapping[str, Any] | None,
    *,
    to_number: str,
) -> None:
    if verified_number is None:
        raise TelnyxSetupError(
            f"{to_number} is not verified in Telnyx. Restricted accounts must "
            "verify destination numbers before outbound dialing."
        )


def preflight_outbound(
    client: TelnyxClient,
    *,
    config: TelnyxConfig,
    to_number: str,
    check_verified_destination: bool = True,
) -> None:
    expected_webhook_url = webhook_url(config.ngrok_url)
    app = client.retrieve_call_control_app(config.call_control_app_id)
    validate_call_control_app(
        app,
        app_id=config.call_control_app_id,
        expected_webhook_url=expected_webhook_url,
    )
    if check_verified_destination:
        validate_verified_destination(
            client.get_verified_number(to_number),
            to_number=to_number,
        )


def verify_telnyx_webhook(
    payload: bytes,
    signature: str | None,
    timestamp: str | None,
    public_key: str,
    *,
    tolerance_seconds: int = 300,
) -> None:
    """Verify a Telnyx webhook using Ed25519 signature headers."""
    if not signature or not timestamp:
        raise TelnyxWebhookVerificationError("Missing Telnyx webhook signature headers")

    try:
        now = int(time.time())
        webhook_timestamp = int(timestamp)
        payload_text = payload.decode("utf-8")
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key))
        signature_bytes = base64.b64decode(signature)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise TelnyxWebhookVerificationError("Invalid Telnyx webhook signature") from exc

    if abs(now - webhook_timestamp) > tolerance_seconds:
        raise TelnyxWebhookVerificationError(
            "Telnyx webhook timestamp outside tolerance window"
        )

    signed_payload = f"{timestamp}|{payload_text}".encode("utf-8")
    try:
        key.verify(signature_bytes, signed_payload)
    except InvalidSignature as exc:
        raise TelnyxWebhookVerificationError(
            "Invalid Telnyx webhook signature"
        ) from exc


async def parse_verified_telnyx_webhook(
    request: Request,
    public_key: str,
) -> dict[str, object]:
    """Read and verify a Telnyx webhook request body."""
    payload = await request.body()
    try:
        verify_telnyx_webhook(
            payload,
            request.headers.get("telnyx-signature-ed25519"),
            request.headers.get("telnyx-timestamp"),
            public_key,
        )
    except TelnyxWebhookVerificationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid Telnyx webhook payload") from exc


def require_telnyx_public_key(env: Mapping[str, str] | None = None) -> str:
    values = require_env(["TELNYX_PUBLIC_KEY"], env)
    return values["TELNYX_PUBLIC_KEY"]


def preflight_inbound(
    client: TelnyxClient,
    *,
    config: TelnyxConfig,
    telnyx_phone_number_id: str,
) -> None:
    expected_webhook_url = webhook_url(config.ngrok_url)
    app = client.retrieve_call_control_app(config.call_control_app_id)
    validate_call_control_app(
        app,
        app_id=config.call_control_app_id,
        expected_webhook_url=expected_webhook_url,
    )
    phone_number = client.retrieve_phone_number(telnyx_phone_number_id)
    validate_phone_number_routing(
        phone_number,
        phone_number_id=telnyx_phone_number_id,
        expected_connection_id=config.call_control_app_id,
    )
