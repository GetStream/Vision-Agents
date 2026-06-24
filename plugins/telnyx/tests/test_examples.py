"""Tests for the minimal Telnyx example helpers."""

import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from telnyx_helpers import (  # noqa: E402
    cleanup_telnyx_example_setup,
    TelnyxConfig,
    TelnyxSetupError,
    load_config,
    media_stream_url,
    prepare_telnyx_example_setup,
    preflight_inbound,
    preflight_outbound,
    require_env,
    validate_call_control_app,
    validate_phone_number_routing,
    validate_verified_destination,
    webhook_url,
)


class FakeTelnyxClient:
    def __init__(
        self,
        *,
        app=None,
        phone_number=None,
        verified_number=None,
        outbound_voice_profile_id="profile-id",
    ):
        self.app = app
        self.phone_number = phone_number
        self.verified_number = verified_number
        self.outbound_voice_profile_id = outbound_voice_profile_id
        self.created_apps = []
        self.deleted_apps = []
        self.updated_connections = []

    def retrieve_call_control_app(self, _app_id):
        return self.app

    def retrieve_phone_number(self, _phone_number_id):
        return self.phone_number

    def find_phone_number(self, _phone_number):
        return self.phone_number

    def get_first_outbound_voice_profile_id(self):
        return self.outbound_voice_profile_id

    def create_call_control_app(
        self,
        *,
        application_name,
        webhook_event_url,
        outbound_voice_profile_id,
    ):
        app = {
            "id": "created-app-id",
            "application_name": application_name,
            "webhook_event_url": webhook_event_url,
            "outbound": {"outbound_voice_profile_id": outbound_voice_profile_id},
        }
        self.created_apps.append(app)
        return {"data": app}

    def delete_call_control_app(self, app_id):
        self.deleted_apps.append(app_id)

    def update_phone_number_connection(self, phone_number_id, connection_id):
        self.updated_connections.append((phone_number_id, connection_id))

    def get_verified_number(self, _phone_number):
        return self.verified_number


def test_url_builders_normalize_scheme_and_trailing_slash():
    assert webhook_url("https://example.ngrok-free.app/") == (
        "https://example.ngrok-free.app/telnyx/events"
    )
    assert media_stream_url("http://example.ngrok-free.app/", "call-1", "token-1") == (
        "wss://example.ngrok-free.app/telnyx/media/call-1/token-1"
    )


def test_require_env_raises_for_missing_values():
    with pytest.raises(TelnyxSetupError, match="TELNYX_API_KEY"):
        require_env(["TELNYX_API_KEY"], env={})


def test_load_config_reads_required_values_from_mapping():
    config = load_config(
        {
            "TELNYX_API_KEY": "key",
            "TELNYX_CALL_CONTROL_APP_ID": "app-id",
            "TELNYX_PHONE_NUMBER": "+15551234567",
            "NGROK_URL": "example.ngrok-free.app",
        }
    )

    assert config.api_key == "key"
    assert config.call_control_app_id == "app-id"
    assert config.phone_number == "+15551234567"
    assert config.ngrok_url == "example.ngrok-free.app"


def test_validate_call_control_app_requires_matching_webhook():
    with pytest.raises(TelnyxSetupError, match="webhook URL mismatch"):
        validate_call_control_app(
            {
                "data": {
                    "id": "app-id",
                    "record_type": "call_control_application",
                    "active": True,
                    "webhook_event_url": "https://old.example/telnyx/events",
                }
            },
            app_id="app-id",
            expected_webhook_url="https://new.example/telnyx/events",
        )


def test_validate_call_control_app_rejects_inactive_app():
    with pytest.raises(TelnyxSetupError, match="inactive"):
        validate_call_control_app(
            {
                "data": {
                    "id": "app-id",
                    "record_type": "call_control_application",
                    "active": False,
                    "webhook_event_url": "https://example/telnyx/events",
                }
            },
            app_id="app-id",
            expected_webhook_url="https://example/telnyx/events",
        )


def test_validate_phone_number_routing_requires_call_control_app_connection():
    with pytest.raises(TelnyxSetupError, match="not routed"):
        validate_phone_number_routing(
            {
                "data": {
                    "id": "phone-id",
                    "phone_number": "+15551234567",
                    "connection_id": "forward-only-id",
                }
            },
            phone_number_id="phone-id",
            expected_connection_id="call-control-app-id",
        )


def test_validate_verified_destination_rejects_missing_number():
    with pytest.raises(TelnyxSetupError, match="not verified"):
        validate_verified_destination(None, to_number="+15557654321")


def test_prepare_telnyx_example_setup_requires_app_or_setup_flag():
    client = FakeTelnyxClient()

    with pytest.raises(TelnyxSetupError, match="--setup-telnyx"):
        prepare_telnyx_example_setup(
            client,
            api_key="key",
            phone_number="+15551234567",
            ngrok_url="example.ngrok-free.app",
        )


def test_prepare_telnyx_example_setup_creates_temp_outbound_app():
    client = FakeTelnyxClient(
        phone_number={
            "id": "phone-id",
            "phone_number": "+15551234567",
            "connection_id": "original-app-id",
        },
    )

    setup = prepare_telnyx_example_setup(
        client,
        api_key="key",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
        setup_telnyx=True,
    )

    assert setup.config.call_control_app_id == "created-app-id"
    assert setup.created_call_control_app_id == "created-app-id"
    assert setup.phone_number_id == "phone-id"
    assert setup.original_connection_id is None
    assert client.created_apps[0]["webhook_event_url"] == (
        "https://example.ngrok-free.app/telnyx/events"
    )
    assert client.updated_connections == []

    cleanup_telnyx_example_setup(client, setup)

    assert client.deleted_apps == ["created-app-id"]


def test_prepare_telnyx_example_setup_routes_and_restores_inbound_number():
    client = FakeTelnyxClient(
        phone_number={
            "id": "phone-id",
            "phone_number": "+15551234567",
            "connection_id": "original-app-id",
        },
    )

    setup = prepare_telnyx_example_setup(
        client,
        api_key="key",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
        setup_telnyx=True,
        route_phone_number=True,
    )

    assert setup.phone_number_id == "phone-id"
    assert setup.original_connection_id == "original-app-id"
    assert client.updated_connections == [("phone-id", "created-app-id")]

    cleanup_telnyx_example_setup(client, setup)

    assert client.updated_connections[-1] == ("phone-id", "original-app-id")
    assert client.deleted_apps == ["created-app-id"]


def test_preflight_outbound_passes_with_matching_app_and_verified_destination():
    config = TelnyxConfig(
        api_key="key",
        call_control_app_id="app-id",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
    )
    client = FakeTelnyxClient(
        app={
            "data": {
                "id": "app-id",
                "record_type": "call_control_application",
                "active": True,
                "webhook_event_url": "https://example.ngrok-free.app/telnyx/events",
            }
        },
        verified_number={"data": {"phone_number": "+15557654321"}},
    )

    preflight_outbound(client, config=config, to_number="+15557654321")


def test_preflight_inbound_requires_phone_number_to_route_to_app():
    config = TelnyxConfig(
        api_key="key",
        call_control_app_id="app-id",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
    )
    client = FakeTelnyxClient(
        app={
            "data": {
                "id": "app-id",
                "record_type": "call_control_application",
                "active": True,
                "webhook_event_url": "https://example.ngrok-free.app/telnyx/events",
            }
        },
        phone_number={
            "data": {
                "id": "phone-id",
                "phone_number": "+15551234567",
                "connection_id": "other-app-id",
            }
        },
    )

    with pytest.raises(TelnyxSetupError, match="not routed"):
        preflight_inbound(client, config=config, telnyx_phone_number_id="phone-id")
