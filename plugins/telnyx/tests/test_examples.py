"""Tests for the minimal Telnyx example helpers."""

import base64
import signal
import subprocess
import sys
import threading
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from vision_agents.plugins.telnyx.example_helpers import (
    TelnyxAPIError,
    TelnyxConfig,
    TelnyxExampleResources,
    TelnyxSetupError,
    TelnyxWebhookVerificationError,
    cleanup_telnyx_example_setup,
    load_config,
    media_stream_url,
    prepare_telnyx_example_setup,
    preflight_inbound,
    preflight_outbound,
    require_env,
    telnyx_example_cleanup,
    validate_call_control_app,
    validate_phone_number_routing,
    validate_verified_destination,
    verify_telnyx_webhook,
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
        update_failures=0,
        delete_failures=0,
    ):
        self.app = app
        self.phone_number = phone_number
        self.verified_number = verified_number
        self.outbound_voice_profile_id = outbound_voice_profile_id
        self.call_control_apps: dict[str, dict] = {}
        self.deleted_app_ids: list[str] = []
        self.connection_updates: list[tuple[str, str]] = []
        self.update_failures = update_failures
        self.delete_failures = delete_failures
        self.phone_connection_id = (
            phone_number.get("connection_id") if phone_number else None
        )

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
        self.call_control_apps[app["id"]] = app
        return {"data": app}

    def delete_call_control_app(self, app_id):
        if self.delete_failures:
            self.delete_failures -= 1
            raise TelnyxAPIError(500, "delete failed")
        self.deleted_app_ids.append(app_id)
        self.call_control_apps.pop(app_id, None)

    def update_phone_number_connection(self, phone_number_id, connection_id):
        if self.update_failures:
            self.update_failures -= 1
            raise TelnyxAPIError(500, "update failed")
        self.connection_updates.append((phone_number_id, connection_id))
        self.phone_connection_id = connection_id
        if self.phone_number is not None:
            self.phone_number["connection_id"] = connection_id
            self.phone_number["id"] = phone_number_id

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


def test_verify_telnyx_webhook_accepts_valid_signature():
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(private_key.public_key().public_bytes_raw()).decode(
        "ascii"
    )
    payload = b'{"data":{"event_type":"call.initiated"}}'
    timestamp = str(int(time.time()))
    signed_payload = f"{timestamp}|{payload.decode('utf-8')}".encode("utf-8")
    signature = base64.b64encode(private_key.sign(signed_payload)).decode("ascii")

    verify_telnyx_webhook(payload, signature, timestamp, public_key)


def test_verify_telnyx_webhook_rejects_invalid_signature():
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(private_key.public_key().public_bytes_raw()).decode(
        "ascii"
    )
    payload = b'{"data":{"event_type":"call.initiated"}}'
    timestamp = str(int(time.time()))

    with pytest.raises(TelnyxWebhookVerificationError, match="Invalid Telnyx"):
        verify_telnyx_webhook(payload, "invalid", timestamp, public_key)


def test_verify_telnyx_webhook_rejects_malformed_timestamp():
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(private_key.public_key().public_bytes_raw()).decode(
        "ascii"
    )
    payload = b'{"data":{"event_type":"call.initiated"}}'

    with pytest.raises(TelnyxWebhookVerificationError, match="Invalid Telnyx"):
        verify_telnyx_webhook(payload, "invalid", "not-a-timestamp", public_key)


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
    assert client.call_control_apps["created-app-id"]["webhook_event_url"] == (
        "https://example.ngrok-free.app/telnyx/events"
    )
    assert client.phone_connection_id == "original-app-id"

    cleanup_telnyx_example_setup(client, setup)

    assert client.deleted_app_ids == ["created-app-id"]
    assert "created-app-id" not in client.call_control_apps


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
    assert client.phone_connection_id == "created-app-id"

    cleanup_telnyx_example_setup(client, setup)

    assert client.phone_connection_id == "original-app-id"
    assert client.deleted_app_ids == ["created-app-id"]


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


def _routed_setup():
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
    return client, setup


def test_telnyx_example_cleanup_restores_resources_on_normal_exit():
    client, setup = _routed_setup()

    with telnyx_example_cleanup(client, setup):
        assert client.phone_connection_id == "created-app-id"

    assert client.phone_connection_id == "original-app-id"
    assert client.deleted_app_ids == ["created-app-id"]


def test_telnyx_example_cleanup_runs_once_when_the_body_raises():
    client, setup = _routed_setup()

    with pytest.raises(KeyboardInterrupt):
        with telnyx_example_cleanup(client, setup):
            raise KeyboardInterrupt

    # Deleting the app twice would 404, so cleanup must happen exactly once.
    assert client.deleted_app_ids == ["created-app-id"]


def test_telnyx_example_cleanup_installs_and_restores_sigterm_handler():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers can only be installed on the main thread")

    client, setup = _routed_setup()
    original_handler = signal.getsignal(signal.SIGTERM)

    with telnyx_example_cleanup(client, setup):
        assert signal.getsignal(signal.SIGTERM) is not original_handler

    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_prepare_telnyx_example_setup_tracks_resources_while_creating_them():
    client = FakeTelnyxClient(
        phone_number={
            "id": "phone-id",
            "phone_number": "+15551234567",
            "connection_id": "original-app-id",
        },
    )
    resources = TelnyxExampleResources()
    tracked_at_reroute: list[tuple[str | None, str | None]] = []
    reroute = client.update_phone_number_connection

    def recording_reroute(phone_number_id, connection_id):
        tracked_at_reroute.append(
            (resources.created_call_control_app_id, resources.original_connection_id)
        )
        reroute(phone_number_id, connection_id)

    client.update_phone_number_connection = recording_reroute  # type: ignore[method-assign]

    setup = prepare_telnyx_example_setup(
        client,
        api_key="key",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
        setup_telnyx=True,
        route_phone_number=True,
        resources=resources,
    )

    # Both resources were already on the tracker before the re-route ran, so a
    # signal arriving mid-setup still has everything it needs to clean up.
    assert tracked_at_reroute == [("created-app-id", "original-app-id")]
    assert resources == TelnyxExampleResources(
        phone_number_id="phone-id",
        created_call_control_app_id="created-app-id",
        original_connection_id="original-app-id",
    )
    assert setup.created_call_control_app_id == "created-app-id"
    assert setup.original_connection_id == "original-app-id"


def test_prepare_telnyx_example_setup_rollback_is_not_repeated_by_the_guard():
    # The re-route fails once, so setup rolls itself back while the cleanup
    # guard is already active. Deleting the app twice would 404.
    client = FakeTelnyxClient(
        phone_number={
            "id": "phone-id",
            "phone_number": "+15551234567",
            "connection_id": "original-app-id",
        },
        update_failures=1,
    )
    resources = TelnyxExampleResources()

    with pytest.raises(TelnyxAPIError):
        with telnyx_example_cleanup(client, resources):
            prepare_telnyx_example_setup(
                client,
                api_key="key",
                phone_number="+15551234567",
                ngrok_url="example.ngrok-free.app",
                setup_telnyx=True,
                route_phone_number=True,
                resources=resources,
            )

    assert client.deleted_app_ids == ["created-app-id"]
    assert resources.created_call_control_app_id is None


def test_telnyx_example_cleanup_retries_only_the_step_that_failed():
    client = FakeTelnyxClient(
        phone_number={
            "id": "phone-id",
            "phone_number": "+15551234567",
            "connection_id": "original-app-id",
        },
        delete_failures=1,
    )
    resources = TelnyxExampleResources(
        phone_number_id="phone-id",
        created_call_control_app_id="created-app-id",
        original_connection_id="original-app-id",
    )

    with pytest.raises(TelnyxSetupError, match="delete temporary"):
        with telnyx_example_cleanup(client, resources):
            pass

    # Routing is restored and forgotten; the app deletion is still outstanding,
    # so a partial cleanup is never recorded as finished.
    assert client.connection_updates == [("phone-id", "original-app-id")]
    assert resources.original_connection_id is None
    assert resources.created_call_control_app_id == "created-app-id"

    with telnyx_example_cleanup(client, resources):
        pass

    assert client.connection_updates == [("phone-id", "original-app-id")]
    assert client.deleted_app_ids == ["created-app-id"]


# Runs in a subprocess: the handler ends the process with the default SIGTERM
# disposition, which cannot be observed in-process.
SIGTERM_EXAMPLE_SCRIPT = '''
import signal
import sys

from vision_agents.plugins.telnyx.example_helpers import (
    TelnyxConfig,
    TelnyxExampleSetup,
    telnyx_example_cleanup,
)

marker = sys.argv[1]


class MarkerClient:
    """Stands in for TelnyxClient and records the cleanup calls it receives."""

    def _record(self, line):
        with open(marker, "a") as handle:
            handle.write(line + "\\n")

    def update_phone_number_connection(self, phone_number_id, connection_id):
        self._record(f"restore-routing:{phone_number_id}:{connection_id}")

    def delete_call_control_app(self, app_id):
        self._record(f"delete-app:{app_id}")


setup = TelnyxExampleSetup(
    config=TelnyxConfig(
        api_key="key",
        call_control_app_id="created-app-id",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
    ),
    phone_number_id="phone-id",
    created_call_control_app_id="created-app-id",
    original_connection_id="original-app-id",
)

with telnyx_example_cleanup(MarkerClient(), setup):
    print("READY", flush=True)
    signal.pause()
'''


# SIGTERM is delivered from inside the fake client, i.e. while
# prepare_telnyx_example_setup is still running.
SIGTERM_DURING_SETUP_SCRIPT = '''
import os
import signal
import sys

from vision_agents.plugins.telnyx.example_helpers import (
    TelnyxExampleResources,
    prepare_telnyx_example_setup,
    telnyx_example_cleanup,
)

marker = sys.argv[1]


class MarkerClient:
    """Fake TelnyxClient that gets SIGTERMed part-way through setup."""

    def _record(self, line):
        with open(marker, "a") as handle:
            handle.write(line + "\\n")

    def find_phone_number(self, phone_number):
        return {
            "id": "phone-id",
            "phone_number": phone_number,
            "connection_id": "original-app-id",
        }

    def get_first_outbound_voice_profile_id(self):
        return "profile-id"

    def create_call_control_app(
        self, *, application_name, webhook_event_url, outbound_voice_profile_id
    ):
        self._record("create-app:created-app-id")
        return {"data": {"id": "created-app-id"}}

    def update_phone_number_connection(self, phone_number_id, connection_id):
        self._record(f"route:{phone_number_id}:{connection_id}")
        if connection_id == "created-app-id":
            # The app exists and the number is re-routed: exactly the window the
            # old code left uncovered.
            os.kill(os.getpid(), signal.SIGTERM)

    def delete_call_control_app(self, app_id):
        self._record(f"delete-app:{app_id}")


client = MarkerClient()
resources = TelnyxExampleResources()

with telnyx_example_cleanup(client, resources):
    prepare_telnyx_example_setup(
        client,
        api_key="key",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
        setup_telnyx=True,
        route_phone_number=True,
        resources=resources,
    )
    raise AssertionError("SIGTERM should have ended the process during setup")
'''


# A second SIGTERM is delivered from inside the first cleanup.
SECOND_SIGTERM_SCRIPT = '''
import os
import signal
import sys
import time

from vision_agents.plugins.telnyx.example_helpers import (
    TelnyxConfig,
    TelnyxExampleSetup,
    telnyx_example_cleanup,
)

marker = sys.argv[1]


class MarkerClient:
    """Fake TelnyxClient that gets a second SIGTERM while cleaning up."""

    def _record(self, line):
        with open(marker, "a") as handle:
            handle.write(line + "\\n")

    def update_phone_number_connection(self, phone_number_id, connection_id):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
        self._record(f"restore-routing:{phone_number_id}:{connection_id}")

    def delete_call_control_app(self, app_id):
        self._record(f"delete-app:{app_id}")


setup = TelnyxExampleSetup(
    config=TelnyxConfig(
        api_key="key",
        call_control_app_id="created-app-id",
        phone_number="+15551234567",
        ngrok_url="example.ngrok-free.app",
    ),
    phone_number_id="phone-id",
    created_call_control_app_id="created-app-id",
    original_connection_id="original-app-id",
)

with telnyx_example_cleanup(MarkerClient(), setup):
    print("READY", flush=True)
    signal.pause()
'''


def _run_sigterm_example(tmp_path, source, *, send_sigterm):
    marker = tmp_path / "cleanup_calls.txt"
    script = tmp_path / "sigterm_example.py"
    script.write_text(source)

    process = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert process.stdout is not None
        if send_sigterm:
            assert process.stdout.readline().strip() == "READY"
            # signal.pause() returns only once the handler has run.
            time.sleep(0.1)
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    lines = marker.read_text().splitlines() if marker.exists() else []
    return process.returncode, lines


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX-only")
def test_telnyx_example_cleanup_runs_on_sigterm(tmp_path):
    returncode, lines = _run_sigterm_example(
        tmp_path, SIGTERM_EXAMPLE_SCRIPT, send_sigterm=True
    )

    # Killed by SIGTERM, i.e. the conventional 128 + 15 exit status.
    assert returncode == -signal.SIGTERM
    assert lines == [
        "restore-routing:phone-id:original-app-id",
        "delete-app:created-app-id",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX-only")
def test_telnyx_example_cleanup_covers_sigterm_during_setup(tmp_path):
    returncode, lines = _run_sigterm_example(
        tmp_path, SIGTERM_DURING_SETUP_SCRIPT, send_sigterm=False
    )

    assert returncode == -signal.SIGTERM
    assert lines == [
        "create-app:created-app-id",
        "route:phone-id:created-app-id",
        "route:phone-id:original-app-id",
        "delete-app:created-app-id",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX-only")
def test_second_sigterm_does_not_interrupt_cleanup(tmp_path):
    returncode, lines = _run_sigterm_example(
        tmp_path, SECOND_SIGTERM_SCRIPT, send_sigterm=True
    )

    assert returncode == -signal.SIGTERM
    assert lines == [
        "restore-routing:phone-id:original-app-id",
        "delete-app:created-app-id",
    ]
