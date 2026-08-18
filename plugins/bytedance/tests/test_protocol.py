import gzip
import json

from vision_agents.plugins.bytedance import _ast, _v3


class TestV3Sauc:
    """Encode/decode of the sequence-based ASR (sauc) framing."""

    def test_full_client_request_roundtrip(self):
        payload = {"audio": {"format": "pcm", "rate": 16000}, "request": {"x": 1}}
        frame = _v3.build_full_client_request(payload, sequence=1)

        assert frame[0] == (0b0001 << 4) | 0b0001
        assert frame[1] >> 4 == _v3.MsgType.FULL_CLIENT_REQUEST
        assert frame[1] & 0x0F == _v3.Flags.POS_SEQ

        parsed = _v3.parse_response(frame)
        assert parsed.type == _v3.MsgType.FULL_CLIENT_REQUEST
        assert parsed.sequence == 1
        assert parsed.payload == payload

    def test_audio_only_request_last_packet_negative_sequence(self):
        frame = _v3.build_audio_only_request(b"pcmbytes", sequence=5, last=True)

        parsed = _v3.parse_response(frame)
        assert parsed.type == _v3.MsgType.AUDIO_ONLY_CLIENT
        assert parsed.sequence == -5

    def test_parse_server_full_response(self):
        body = gzip.compress(json.dumps({"result": {"text": "hello"}}).encode())
        header = bytes(
            [
                (0b0001 << 4) | 0b0001,
                (int(_v3.MsgType.FULL_SERVER_RESPONSE) << 4) | _v3.Flags.POS_SEQ,
                (int(_v3.Serialization.JSON) << 4) | int(_v3.Compression.GZIP),
                0,
            ]
        )
        frame = (
            header
            + (2).to_bytes(4, "big", signed=True)
            + len(body).to_bytes(4, "big")
            + body
        )

        parsed = _v3.parse_response(frame)
        assert parsed.type == _v3.MsgType.FULL_SERVER_RESPONSE
        assert parsed.sequence == 2
        assert parsed.payload == {"result": {"text": "hello"}}

    def test_parse_error_response(self):
        body = gzip.compress(b'{"error":"bad"}')
        header = bytes(
            [
                (0b0001 << 4) | 0b0001,
                (int(_v3.MsgType.ERROR) << 4) | _v3.Flags.NO_SEQ,
                (int(_v3.Serialization.JSON) << 4) | int(_v3.Compression.GZIP),
                0,
            ]
        )
        frame = (
            header + (45000000).to_bytes(4, "big") + len(body).to_bytes(4, "big") + body
        )

        parsed = _v3.parse_response(frame)
        assert parsed.type == _v3.MsgType.ERROR
        assert parsed.code == 45000000
        assert parsed.payload == {"error": "bad"}


class TestV3Event:
    """Encode/decode of the event-based TTS framing."""

    def test_connection_event_has_no_session_id(self):
        frame = _v3.build_event_message(
            _v3.MsgType.FULL_CLIENT_REQUEST,
            _v3.EventType.START_CONNECTION,
            payload=b"{}",
        )
        parsed = _v3.parse_response(frame)
        assert parsed.event == _v3.EventType.START_CONNECTION
        assert parsed.session_id is None
        assert parsed.payload == {}

    def test_session_event_roundtrip_json(self):
        payload = json.dumps({"req_params": {"text": "hi"}}).encode()
        frame = _v3.build_event_message(
            _v3.MsgType.FULL_CLIENT_REQUEST,
            _v3.EventType.TASK_REQUEST,
            payload=payload,
            session_id="sess-1",
        )
        parsed = _v3.parse_response(frame)
        assert parsed.event == _v3.EventType.TASK_REQUEST
        assert parsed.session_id == "sess-1"
        assert parsed.payload == {"req_params": {"text": "hi"}}

    def test_audio_server_response_keeps_bytes(self):
        frame = _v3.build_event_message(
            _v3.MsgType.AUDIO_ONLY_SERVER,
            _v3.EventType.TTS_RESPONSE,
            payload=b"\x01\x02\x03\x04",
            session_id="sess-1",
            serialization=_v3.Serialization.RAW,
            compression=_v3.Compression.NONE,
        )
        parsed = _v3.parse_response(frame)
        assert parsed.type == _v3.MsgType.AUDIO_ONLY_SERVER
        assert parsed.event == _v3.EventType.TTS_RESPONSE
        assert parsed.payload == b"\x01\x02\x03\x04"


class TestAstProto:
    """Encode/decode of the AST 2.0 protobuf messages."""

    def test_translate_request_encodes_expected_wire_bytes(self):
        req = _ast.TranslateRequest(
            event=100,
            session_id="sid",
            request=_ast.ReqParams(
                mode="s2s", source_language="zh", target_language="en"
            ),
        )
        encoded = req.encode()

        # request_meta(1) { session_id(6): "sid" }
        assert b"\x0a\x05\x32\x03sid" in encoded
        # event(2) varint 100
        assert b"\x10\x64" in encoded

    def test_translate_response_roundtrip(self):
        meta = _ast._message_field(
            1, _ast._string_field(1, "sid") + _ast._int_field(3, 20000000)
        )
        raw = meta + _ast._int_field(2, 655) + _ast._string_field(4, "translated text")
        response = _ast.TranslateResponse.decode(raw)
        assert response.event == 655
        assert response.text == "translated text"
        assert response.session_id == "sid"
        assert response.status_code == 20000000

    def test_translate_response_audio_payload(self):
        raw = _ast._int_field(2, 352) + _ast._bytes_field(3, b"\x10\x20\x30")
        response = _ast.TranslateResponse.decode(raw)
        assert response.event == 352
        assert response.data == b"\x10\x20\x30"
