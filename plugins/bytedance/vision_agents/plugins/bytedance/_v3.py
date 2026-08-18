"""ByteDance / Volcengine Seed Speech v3 binary WebSocket framing.

Two closely related framings share the same 4-byte header:

* **sauc** (streaming ASR): header + optional 4-byte sequence + payload-size + body.
  See https://docs.volcengine.com/docs/6561/1354869
* **event** (bidirectional TTS): header + 4-byte event + optional session id +
  payload-size + body. See https://www.volcengine.com/docs/6561/1329505

All integers are big-endian. JSON bodies are gzip-compressed.
"""

import gzip
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001


class MsgType(IntEnum):
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_CLIENT = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    FRONTEND_RESULT = 0b1100
    ERROR = 0b1111


class Flags(IntEnum):
    NO_SEQ = 0b0000
    POS_SEQ = 0b0001
    LAST_NO_SEQ = 0b0010
    NEG_SEQ = 0b0011
    WITH_EVENT = 0b0100


class Serialization(IntEnum):
    RAW = 0b0000
    JSON = 0b0001


class Compression(IntEnum):
    NONE = 0b0000
    GZIP = 0b0001


class EventType(IntEnum):
    """Event codes for the bidirectional (event) framing."""

    NONE = 0
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154
    TASK_REQUEST = 200
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359


# Connection-level events carry no session id in the framing.
_CONNECTION_EVENTS = frozenset(
    {
        EventType.START_CONNECTION,
        EventType.FINISH_CONNECTION,
        EventType.CONNECTION_STARTED,
        EventType.CONNECTION_FAILED,
        EventType.CONNECTION_FINISHED,
    }
)


@dataclass
class Message:
    """A parsed server frame."""

    type: MsgType
    payload: Any = None
    event: Optional[int] = None
    session_id: Optional[str] = None
    sequence: Optional[int] = None
    code: Optional[int] = None


def _pack_header(
    msg_type: MsgType,
    flags: Flags,
    serialization: Serialization,
    compression: Compression,
) -> bytes:
    return bytes(
        [
            (PROTOCOL_VERSION << 4) | DEFAULT_HEADER_SIZE,
            (int(msg_type) << 4) | int(flags),
            (int(serialization) << 4) | int(compression),
            0,
        ]
    )


def build_full_client_request(payload: dict, sequence: int = 1) -> bytes:
    """Build a sauc full client request (config packet) with a gzip JSON body."""
    body = gzip.compress(json.dumps(payload).encode())
    header = _pack_header(
        MsgType.FULL_CLIENT_REQUEST, Flags.POS_SEQ, Serialization.JSON, Compression.GZIP
    )
    return (
        header
        + sequence.to_bytes(4, "big", signed=True)
        + len(body).to_bytes(4, "big")
        + body
    )


def build_audio_only_request(audio: bytes, sequence: int, last: bool = False) -> bytes:
    """Build a sauc audio-only request with a gzip audio body.

    Args:
        audio: Raw PCM bytes for this packet.
        sequence: Positive sequence number; negated automatically when ``last``.
        last: Marks the final packet (negative sequence).
    """
    flags = Flags.NEG_SEQ if last else Flags.POS_SEQ
    seq = -sequence if last else sequence
    body = gzip.compress(audio)
    header = _pack_header(
        MsgType.AUDIO_ONLY_CLIENT, flags, Serialization.RAW, Compression.GZIP
    )
    return (
        header
        + seq.to_bytes(4, "big", signed=True)
        + len(body).to_bytes(4, "big")
        + body
    )


def build_event_message(
    msg_type: MsgType,
    event: EventType,
    *,
    payload: bytes = b"",
    session_id: Optional[str] = None,
    serialization: Serialization = Serialization.JSON,
    compression: Compression = Compression.GZIP,
) -> bytes:
    """Build an event-framed message (bidirectional TTS)."""
    out = bytearray(
        _pack_header(msg_type, Flags.WITH_EVENT, serialization, compression)
    )
    out += int(event).to_bytes(4, "big", signed=True)
    if session_id is not None:
        sid = session_id.encode()
        out += len(sid).to_bytes(4, "big") + sid
    body = payload
    if compression == Compression.GZIP and payload:
        body = gzip.compress(payload)
    out += len(body).to_bytes(4, "big") + body
    return bytes(out)


def parse_response(data: bytes) -> Message:
    """Parse a server frame from either framing into a :class:`Message`."""
    header_size = data[0] & 0x0F
    msg_type = MsgType(data[1] >> 4)
    flags = data[1] & 0x0F
    serialization = data[2] >> 4
    compression = data[2] & 0x0F
    body = data[header_size * 4 :]

    event: Optional[int] = None
    session_id: Optional[str] = None
    sequence: Optional[int] = None
    code: Optional[int] = None

    if flags & Flags.WITH_EVENT:
        event = int.from_bytes(body[:4], "big", signed=True)
        body = body[4:]
        if event not in _CONNECTION_EVENTS:
            sid_len = int.from_bytes(body[:4], "big")
            body = body[4:]
            session_id = body[:sid_len].decode()
            body = body[sid_len:]
    elif flags & Flags.POS_SEQ:
        sequence = int.from_bytes(body[:4], "big", signed=True)
        body = body[4:]

    if msg_type == MsgType.ERROR:
        code = int.from_bytes(body[:4], "big")
        size = int.from_bytes(body[4:8], "big")
        body = body[8 : 8 + size]
    else:
        size = int.from_bytes(body[:4], "big")
        body = body[4 : 4 + size]

    if compression == Compression.GZIP and body:
        body = gzip.decompress(body)

    is_audio = msg_type == MsgType.AUDIO_ONLY_SERVER
    payload: Any = body
    if serialization == Serialization.JSON and not is_audio and body:
        payload = json.loads(body)

    return Message(
        type=msg_type,
        payload=payload,
        event=event,
        session_id=session_id,
        sequence=sequence,
        code=code,
    )
