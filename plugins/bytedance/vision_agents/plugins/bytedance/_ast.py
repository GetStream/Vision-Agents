"""Minimal proto3 codec for the AST 2.0 (Live Interpretation) messages.

The wire format follows ``ast.proto`` in this package (field numbers taken
verbatim from the upstream schema). A hand-written codec avoids a hard
dependency on a specific ``protobuf`` runtime version and keeps encode/decode
trivially testable from raw bytes.
"""

from dataclasses import dataclass
from typing import Optional

_WIRE_VARINT = 0
_WIRE_LEN = 2
_WIRE_I64 = 1
_WIRE_I32 = 5


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _bytes_field(field_number: int, value: bytes) -> bytes:
    return _key(field_number, _WIRE_LEN) + _varint(len(value)) + value


def _string_field(field_number: int, value: str) -> bytes:
    return _bytes_field(field_number, value.encode())


def _int_field(field_number: int, value: int) -> bytes:
    return _key(field_number, _WIRE_VARINT) + _varint(value)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _decode_fields(buf: bytes) -> dict[int, list]:
    fields: dict[int, list] = {}
    pos = 0
    n = len(buf)
    value: int | bytes
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == _WIRE_VARINT:
            value, pos = _read_varint(buf, pos)
        elif wire_type == _WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire_type == _WIRE_I32:
            value = buf[pos : pos + 4]
            pos += 4
        elif wire_type == _WIRE_I64:
            value = buf[pos : pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        fields.setdefault(field_number, []).append(value)
    return fields


@dataclass
class Audio:
    format: str = ""
    codec: str = ""
    rate: int = 0
    bits: int = 0
    channel: int = 0
    binary_data: bytes = b""

    def encode(self) -> bytes:
        out = bytearray()
        if self.format:
            out += _string_field(4, self.format)
        if self.codec:
            out += _string_field(5, self.codec)
        if self.rate:
            out += _int_field(7, self.rate)
        if self.bits:
            out += _int_field(8, self.bits)
        if self.channel:
            out += _int_field(9, self.channel)
        if self.binary_data:
            out += _bytes_field(14, self.binary_data)
        return bytes(out)


@dataclass
class ReqParams:
    mode: str = ""
    source_language: str = ""
    target_language: str = ""
    speaker_id: str = ""

    def encode(self) -> bytes:
        out = bytearray()
        if self.mode:
            out += _string_field(1, self.mode)
        if self.source_language:
            out += _string_field(2, self.source_language)
        if self.target_language:
            out += _string_field(3, self.target_language)
        if self.speaker_id:
            out += _string_field(4, self.speaker_id)
        return bytes(out)


@dataclass
class TranslateRequest:
    event: int = 0
    session_id: str = ""
    user_uid: str = ""
    source_audio: Optional[Audio] = None
    target_audio: Optional[Audio] = None
    request: Optional[ReqParams] = None
    denoise: Optional[bool] = None

    def encode(self) -> bytes:
        out = bytearray()
        if self.session_id:
            out += _bytes_field(1, _string_field(6, self.session_id))
        if self.event:
            out += _int_field(2, self.event)
        if self.user_uid:
            out += _bytes_field(3, _string_field(1, self.user_uid))
        if self.source_audio is not None:
            out += _bytes_field(4, self.source_audio.encode())
        if self.target_audio is not None:
            out += _bytes_field(5, self.target_audio.encode())
        if self.request is not None:
            out += _bytes_field(6, self.request.encode())
        if self.denoise is not None:
            out += _int_field(7, 1 if self.denoise else 0)
        return bytes(out)


@dataclass
class TranslateResponse:
    event: int = 0
    data: bytes = b""
    text: str = ""
    session_id: str = ""
    status_code: int = 0
    message: str = ""

    @classmethod
    def decode(cls, buf: bytes) -> "TranslateResponse":
        fields = _decode_fields(buf)
        response = cls()
        if 2 in fields:
            response.event = fields[2][-1]
        if 3 in fields:
            response.data = bytes(fields[3][-1])
        if 4 in fields:
            response.text = fields[4][-1].decode()
        if 1 in fields:
            meta = _decode_fields(fields[1][-1])
            if 1 in meta:
                response.session_id = meta[1][-1].decode()
            if 3 in meta:
                response.status_code = meta[3][-1]
            if 4 in meta:
                response.message = meta[4][-1].decode()
        return response
