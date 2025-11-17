"""Utility to generate static SFU event dataclass definitions.

This script inspects ``events_pb2`` at runtime and materialises a static Python
module containing dataclass wrappers for every protobuf message emitted by the
SFU. Run this script whenever the upstream protobuf schema changes.
"""

from __future__ import annotations

import pathlib
from typing import (
    Dict as TypingDict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2


HEADER_LINES: Sequence[str] = (
    "from __future__ import annotations",
    "",
    "import uuid",
    "from dataclasses import dataclass, field",
    "from datetime import datetime, timezone",
    "from typing import Any, Dict, List, Optional",
    "",
    "from dataclasses_json import DataClassJsonMixin",
    "from google.protobuf.json_format import MessageToDict",
    "from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2",
    "from vision_agents.core.events.base import BaseEvent",
    "",
    "# Note: For enum fields typed as 'int', use the corresponding enum from:",
    "# getstream.video.rtc.pb.stream.video.sfu.models.models_pb2",
    "# Available enums: TrackType, ConnectionQuality, ErrorCode, PeerType, etc.",
    "",
    "",
    "def _to_dict(message) -> Dict[str, Any]:",
    "    try:",
    "        return MessageToDict(message, preserving_proto_field_name=True)",
    "    except Exception:",
    "        return {}",
    "",
)


def _iter_protobuf_messages() -> Iterable[Tuple[str, Type[Message]]]:
    for name in sorted(dir(events_pb2)):
        attr = getattr(events_pb2, name)
        if isinstance(attr, type) and issubclass(attr, Message):
            yield name, attr


def _class_name(proto_name: str) -> str:
    return proto_name if proto_name.endswith("Event") else proto_name + "Event"


def _collect_message_types() -> TypingDict[str, Type[Message]]:
    """Collect all message types referenced in event fields (recursively)."""
    message_types: TypingDict[str, Type[Message]] = {}
    to_process: Set[str] = set()

    # Collect all message types from events
    for proto_name, message_cls in _iter_protobuf_messages():
        for field_desc in message_cls.DESCRIPTOR.fields:
            if field_desc.type == FieldDescriptor.TYPE_MESSAGE:
                message_type_name = field_desc.message_type.full_name
                if message_type_name.startswith("stream.video.sfu.model"):
                    class_name = message_type_name.split(".")[-1]
                    to_process.add(class_name)

    # Process messages and their nested message types recursively
    processed: Set[str] = set()
    while to_process:
        class_name = to_process.pop()
        if class_name in processed:
            continue
        processed.add(class_name)

        if hasattr(models_pb2, class_name):
            message_cls = getattr(models_pb2, class_name)
            message_types[class_name] = message_cls

            # Check for nested message types
            for field_desc in message_cls.DESCRIPTOR.fields:
                if field_desc.type == FieldDescriptor.TYPE_MESSAGE:
                    nested_type_name = field_desc.message_type.full_name
                    if nested_type_name.startswith("stream.video.sfu.model"):
                        nested_class_name = nested_type_name.split(".")[-1]
                        if nested_class_name not in processed:
                            to_process.add(nested_class_name)

    return message_types


def _get_message_type_name(field_descriptor: FieldDescriptor) -> Optional[str]:
    """Get the wrapper class name for a message type field."""
    if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
        message_type_name = field_descriptor.message_type.full_name

        # Map known message types to their wrapper classes
        if message_type_name.startswith("stream.video.sfu.model"):
            class_name = message_type_name.split(".")[-1]
            # Return the class name which will be defined at the top
            return class_name

    return None


def _get_enum_type_name(field_descriptor: FieldDescriptor) -> Optional[str]:
    """Get the enum class name for an enum type field for documentation purposes."""
    if field_descriptor.type == FieldDescriptor.TYPE_ENUM:
        enum_type_name = field_descriptor.enum_type.full_name
        # Extract just the class name from the full name (e.g., stream.video.sfu.models.TrackType -> TrackType)
        if enum_type_name.startswith("stream.video.sfu.models."):
            return enum_type_name.split(".")[-1]
    return None


def _get_python_type_from_protobuf_field(field_descriptor: FieldDescriptor) -> str:
    """Determine Python type from protobuf field descriptor.

    Maps protobuf field types to their corresponding Python types.
    All fields are returned as Optional since we want optional semantics.
    """
    # Map protobuf types to Python types
    type_map = {
        FieldDescriptor.TYPE_DOUBLE: "float",
        FieldDescriptor.TYPE_FLOAT: "float",
        FieldDescriptor.TYPE_INT64: "int",
        FieldDescriptor.TYPE_UINT64: "int",
        FieldDescriptor.TYPE_INT32: "int",
        FieldDescriptor.TYPE_FIXED64: "int",
        FieldDescriptor.TYPE_FIXED32: "int",
        FieldDescriptor.TYPE_BOOL: "bool",
        FieldDescriptor.TYPE_STRING: "str",
        FieldDescriptor.TYPE_BYTES: "bytes",
        FieldDescriptor.TYPE_UINT32: "int",
        FieldDescriptor.TYPE_SFIXED32: "int",
        FieldDescriptor.TYPE_SFIXED64: "int",
        FieldDescriptor.TYPE_SINT32: "int",
        FieldDescriptor.TYPE_SINT64: "int",
    }

    # Handle repeated fields (lists)
    if field_descriptor.is_repeated:
        # For enum types in repeated fields - use int, documented in docstring
        if field_descriptor.type == FieldDescriptor.TYPE_ENUM:
            return "Optional[List[int]]"
        # For message types in repeated fields
        elif field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
            message_type = _get_message_type_name(field_descriptor)
            base_type = message_type if message_type else "Any"
            return f"Optional[List[{base_type}]]"
        # For scalar types in repeated fields
        else:
            base_type = type_map.get(field_descriptor.type, "Any")
            return f"Optional[List[{base_type}]]"

    # Handle message types (nested protobuf messages)
    if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
        message_type = _get_message_type_name(field_descriptor)
        return f"Optional[{message_type}]" if message_type else "Optional[Any]"

    # Handle enum types - use int, documented in docstring
    if field_descriptor.type == FieldDescriptor.TYPE_ENUM:
        return "Optional[int]"

    # Handle scalar types - all made optional
    base_type = type_map.get(field_descriptor.type, "Any")
    return f"Optional[{base_type}]"


def _render_message_wrapper(class_name: str, message_cls: Type[Message]) -> List[str]:
    """Generate a dataclass wrapper for a protobuf message type (like Participant)."""
    lines = ["@dataclass", f"class {class_name}(DataClassJsonMixin):"]

    # Build docstring with enum field documentation
    docstring_lines = [f"Wrapper for {message_cls.DESCRIPTOR.full_name}."]
    field_descriptors = message_cls.DESCRIPTOR.fields
    enum_fields = []
    for field_desc in field_descriptors:
        if field_desc.type == FieldDescriptor.TYPE_ENUM:
            enum_type_name = _get_enum_type_name(field_desc)
            if enum_type_name:
                enum_fields.append((field_desc.name, enum_type_name))

    if enum_fields:
        docstring_lines.append("")
        docstring_lines.append("Enum fields (use values from models_pb2):")
        for field_name, enum_name in enum_fields:
            docstring_lines.append(f"    - {field_name}: {enum_name}")

    # Add docstring
    if len(docstring_lines) == 1:
        lines.append(f'    """{docstring_lines[0]}"""')
    else:
        lines.append('    """' + docstring_lines[0])
        for line in docstring_lines[1:]:
            lines.append("    " + line)
        lines.append('    """')

    if not field_descriptors:
        lines.append("    pass")
        lines.append("")
        return lines

    # Generate fields with proper types
    for field_desc in field_descriptors:
        field_name = field_desc.name
        python_type = _get_python_type_from_protobuf_field(field_desc)

        # Use proper default for optional fields
        lines.append(f"    {field_name}: {python_type} = None")

    lines.append("")
    lines.append("    @classmethod")
    lines.append(f"    def from_proto(cls, proto_obj) -> '{class_name}':")
    lines.append(f'        """Create from protobuf {class_name}."""')
    lines.append("        if proto_obj is None:")
    lines.append("            return cls()")
    lines.append("        return cls(")

    # Generate field assignments
    for i, field_desc in enumerate(field_descriptors):
        field_name = field_desc.name
        comma = "," if i < len(field_descriptors) - 1 else ""

        # Handle different field types
        if field_desc.type == FieldDescriptor.TYPE_MESSAGE and field_desc.is_repeated:
            # Repeated message fields
            message_type = _get_message_type_name(field_desc)
            if message_type:
                lines.append(
                    f"            {field_name}=[{message_type}.from_proto(item) for item in proto_obj.{field_name}]{comma}"
                )
            else:
                lines.append(
                    f"            {field_name}=list(proto_obj.{field_name}){comma}"
                )
        elif field_desc.type == FieldDescriptor.TYPE_MESSAGE:
            # Single message field
            message_type = _get_message_type_name(field_desc)
            if message_type:
                lines.append(
                    f"            {field_name}={message_type}.from_proto(proto_obj.{field_name}) if proto_obj.HasField('{field_name}') else None{comma}"
                )
            else:
                lines.append(
                    f"            {field_name}=proto_obj.{field_name} if proto_obj.HasField('{field_name}') else None{comma}"
                )
        elif field_desc.is_repeated:
            # Repeated scalar/enum fields - convert to list of ints for enums
            lines.append(
                f"            {field_name}=list(proto_obj.{field_name}){comma}"
            )
        else:
            # Regular scalar/enum fields
            lines.append(f"            {field_name}=proto_obj.{field_name}{comma}")

    lines.append("        )")
    lines.append("")
    return lines


def _render_class(proto_name: str, message_cls: Type[Message]) -> List[str]:
    class_name = _class_name(proto_name)
    event_type = message_cls.DESCRIPTOR.full_name

    # Get field descriptors for this message
    field_descriptors = message_cls.DESCRIPTOR.fields

    lines = ["@dataclass", f"class {class_name}(BaseEvent):"]
    lines.append(
        f'    """Dataclass event for {message_cls.__module__}.{message_cls.__name__}."""'
    )

    # Override type field with the specific event type
    lines.append(f'    type: str = field(default="{event_type}", init=False)')

    # Add payload field (optional to match BaseEvent pattern)
    lines.append(
        f"    payload: Optional[events_pb2.{proto_name}] = field(default=None, repr=False)"
    )

    # Add property fields for each protobuf field (skip fields that conflict with BaseEvent)
    base_event_fields = {"type", "event_id", "timestamp", "session_id", "user_metadata"}
    for field_desc in field_descriptors:
        field_name = field_desc.name
        if (
            field_name in base_event_fields
        ):  # Skip fields that conflict with BaseEvent fields
            continue
        type_hint = _get_python_type_from_protobuf_field(field_desc)
        lines.append("")
        lines.append("    @property")
        lines.append(f"    def {field_name}(self) -> {type_hint}:")

        # Build docstring with enum information if applicable
        docstring = f"Access {field_name} field from the protobuf payload."
        if field_desc.type == FieldDescriptor.TYPE_ENUM:
            enum_type_name = _get_enum_type_name(field_desc)
            if enum_type_name:
                docstring += f" Use models_pb2.{enum_type_name} enum."

        lines.append(f'        """{docstring}"""')
        lines.append("        if self.payload is None:")
        lines.append("            return None")

        # Handle message type fields - wrap them in our dataclass
        if field_desc.type == FieldDescriptor.TYPE_MESSAGE:
            message_type = _get_message_type_name(field_desc)
            if message_type:
                if field_desc.is_repeated:
                    lines.append(
                        f"        proto_list = getattr(self.payload, '{field_name}', [])"
                    )
                    lines.append(
                        f"        return [{message_type}.from_proto(item) for item in proto_list] if proto_list else None"
                    )
                else:
                    lines.append(
                        f"        proto_val = getattr(self.payload, '{field_name}', None)"
                    )
                    lines.append(
                        f"        return {message_type}.from_proto(proto_val) if proto_val is not None else None"
                    )
            else:
                lines.append(
                    f"        return getattr(self.payload, '{field_name}', None)"
                )
        else:
            # Scalar or enum fields
            lines.append(f"        return getattr(self.payload, '{field_name}', None)")

    lines.append("")
    lines.append("    @classmethod")
    lines.append(
        "    def from_proto(cls, proto_obj: events_pb2.{0}, **extra):".format(
            proto_name
        )
    )
    lines.append('        """Create event instance from protobuf message."""')
    lines.append("        return cls(payload=proto_obj, **extra)")
    lines.append("")
    lines.append("    def as_dict(self) -> Dict[str, Any]:")
    lines.append('        """Convert protobuf payload to dictionary."""')
    lines.append("        if self.payload is None:")
    lines.append("            return {}")
    lines.append("        return _to_dict(self.payload)")
    lines.append("")
    lines.append("    def __getattr__(self, item: str):")
    lines.append('        """Delegate attribute access to protobuf payload."""')
    lines.append("        if self.payload is not None:")
    lines.append("            return getattr(self.payload, item)")
    lines.append(
        "        raise AttributeError(f\"'{self.__class__.__name__}' object has no attribute '{item}'\")"
    )
    lines.append("")
    return lines


def _render_module_body() -> Tuple[List[str], List[str], List[str]]:
    """Generate message wrappers and event classes.

    Returns:
        Tuple of (message_wrapper_blocks, event_class_blocks, event_class_names)
    """
    # Collect all message types used in events
    message_types = _collect_message_types()

    # Generate message wrapper classes
    message_wrapper_blocks: List[str] = []
    for class_name in sorted(message_types.keys()):
        message_cls = message_types[class_name]
        wrapper_lines = _render_message_wrapper(class_name, message_cls)
        message_wrapper_blocks.append("\n".join(wrapper_lines))

    # Generate event classes
    event_class_blocks: List[str] = []
    event_class_names: List[str] = []

    for proto_name, message_cls in _iter_protobuf_messages():
        class_name = _class_name(proto_name)
        event_class_names.append(class_name)
        class_lines = _render_class(proto_name, message_cls)
        event_class_blocks.append("\n".join(class_lines))

    return message_wrapper_blocks, event_class_blocks, event_class_names


def _build_module() -> str:
    message_wrappers, event_classes, event_names = _render_module_body()

    parts: List[str] = [
        '"""Auto-generated SFU event dataclasses. Do not edit manually."""',
        "# Generated by _generate_sfu_events.py",
        *HEADER_LINES,
    ]

    # Add section header for message wrappers
    if message_wrappers:
        parts.extend(
            [
                "# " + "=" * 78,
                "# Message Type Wrappers",
                "# These are wrappers for protobuf message types used in events",
                "# " + "=" * 78,
                "",
            ]
        )
        parts.extend(message_wrappers)
        parts.append("")

    # Add section header for event classes
    parts.extend(
        [
            "# " + "=" * 78,
            "# Event Classes",
            "# " + "=" * 78,
            "",
        ]
    )
    parts.extend(event_classes)

    # Add exports
    exports_section = [
        "",
        "__all__ = (",
        *[f'    "{name}",' for name in event_names],
        ")",
    ]

    parts.extend(exports_section)

    return "\n".join(parts) + "\n"


def verify_generated_classes() -> bool:
    """Verify that generated classes match protobuf definitions.

    Returns:
        True if all checks pass, False otherwise.
    """
    import importlib.util
    import sys

    # Import the generated module
    target_path = pathlib.Path(__file__).parent / "core" / "edge" / "sfu_events.py"
    if not target_path.exists():
        print("Error: sfu_events.py not found. Run generation first.")
        return False

    # Dynamically load the module
    spec = importlib.util.spec_from_file_location("sfu_events", target_path)
    if spec is None or spec.loader is None:
        print("Error: Could not load sfu_events module")
        return False

    sfu_events = importlib.util.module_from_spec(spec)
    sys.modules["sfu_events"] = sfu_events
    spec.loader.exec_module(sfu_events)

    all_valid = True

    for proto_name, message_cls in _iter_protobuf_messages():
        class_name = _class_name(proto_name)

        # Check if class exists in generated module
        if not hasattr(sfu_events, class_name):
            print(f"✗ Class {class_name} not found in generated module")
            all_valid = False
            continue

        event_class = getattr(sfu_events, class_name)

        # Verify it's a BaseEvent subclass
        if not hasattr(event_class, "__mro__"):
            print(f"✗ {class_name} is not a class")
            all_valid = False
            continue

        # Check field correspondence
        proto_fields = {f.name: f for f in message_cls.DESCRIPTOR.fields}

        # Check that all protobuf fields are accessible via properties
        for field_name, field_desc in proto_fields.items():
            if field_name in {
                "type",
                "event_id",
                "timestamp",
                "session_id",
                "user_metadata",
            }:
                continue  # Skip BaseEvent fields

            if not hasattr(event_class, field_name):
                print(
                    f"✗ {class_name} missing property for protobuf field: {field_name}"
                )
                all_valid = False
                continue

            # Verify it's a property (check on the class itself, not an instance)
            attr = getattr(event_class, field_name, None)
            if not isinstance(attr, property):
                print(
                    f"✗ {class_name}.{field_name} is not a property (type: {type(attr).__name__})"
                )
                all_valid = False
                continue

        print(f"✓ {class_name} verified ({len(proto_fields)} protobuf fields)")

    return all_valid


def verify_field_types() -> None:
    """Verify and display field type mappings for all protobuf messages."""
    print("\n" + "=" * 80)
    print("Field Type Verification Report")
    print("=" * 80 + "\n")

    for proto_name, message_cls in _iter_protobuf_messages():
        class_name = _class_name(proto_name)
        print(f"\n{class_name} ({proto_name}):")
        print(f"  Protobuf type: {message_cls.DESCRIPTOR.full_name}")

        field_descriptors = message_cls.DESCRIPTOR.fields
        if not field_descriptors:
            print("  (no fields)")
            continue

        for field_desc in field_descriptors:
            field_name = field_desc.name
            if field_name in {
                "type",
                "event_id",
                "timestamp",
                "session_id",
                "user_metadata",
            }:
                continue

            python_type = _get_python_type_from_protobuf_field(field_desc)
            proto_type_name = field_desc.type
            label = (
                "repeated"
                if field_desc.is_repeated
                else "optional"
                if hasattr(field_desc, "is_optional")
                else "required"
            )

            print(f"  - {field_name}: type={proto_type_name} ({label}) → {python_type}")


def main() -> None:
    import sys

    # Generate sfu_events.py in the core/edge directory
    target_path = pathlib.Path(__file__).parent / "core" / "edge" / "sfu_events.py"
    target_path.write_text(_build_module(), encoding="utf-8")
    print(f"Regenerated {target_path}")

    # Verify field types
    if "--verify-types" in sys.argv:
        verify_field_types()

    # Verify generated classes
    if "--verify" in sys.argv:
        print("\nVerifying generated classes...")
        if verify_generated_classes():
            print("\n✓ All verifications passed!")
        else:
            print("\n✗ Some verifications failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
