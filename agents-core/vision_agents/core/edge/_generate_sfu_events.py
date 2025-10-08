"""Utility to generate static SFU event dataclass definitions.

This script inspects ``events_pb2`` at runtime and materialises a static Python
module containing dataclass wrappers for every protobuf message emitted by the
SFU. Run this script whenever the upstream protobuf schema changes.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, List, Optional, Sequence, Tuple, Type

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2


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
    "from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant",
    "from vision_agents.core.events.base import BaseEvent",
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
    if field_descriptor.label == FieldDescriptor.LABEL_REPEATED:
        base_type = type_map.get(field_descriptor.type, "Any")
        # For message types in repeated fields
        if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
            base_type = "Any"  # Could be refined to specific message type
        return f"Optional[List[{base_type}]]"
    
    # Handle message types (nested protobuf messages)
    if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
        return "Optional[Any]"  # Could be refined to specific message type
    
    # Handle enum types
    if field_descriptor.type == FieldDescriptor.TYPE_ENUM:
        return "Optional[int]"  # Enums are represented as ints
    
    # Handle scalar types - all made optional
    base_type = type_map.get(field_descriptor.type, "Any")
    return f"Optional[{base_type}]"


def _render_class(proto_name: str, message_cls: Type[Message]) -> List[str]:
    class_name = _class_name(proto_name)
    event_type = message_cls.DESCRIPTOR.full_name
    
    # Get field descriptors for this message
    field_descriptors = message_cls.DESCRIPTOR.fields

    lines = ["@dataclass", f"class {class_name}(BaseEvent):"]
    lines.append(f"    \"\"\"Dataclass event for {message_cls.__module__}.{message_cls.__name__}.\"\"\"")
    
    # Override type field with the specific event type
    lines.append(f"    type: str = field(default=\"{event_type}\", init=False)")
    
    # Add payload field (optional to match BaseEvent pattern)
    lines.append(f"    payload: Optional[events_pb2.{proto_name}] = field(default=None, repr=False)")
    
    # Add property fields for each protobuf field (skip fields that conflict with BaseEvent)
    base_event_fields = {"type", "event_id", "timestamp", "session_id", "user_metadata"}
    for field_desc in field_descriptors:
        field_name = field_desc.name
        if field_name in base_event_fields:  # Skip fields that conflict with BaseEvent fields
            continue
        type_hint = _get_python_type_from_protobuf_field(field_desc)
        lines.append("")
        lines.append("    @property")
        lines.append(f"    def {field_name}(self) -> {type_hint}:")
        lines.append(f"        \"\"\"Access {field_name} field from the protobuf payload.\"\"\"")
        lines.append(f"        if self.payload is None:")
        lines.append(f"            return None")
        lines.append(f"        return getattr(self.payload, '{field_name}', None)")
    
    lines.append("")
    lines.append("    @classmethod")
    lines.append("    def from_proto(cls, proto_obj: events_pb2.{0}, **extra):".format(proto_name))
    lines.append("        \"\"\"Create event instance from protobuf message.\"\"\"")
    lines.append("        return cls(payload=proto_obj, **extra)")
    lines.append("")
    lines.append("    def as_dict(self) -> Dict[str, Any]:")
    lines.append("        \"\"\"Convert protobuf payload to dictionary.\"\"\"")
    lines.append("        if self.payload is None:")
    lines.append("            return {}")
    lines.append("        return _to_dict(self.payload)")
    lines.append("")
    lines.append("    def __getattr__(self, item: str):")
    lines.append("        \"\"\"Delegate attribute access to protobuf payload.\"\"\"")
    lines.append("        if self.payload is not None:")
    lines.append("            return getattr(self.payload, item)")
    lines.append("        raise AttributeError(f\"'{self.__class__.__name__}' object has no attribute '{item}'\")")
    lines.append("")
    return lines


def _render_module_body() -> Tuple[List[str], List[str]]:
    class_blocks: List[str] = []
    registry_names: List[str] = []

    for proto_name, message_cls in _iter_protobuf_messages():
        class_name = _class_name(proto_name)
        registry_names.append(class_name)
        class_lines = _render_class(proto_name, message_cls)
        class_blocks.append("\n".join(class_lines))

    return class_blocks, registry_names


def _build_module() -> str:
    class_blocks, exported_names = _render_module_body()

    parts: List[str] = [
        "\"\"\"Auto-generated SFU event dataclasses. Do not edit manually.\"\"\"",
        "# Generated by _generate_sfu_events.py",
        *HEADER_LINES,
        *class_blocks,
    ]

    exports_section = [
        "",
        "__all__ = (",
        *[f"    \"{name}\"," for name in exported_names],
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
    target_path = pathlib.Path(__file__).with_name("sfu_events.py")
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
        if not hasattr(event_class, '__mro__'):
            print(f"✗ {class_name} is not a class")
            all_valid = False
            continue
            
        # Check field correspondence
        proto_fields = {f.name: f for f in message_cls.DESCRIPTOR.fields}
        
        # Check that all protobuf fields are accessible via properties
        for field_name, field_desc in proto_fields.items():
            if field_name in {"type", "event_id", "timestamp", "session_id", "user_metadata"}:
                continue  # Skip BaseEvent fields
                
            if not hasattr(event_class, field_name):
                print(f"✗ {class_name} missing property for protobuf field: {field_name}")
                all_valid = False
                continue
            
            # Verify it's a property
            attr = getattr(type(event_class), field_name, None)
            if not isinstance(attr, property):
                print(f"✗ {class_name}.{field_name} is not a property")
                all_valid = False
                continue
        
        print(f"✓ {class_name} verified ({len(proto_fields)} protobuf fields)")
    
    return all_valid


def verify_field_types() -> None:
    """Verify and display field type mappings for all protobuf messages."""
    print("\n" + "="*80)
    print("Field Type Verification Report")
    print("="*80 + "\n")
    
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
            if field_name in {"type", "event_id", "timestamp", "session_id", "user_metadata"}:
                continue
                
            python_type = _get_python_type_from_protobuf_field(field_desc)
            proto_type_name = FieldDescriptor.Type.Name(field_desc.type)
            label_name = FieldDescriptor.Label.Name(field_desc.label)
            
            print(f"  - {field_name}: {proto_type_name} ({label_name}) → {python_type}")


def main() -> None:
    import sys
    
    target_path = pathlib.Path(__file__).with_name("sfu_events.py")
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

