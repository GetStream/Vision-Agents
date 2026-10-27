"""A registered function's arguments as a Pydantic model: what the model is told, and what
each call's arguments are held to before the function runs."""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Callable, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticSchemaGenerationError,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
    create_model,
)
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.fields import FieldInfo

# What an argument with no type, or a type Pydantic cannot describe, is told to the model as,
# and taken as it comes.
_UNDESCRIBED = Annotated[Any, WithJsonSchema({"type": "string"})]

# A number the model sends for a string, such as a zip code, is taken as its digits.
_CONFIG = ConfigDict(coerce_numbers_to_str=True, arbitrary_types_allowed=True)


class ToolArgumentsError(TypeError):
    """A call's arguments are not what its function takes. The message says what is wrong
    with each, in words the model can correct."""


class Arguments:
    """A function's arguments as one Pydantic model.

    A function whose only parameter is a model takes that model's fields as its arguments
    and is handed the model; any other function takes its parameters as the fields.
    """

    def __init__(self, name: str, func: Callable, type_hints: dict[str, Any]):
        self.name = name
        parameters = [
            parameter
            for parameter in inspect.signature(func).parameters.values()
            if parameter.name != "self"
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        self.model_parameter: Optional[str] = None
        if len(parameters) == 1:
            only = type_hints.get(parameters[0].name)
            if inspect.isclass(only) and issubclass(only, BaseModel):
                self.model_parameter = parameters[0].name
                self.model: type[BaseModel] = only
                return

        fields: dict[str, Any] = {}
        for parameter in parameters:
            annotation = _describable(type_hints.get(parameter.name, _UNDESCRIBED))
            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            fields[parameter.name] = (annotation, default)
        self.model = create_model(f"{name}_arguments", __config__=_CONFIG, **fields)

    @property
    def names(self) -> set[str]:
        """The arguments the model fills in."""
        return set(self.model.model_fields)

    def json_schema(self) -> dict[str, Any]:
        """The arguments as the model is told them: the model's JSON Schema with every
        reference written out in place and Pydantic's titles left out, since some providers
        refuse `$defs`."""
        schema = self.model.model_json_schema()
        definitions = schema.pop("$defs", {})
        return _plain(schema, definitions)

    def parse(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """The keyword arguments to call the function with, from what the model sent.

        Arguments the function does not take are dropped, since a model sometimes adds one
        it saw on another tool. An argument left out is left to the function's own default.
        """
        try:
            parsed = self.model.model_validate(arguments)
        except ValidationError as invalid:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
                for error in invalid.errors()
            )
            raise ToolArgumentsError(
                f"{self.name} was asked for with arguments it cannot take: {problems}"
            ) from None
        if self.model_parameter is not None:
            return {self.model_parameter: parsed}
        return {field: getattr(parsed, field) for field in parsed.model_fields_set}


def _describable(annotation: Any) -> Any:
    """The annotation, or one taken as it comes when Pydantic cannot describe it."""
    try:
        TypeAdapter(annotation, config=_CONFIG).json_schema()
    except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema):
        return _UNDESCRIBED
    except Exception:
        # A TypeAdapter refuses config for a model or dataclass, which carries its own.
        try:
            TypeAdapter(annotation).json_schema()
        except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema):
            return _UNDESCRIBED
    if isinstance(annotation, FieldInfo):
        return _UNDESCRIBED
    return annotation


def _plain(schema: Any, definitions: dict[str, Any]) -> Any:
    """`schema` with each `$ref` replaced by its definition, titles dropped, and an optional
    value's `anyOf: [X, null]` read as X, which is how a function's optional argument was
    always told to the model."""
    if isinstance(schema, list):
        return [_plain(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema

    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        return _plain(
            {**definitions[reference.removeprefix("#/$defs/")], **siblings}, definitions
        )

    plain: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties" and isinstance(value, dict):
            plain[key] = {
                name: _plain(field, definitions) for name, field in value.items()
            }
        else:
            plain[key] = _plain(value, definitions)

    options = plain.get("anyOf")
    if isinstance(options, list) and len(options) == 2 and {"type": "null"} in options:
        (kept,) = [option for option in options if option != {"type": "null"}]
        del plain["anyOf"]
        plain = {**kept, **plain}
    return plain
