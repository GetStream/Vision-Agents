from datetime import date
from enum import Enum
from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, Field
from vision_agents.core.llm import FunctionRegistry, ToolArgumentsError


def _parameters(registry: FunctionRegistry) -> dict:
    return registry.get_tool_schemas()[0]["parameters_schema"]


class BookFlight(BaseModel):
    flight_id: str = Field(pattern=r"^[A-Z]{2}\d{1,4}$", description="e.g. LH400")
    seats: int = Field(ge=1, le=9)
    cabin: Literal["economy", "business", "first"] = "economy"


BOOK_FLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "flight_id": {
            "type": "string",
            "pattern": r"^[A-Z]{2}\d{1,4}$",
            "description": "e.g. LH400",
        },
        "seats": {"type": "integer", "minimum": 1, "maximum": 9},
        "cabin": {
            "type": "string",
            "enum": ["economy", "business", "first"],
            "default": "economy",
        },
    },
    "required": ["flight_id", "seats"],
}


class TestASimpleFunction:
    def test_a_plain_argument_is_its_type(self):
        registry = FunctionRegistry()

        @registry.register(description="Current weather in a city")
        async def get_weather(city: str) -> str:
            return city

        assert _parameters(registry) == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    def test_an_annotated_field_says_what_the_argument_means(self):
        registry = FunctionRegistry()

        @registry.register(description="Current weather in a city")
        async def get_weather(
            city: Annotated[str, Field(description="City name, e.g. Amsterdam")],
        ) -> str:
            return city

        assert _parameters(registry)["properties"]["city"] == {
            "type": "string",
            "description": "City name, e.g. Amsterdam",
        }

    def test_an_optional_argument_is_its_type_and_not_required(self):
        registry = FunctionRegistry()

        @registry.register(description="Search")
        async def search(query: str, limit: Optional[int] = None) -> str:
            return query

        assert _parameters(registry) == {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": None},
            },
            "required": ["query"],
        }

    def test_a_type_with_no_schema_is_described_as_a_string_as_before(self):
        class Opaque:
            pass

        registry = FunctionRegistry()

        @registry.register(description="Takes something")
        async def take(thing: Opaque) -> str:
            return "taken"

        assert _parameters(registry)["properties"]["thing"] == {"type": "string"}

    async def test_a_number_for_a_string_is_taken_as_its_digits(self):
        registry = FunctionRegistry()

        @registry.register(description="Look up a zip code")
        async def lookup(zip_code: str) -> str:
            return zip_code

        assert await registry.call_function("lookup", {"zip_code": 10115}) == "10115"


class TestConstrainedArguments:
    def test_fields_with_constraints_are_told_to_the_model(self):
        registry = FunctionRegistry()

        @registry.register(description="Book a flight for the caller")
        async def book_flight(
            flight_id: Annotated[
                str, Field(pattern=r"^[A-Z]{2}\d{1,4}$", description="e.g. LH400")
            ],
            seats: Annotated[int, Field(ge=1, le=9)],
            cabin: Literal["economy", "business", "first"] = "economy",
        ) -> str:
            return "booked"

        assert _parameters(registry) == BOOK_FLIGHT_SCHEMA

    async def test_arguments_the_schema_refuses_fail_naming_each_without_running(self):
        registry = FunctionRegistry()
        ran = []

        @registry.register(description="Book a flight for the caller")
        async def book_flight(
            flight_id: Annotated[str, Field(pattern=r"^[A-Z]{2}\d{1,4}$")],
            seats: Annotated[int, Field(ge=1, le=9)],
        ) -> str:
            ran.append(flight_id)
            return "booked"

        with pytest.raises(ToolArgumentsError) as refused:
            await registry.call_function(
                "book_flight", {"flight_id": "lh400", "seats": 12}
            )

        message = str(refused.value)
        assert message.startswith(
            "book_flight was asked for with arguments it cannot take:"
        )
        assert "flight_id: " in message
        assert "seats: " in message
        assert ran == []
        assert isinstance(refused.value, TypeError)

    async def test_arguments_arrive_as_their_types(self):
        class Cabin(Enum):
            ECONOMY = "economy"
            BUSINESS = "business"

        class Passenger(BaseModel):
            name: str = Field(min_length=1)

        registry = FunctionRegistry()
        seen = {}

        @registry.register(description="Plan a trip")
        async def plan(
            depart_on: date, cabin: Cabin, passengers: list[Passenger]
        ) -> str:
            seen.update(depart_on=depart_on, cabin=cabin, passengers=passengers)
            return "planned"

        await registry.call_function(
            "plan",
            {
                "depart_on": "2026-11-03",
                "cabin": "business",
                "passengers": [{"name": "Sam"}],
            },
        )

        assert seen == {
            "depart_on": date(2026, 11, 3),
            "cabin": Cabin.BUSINESS,
            "passengers": [Passenger(name="Sam")],
        }

    def test_a_nested_model_is_written_out_in_place(self):
        class Passenger(BaseModel):
            name: str = Field(min_length=1)
            age: int = Field(ge=0)

        registry = FunctionRegistry()

        @registry.register(description="Check passengers in")
        async def check_in(passengers: list[Passenger]) -> str:
            return "checked in"

        assert _parameters(registry) == {
            "type": "object",
            "properties": {
                "passengers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "age": {"type": "integer", "minimum": 0},
                        },
                        "required": ["name", "age"],
                    },
                }
            },
            "required": ["passengers"],
        }

    def test_a_field_named_title_is_kept(self):
        registry = FunctionRegistry()

        @registry.register(description="Name a book")
        async def name_book(title: str) -> str:
            return title

        assert _parameters(registry)["properties"] == {"title": {"type": "string"}}


class TestAFunctionTakingOneModel:
    def test_the_models_fields_are_the_tools_arguments(self):
        registry = FunctionRegistry()

        @registry.register(description="Book a flight for the caller")
        async def book_flight(booking: BookFlight) -> str:
            return "booked"

        assert _parameters(registry) == BOOK_FLIGHT_SCHEMA

    async def test_it_is_handed_the_model(self):
        registry = FunctionRegistry()
        seen = []

        @registry.register(description="Book a flight for the caller")
        async def book_flight(booking: BookFlight) -> str:
            seen.append(booking)
            return "booked"

        await registry.call_function("book_flight", {"flight_id": "LH400", "seats": 2})

        assert seen == [BookFlight(flight_id="LH400", seats=2, cabin="economy")]

    async def test_arguments_the_model_refuses_fail_without_running(self):
        registry = FunctionRegistry()
        ran = []

        @registry.register(description="Book a flight for the caller")
        async def book_flight(booking: BookFlight) -> str:
            ran.append(booking)
            return "booked"

        with pytest.raises(ToolArgumentsError, match="seats: "):
            await registry.call_function(
                "book_flight", {"flight_id": "LH400", "seats": 0}
            )
        assert ran == []

    def test_its_approval_names_the_models_fields(self):
        registry = FunctionRegistry()

        @registry.register(
            description="Book a flight", needs_approval="Book {flight_id}?"
        )
        async def book_flight(booking: BookFlight) -> str:
            return "booked"

        with pytest.raises(ValueError, match="names {booking}"):

            @registry.register(
                description="Book a flight", needs_approval="Book {booking}?"
            )
            async def book_again(booking: BookFlight) -> str:
                return "booked"
