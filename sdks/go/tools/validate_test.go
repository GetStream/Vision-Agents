package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/suite"
)

// ValidateSuite is a call's arguments held to the schema the model was told.
type ValidateSuite struct {
	suite.Suite
}

func TestValidateSuite(t *testing.T) {
	suite.Run(t, new(ValidateSuite))
}

type booking struct {
	FlightID string `json:"flight_id" schema:"e.g. LH400" pattern:"^[A-Z]{2}[0-9]{1,4}$"`
	Seats    int    `json:"seats" minimum:"1" maximum:"9"`
	Cabin    string `json:"cabin,omitempty" enum:"economy,business,first" default:"economy"`
}

// bookings registers book_flight, which records each booking it is handed.
func (s *ValidateSuite) bookings() (*Registry, *[]booking) {
	registry := NewRegistry()
	var booked []booking
	s.Require().NoError(Register(registry, "book_flight", "Book a flight for the caller",
		func(_ context.Context, in booking) (any, error) {
			booked = append(booked, in)
			return "booked", nil
		}))
	return registry, &booked
}

func (s *ValidateSuite) TestArgumentsTheSchemaTakesRunTheFunction() {
	registry, booked := s.bookings()

	output, err := registry.Call(s.T().Context(), "book_flight", `{"flight_id":"LH400","seats":2}`)

	s.Require().NoError(err)
	s.Equal("booked", output.Text)
	s.Equal([]booking{{FlightID: "LH400", Seats: 2}}, *booked)
}

func (s *ValidateSuite) TestArgumentsTheSchemaRefusesFailNamingEachWithoutRunning() {
	registry, booked := s.bookings()

	_, err := registry.Call(s.T().Context(), "book_flight", `{"flight_id":"lh400","seats":12,"cabin":"steerage"}`)

	s.Require().Error(err)
	s.Contains(err.Error(), "book_flight was asked for with arguments it cannot take:")
	s.Contains(err.Error(), "/flight_id: ")
	s.Contains(err.Error(), "/seats: ")
	s.Contains(err.Error(), "/cabin: ")
	s.Empty(*booked)
}

func (s *ValidateSuite) TestARequiredArgumentLeftOutIsRefused() {
	registry, booked := s.bookings()

	_, err := registry.Call(s.T().Context(), "book_flight", `{"flight_id":"LH400"}`)

	s.Require().Error(err)
	s.Contains(err.Error(), "seats")
	s.Empty(*booked)
}

func (s *ValidateSuite) TestAFormatIsHeldTo() {
	type In struct {
		Day string `json:"day" format:"date"`
	}
	registry := NewRegistry()
	s.Require().NoError(Register(registry, "plan", "Plan a day",
		func(context.Context, In) (any, error) { return "planned", nil }))

	_, err := registry.Call(s.T().Context(), "plan", `{"day":"next tuesday"}`)
	s.Require().Error(err)

	_, err = registry.Call(s.T().Context(), "plan", `{"day":"2026-11-03"}`)
	s.NoError(err)
}

func (s *ValidateSuite) TestMoreConstraintsAreCarried() {
	type In struct {
		Name   string   `json:"name" minLength:"1" maxLength:"40"`
		Tags   []string `json:"tags" minItems:"1" maxItems:"5"`
		Price  float64  `json:"price" exclusiveMinimum:"0" multipleOf:"0.01"`
		Copies int      `json:"copies,omitempty" exclusiveMaximum:"100" default:"1"`
	}

	schema, err := Schema(reflect.TypeOf(In{}))

	s.Require().NoError(err)
	properties := schema["properties"].(map[string]any)
	s.Equal(map[string]any{"type": "string", "minLength": 1, "maxLength": 40}, properties["name"])
	s.Equal(map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "minItems": 1, "maxItems": 5}, properties["tags"])
	s.Equal(map[string]any{"type": "number", "exclusiveMinimum": 0.0, "multipleOf": 0.01}, properties["price"])
	s.Equal(map[string]any{"type": "integer", "exclusiveMaximum": 100.0, "default": int64(1)}, properties["copies"])
}

func (s *ValidateSuite) TestTheJSONSchemaTagIsRead() {
	type In struct {
		Name  string `json:"name" jsonschema:"description=Full name\\, as on the passport,minLength=1"`
		Cabin string `json:"cabin" jsonschema:"enum=economy,enum=business,default=economy"`
		Code  string `json:"code" jsonschema:"pattern=^[A-Z]{2}[0-9]{1\\,4}$,title=Flight"`
	}

	schema, err := Schema(reflect.TypeOf(In{}))

	s.Require().NoError(err)
	properties := schema["properties"].(map[string]any)
	s.Equal(map[string]any{"type": "string", "description": "Full name, as on the passport", "minLength": 1}, properties["name"])
	s.Equal(map[string]any{"type": "string", "enum": []any{"economy", "business"}, "default": "economy"}, properties["cabin"])
	s.Equal(map[string]any{"type": "string", "pattern": "^[A-Z]{2}[0-9]{1,4}$", "title": "Flight"}, properties["code"])
}

const findOrder = `{
  "type": "object",
  "properties": {
    "lookup": {
      "oneOf": [
        {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}, "required": ["email"]}
      ]
    }
  },
  "required": ["lookup"]
}`

func (s *ValidateSuite) TestASchemaWrittenAsJSONIsOfferedAndHeldTo() {
	registry := NewRegistry()
	s.Require().NoError(RegisterSchema(registry, "find_order", "Find an order by id or by email", []byte(findOrder),
		func(_ context.Context, args json.RawMessage) (any, error) { return string(args), nil }))

	var want map[string]any
	s.Require().NoError(json.Unmarshal([]byte(findOrder), &want))
	s.Equal(want, registry.List()[0].Parameters)

	output, err := registry.Call(s.T().Context(), "find_order", `{"lookup":{"email":"sam@example.com"}}`)
	s.Require().NoError(err)
	s.Equal(`{"lookup":{"email":"sam@example.com"}}`, output.Text)

	_, err = registry.Call(s.T().Context(), "find_order", `{"lookup":{"email":"not an address"}}`)
	s.Error(err)
}

func (s *ValidateSuite) TestASchemaAnotherLibraryBuiltIsTakenAsTheJSONItMarshalsTo() {
	type schema struct {
		Type       string                    `json:"type"`
		Properties map[string]map[string]any `json:"properties"`
	}
	registry := NewRegistry()

	err := RegisterSchema(registry, "ping", "Ping a host",
		schema{Type: "object", Properties: map[string]map[string]any{"host": {"type": "string"}}},
		func(context.Context, json.RawMessage) (any, error) { return "pong", nil })

	s.Require().NoError(err)
	s.Equal(map[string]any{"type": "object", "properties": map[string]any{"host": map[string]any{"type": "string"}}},
		registry.List()[0].Parameters)
}

func (s *ValidateSuite) TestASchemaWrittenByHandCanStillDecodeIntoAStruct() {
	type In struct {
		Lookup struct {
			OrderID string `json:"order_id"`
			Email   string `json:"email"`
		} `json:"lookup"`
	}
	registry := NewRegistry()
	var seen In
	s.Require().NoError(RegisterWithSchema(registry, "find_order", "Find an order by id or by email", []byte(findOrder),
		func(_ context.Context, in In) (any, error) {
			seen = in
			return "found", nil
		}))

	_, err := registry.Call(s.T().Context(), "find_order", `{"lookup":{"order_id":"1042"}}`)

	s.Require().NoError(err)
	s.Equal("1042", seen.Lookup.OrderID)
}

func (s *ValidateSuite) TestASchemaThatIsNotOneIsRefusedWhenRegistered() {
	err := RegisterSchema(NewRegistry(), "broken", "Takes something",
		[]byte(`{"type":"object","properties":{"n":{"type":"integer","minimum":"one"}}}`),
		func(context.Context, json.RawMessage) (any, error) { return "", nil })

	s.Error(err)
}
