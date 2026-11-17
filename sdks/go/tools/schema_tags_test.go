package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/suite"
)

// SchemaTagsSuite is what a model is told about arguments beyond their names and types.
type SchemaTagsSuite struct {
	suite.Suite
}

func TestSchemaTagsSuite(t *testing.T) {
	suite.Run(t, new(SchemaTagsSuite))
}

type seatClass string

func (seatClass) JSONSchema() map[string]any {
	return map[string]any{"type": "string", "enum": []any{"economy", "business"}}
}

type bookFlight struct {
	FlightID string    `json:"flight_id" schema:"the flight the user picked" pattern:"^[A-Z]{2}[0-9]+$" example:"LH400"`
	Seats    int       `json:"seats" minimum:"1" maximum:"9"`
	Meal     string    `json:"meal" enum:"none,vegetarian,vegan"`
	Email    string    `json:"email,omitempty" format:"email"`
	Class    seatClass `json:"class" schema:"the cabin"`
	Floor    int       `json:"floor" enum:"1,2,3"`
}

// property is the schema of one of bookFlight's arguments.
func (s *SchemaTagsSuite) property(name string) map[string]any {
	schema, err := Schema(reflect.TypeOf(bookFlight{}))
	s.Require().NoError(err)
	return schema["properties"].(map[string]any)[name].(map[string]any)
}

func (s *SchemaTagsSuite) TestAPatternAndAnExampleAreCarried() {
	flight := s.property("flight_id")

	s.Equal("^[A-Z]{2}[0-9]+$", flight["pattern"])
	s.Equal([]any{"LH400"}, flight["examples"])
	s.Equal("the flight the user picked", flight["description"])
}

func (s *SchemaTagsSuite) TestBoundsAreNumbers() {
	seats := s.property("seats")

	s.Equal(1.0, seats["minimum"])
	s.Equal(9.0, seats["maximum"])
}

func (s *SchemaTagsSuite) TestAnEnumIsItsValues() {
	s.Equal([]any{"none", "vegetarian", "vegan"}, s.property("meal")["enum"])
}

func (s *SchemaTagsSuite) TestAnEnumOnANumberIsNumbers() {
	s.Equal([]any{int64(1), int64(2), int64(3)}, s.property("floor")["enum"])
}

func (s *SchemaTagsSuite) TestAFormatIsCarried() {
	s.Equal("email", s.property("email")["format"])
}

func (s *SchemaTagsSuite) TestATypeDescribingItselfKeepsItsDescriptionBeside() {
	class := s.property("class")

	s.Equal([]any{"economy", "business"}, class["enum"])
	s.Equal("the cabin", class["description"])
	s.NotContains(seatClass("").JSONSchema(), "description", "the type's own map was written into")
}

func (s *SchemaTagsSuite) TestATagThatDoesNotFitItsFieldIsRefused() {
	type In struct {
		Seats int `json:"seats" minimum:"one"`
	}

	_, err := Schema(reflect.TypeOf(In{}))

	s.Error(err)
}

func (s *SchemaTagsSuite) TestASchemaWrittenByHandIsOfferedAsWritten() {
	registry := NewRegistry()
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"route": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "array"},
			}},
		},
	}

	s.Require().NoError(RegisterSchema(registry, "plan_route", "Plan a route", parameters,
		func(context.Context, json.RawMessage) (any, error) { return "planned", nil }))

	s.Equal(parameters, registry.List()[0].Parameters)
}

func (s *SchemaTagsSuite) TestAFunctionRegisteredBySchemaIsHandedTheRawArguments() {
	registry := NewRegistry()
	var got json.RawMessage
	s.Require().NoError(RegisterSchema(registry, "plan_route", "Plan a route", map[string]any{"type": "object"},
		func(_ context.Context, arguments json.RawMessage) (any, error) {
			got = arguments
			return "planned", nil
		}))

	output, err := registry.Call(s.T().Context(), "plan_route", `{"route":["A","B"]}`)

	s.Require().NoError(err)
	s.Equal("planned", output.Text)
	s.JSONEq(`{"route":["A","B"]}`, string(got))
}

func (s *SchemaTagsSuite) TestParametersThatAreNotAnObjectAreRefused() {
	err := RegisterSchema(NewRegistry(), "bad", "Bad", map[string]any{"type": "string"},
		func(context.Context, json.RawMessage) (any, error) { return nil, nil })

	s.Error(err)
}
