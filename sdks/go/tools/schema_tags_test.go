package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

type seatClass string

func (seatClass) JSONSchema() map[string]any {
	return map[string]any{"type": "string", "enum": []any{"economy", "business"}}
}

func TestTagsCarryMoreThanADescription(t *testing.T) {
	type BookFlight struct {
		FlightID string    `json:"flight_id" schema:"the flight the user picked" pattern:"^[A-Z]{2}[0-9]+$" example:"LH400"`
		Seats    int       `json:"seats" minimum:"1" maximum:"9"`
		Meal     string    `json:"meal" enum:"none,vegetarian,vegan"`
		Email    string    `json:"email,omitempty" format:"email"`
		Class    seatClass `json:"class" schema:"the cabin"`
	}
	schema, err := Schema(reflect.TypeOf(BookFlight{}))
	if err != nil {
		t.Fatal(err)
	}
	properties := schema["properties"].(map[string]any)
	property := func(name string) map[string]any { return properties[name].(map[string]any) }

	if got := property("flight_id"); got["pattern"] != "^[A-Z]{2}[0-9]+$" || !reflect.DeepEqual(got["examples"], []any{"LH400"}) {
		t.Errorf("flight_id is %v", got)
	}
	if got := property("seats"); got["minimum"] != 1.0 || got["maximum"] != 9.0 {
		t.Errorf("seats is %v", got)
	}
	if got := property("meal"); !reflect.DeepEqual(got["enum"], []any{"none", "vegetarian", "vegan"}) {
		t.Errorf("meal is %v", got)
	}
	if got := property("email"); got["format"] != "email" {
		t.Errorf("email is %v", got)
	}
	if got := property("class"); !reflect.DeepEqual(got["enum"], []any{"economy", "business"}) || got["description"] != "the cabin" {
		t.Errorf("a type describing itself became %v", got)
	}
}

func TestAnEnumOnANumberIsANumber(t *testing.T) {
	type In struct {
		Floor int `json:"floor" enum:"1,2,3"`
	}
	schema, err := Schema(reflect.TypeOf(In{}))
	if err != nil {
		t.Fatal(err)
	}
	floor := schema["properties"].(map[string]any)["floor"].(map[string]any)
	if !reflect.DeepEqual(floor["enum"], []any{int64(1), int64(2), int64(3)}) {
		t.Errorf("floor is %v", floor)
	}
}

func TestASchemaWrittenByHandIsOfferedAsWrittenAndHandedTheRawArguments(t *testing.T) {
	registry := NewRegistry()
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"route": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "array"},
			}},
		},
	}
	var got json.RawMessage
	err := RegisterSchema(registry, "plan_route", "Plan a route", parameters,
		func(_ context.Context, arguments json.RawMessage) (any, error) {
			got = arguments
			return "planned", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if offered := registry.List()[0].Parameters; !reflect.DeepEqual(offered, parameters) {
		t.Errorf("the model was offered %v", offered)
	}
	output, err := registry.Call(t.Context(), "plan_route", `{"route":["A","B"]}`)
	if err != nil || output.Text != "planned" || string(got) != `{"route":["A","B"]}` {
		t.Errorf("got %q, %v, and the function saw %s", output.Text, err, got)
	}
	if err := RegisterSchema(registry, "bad", "Bad", map[string]any{"type": "string"},
		func(context.Context, json.RawMessage) (any, error) { return nil, nil }); err == nil {
		t.Error("parameters that are not an object were accepted")
	}
}

func TestATagThatDoesNotFitItsFieldIsRefused(t *testing.T) {
	type In struct {
		Seats int `json:"seats" minimum:"one"`
	}
	if _, err := Schema(reflect.TypeOf(In{})); err == nil {
		t.Error("a minimum that is not a number was accepted")
	}
}
