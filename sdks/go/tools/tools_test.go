package tools

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestASchemaSaysWhatEveryFieldMeans(t *testing.T) {
	type weather struct {
		Location string `json:"location" schema:"the city and state"`
		Days     int    `json:"days,omitempty" schema:"how far ahead to look"`
	}

	schema, err := Schema(reflect.TypeFor[weather]())
	if err != nil {
		t.Fatal(err)
	}

	properties := schema["properties"].(map[string]any)
	location := properties["location"].(map[string]any)
	if location["type"] != "string" {
		t.Errorf("location is %v, want a string", location["type"])
	}
	if location["description"] != "the city and state" {
		t.Errorf("location says %q, want the schema tag", location["description"])
	}
	if properties["days"].(map[string]any)["type"] != "integer" {
		t.Error("days is not an integer")
	}
}

func TestOmitemptyAndPointersAreWhatMakesAFieldOptional(t *testing.T) {
	type search struct {
		Query    string  `json:"query"`
		Limit    int     `json:"limit,omitempty"`
		AreaCode *string `json:"area_code"`
	}

	schema, err := Schema(reflect.TypeFor[search]())
	if err != nil {
		t.Fatal(err)
	}

	required, _ := schema["required"].([]string)
	if len(required) != 1 || required[0] != "query" {
		t.Errorf("required is %v, want only query", required)
	}
}

func TestAFieldHiddenFromJSONIsHiddenFromTheModel(t *testing.T) {
	type request struct {
		Question string `json:"question"`
		Secret   string `json:"-"`
		hidden   string
	}

	schema, err := Schema(reflect.TypeFor[request]())
	if err != nil {
		t.Fatal(err)
	}

	properties := schema["properties"].(map[string]any)
	if len(properties) != 1 {
		t.Errorf("the model is offered %v, want only the question", properties)
	}
	_ = request{}.hidden
}

func TestAnEmbeddedStructReadsAsThoughItsFieldsWereDeclaredHere(t *testing.T) {
	type page struct {
		Limit int `json:"limit,omitempty"`
	}
	type listing struct {
		page
		Status string `json:"status"`
	}

	schema, err := Schema(reflect.TypeFor[listing]())
	if err != nil {
		t.Fatal(err)
	}

	properties := schema["properties"].(map[string]any)
	if _, ok := properties["limit"]; !ok {
		t.Errorf("the embedded field is missing from %v", properties)
	}
}

func TestATypeThatContainsItselfIsRefusedRatherThanUnrolled(t *testing.T) {
	type node struct {
		Name  string `json:"name"`
		Child *node  `json:"child,omitempty"`
	}

	if _, err := Schema(reflect.TypeFor[node]()); err == nil {
		t.Fatal("a schema for a self-referential type would never end")
	}
}

func TestAFunctionRunsWithTheArgumentsTheModelWrote(t *testing.T) {
	registry := NewRegistry()
	err := Register(registry, "get_weather", "Get current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state"`
		}) (any, error) {
			return "sunny in " + in.Location, nil
		})
	if err != nil {
		t.Fatal(err)
	}

	output, err := registry.Call(t.Context(), "get_weather", `{"location":"Boulder, CO"}`)
	if err != nil {
		t.Fatal(err)
	}
	if output != "sunny in Boulder, CO" {
		t.Errorf("the function said %q", output)
	}
}

func TestAnythingButAStringComesBackAsJSON(t *testing.T) {
	registry := NewRegistry()
	err := Register(registry, "counts", "How many of each",
		func(_ context.Context, _ struct{}) (any, error) {
			return map[string]int{"calls": 2}, nil
		})
	if err != nil {
		t.Fatal(err)
	}

	output, err := registry.Call(t.Context(), "counts", "")
	if err != nil {
		t.Fatal(err)
	}
	if output != `{"calls":2}` {
		t.Errorf("the function said %q", output)
	}
}

func TestAFailingFunctionSaysSoRatherThanReturningNothing(t *testing.T) {
	registry := NewRegistry()
	err := Register(registry, "lookup", "Look something up",
		func(_ context.Context, _ struct{}) (any, error) {
			return nil, errors.New("the database is down")
		})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := registry.Call(t.Context(), "lookup", "{}"); err == nil {
		t.Fatal("a tool that failed has to say so, since the model is waiting on it")
	}
}

func TestRegisteringTwoFunctionsUnderOneNameIsRefused(t *testing.T) {
	registry := NewRegistry()
	run := func(_ context.Context, _ struct{}) (any, error) { return "", nil }

	if err := Register(registry, "one", "The first", run); err != nil {
		t.Fatal(err)
	}
	if err := Register(registry, "one", "The second", run); err == nil {
		t.Fatal("the model would have no way to say which it meant")
	}
}

func TestFunctionsComeBackInTheOrderTheyWereRegistered(t *testing.T) {
	registry := NewRegistry()
	run := func(_ context.Context, _ struct{}) (any, error) { return "", nil }
	for _, name := range []string{"first", "second", "third"} {
		if err := Register(registry, name, "does something", run); err != nil {
			t.Fatal(err)
		}
	}

	listed := registry.List()
	for index, want := range []string{"first", "second", "third"} {
		if listed[index].Name != want {
			t.Errorf("position %d holds %s, want %s", index, listed[index].Name, want)
		}
	}
}

func TestAFunctionNothingIsRegisteredUnderIsAnError(t *testing.T) {
	registry := NewRegistry()

	if _, err := registry.Call(t.Context(), "nothing", "{}"); err == nil {
		t.Fatal("calling a function that does not exist has to fail")
	}
}

// lookupOrder is a tool whose argument is its exported field and whose orders are its own.
type lookupOrder struct {
	OrderID string `json:"order_id" schema:"the order number"`
	orders  map[string]string
}

func (lookupOrder) Name() string        { return "lookup_order" }
func (lookupOrder) Description() string { return "Look up an order by its number" }
func (l lookupOrder) Run(context.Context) (any, error) {
	status, ok := l.orders[l.OrderID]
	if !ok {
		return nil, errors.New("no such order")
	}
	return map[string]string{"order_id": l.OrderID, "status": status}, nil
}

func TestAToolIsDeclaredByItsFieldsAndRunsWithTheModelsArguments(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Add(lookupOrder{orders: map[string]string{"1042": "shipped"}}); err != nil {
		t.Fatal(err)
	}

	listed := registry.List()
	if len(listed) != 1 || listed[0].Name != "lookup_order" || listed[0].Description != "Look up an order by its number" {
		t.Fatalf("the registry offers %+v, want lookup_order", listed)
	}
	properties := listed[0].Parameters["properties"].(map[string]any)
	if len(properties) != 1 || properties["order_id"] == nil {
		t.Errorf("the model is offered %v, want only order_id", properties)
	}

	output, err := registry.Call(t.Context(), "lookup_order", `{"order_id":"1042"}`)
	if err != nil {
		t.Fatal(err)
	}
	if output != `{"order_id":"1042","status":"shipped"}` {
		t.Errorf("the model was answered %s", output)
	}
}

func TestOneToolCallsArgumentsDoNotLeakIntoTheNext(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Add(&lookupOrder{orders: map[string]string{"1042": "shipped"}}); err != nil {
		t.Fatal(err)
	}

	if _, err := registry.Call(t.Context(), "lookup_order", `{"order_id":"1042"}`); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Call(t.Context(), "lookup_order", `{}`); err == nil {
		t.Error("a call without an order id found the previous call's order")
	}
}

type unnamed struct{}

func (unnamed) Name() string                     { return "" }
func (unnamed) Description() string              { return "does nothing" }
func (unnamed) Run(context.Context) (any, error) { return nil, nil }

type notAStruct string

func (notAStruct) Name() string                     { return "shout" }
func (notAStruct) Description() string              { return "shouts" }
func (notAStruct) Run(context.Context) (any, error) { return nil, nil }

func TestAToolTheModelCouldNotBeOfferedIsRefused(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Add(unnamed{}); err == nil {
		t.Error("a tool without a name was added")
	}
	if err := registry.Add(notAStruct("")); err == nil {
		t.Error("a tool with no fields to hold arguments was added")
	}
	if err := registry.Add((*lookupOrder)(nil)); err == nil {
		t.Error("a nil tool was added")
	}
	if len(registry.List()) != 0 {
		t.Errorf("the registry offers %d tools after refusing all of them", len(registry.List()))
	}
}
