package harness

import (
	"encoding/json"
	"testing"
)

func TestCoerceWritesArgumentsAsTheirSchemaTypes(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"time":      map[string]any{"type": "string"},
			"guests":    map[string]any{"type": "integer"},
			"breakfast": map[string]any{"type": "boolean"},
			"pin":       map[string]any{"type": "string"},
		},
	}
	got := Coerce(schema, `{"time": "18:45", "guests": "4", "breakfast": "True", "pin": 3825}`)
	var values map[string]any
	if err := json.Unmarshal([]byte(got), &values); err != nil {
		t.Fatalf("coerced arguments do not parse: %v", err)
	}
	if values["breakfast"] != true || values["guests"] != float64(4) || values["pin"] != "3825" || values["time"] != "18:45" {
		t.Fatalf("Coerce = %s", got)
	}

	for _, unchanged := range []string{`{"breakfast": "maybe"}`, `{"time": "18:45"}`, `not json`, ``} {
		if got := Coerce(schema, unchanged); got != unchanged {
			t.Errorf("Coerce(%q) = %q, want it unchanged", unchanged, got)
		}
	}
}
