package harness

import (
	"encoding/json"
	"testing"
)

func TestCoerceWritesArgumentsAsTheirSchemaTypes(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"time":       map[string]any{"type": "string"},
			"party_size": map[string]any{"type": "integer"},
			"patio":      map[string]any{"type": "boolean"},
			"pin":        map[string]any{"type": "string"},
		},
	}
	got := Coerce(schema, `{"time": "7:30", "party_size": "4", "patio": "True", "pin": 4471}`)
	var values map[string]any
	if err := json.Unmarshal([]byte(got), &values); err != nil {
		t.Fatalf("coerced arguments do not parse: %v", err)
	}
	if values["patio"] != true || values["party_size"] != float64(4) || values["pin"] != "4471" || values["time"] != "7:30" {
		t.Fatalf("Coerce = %s", got)
	}

	for _, unchanged := range []string{`{"patio": "maybe"}`, `{"time": "7:30"}`, `not json`, ``} {
		if got := Coerce(schema, unchanged); got != unchanged {
			t.Errorf("Coerce(%q) = %q, want it unchanged", unchanged, got)
		}
	}
}
