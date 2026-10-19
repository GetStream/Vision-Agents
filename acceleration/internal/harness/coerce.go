package harness

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// Coerce rewrites a call's top-level arguments to the types its schema declares, where the
// model wrote one as another: Gemma passed patio as "True" and a party size as "4", and
// a tool that checks types refuses the call or reads the string as false. Arguments that do
// not parse, and values with no sensible reading, are left as the model wrote them.
func Coerce(parameters map[string]any, arguments string) string {
	properties, _ := parameters["properties"].(map[string]any)
	if len(properties) == 0 || strings.TrimSpace(arguments) == "" {
		return arguments
	}
	var values map[string]any
	if err := json.Unmarshal([]byte(arguments), &values); err != nil {
		return arguments
	}
	changed := false
	for name, value := range values {
		schema, _ := properties[name].(map[string]any)
		kind, _ := schema["type"].(string)
		if coerced, ok := coerceValue(kind, value); ok {
			values[name] = coerced
			changed = true
		}
	}
	if !changed {
		return arguments
	}
	rewritten, err := json.Marshal(values)
	if err != nil {
		return arguments
	}
	return string(rewritten)
}

func coerceValue(kind string, value any) (any, bool) {
	switch kind {
	case "boolean":
		text, ok := value.(string)
		if !ok {
			return nil, false
		}
		switch strings.ToLower(strings.TrimSpace(text)) {
		case "true", "yes":
			return true, true
		case "false", "no":
			return false, true
		}
	case "integer":
		switch typed := value.(type) {
		case string:
			if n, err := strconv.Atoi(strings.TrimSpace(typed)); err == nil {
				return n, true
			}
		case float64:
			if typed != math.Trunc(typed) {
				return nil, false
			}
		}
	case "number":
		if text, ok := value.(string); ok {
			if n, err := strconv.ParseFloat(strings.TrimSpace(text), 64); err == nil {
				return n, true
			}
		}
	case "string":
		switch typed := value.(type) {
		case float64:
			return strconv.FormatFloat(typed, 'f', -1, 64), true
		case bool:
			return strconv.FormatBool(typed), true
		}
	}
	return nil, false
}
