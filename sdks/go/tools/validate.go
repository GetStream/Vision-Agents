package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// compile readies a function's parameters for holding its calls' arguments to them, refusing
// a schema that is not one.
func compile(parameters map[string]any) (*jsonschema.Schema, error) {
	encoded, err := json.Marshal(parameters)
	if err != nil {
		return nil, fmt.Errorf("its parameters cannot be written as JSON: %w", err)
	}
	// The library reads numbers its own way, so the schema goes through its reader too.
	document, err := jsonschema.UnmarshalJSON(bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("its parameters are not JSON: %w", err)
	}
	const location = "parameters.json"
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.AssertFormat()
	if err := compiler.AddResource(location, document); err != nil {
		return nil, fmt.Errorf("its parameters are not a JSON Schema: %w", err)
	}
	schema, err := compiler.Compile(location)
	if err != nil {
		return nil, fmt.Errorf("its parameters are not a JSON Schema: %w", err)
	}
	return schema, nil
}

// check holds a call's arguments to its function's schema, saying what is wrong with each
// one it refuses in words the model can correct, such as "/seats: maximum: got 12, want 9".
// Empty arguments are no arguments.
func check(schema *jsonschema.Schema, arguments []byte) error {
	if len(bytes.TrimSpace(arguments)) == 0 {
		arguments = []byte("{}")
	}
	instance, err := jsonschema.UnmarshalJSON(bytes.NewReader(arguments))
	if err != nil {
		return fmt.Errorf("they are not JSON: %w", err)
	}
	err = schema.Validate(instance)
	var invalid *jsonschema.ValidationError
	if !errors.As(err, &invalid) {
		return err
	}
	var problems []string
	for _, unit := range invalid.BasicOutput().Errors {
		if unit.Error == nil {
			continue
		}
		at := unit.InstanceLocation
		if at == "" {
			at = "/"
		}
		problems = append(problems, at+": "+unit.Error.String())
	}
	if len(problems) == 0 {
		return invalid
	}
	return errors.New(strings.Join(problems, "; "))
}

// parametersOf reads a schema written by hand: a map as it is, raw JSON as what it says, and
// anything else, such as another library's schema type, as the JSON it marshals to.
func parametersOf(parameters any) (map[string]any, error) {
	var raw []byte
	switch schema := parameters.(type) {
	case nil:
		return nil, errors.New("a tool's parameters are a JSON Schema object, and none was given")
	case map[string]any:
		return schema, nil
	case json.RawMessage:
		raw = schema
	case []byte:
		raw = schema
	case string:
		raw = []byte(schema)
	default:
		encoded, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("a tool's parameters cannot be written as JSON: %w", err)
		}
		raw = encoded
	}
	var read map[string]any
	if err := json.Unmarshal(raw, &read); err != nil {
		return nil, fmt.Errorf("a tool's parameters are a JSON Schema object: %w", err)
	}
	return read, nil
}
