// Package tools holds the caller's own functions: what the model is told about them, and
// what runs when it asks for one.
package tools

import (
	"fmt"
	"reflect"
	"strings"
	"time"
)

// DescriptionTag is the struct tag carrying what a field means. The model chooses arguments
// by these, so a field without one is a field it has to guess at.
const DescriptionTag = "schema"

var timeType = reflect.TypeOf(time.Time{})

// Schema renders a Go type as the JSON Schema object a model is offered.
//
// Field names come from the `json` tag and descriptions from the `schema` tag. A field is
// required unless it is a pointer or its `json` tag says omitempty, which is the only way
// Go says "this one may be left out".
func Schema(t reflect.Type) (map[string]any, error) {
	return schemaOf(t, map[reflect.Type]bool{})
}

func schemaOf(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}, nil
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Interface:
		// An empty interface says nothing about the value, and a schema that says nothing
		// is how JSON Schema spells that.
		if t.NumMethod() == 0 {
			return map[string]any{}, nil
		}
		return nil, fmt.Errorf("tools: %s is an interface, which has no schema", t)
	case reflect.Slice, reflect.Array:
		// A []byte arrives as base64 in JSON, so it is a string to whoever is filling it in.
		if t.Elem().Kind() == reflect.Uint8 && t.Kind() == reflect.Slice {
			return map[string]any{"type": "string", "contentEncoding": "base64"}, nil
		}
		items, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("tools: %s is keyed by %s, and JSON objects are keyed by strings", t, t.Key())
		}
		values, err := schemaOf(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": values}, nil
	case reflect.Struct:
		return structSchema(t, seen)
	default:
		return nil, fmt.Errorf("tools: %s cannot be described to a model", t)
	}
}

func structSchema(t reflect.Type, seen map[reflect.Type]bool) (map[string]any, error) {
	if seen[t] {
		return nil, fmt.Errorf("tools: %s contains itself, and a schema for it would never end", t)
	}
	seen[t] = true
	defer delete(seen, t)

	properties := map[string]any{}
	var required []string
	if err := collect(t, seen, properties, &required); err != nil {
		return nil, err
	}

	schema := map[string]any{"type": "object", "properties": properties}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema, nil
}

// collect walks a struct's fields into properties, following embedded structs as though
// their fields were declared here, which is what encoding/json does with them.
func collect(t reflect.Type, seen map[reflect.Type]bool, properties map[string]any, required *[]string) error {
	for i := range t.NumField() {
		field := t.Field(i)

		name, omitempty, skip := fieldName(field)
		if skip {
			continue
		}

		// An embedded struct reads as though its fields were declared here, which is what
		// encoding/json does with one, and it does so even when the type itself is
		// unexported.
		if field.Anonymous && name == "" {
			embedded := field.Type
			for embedded.Kind() == reflect.Pointer {
				embedded = embedded.Elem()
			}
			if embedded.Kind() == reflect.Struct {
				if err := collect(embedded, seen, properties, required); err != nil {
					return err
				}
				continue
			}
		}
		if !field.IsExported() {
			continue
		}
		if name == "" {
			name = field.Name
		}

		schema, err := schemaOf(field.Type, seen)
		if err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), field.Name, err)
		}
		if description := field.Tag.Get(DescriptionTag); description != "" {
			schema["description"] = description
		}
		properties[name] = schema

		if !omitempty && field.Type.Kind() != reflect.Pointer {
			*required = append(*required, name)
		}
	}
	return nil
}

// fieldName reads the `json` tag: the name it gives, whether it may be left out, and
// whether the field is hidden from JSON altogether.
func fieldName(field reflect.StructField) (name string, omitempty bool, skip bool) {
	tag, ok := field.Tag.Lookup("json")
	if !ok {
		if field.Anonymous {
			return "", false, false
		}
		return field.Name, false, false
	}
	if tag == "-" {
		return "", false, true
	}

	name, options, _ := strings.Cut(tag, ",")
	for _, option := range strings.Split(options, ",") {
		if option == "omitempty" {
			omitempty = true
		}
	}
	if name == "" && !field.Anonymous {
		name = field.Name
	}
	return name, omitempty, false
}
