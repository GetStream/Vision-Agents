// Package tools holds the caller's own functions: what the model is told about them, and
// what runs when it asks for one.
package tools

import (
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
)

// DescriptionTag is the struct tag carrying what a field means. The model chooses arguments
// by these, so a field without one is a field it has to guess at.
const DescriptionTag = "schema"

var timeType = reflect.TypeOf(time.Time{})

// Schemer is a type that describes itself to a model, for whatever reflection cannot say:
// a string that is one of a few values, a union, a shape only its own package knows.
type Schemer interface {
	JSONSchema() map[string]any
}

var schemerType = reflect.TypeOf((*Schemer)(nil)).Elem()

// selfDescribed is the schema a type gives itself, or nil for one that gives none.
func selfDescribed(t reflect.Type) map[string]any {
	var value any
	switch {
	case t.Implements(schemerType):
		value = reflect.Zero(t).Interface()
	case reflect.PointerTo(t).Implements(schemerType):
		value = reflect.New(t).Interface()
	default:
		return nil
	}
	described := value.(Schemer).JSONSchema()
	if described == nil {
		return nil
	}
	// A copy, so a description added beside it never writes into the type's own map.
	schema := make(map[string]any, len(described)+1)
	for key, value := range described {
		schema[key] = value
	}
	return schema
}

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
	if schema := selfDescribed(t); schema != nil {
		return schema, nil
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
		if err := constrain(schema, field.Tag); err != nil {
			return fmt.Errorf("%s.%s: %w", t.Name(), field.Name, err)
		}
		properties[name] = schema

		if !omitempty && field.Type.Kind() != reflect.Pointer {
			*required = append(*required, name)
		}
	}
	return nil
}

// JSONSchemaTag is the struct tag other Go JSON Schema libraries read, such as
// invopop/jsonschema: comma-separated key=value pairs, `enum` once per value, and `\,` for a
// comma inside a value, as in `jsonschema:"description=Full name,minLength=1"`.
const JSONSchemaTag = "jsonschema"

// numberKeywords and countKeywords are the constraints a tag gives as a number and as a whole
// count; textKeywords are taken as written.
var (
	numberKeywords = []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
	countKeywords  = []string{"minLength", "maxLength", "minItems", "maxItems"}
	textKeywords   = []string{"pattern", "format", "title"}
)

// constrain adds what a field's tags say beyond its description: `enum` as comma-separated
// values, the bounds in numberKeywords and countKeywords, `pattern`, `format`, a `default`
// and one `example`, each as a tag of its own or inside a `jsonschema` tag. Values are read
// as the field's own type, so an enum on an int is numbers.
func constrain(schema map[string]any, tag reflect.StructTag) error {
	kind, _ := schema["type"].(string)
	tags, err := keywordsOf(tag)
	if err != nil {
		return err
	}
	if description, ok := tags["description"]; ok && schema["description"] == nil {
		schema["description"] = description[0]
	}
	if values, ok := tags["enum"]; ok {
		var enum []any
		for _, value := range values {
			parsed, err := valueOf(kind, strings.TrimSpace(value))
			if err != nil {
				return fmt.Errorf("enum: %w", err)
			}
			enum = append(enum, parsed)
		}
		schema["enum"] = enum
	}
	for _, keyword := range numberKeywords {
		if value, ok := tags[keyword]; ok {
			number, err := strconv.ParseFloat(value[0], 64)
			if err != nil {
				return fmt.Errorf("%s %q is not a number", keyword, value[0])
			}
			schema[keyword] = number
		}
	}
	for _, keyword := range countKeywords {
		if value, ok := tags[keyword]; ok {
			count, err := strconv.Atoi(value[0])
			if err != nil || count < 0 {
				return fmt.Errorf("%s %q is not a count", keyword, value[0])
			}
			schema[keyword] = count
		}
	}
	for _, keyword := range textKeywords {
		if value, ok := tags[keyword]; ok {
			schema[keyword] = value[0]
		}
	}
	if value, ok := tags["default"]; ok {
		parsed, err := valueOf(kind, value[0])
		if err != nil {
			return fmt.Errorf("default: %w", err)
		}
		schema["default"] = parsed
	}
	if value, ok := tags["example"]; ok {
		parsed, err := valueOf(kind, value[0])
		if err != nil {
			return fmt.Errorf("example: %w", err)
		}
		schema["examples"] = []any{parsed}
	}
	return nil
}

// keywordsOf gathers a field's constraints from its tags of their own and its `jsonschema`
// tag, each keyword with its values: several for an enum, one for the rest.
func keywordsOf(tag reflect.StructTag) (map[string][]string, error) {
	keywords := map[string][]string{}
	if values, ok := tag.Lookup("enum"); ok {
		keywords["enum"] = strings.Split(values, ",")
	}
	for _, keyword := range slices.Concat(numberKeywords, countKeywords, textKeywords, []string{"default", "example"}) {
		if value, ok := tag.Lookup(keyword); ok {
			keywords[keyword] = []string{value}
		}
	}
	written, ok := tag.Lookup(JSONSchemaTag)
	if !ok {
		return keywords, nil
	}
	for _, pair := range splitUnescaped(written) {
		key, value, ok := strings.Cut(pair, "=")
		if !ok {
			// A bare word, such as required or nullable, says nothing a schema here carries.
			continue
		}
		key = strings.TrimSpace(key)
		if key == "enum" {
			keywords[key] = append(keywords[key], value)
			continue
		}
		if key == "examples" {
			key = "example"
		}
		keywords[key] = []string{value}
	}
	return keywords, nil
}

// splitUnescaped splits a `jsonschema` tag at each comma not written as `\,`.
func splitUnescaped(tag string) []string {
	var parts []string
	var part strings.Builder
	for i := 0; i < len(tag); i++ {
		switch {
		case tag[i] == '\\' && i+1 < len(tag) && tag[i+1] == ',':
			part.WriteByte(',')
			i++
		case tag[i] == ',':
			parts = append(parts, part.String())
			part.Reset()
		default:
			part.WriteByte(tag[i])
		}
	}
	return append(parts, part.String())
}

// valueOf reads a tag's value as a JSON Schema type names it.
func valueOf(kind, value string) (any, error) {
	switch kind {
	case "integer":
		return strconv.ParseInt(value, 10, 64)
	case "number":
		return strconv.ParseFloat(value, 64)
	case "boolean":
		return strconv.ParseBool(value)
	}
	return value, nil
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
