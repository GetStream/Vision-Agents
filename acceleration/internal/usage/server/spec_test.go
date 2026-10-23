package server

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestTheCommittedSpecIsRenderedFromTheOperations(t *testing.T) {
	// The spec is what staff tools are typed from, so one edited by hand or left behind by a
	// change to an operation is a tool out of step with the service.
	committed, err := os.ReadFile("openapi.yaml")
	require.NoError(t, err)
	rendered, err := Spec()
	require.NoError(t, err)

	require.Equal(t, string(rendered), string(committed), "run go run ./cmd/usage openapi in acceleration/")
}

func TestTheSpecWritesNumbersAsNumbers(t *testing.T) {
	rendered, err := Spec()
	require.NoError(t, err)
	var document any
	require.NoError(t, yaml.Unmarshal(rendered, &document))

	var quoted []string
	var walk func(path string, value any)
	walk = func(path string, value any) {
		switch value := value.(type) {
		case map[string]any:
			numeric := value["type"] == "integer" || value["type"] == "number"
			for key, child := range value {
				_, isString := child.(string)
				switch key {
				case "minLength", "maxLength", "minItems", "maxItems":
					if isString {
						quoted = append(quoted, path+"."+key)
					}
				case "minimum", "maximum", "default", "example":
					if isString && numeric {
						quoted = append(quoted, path+"."+key)
					}
				}
				walk(path+"."+key, child)
			}
		case []any:
			for _, child := range value {
				walk(path+"[]", child)
			}
		}
	}
	walk("", document)

	require.Empty(t, quoted, "a numeric bound or default written as a string")
}
