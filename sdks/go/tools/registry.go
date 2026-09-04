package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

// Function is one of the caller's own functions, as the model is offered it and as it runs.
type Function struct {
	// Name is how the model asks for it.
	Name string
	// Description is what the model is told it does, which is the whole of how it decides
	// when to reach for one.
	Description string
	// Parameters is the JSON Schema object describing the arguments.
	Parameters map[string]any

	run func(ctx context.Context, arguments []byte) (string, error)
}

// Registry holds the functions a session offers, in the order they were registered.
type Registry struct {
	mu        sync.RWMutex
	functions map[string]*Function
	order     []string
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{functions: map[string]*Function{}}
}

// Register adds a function to the registry.
//
// The argument type is described to the model by reflection, so a struct with `json` and
// `schema` tags is the whole declaration:
//
//	tools.Register(registry, "get_weather", "Get current weather for a location",
//	    func(ctx context.Context, in struct {
//	        Location string `json:"location" schema:"the city and state"`
//	    }) (any, error) {
//	        return weatherAt(ctx, in.Location)
//	    })
func Register[In any](registry *Registry, name, description string, run func(context.Context, In) (any, error)) error {
	if registry == nil {
		return fmt.Errorf("tools: %s has no registry to go in", name)
	}
	if name == "" {
		return fmt.Errorf("tools: a function needs a name")
	}
	if description == "" {
		return fmt.Errorf("tools: %s needs a description, since it is all the model has to choose by", name)
	}
	if run == nil {
		return fmt.Errorf("tools: %s has nothing to run", name)
	}

	var zero In
	parameters, err := Schema(reflect.TypeOf(&zero).Elem())
	if err != nil {
		return fmt.Errorf("tools: %s: %w", name, err)
	}

	return registry.add(&Function{
		Name:        name,
		Description: description,
		Parameters:  parameters,
		run: func(ctx context.Context, arguments []byte) (string, error) {
			var in In
			if len(arguments) > 0 {
				if err := json.Unmarshal(arguments, &in); err != nil {
					return "", fmt.Errorf("tools: %s was asked for with arguments it cannot take: %w", name, err)
				}
			}
			output, err := run(ctx, in)
			if err != nil {
				return "", err
			}
			return Render(output), nil
		},
	})
}

func (r *Registry) add(function *Function) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.functions == nil {
		r.functions = map[string]*Function{}
	}
	if _, taken := r.functions[function.Name]; taken {
		return fmt.Errorf("tools: %s is registered twice", function.Name)
	}
	r.functions[function.Name] = function
	r.order = append(r.order, function.Name)
	return nil
}

// List returns the functions in the order they were registered.
func (r *Registry) List() []*Function {
	r.mu.RLock()
	defer r.mu.RUnlock()

	listed := make([]*Function, 0, len(r.order))
	for _, name := range r.order {
		listed = append(listed, r.functions[name])
	}
	return listed
}

// Call runs one function and renders what it returned in words the model can use.
//
// Arguments are the JSON object the model wrote. Empty is treated as no arguments, since a
// model calling a function that takes none often sends nothing at all.
func (r *Registry) Call(ctx context.Context, name string, arguments string) (string, error) {
	r.mu.RLock()
	function := r.functions[name]
	r.mu.RUnlock()

	if function == nil {
		return "", fmt.Errorf("tools: nothing is registered as %s", name)
	}
	if arguments == "" {
		arguments = "{}"
	}
	return function.run(ctx, []byte(arguments))
}

// Render turns what a function returned into words the model can use. A string is already
// that; everything else becomes JSON.
func Render(output any) string {
	if text, ok := output.(string); ok {
		return text
	}
	encoded, err := json.Marshal(output)
	if err != nil {
		return fmt.Sprintf("%v", output)
	}
	return string(encoded)
}
