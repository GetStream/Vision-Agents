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

	run func(ctx context.Context, arguments []byte) (Output, error)
}

// Output is what a tool returns to show the model pictures beside its words, such as a frame
// of the caller's camera. The router hands the pictures to the agent's vision skill, which
// tells the conversation what it found; an agent without one is told the tool failed.
type Output struct {
	Text   string
	Images []Image
}

// Image is one picture a tool shows.
type Image struct {
	// URL is an https URL or a data URI, such as data:image/jpeg;base64,... One socket
	// message holds at most 5 MB, so shrink a photo before returning it.
	URL string
	// Detail is how closely the model looks: auto, low or high. Empty lets it decide.
	Detail string
}

// Wire is the output as a tool_result carries it: the words alone, or an array of parts
// when there are pictures.
func (o Output) Wire() any {
	if len(o.Images) == 0 {
		return o.Text
	}
	parts := make([]map[string]any, 0, len(o.Images)+1)
	if o.Text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": o.Text})
	}
	for _, image := range o.Images {
		url := map[string]any{"url": image.URL}
		if image.Detail != "" {
			url["detail"] = image.Detail
		}
		parts = append(parts, map[string]any{"type": "image_url", "image_url": url})
	}
	return parts
}

// Tool is one of the caller's functions, declared as a type.
//
// Its exported fields are the arguments the model fills in, described by their `json` and
// `schema` tags. Unexported fields are the caller's own, for whatever Run needs to reach:
//
//	type LookupOrder struct {
//	    OrderID string `json:"order_id" schema:"the order number, e.g. 1042"`
//	    orders  *Orders
//	}
//
//	func (LookupOrder) Name() string        { return "lookup_order" }
//	func (LookupOrder) Description() string { return "Look up an order by its number" }
//	func (l LookupOrder) Run(ctx context.Context) (any, error) {
//	    return l.orders.Find(ctx, l.OrderID)
//	}
type Tool interface {
	// Name is how the model asks for it.
	Name() string
	// Description is what the model is told it does.
	Description() string
	// Run does the work, with the model's arguments already in the fields. Returning an
	// Output shows the model pictures beside the words.
	Run(ctx context.Context) (any, error)
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
		run: func(ctx context.Context, arguments []byte) (Output, error) {
			var in In
			if len(arguments) > 0 {
				if err := json.Unmarshal(arguments, &in); err != nil {
					return Output{}, fmt.Errorf("tools: %s was asked for with arguments it cannot take: %w", name, err)
				}
			}
			output, err := run(ctx, in)
			if err != nil {
				return Output{}, err
			}
			return outputOf(output), nil
		},
	})
}

// Add offers a tool to the model.
//
// Every call runs on a copy of the value added, with the model's arguments decoded into it,
// so calls running at once never share their arguments.
func (r *Registry) Add(tool Tool) error {
	if r == nil {
		return fmt.Errorf("tools: there is no registry to add to")
	}
	value := reflect.ValueOf(tool)
	if tool == nil || (value.Kind() == reflect.Pointer && value.IsNil()) {
		return fmt.Errorf("tools: a nil tool cannot be added")
	}
	name, description := tool.Name(), tool.Description()
	if name == "" {
		return fmt.Errorf("tools: %T needs a name", tool)
	}
	if description == "" {
		return fmt.Errorf("tools: %s needs a description, since it is all the model has to choose by", name)
	}

	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return fmt.Errorf("tools: %s is a %s, and a tool's arguments are the fields of a struct", name, value.Kind())
	}
	parameters, err := Schema(value.Type())
	if err != nil {
		return fmt.Errorf("tools: %s: %w", name, err)
	}

	return r.add(&Function{
		Name:        name,
		Description: description,
		Parameters:  parameters,
		run: func(ctx context.Context, arguments []byte) (Output, error) {
			call := reflect.New(value.Type())
			call.Elem().Set(value)
			if len(arguments) > 0 {
				if err := json.Unmarshal(arguments, call.Interface()); err != nil {
					return Output{}, fmt.Errorf("tools: %s was asked for with arguments it cannot take: %w", name, err)
				}
			}
			output, err := call.Interface().(Tool).Run(ctx)
			if err != nil {
				return Output{}, err
			}
			return outputOf(output), nil
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

// Call runs one function and returns what it said: words the model can use, and any
// pictures it returned as an Output.
//
// Arguments are the JSON object the model wrote. Empty is treated as no arguments, since a
// model calling a function that takes none often sends nothing at all.
func (r *Registry) Call(ctx context.Context, name string, arguments string) (Output, error) {
	r.mu.RLock()
	function := r.functions[name]
	r.mu.RUnlock()

	if function == nil {
		return Output{}, fmt.Errorf("tools: nothing is registered as %s", name)
	}
	if arguments == "" {
		arguments = "{}"
	}
	return function.run(ctx, []byte(arguments))
}

// outputOf is what a function returned as the model is given it: an Output as it is, and
// anything else as words.
func outputOf(returned any) Output {
	switch output := returned.(type) {
	case Output:
		return output
	case *Output:
		if output != nil {
			return *output
		}
	}
	return Output{Text: Render(returned)}
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
