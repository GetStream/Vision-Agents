package agents

import (
	"context"

	"github.com/GetStream/Vision-Agents/agents-core-go/tools"
)

// Registrar is anything holding a function registry. The pipeline is the one that matters:
// the functions live here, so the model asks for them over the session's socket and this
// process answers.
type Registrar interface {
	Functions() *tools.Registry
}

// RegisterFunction offers one of your functions to the model.
//
// The argument type is described to the model by reflection over its `json` and `schema`
// tags, so declaring the struct is declaring the tool:
//
//	agents.RegisterFunction(llm, "get_weather", "Get current weather for a location",
//	    func(ctx context.Context, in struct {
//	        Location string `json:"location" schema:"the city and state"`
//	    }) (any, error) {
//	        return weatherAt(ctx, in.Location)
//	    })
func RegisterFunction[In any](target Registrar, name, description string, run func(context.Context, In) (any, error)) error {
	return tools.Register(target.Functions(), name, description, run)
}
