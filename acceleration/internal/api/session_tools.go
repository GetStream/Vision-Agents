package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var errNoSessionTools = notFound("the session recorded no tools: it ended before they were kept, or kept nothing")

type listSessionToolsRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listSessionToolsResponse struct {
	Body OfferedTools
}

// OfferedTools The tools a session's conversation model is offered, as they are sent to it.
type OfferedTools struct {
	Tools  []OfferedTool `json:"tools" doc:"Every tool, in the order the model is offered them."`
	Tokens int64         `json:"tokens" doc:"Roughly what offering them all costs on every request, in tokens."`
}

func (*OfferedTools) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The tools a session's conversation model is offered, as they are sent to it."
	return schema
}

// OfferedTool One tool as the model sees it.
type OfferedTool struct {
	Name        string         `json:"name" doc:"How the model asks for it."`
	Description string         `json:"description" doc:"What the model is told the tool does."`
	Parameters  map[string]any `json:"parameters,omitempty" doc:"The JSON Schema of its arguments."`
	Tokens      int64          `json:"tokens" doc:"Roughly what offering it costs on every request, in tokens: its name, description and schema at four characters a token."`
	Source      string         `json:"source,omitempty" enum:"session,worker,connector,builtin" doc:"Where the tool came from: session, declared by whoever opened the session; worker, hosted by a worker connected when the session opened; connector, opened by one of the agent's connector bindings; builtin, the router's own. A session's tools are fixed when it opens, so a worker that was not connected then leaves none of its tools here. Absent for a session kept before sources were recorded."`
}

func (*OfferedTool) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One tool as the model sees it."
	return schema
}

func (s *Server) registerSessionTools(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listSessionTools",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/tools",
		Summary:     "List the tools a session's model is offered",
		Description: "The tools the conversation model is offered on every request, with the description " +
			"and schema each is sent with: the agent's own, its connectors' and the built-in ones this " +
			"session can carry out, each with where it came from. A worker's tools are here only if a " +
			"worker hosting them was connected when the session opened, so a session missing them was " +
			"opened while none was. A session the router no longer holds answers what it was offered " +
			"when it last opened; one that ended before that was kept, or kept nothing, is a 404.",
		Responses: map[string]*huma.Response{"200": {Description: "The tools"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.listSessionTools)
}

// listSessionTools answers what a session's conversation model is offered: from the session
// while the router holds it, and from what was recorded when it opened once it does not.
func (s *Server) listSessionTools(ctx context.Context, request *listSessionToolsRequest) (*listSessionToolsResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	var tools []llm.Tool
	sources := map[string]string{}
	if found.Live != nil {
		tools = found.Live.ToolDefinitions()
		for _, tool := range tools {
			sources[tool.Name] = string(found.Live.ToolSource(tool.Name))
		}
	} else {
		stored, err := s.store.SessionTools(ctx, request.Id)
		if errors.Is(err, store.ErrNoSessionTools) {
			return nil, errNoSessionTools
		}
		if err != nil {
			return nil, err
		}
		for _, tool := range stored {
			tools = append(tools, llm.Tool{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters})
			sources[tool.Name] = tool.Source
		}
	}

	offered := OfferedTools{Tools: []OfferedTool{}}
	for _, tool := range tools {
		tokens := llm.ToolTokens(tool)
		offered.Tools = append(offered.Tools, OfferedTool{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters, Tokens: tokens,
			Source: sources[tool.Name],
		})
		offered.Tokens += tokens
	}
	return &listSessionToolsResponse{Body: offered}, nil
}
