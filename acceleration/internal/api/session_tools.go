package api

import (
	"context"
	"errors"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

var errNoSessionTools = notFound("the session recorded no tools: it ended before they were kept, or kept nothing")

// errOthersTools refuses a change to a session's tools by a caller of another kind than the one
// that opened it. Its tools are run on that side (session.Seat), so a backend's would go
// unanswered by a device, and a device's by the backend.
var errOthersTools = forbidden("a session's tools are declared by the kind of caller that opened it: " +
	"a device's session by the device, and a backend's by a backend")

var errNoCallerTool = notFound("the session's caller has no tool by that name")

// errToolsFixed refuses a change to the tools of a session on a speech-to-speech model that
// takes them only when it opens, such as Gemini Live.
var errToolsFixed = invalidRequest("the session's speech-to-speech model takes its tools only when it " +
	"opens, so they cannot change during the call")

type addSessionToolsRequest struct {
	ID   string `path:"id" doc:"The session, as returned when it was created."`
	Body AddSessionToolsRequest
}

// AddSessionToolsRequest is tools to give a running session's caller.
type AddSessionToolsRequest struct {
	Tools []SessionTool `json:"tools" minItems:"1" doc:"The tools, each declared as it is when a session is created. One named as a tool the caller already has replaces it."`
}

func (*AddSessionToolsRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Tools to give a running session's caller."
	return schema
}

type removeSessionToolRequest struct {
	ID   string `path:"id" doc:"The session, as returned when it was created."`
	Name string `path:"name" doc:"The tool's name, as the caller declared it."`
}

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
	huma.Register(api, huma.Operation{
		OperationID:   "addSessionTools",
		Method:        http.MethodPost,
		Path:          "/v1/agents/sessions/{id}/tools",
		Summary:       "Give a running session more of the caller's tools",
		DefaultStatus: http.StatusNoContent,
		Description: "Declares tools on a running session as though they had been declared when it was " +
			"created, for tools that exist only partway through a conversation, such as the ones a card " +
			"shown in it brings. Whoever runs the caller's tools is asked to run them over the events " +
			"socket, and the model is offered them from its next request: a reply being written finishes " +
			"with the tools it started with. A tool named as one the caller already has replaces it.\n\n" +
			"The kind of caller that opened the session declares its tools: a device those of the session " +
			"it opened, and a backend those of a session a backend opened. Anyone else is refused with a " +
			"403. A name the session offers that is not the caller's (the router's own, a connector's, a " +
			"worker's or another agent's) is a 409. A session that is not running is a 404. A " +
			"speech-to-speech model that takes its tools only when it opens, such as Gemini Live, cannot " +
			"be given more during the call: 400.",
		Responses: map[string]*huma.Response{"204": {Description: "The model is offered them from its next request"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.addSessionTools)
	huma.Register(api, huma.Operation{
		OperationID:   "removeSessionTool",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/sessions/{id}/tools/{name}",
		Summary:       "Withdraw one of the caller's tools from a running session",
		DefaultStatus: http.StatusNoContent,
		Description: "The model is no longer offered the tool from its next request. A call it made " +
			"before still goes to the caller to answer. Only the caller's own tools are withdrawn, whether " +
			"declared when the session was created or added since, and by the kind of caller that opened " +
			"the session, as for adding: anyone else is refused with a 403. A name the caller has no tool " +
			"by, and a session that is not running, are a 404.",
		Responses: map[string]*huma.Response{"204": {Description: "The model is no longer offered it"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.removeSessionTool)
}

// addSessionTools gives a running session's caller more tools.
func (s *Server) addSessionTools(ctx context.Context, request *addSessionToolsRequest) (*struct{}, error) {
	live, failure := s.declaringSession(ctx, request.ID)
	if failure != nil {
		return nil, failure
	}
	tools := make([]harness.Tool, 0, len(request.Body.Tools))
	for _, tool := range request.Body.Tools {
		tools = append(tools, toolOf(tool))
	}
	return nil, toolsChanged(live.AddTools(ctx, tools))
}

// removeSessionTool withdraws one of a running session's caller's tools.
func (s *Server) removeSessionTool(ctx context.Context, request *removeSessionToolRequest) (*struct{}, error) {
	live, failure := s.declaringSession(ctx, request.ID)
	if failure != nil {
		return nil, failure
	}
	return nil, toolsChanged(live.RemoveTool(ctx, request.Name))
}

// declaringSession is a running session of the caller's whose tools it may change.
func (s *Server) declaringSession(ctx context.Context, id string) (*session.Session, error) {
	live, failure := s.session(ctx, id)
	if failure != nil {
		return nil, failure
	}
	if live.Spec().CallerKind != KindFrom(ctx) {
		return nil, errOthersTools
	}
	return live, nil
}

// toolsChanged answers a change to a session's tools that ended in err.
func toolsChanged(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, session.ErrToolTaken):
		return conflict(err.Error())
	case errors.Is(err, session.ErrNoCallerTool):
		return errNoCallerTool
	case errors.Is(err, sts.ErrToolsFixed):
		return errToolsFixed
	default:
		// What is left is the tools themselves, such as one with no description.
		return invalidRequest(err.Error())
	}
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
