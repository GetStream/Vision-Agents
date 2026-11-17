//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A tool a caller declares with an approval reaches the session's spec, which is what makes
// its calls wait for a person; one with no question asks nothing.
func TestADeclaredToolsApprovalReachesTheSpec(t *testing.T) {
	client := SessionToolExecutorClient
	message, reason := "Only your city is shared.", "purpose"
	tools := []SessionTool{
		{Name: "athena_device_location", Description: "Where they are", Executor: &client,
			Approval: &SessionToolApproval{Title: "Share your location?", Message: &message, ReasonArgument: &reason}},
		{Name: "athena_read_canvas", Description: "Read a canvas", Approval: &SessionToolApproval{}},
	}
	spec := specOf(CreateSessionRequest{Tools: &tools}, "acme", nil)
	require.Len(t, spec.Tools, 2)
	require.True(t, spec.Tools[0].Client)
	require.Equal(t, &harness.ToolApproval{Title: "Share your location?", Message: "Only your city is shared.", ReasonArgument: "purpose"}, spec.Tools[0].Approval)
	require.Nil(t, spec.Tools[1].Approval)
}

type SessionToolsSuite struct {
	RouterSuite
}

func TestSessionToolsSuite(t *testing.T) {
	runSuite(t, new(SessionToolsSuite))
}

// SetupSuite has sessions offer what the suite's workers host, as cmd/router does.
func (s *SessionToolsSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.manager.HostTools(s.dispatch)
}

func (s *SessionToolsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionToolsSuite) TestADeclaredToolsPoliciesReachTheSpec() {
	interruptible, timeout := true, 5000
	tools := []SessionTool{
		{Name: "search_flights", Description: "Search for flights", Interruptible: &interruptible, TimeoutMs: &timeout},
		{Name: "book_flight", Description: "Book a flight"},
	}

	spec := specOf(CreateSessionRequest{Tools: &tools}, "acme", nil)

	s.Require().Len(spec.Tools, 2)
	s.True(spec.Tools[0].Interruptible)
	s.Equal(5*time.Second, spec.Tools[0].Timeout)
	s.False(spec.Tools[1].Interruptible, "a tool outlives an interruption unless it says otherwise")
	s.Zero(spec.Tools[1].Timeout, "a tool naming no timeout takes the session's")
}

func (s *SessionToolsSuite) TestANegativeToolTimeoutIsRefused() {
	request := textSession(nil)
	timeout := -1
	request.Tools = &[]SessionTool{{Name: "search_flights", Description: "Search for flights", TimeoutMs: &timeout}}

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPost, "/v1/agents/sessions", request, nil))
}

func (s *SessionToolsSuite) TestASessionSaysWhatItsModelIsOffered() {
	request := textSession(nil)
	request.Tools = &[]SessionTool{{
		Name: "lookup_order", Description: "find an order by its number",
		Parameters: &map[string]any{"type": "object", "properties": map[string]any{"number": map[string]any{"type": "string"}}},
	}}
	opened := s.serverClient.createSession(request)

	var offered OfferedTools
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+opened.Id+"/tools", nil, &offered))
	s.Require().NotEmpty(offered.Tools)
	s.Equal("lookup_order", offered.Tools[0].Name)
	s.Equal("find an order by its number", offered.Tools[0].Description)
	s.Equal("object", offered.Tools[0].Parameters["type"])
	var total int64
	for _, tool := range offered.Tools {
		s.Positive(tool.Tokens, tool.Name)
		total += tool.Tokens
	}
	s.Equal(total, offered.Tokens)
}

func (s *SessionToolsSuite) TestASessionSaysWhereEachToolItIsOfferedCameFrom() {
	agent := "agent-" + s.utils.uuid()
	s.await(s.serverClient.opens("/v1/dispatch"), "ready")
	s.Require().NoError(s.dispatch.Host(s.dispatch.Workers(s.customerID())[0], agent,
		[]dispatch.Tool{{Name: "investigate_sdk", Description: "Read SDK source"}}, time.Minute))

	request := textSession(nil)
	request.AgentId = &agent
	request.Tools = &[]SessionTool{{Name: "lookup_order", Description: "find an order by its number"}}
	opened := s.serverClient.createSession(request)

	sources := s.toolSources(opened.Id)
	s.Equal("session", sources["lookup_order"])
	s.Equal("worker", sources["investigate_sdk"])
	for name, source := range sources {
		if name != "lookup_order" && name != "investigate_sdk" {
			s.Equal("builtin", source, name)
		}
	}
}

func (s *SessionToolsSuite) TestASessionOpenedWithNoWorkerHostingItsAgentHasNoWorkerTools() {
	agent := "agent-" + s.utils.uuid()
	request := textSession(nil)
	request.AgentId = &agent
	opened := s.serverClient.createSession(request)

	for name, source := range s.toolSources(opened.Id) {
		s.NotEqual("worker", source, name)
	}
}

func (s *SessionToolsSuite) TestASessionTheRouterLetGoOfSaysWhereItsToolsCameFrom() {
	id := s.utils.uuid()
	s.Require().NoError(s.store.SaveSession(context.Background(), &store.AgentSession{
		ID: id, CustomerID: s.customerID(), AgentName: "support", AgentID: "support", State: store.SessionClosed,
	}))
	s.Require().NoError(s.store.SaveSessionTools(context.Background(), id, []store.ToolDefinition{
		{Name: "investigate_sdk", Description: "Read SDK source", Source: "worker"},
		{Name: "lookup_order", Description: "find an order by its number"},
	}))

	sources := s.toolSources(id)
	s.Equal("worker", sources["investigate_sdk"])
	s.Empty(sources["lookup_order"], "a tool kept before sources were recorded names none")
}

// toolSources is where each tool a session is offered came from, by name.
func (s *SessionToolsSuite) toolSources(sessionID string) map[string]string {
	var offered OfferedTools
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+sessionID+"/tools", nil, &offered))
	sources := map[string]string{}
	for _, tool := range offered.Tools {
		sources[tool.Name] = tool.Source
	}
	return sources
}

func (s *SessionToolsSuite) TestASessionTheRouterLetGoOfListsWhatItWasOffered() {
	id := s.utils.uuid()
	s.Require().NoError(s.store.SaveSession(context.Background(), &store.AgentSession{
		ID: id, CustomerID: s.customerID(), AgentName: "support", AgentID: "support", State: store.SessionClosed,
	}))
	s.Require().NoError(s.store.SaveSessionTools(context.Background(), id, []store.ToolDefinition{
		{Name: "lookup_order", Description: "find an order by its number", Parameters: map[string]any{"type": "object"}},
	}))

	var offered OfferedTools
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+id+"/tools", nil, &offered))
	s.Require().Len(offered.Tools, 1)
	s.Equal("lookup_order", offered.Tools[0].Name)
	s.Equal("object", offered.Tools[0].Parameters["type"])
	s.Equal(offered.Tools[0].Tokens, offered.Tokens)
}

func (s *SessionToolsSuite) TestASessionTheRouterDoesNotHoldHasNoToolsToList() {
	s.Equal(http.StatusNotFound,
		s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+s.utils.uuid()+"/tools", nil, nil))
}
