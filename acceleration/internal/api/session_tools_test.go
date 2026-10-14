//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

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

func (s *SessionToolsSuite) SetupTest() {
	s.useApp(s.data.createApp())
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
