package session

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

func TestCallTools(t *testing.T) {
	suite.Run(t, new(CallToolsSuite))
}

type CallToolsSuite struct {
	suite.Suite
	builtin []harness.Tool
}

func (s *CallToolsSuite) SetupTest() {
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.builtin = tools.Tools
}

func (s *CallToolsSuite) offered(spec Spec) []string {
	var names []string
	for _, tool := range spec.callTools(s.builtin) {
		names = append(names, tool.Name)
	}
	return names
}

func (s *CallToolsSuite) TestAPhoneSessionOnAConfigWithoutCallsOffersTransferAndPress() {
	names := s.offered(FromConfig(store.AgentConfig{ID: "cfg-1"}))

	s.Contains(names, "transfer")
	s.Contains(names, "press")
}

func (s *CallToolsSuite) TestAPhoneSessionWithTransferOffIsNotOfferedTransfer() {
	config := store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{Transfer: &store.TransferSettings{Enabled: false}}}

	names := s.offered(FromConfig(config))

	s.NotContains(names, "transfer")
	s.Contains(names, "press")
}

func (s *CallToolsSuite) TestAPhoneSessionWithPressOffIsNotOfferedPress() {
	config := store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{Press: &store.PressSettings{Enabled: false}}}

	names := s.offered(FromConfig(config))

	s.NotContains(names, "press")
	s.Contains(names, "transfer")
}

func (s *CallToolsSuite) TestASessionWithoutAConfigKeepsTodaysTools() {
	spec := Spec{Calls: store.AgentCalls{
		Transfer: &store.TransferSettings{Enabled: false},
		Press:    &store.PressSettings{Enabled: false},
	}}

	names := s.offered(spec)

	s.Contains(names, "transfer")
	s.Contains(names, "press")
	s.Empty(spec.destinations())
}

func (s *CallToolsSuite) TestAPhoneSessionWithDestinationsOffersThemByName() {
	config := store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{Transfer: &store.TransferSettings{
		Enabled: true,
		Destinations: []store.TransferDestination{
			{Name: "billing", Number: "+15551230001", When: "a payment question", Warm: true},
			{Name: "support", Number: "+15551230002"},
		},
	}}}

	spec := FromConfig(config)

	s.Contains(s.offered(spec), "transfer")
	s.Equal([]agent.Destination{
		{Name: "billing", Number: "+15551230001", When: "a payment question", Warm: true},
		{Name: "support", Number: "+15551230002"},
	}, spec.destinations())
}

func (s *CallToolsSuite) TestASessionsOwnTransferIgnoresTheConfigsDestinations() {
	config := store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{Transfer: &store.TransferSettings{
		Enabled:      true,
		Destinations: []store.TransferDestination{{Name: "billing", Number: "+15551230001"}},
	}}}
	spec := FromConfig(config)
	spec.Tools = []harness.Tool{{Name: "transfer", Description: "the caller's own"}}

	s.Empty(spec.destinations())
}

func (s *CallToolsSuite) TestDestinationsOfATransferTurnedOffAreNotOffered() {
	config := store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{Transfer: &store.TransferSettings{
		Enabled:      false,
		Destinations: []store.TransferDestination{{Name: "billing", Number: "+15551230001"}},
	}}}

	s.Empty(FromConfig(config).destinations())
}

// endsCalls is a config's calls settings with end_call on.
func endsCalls() store.AgentCalls {
	return store.AgentCalls{EndCall: &store.EndCallSettings{Enabled: true}}
}

// hangingUp has sessions act in a Stream app served by the returned server, with a model
// that says goodbye and calls end_call on its first turn.
func (s *SessionSuite) hangingUp() *chattest.Server {
	server := chattest.NewServer(s.T())
	s.apps = streamapp.NewClients(streamapp.NewStatic(map[string]streamapp.Identity{
		"acme": {CustomerID: "acme", StreamApp: 1, APIKey: "acme-key", Secret: streamapp.NewSecret("acme-secret"), BaseURL: server.URL},
	}, streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret"})), streamapp.ClientsOptions{})
	s.manages()
	s.model.reply = "Goodbye."
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: "end_call", Arguments: "{}"}}
	return server
}

// hangsUp has the caller take their leave and waits for the agent to hang up on them.
func (s *SessionSuite) hangsUp(created *Session) {
	s.Contains(offered(created), "end_call")
	s.says(created, "that is all, thanks")
	s.eventually(func() bool {
		return created.State() == Ended && !s.manager.Running(created.ID())
	}, "the agent hung up and the session was not closed")
	s.Equal(EndAgentHangup, created.endReason())
}

// endedCalls is every call the server was asked to end for everybody.
func endedCalls(server *chattest.Server) []string {
	var ended []string
	for _, request := range server.Requests("acme-key") {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/mark_ended") {
			parts := strings.Split(request.Path, "/")
			ended = append(ended, parts[len(parts)-3]+":"+parts[len(parts)-2])
		}
	}
	return ended
}

func offered(created *Session) []string {
	var names []string
	for _, tool := range created.ToolDefinitions() {
		names = append(names, tool.Name)
	}
	return names
}

func (s *SessionSuite) TestAnAgentThatHangsUpEndsTheWebCallItCreated() {
	server := s.hangingUp()
	created := s.joins(Spec{ID: "session-1", CallID: "session-1", ConfigID: "cfg-1", Calls: endsCalls()})

	s.hangsUp(created)

	s.Equal([]string{"agent:session-1"}, endedCalls(server), "the call joining created was left running")
}

func (s *SessionSuite) TestAnAgentThatHangsUpOnANamedCallOnlyLeaves() {
	server := s.hangingUp()
	created := s.joins(Spec{ID: "session-1", CallID: "their-call", ConfigID: "cfg-1", Calls: endsCalls()})

	s.hangsUp(created)

	s.Empty(endedCalls(server), "a call the caller named was ended for everybody on it")
	s.True(s.edges[0].gone(), "the agent is still in the call")
}

func (s *SessionSuite) TestEndCallIsNotOfferedInWriting() {
	s.manages()
	created := s.writes(Spec{ConfigID: "cfg-1", Calls: endsCalls()})

	s.NotContains(offered(created), "end_call")

	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	s.Contains(offered(created), "end_call", "starting voice did not offer it")
}

func (s *SessionSuite) TestEndCallIsNotOfferedWithoutAConfig() {
	s.manages()
	created := s.joins(Spec{Calls: endsCalls()})

	s.NotContains(offered(created), "end_call")
}

func (s *SessionSuite) TestEndCallIsNotOfferedWhenTheConfigLeavesItOff() {
	s.manages()
	created := s.joins(Spec{ConfigID: "cfg-1"})

	s.NotContains(offered(created), "end_call")
}

func (s *SessionSuite) TestTheFirstEndReasonWins() {
	s.manages()
	created := s.joins(Spec{})
	created.noteEnd(EndTransferred)

	created.SetTimeout(0, TimeoutHangup)

	s.timedOut(created)
	s.Equal(EndTransferred, created.endReason())
}

func (s *SessionSuite) TestATimeoutKeepsItsReason() {
	s.manages()
	created := s.joins(Spec{})

	created.SetTimeout(0, TimeoutHangup)

	s.timedOut(created)
	s.Equal(TimeoutHangup, created.endReason())
}
