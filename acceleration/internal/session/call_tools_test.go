package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
