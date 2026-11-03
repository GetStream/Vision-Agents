//go:build integration

package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorInstructionsSuite is what the servers of a session's custom connectors say about
// their tools, for the agent's instructions. A built-in's are left out (internal/api's
// SessionConnectorsSuite).
type ConnectorInstructionsSuite struct {
	connectorFixture
}

func TestConnectorInstructionsSuite(t *testing.T) {
	suite.Run(t, new(ConnectorInstructionsSuite))
}

func (s *ConnectorInstructionsSuite) TestWhatACustomConnectorsServerSaysFollowsTheAgentsInstructions() {
	s.provider.instruct("primary", "Quote ticket ids whole.")

	s.Equal("Be brief.\n\nThe crm tools (crm__*) come with these instructions from their server:\n\nQuote ticket ids whole.",
		s.told(s.fixed("crm", s.connection("", "primary"), "whoami")))
}

func (s *ConnectorInstructionsSuite) TestEachBindingSaysWhatItsOwnServerSaidInTheConfigsOrder() {
	s.provider.instruct("primary", "Quote ticket ids whole.")
	s.provider.instruct("secondary", "Dates are UTC.")

	s.Equal("Be brief.\n\n"+
		"The crm tools (crm__*) come with these instructions from their server:\n\nQuote ticket ids whole.\n\n"+
		"The archive tools (archive__*) come with these instructions from their server:\n\nDates are UTC.",
		s.told(s.fixed("crm", s.connection("", "primary"), "whoami"), s.fixed("archive", s.connection("", "secondary"), "whoami")))
}

func (s *ConnectorInstructionsSuite) TestAServerThatSaysNothingAddsNothing() {
	s.Equal("Be brief.", s.told(s.fixed("crm", s.connection("", "primary"), "whoami")))
}

func (s *ConnectorInstructionsSuite) TestAServerThatSaysOnlyWhitespaceAddsNothing() {
	s.provider.instruct("primary", "  \n\t ")

	s.Equal("Be brief.", s.told(s.fixed("crm", s.connection("", "primary"), "whoami")))
}

func (s *ConnectorInstructionsSuite) TestWhatAServerSaysIsAddedWithoutItsPadding() {
	s.provider.instruct("primary", "\n  Quote ticket ids whole. \t\n")

	s.Equal("Be brief.\n\nThe crm tools (crm__*) come with these instructions from their server:\n\nQuote ticket ids whole.",
		s.told(s.fixed("crm", s.connection("", "primary"), "whoami")))
}

// A server none of whose tools the model is offered has nothing to advise on.
func (s *ConnectorInstructionsSuite) TestAServerWhoseToolsAreAllLeftOutAddsNothing() {
	s.provider.instruct("primary", "Quote ticket ids whole.")
	s.provider.redescribe("primary", "Says which account this is, now in full.")

	s.Equal("Be brief.", s.told(s.fixed("crm", s.connection("", "primary"), "whoami")))
}

// told is what the agent of a session of a config with bindings, told "Be brief.", is told.
func (s *ConnectorInstructionsSuite) told(bindings ...store.ConnectorBinding) string {
	d, _, _, err := s.attach(s.spec(s.config(bindings...), "", nil))
	s.Require().NoError(err)
	spec := Spec{Instructions: "Be brief.", ServerInstructions: serverInstructions(nil, nil, d.said())}
	return spec.prompt()
}
