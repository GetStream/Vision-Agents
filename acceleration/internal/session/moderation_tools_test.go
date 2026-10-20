package session

import (
	"context"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/moderator"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

func TestModerationTools(t *testing.T) {
	suite.Run(t, new(ModerationToolsSuite))
}

type ModerationToolsSuite struct {
	suite.Suite
	own    streamapp.Bound
	caller *callerTools
}

func (s *ModerationToolsSuite) SetupTest() {
	client, err := getstream.NewClient("acme-key", "acme-secret")
	s.Require().NoError(err)
	s.own = streamapp.Bound{Identity: streamapp.Identity{StreamApp: 77, Registered: true}, Client: client}
	s.caller = &callerTools{}
}

func (s *ModerationToolsSuite) names(tools []harness.Tool) []string {
	var names []string
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	return names
}

func (s *ModerationToolsSuite) TestTheModeratorIsOfferedTheModerationReads() {
	tools, _ := moderationTools(Spec{Role: store.RoleAgentModerator}, s.own, nil, s.caller)

	s.Equal(s.names(moderator.Tools()), s.names(tools))
}

func (s *ModerationToolsSuite) TestAnOrdinaryAgentIsOfferedNone() {
	tools, runner := moderationTools(Spec{}, s.own, nil, s.caller)

	s.Empty(tools)
	s.Same(s.caller, runner)
}

func (s *ModerationToolsSuite) TestTheDeploymentsSharedAppIsNeverRead() {
	shared := s.own
	shared.Identity.Registered = false

	tools, _ := moderationTools(Spec{Role: store.RoleAgentModerator}, shared, nil, s.caller)

	s.Empty(tools, "the shared app holds every other customer's moderation data")
}

func (s *ModerationToolsSuite) TestASessionWithNoStreamAppIsOfferedNone() {
	tools, _ := moderationTools(Spec{Role: store.RoleAgentModerator}, streamapp.Bound{}, nil, s.caller)

	s.Empty(tools)
}

func (s *ModerationToolsSuite) TestANameTheCallerDeclaredStaysTheCallers() {
	declared := []harness.Tool{{Name: moderator.ListBlocklistsTool, Description: "The caller's own"}}

	tools, runner := moderationTools(Spec{Role: store.RoleAgentModerator}, s.own, declared, s.caller)
	parts, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: moderator.ListBlocklistsTool})

	s.Require().NoError(err)
	s.Len(tools, len(moderator.Tools()))
	s.Equal("The caller's own", tools[0].Description)
	s.Equal("from the caller", parts[0].Text)
}

func (s *ModerationToolsSuite) TestAConfigsRoleReachesTheSpec() {
	spec := FromConfig(store.AgentConfig{CustomerID: "acme", Role: store.RoleAgentModerator})

	s.Equal(store.RoleAgentModerator, spec.Role)
}
