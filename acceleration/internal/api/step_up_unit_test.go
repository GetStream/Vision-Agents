package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SteppedUpSuite is what a step-up's authorize request asks for (steppedUp, AI-854).
type SteppedUpSuite struct {
	suite.Suite
	connection store.ConnectorConnection
}

func TestSteppedUpSuite(t *testing.T) {
	suite.Run(t, new(SteppedUpSuite))
}

func (s *SteppedUpSuite) SetupTest() {
	s.connection = store.ConnectorConnection{GrantedScopes: []string{"files:read", "profile"}}
}

func (s *SteppedUpSuite) TestAUnionManifestAsksForItsListTheGrantAndTheMissingScopes() {
	manifest := core.ResolvedManifest{Scopes: core.ScopePolicy{List: []string{"files:read"}, StepUpUnion: true}}

	asked := steppedUp(manifest, s.connection, core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"files:write"}})

	s.Equal([]string{"files:read", "profile", "files:write"}, asked.Scopes.List)
}

func (s *SteppedUpSuite) TestWithoutUnionOnlyTheScopesTheProviderAskedForAreAsked() {
	manifest := core.ResolvedManifest{Scopes: core.ScopePolicy{List: []string{"files:read"}}}

	asked := steppedUp(manifest, s.connection, core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"files:write"}})

	s.Equal([]string{"files:write"}, asked.Scopes.List)
}

func (s *SteppedUpSuite) TestAClaimsChallengeIsTheClaimsParameterAndLeavesTheManifestsOwnAlone() {
	params := map[string]string{"access_type": "offline"}
	manifest := core.ResolvedManifest{Scopes: core.ScopePolicy{List: []string{"files:read"}}, AuthorizeParams: params}

	asked := steppedUp(manifest, s.connection, core.Outcome{Kind: core.OutcomeScopeRequired, Claims: `{"access_token":{}}`})

	s.Equal(map[string]string{"access_type": "offline", "claims": `{"access_token":{}}`}, asked.AuthorizeParams)
	s.Equal(map[string]string{"access_type": "offline"}, params, "the manifest's map is not written")
	s.Equal([]string{"files:read"}, asked.Scopes.List, "no scope asked, so the manifest's own")
}
