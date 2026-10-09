//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"testing"
)

// NamesSuite holds what a create or a rename to a name that is already taken is answered
// with, for each resource a customer names.
type NamesSuite struct {
	RouterSuite
}

func TestNamesSuite(t *testing.T) {
	runSuite(t, new(NamesSuite))
}

// SetupTest gives every test an app of its own, so the names it takes are its own.
func (s *NamesSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// refused calls the router and returns the failure it answered with.
func (s *NamesSuite) refused(method, path string, body any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(method, path, body)
	var answer ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answer), string(payload))
	return status, answer.Error
}

// taken checks a failure is the 409 for a name already taken, saying so in its own words.
func (s *NamesSuite) taken(status int, failure ErrorDetail, message string) {
	s.Equal(http.StatusConflict, status)
	s.Equal(ErrorTypeConflict, failure.Type)
	s.Equal(codeNameTaken, failure.Code)
	s.Equal(message, failure.Message)
}

func (s *NamesSuite) agent(name string) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": name}, &created))
	return created
}

func (s *NamesSuite) TestAnAgentCannotBeCreatedUnderANameAnotherHas() {
	s.agent("support")

	status, failure := s.refused(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "support"})

	s.taken(status, failure, "an agent with this name already exists")
}

func (s *NamesSuite) TestAnAgentCannotBeRenamedToANameAnotherHas() {
	s.agent("support")
	sales := s.agent("sales")

	status, failure := s.refused(http.MethodPut, "/v1/agents/configs/"+sales.Id,
		map[string]any{"name": "support"})
	s.taken(status, failure, "an agent with this name already exists")

	status, failure = s.refused(http.MethodPatch, "/v1/agents/configs/"+sales.Id,
		map[string]any{"name": "support"})
	s.taken(status, failure, "an agent with this name already exists")
}

func (s *NamesSuite) TestADeletedAgentsNameIsFreeAgain() {
	gone := s.agent("support")
	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+gone.Id, nil, nil))

	s.agent("support")
}

func (s *NamesSuite) skill(configID, name string) (int, ErrorDetail) {
	return s.refused(http.MethodPost, "/v1/agents/skills", map[string]any{
		"config_id": configID, "name": name, "description": "work out what a caller is owed",
		"instructions": "Read the order and the policy, then say what to refund.",
	})
}

func (s *NamesSuite) TestASkillNameIsTakenOnlyOnItsOwnAgent() {
	support, sales := s.agent("support"), s.agent("sales")
	status, _ := s.skill(support.Id, "refund")
	s.Require().Equal(http.StatusCreated, status)

	status, failure := s.skill(support.Id, "refund")
	s.taken(status, failure, "this agent already has a skill with this name")

	status, _ = s.skill(sales.Id, "refund")
	s.Equal(http.StatusCreated, status, "another agent may have a skill of the same name")
}

func (s *NamesSuite) TestARouterCannotTakeANameAnotherHas() {
	var cheap RouterConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "fast"}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "cheap"}, &cheap))

	status, failure := s.refused(http.MethodPost, "/v1/router/configs", map[string]any{"name": "fast"})
	s.taken(status, failure, "a router with this name already exists")

	status, failure = s.refused(http.MethodPut, "/v1/router/configs/"+cheap.Id, map[string]any{"name": "fast"})
	s.taken(status, failure, "a router with this name already exists")
}

func (s *NamesSuite) TestAVoiceCannotTakeANameAnotherHas() {
	var narrator Voice
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/voices",
		map[string]any{"name": "founder"}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/voices",
		map[string]any{"name": "narrator"}, &narrator))

	status, failure := s.refused(http.MethodPost, "/v1/agents/voices", map[string]any{"name": "founder"})
	s.taken(status, failure, "a voice with this name already exists")

	status, failure = s.refused(http.MethodPut, "/v1/agents/voices/"+narrator.Id,
		map[string]any{"name": "founder"})
	s.taken(status, failure, "a voice with this name already exists")
}

func (s *NamesSuite) TestEveryNameIndexTheStoreKnowsIsOneTheMigrationsMake() {
	// The store tells a name taken by the unique index it violated, so an index renamed in
	// a migration would quietly turn a 409 back into a 500.
	for _, index := range []string{
		"agent_configs_name_idx", "skills_name_idx", "router_configs_name_idx", "voices_name_idx",
	} {
		var found bool
		s.Require().NoError(s.store.DB().QueryRowContext(s.T().Context(),
			"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = ?)", index).Scan(&found))
		s.True(found, index)
	}
}
