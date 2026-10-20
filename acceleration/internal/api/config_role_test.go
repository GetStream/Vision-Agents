//go:build integration

package api

import (
	"encoding/json"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A config holds no role unless it says so, and a save that leaves the role out keeps it, so
// a client that does not know the field cannot take the app's moderator away by saving.
func (s *ConfigsSuite) TestAConfigHoldsNoRoleUnlessItSaysSo() {
	ordinary := s.createConfig(map[string]any{"name": "support"})
	s.Nil(ordinary.Role)

	moderator := s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})
	s.Equal(AgentRoleAiModerator, value(moderator.Role))

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+moderator.Id, map[string]any{"name": "nora"}, &saved))
	s.Equal(AgentRoleAiModerator, value(saved.Role), "a role left out on an update stays")

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+moderator.Id, map[string]any{"instructions": "Be fair."}, &patched))
	s.Equal(AgentRoleAiModerator, value(patched.Role), "a role left out on a patch stays")
}

func (s *ConfigsSuite) TestARoleThereIsNoneOfIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "nora", "role": "janitor"})

	s.Equal(http.StatusBadRequest, status)
}

// One live config per app holds a role, whether the second asks for it on create, update or
// patch.
func (s *ConfigsSuite) TestASecondAgentInTheSameRoleIsRefused() {
	s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})
	other := s.createConfig(map[string]any{"name": "support"})

	for name, write := range map[string]func() (int, ErrorDetail){
		"create": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "nora-2", "role": "ai_moderator"})
		},
		"update": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPut, "/v1/agents/configs/"+other.Id, map[string]any{"name": "support", "role": "ai_moderator"})
		},
		"patch": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPatch, "/v1/agents/configs/"+other.Id, map[string]any{"role": "ai_moderator"})
		},
	} {
		status, failure := write()
		s.Equal(http.StatusConflict, status, name)
		s.Equal(codeRoleTaken, failure.Code, name)
	}
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+other.Id, nil, &stored))
	s.Nil(stored.Role, "the refused update and patch wrote nothing")
}

// A test copy carries the role of the config it copies, and the lookup still finds the live one.
func (s *ConfigsSuite) TestATestCopyMayShareItsConfigsRole() {
	live := s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})
	s.createConfig(map[string]any{"name": "nora-draft", "role": "ai_moderator",
		"tags": map[string]string{store.DraftOfTag: live.Id}})

	found := s.configsInRole("ai_moderator")

	s.Require().Len(found, 1)
	s.Equal(live.Id, found[0].Id)
}

func (s *ConfigsSuite) TestADeletedConfigFreesItsRole() {
	gone := s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+gone.Id, nil, nil))

	successor := s.createConfig(map[string]any{"name": "nora-2", "role": "ai_moderator"})

	found := s.configsInRole("ai_moderator")
	s.Require().Len(found, 1)
	s.Equal(successor.Id, found[0].Id)
}

func (s *ConfigsSuite) TestAnAppWithNoModeratorFindsNoneByRole() {
	s.createConfig(map[string]any{"name": "support"})

	s.Empty(s.configsInRole("ai_moderator"))
}

// Another app's moderator is not this one's.
func (s *ConfigsSuite) TestTheLookupByRoleIsScopedToTheApp() {
	s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})

	s.useApp(s.data.createApp())

	s.Empty(s.configsInRole("ai_moderator"))
	s.createConfig(map[string]any{"name": "nora", "role": "ai_moderator"})
}

func (s *SyncSuite) TestASyncGivesTheAgentItsRoleAndLeavingItOutKeepsIt() {
	synced := s.sync(map[string]any{"name": "nora", "hash": "v1", "role": "ai_moderator"})
	s.Equal(AgentRoleAiModerator, value(synced.Config.Role))

	resynced := s.sync(map[string]any{"name": "nora", "hash": "v2", "instructions": "Be fair."})
	s.Equal(AgentRoleAiModerator, value(resynced.Config.Role))
}

// refusal is the status and error a write the router refuses is answered with.
func (s *ConfigsSuite) refusal(method, path string, body any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(method, path, body)
	var answer ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answer), string(payload))
	return status, answer.Error
}

// configsInRole is what listing the app's configs by role answers.
func (s *ConfigsSuite) configsInRole(role string) []AgentConfig {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs?role="+role, nil, &listed))
	return listed
}
