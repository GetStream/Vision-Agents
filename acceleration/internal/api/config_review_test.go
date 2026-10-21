//go:build integration

package api

import (
	"net/http"
)

func (s *ConfigsSuite) TestAConfigsReviewInstructionsAreStoredAndReadBack() {
	created := s.createConfig(map[string]any{"name": "support",
		"review": map[string]any{"instructions": "Score how well the agent sold the upgrade."}})

	s.Require().NotNil(created.Review)
	s.Equal("Score how well the agent sold the upgrade.", value(created.Review.Instructions))
	s.Equal("Score how well the agent sold the upgrade.", s.reviewInstructions(created.Id))
}

func (s *ConfigsSuite) TestAConfigWithoutReviewInstructionsHasNoReview() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.Nil(created.Review)
}

// A client that does not know the field must not take a config's review guide away by saving.
func (s *ConfigsSuite) TestAnUpdateLeavingTheReviewOutKeepsIt() {
	created := s.createConfig(map[string]any{"name": "support",
		"review": map[string]any{"instructions": "Score the upsell."}})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+created.Id, map[string]any{"name": "support"}, nil))

	s.Equal("Score the upsell.", s.reviewInstructions(created.Id))
}

func (s *ConfigsSuite) TestAnUpdateWithEmptyReviewInstructionsRemovesThem() {
	created := s.createConfig(map[string]any{"name": "support",
		"review": map[string]any{"instructions": "Score the upsell."}})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+created.Id, map[string]any{"name": "support", "review": map[string]any{"instructions": ""}}, nil))

	s.Empty(s.reviewInstructions(created.Id))
}

func (s *ConfigsSuite) TestAPatchSetsKeepsAndRemovesTheReviewInstructions() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.patch(created.Id, map[string]any{"review": map[string]any{"instructions": "Score the upsell."}})
	s.Equal("Score the upsell.", s.reviewInstructions(created.Id), "a patch sets them")

	s.patch(created.Id, map[string]any{"instructions": "Be brief."})
	s.Equal("Score the upsell.", s.reviewInstructions(created.Id), "a patch leaving them out keeps them")

	s.patch(created.Id, map[string]any{"review": map[string]any{}})
	s.Equal("Score the upsell.", s.reviewInstructions(created.Id), "a review naming no instructions keeps them")

	s.patch(created.Id, map[string]any{"review": map[string]any{"instructions": ""}})
	s.Empty(s.reviewInstructions(created.Id), "an empty string removes them")
}

// agent.yaml has no review guide, so a sync keeps the one the dashboard set.
func (s *SyncSuite) TestASyncKeepsTheReviewInstructionsTheDashboardSet() {
	synced := s.sync(map[string]any{"name": "support", "hash": "v1"})
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+synced.Config.Id,
		map[string]any{"review": map[string]any{"instructions": "Score the upsell."}}, nil))

	s.sync(map[string]any{"name": "support", "hash": "v2", "instructions": "Be brief."})

	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &stored))
	s.Require().NotNil(stored.Review)
	s.Equal("Score the upsell.", value(stored.Review.Instructions))
}

func (s *ConfigsSuite) patch(id string, body map[string]any) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+id, body, nil))
}

// reviewInstructions is the review guide the config is read back with, empty when it has none.
func (s *ConfigsSuite) reviewInstructions(id string) string {
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &stored))
	if stored.Review == nil {
		return ""
	}
	return value(stored.Review.Instructions)
}
