//go:build integration

package api

import (
	"net/http"
	"testing"
)

type SyncSuite struct {
	RouterSuite
}

func TestSyncSuite(t *testing.T) {
	runSuite(t, new(SyncSuite))
}

// SetupTest gives every test an app of its own, because a sync is named by what the agent
// is called and applying one again is what it means to change it.
func (s *SyncSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SyncSuite) TestSyncingAnAgentStoresItsInstructionsAndSkills() {
	declaration := map[string]any{
		"name": "support", "hash": "v1", "instructions": "Be brief.",
		"skills": []map[string]string{
			{"config_id": "", "name": "refund", "description": "work out a refund", "instructions": "Read the policy."},
		},
	}
	first := s.sync(declaration)
	s.False(first.Unchanged)
	s.Equal("support", first.Config.Name)
	s.Equal("Be brief.", value(first.Config.Instructions))
	s.Equal([]string{"refund"}, value(first.Config.Skills))
	s.Equal("v1", value(first.Config.SyncHash))

	second := s.sync(declaration)
	s.True(second.Unchanged, "the same hash means nothing was written")
	s.Equal(first.Config.Id, second.Config.Id)

	third := s.sync(map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be even briefer.",
	})
	s.False(third.Unchanged)
	s.Equal(first.Config.Id, third.Config.Id)
	s.Equal("Be even briefer.", value(third.Config.Instructions))
}

func (s *SyncSuite) TestSyncingAnAgentStoresWhatItsDeclarationRunsItOn() {
	result := s.sync(map[string]any{
		"name": "analyst", "hash": "v1", "mode": "text",
		"llm": "llm-flow", "subagent": "llm-flow", "tts": "en-low-latency", "voice": "aurora",
		"greeting": "Hello.", "keyterms": []string{"Vision Agents"}, "sandbox": "daytona",
		"tags": map[string]string{"project": "analyst"},
	})

	s.Equal(AgentModeText, result.Config.Mode)
	s.Equal("llm-flow", value(result.Config.Llm))
	s.Equal("llm-flow", value(result.Config.Subagent))
	s.Equal("aurora", value(result.Config.Voice))
	s.Equal("Hello.", value(result.Config.Greeting))
	s.Equal([]string{"Vision Agents"}, value(result.Config.Keyterms))
	s.Equal(Daytona, value(result.Config.Sandbox))
	s.Equal("analyst", value(result.Config.Tags)["project"])
}

func (s *SyncSuite) TestASyncThatNamesNoModelLeavesTheOneStored() {
	first := s.sync(map[string]any{
		"name": "switchboard", "hash": "v1", "llm": "llm-flow", "sandbox": "daytona",
	})

	// A directory that says nothing about a model should not blank the one the dashboard
	// chose, so only what the declaration named is written.
	second := s.sync(map[string]any{
		"name": "switchboard", "hash": "v2", "instructions": "Be brief.",
	})

	s.Equal(first.Config.Id, second.Config.Id)
	s.Equal("llm-flow", value(second.Config.Llm))
	s.Equal(Daytona, value(second.Config.Sandbox))
}

func (s *SyncSuite) TestASyncNamingASandboxNobodyRunsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "analyst", "hash": "v1", "sandbox": "docker"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "docker")
}

func (s *SyncSuite) TestASyncNamingAModeNobodyRunsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "analyst", "hash": "v1", "mode": "txt"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "text")
}

func (s *SyncSuite) TestASyncedDirectoryListsItsFilesAndForgetsTheOnesTakenOut() {
	synced := s.sync(map[string]any{
		"name": "librarian", "hash": "v1",
		"knowledge": []map[string]string{
			{"source": "pricing.md", "text": "# Pricing\n\nA penny."},
			{"source": "refunds.md", "text": "# Refunds\n\nThirty days."},
		},
	})
	s.Require().NotNil(synced.Config.KnowledgeNamespace)
	s.Len(s.documents(*synced.Config.KnowledgeNamespace), 2)

	s.sync(map[string]any{
		"name": "librarian", "hash": "v2",
		"knowledge": []map[string]string{{"source": "pricing.md", "text": "# Pricing\n\nA penny."}},
	})
	listed := s.documents("librarian")
	s.Require().Len(listed, 1)
	s.Equal("pricing.md", listed[0].Source)

	s.sync(map[string]any{"name": "librarian", "hash": "v3"})
	s.Empty(s.documents("librarian"), "a directory with no knowledge left holds none")
}

func (s *SyncSuite) TestASyncedDirectorysPagesAreReadIntoItsKnowledge() {
	synced := s.sync(map[string]any{
		"name": "librarian", "hash": "v1",
		"knowledge_urls": []map[string]string{
			{"url": "https://example.com/pricing", "title": "What a call costs"},
		},
	})
	s.Require().NotNil(synced.Config.KnowledgeNamespace, "pages alone are still a knowledge base")
	s.Equal("librarian", *synced.Config.KnowledgeNamespace)

	var listed []KnowledgeUrl
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/urls?namespace=librarian", nil, &listed))
	s.Require().Len(listed, 1)
	s.Equal("https://example.com/pricing", listed[0].Url)
	s.Equal("What a call costs", value(listed[0].Title))
}

func (s *SyncSuite) TestASyncedDirectorysSimulationsAreFoundByNameAndForgottenWhenTakenOut() {
	first := s.sync(map[string]any{
		"name": "deli", "hash": "v1",
		"simulations": []map[string]any{
			{"name": "change of order", "scenario": "Swap the club for a wrap.", "assertion": "One wrap.", "variations": 3},
			{"name": "off the menu", "scenario": "Ask for a milkshake.", "assertion": "No milkshake."},
		},
	})
	stored := s.simulations(first.Config.Id)
	s.Require().Len(stored, 2)
	changed := stored["change of order"]
	s.Equal(3, changed.Variations)
	s.Equal(SimulationModeText, changed.Mode)

	s.sync(map[string]any{
		"name": "deli", "hash": "v2",
		"simulations": []map[string]any{
			{"name": "change of order", "scenario": "Swap the club for a wrap.", "assertion": "One wrap, no fries.", "mode": "audio"},
		},
	})
	stored = s.simulations(first.Config.Id)
	s.Require().Len(stored, 1, "a simulation no longer declared is deleted")
	s.Equal(changed.Id, stored["change of order"].Id, "found by name, so its runs stay attached")
	s.Equal("One wrap, no fries.", stored["change of order"].Assertion)
	s.Equal(SimulationModeAudio, stored["change of order"].Mode)

	s.sync(map[string]any{"name": "deli", "hash": "v3"})
	s.Len(s.simulations(first.Config.Id), 1, "a directory with no simulations/ leaves them alone")

	s.sync(map[string]any{"name": "deli", "hash": "v4", "simulations": []map[string]any{}})
	s.Empty(s.simulations(first.Config.Id), "an empty simulations/ holds none")
}

func (s *SyncSuite) TestASyncDeclaringTwoSimulationsWithOneNameIsRefused() {
	simulation := map[string]any{"name": "order", "scenario": "Order lunch.", "assertion": "Ordered."}
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "deli", "hash": "v1", "simulations": []map[string]any{simulation, simulation},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "order")
	s.Empty(s.configsNamed("deli"), "nothing is written when a simulation is refused")
}

func (s *SyncSuite) TestASyncDeclaringASimulationWithTooManyVariationsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "deli", "hash": "v1",
		"simulations": []map[string]any{
			{"name": "order", "scenario": "Order lunch.", "assertion": "Ordered.", "variations": 11},
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "variations")
	s.Empty(s.configsNamed("deli"))
}

func (s *SyncSuite) TestOnlyTheAppsOwnBackendMaySyncAnAgent() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sync",
			map[string]any{"name": "agent-" + s.utils.uuid(), "hash": "v1"}, nil)
	})
}

// sync applies a directory's declaration.
func (s *SyncSuite) sync(declaration map[string]any) SyncAgentResult {
	var result SyncAgentResult
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/agents/sync", declaration, &result))
	return result
}

// simulations are the app's simulations of one config, by name.
func (s *SyncSuite) simulations(configID string) map[string]Simulation {
	var listed []Simulation
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/simulations", nil, &listed))
	named := map[string]Simulation{}
	for _, simulation := range listed {
		if simulation.ConfigId == configID {
			named[simulation.Name] = simulation
		}
	}
	return named
}

// configsNamed lists the app's configs called name.
func (s *SyncSuite) configsNamed(name string) []AgentConfig {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs?name="+name, nil, &listed))
	return listed
}

// documents lists what one of the app's knowledge bases holds.
func (s *SyncSuite) documents(namespace string) []IndexedKnowledgeDocument {
	var listed []IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents?namespace="+namespace, nil, &listed))
	return listed
}
