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
			{"name": "refund", "description": "work out a refund", "instructions": "Read the policy."},
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

// documents lists what one of the app's knowledge bases holds.
func (s *SyncSuite) documents(namespace string) []IndexedKnowledgeDocument {
	var listed []IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents?namespace="+namespace, nil, &listed))
	return listed
}
