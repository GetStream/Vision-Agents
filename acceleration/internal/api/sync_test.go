//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

type SyncSuite struct {
	RouterSuite
}

func TestSyncSuite(t *testing.T) {
	runSuite(t, new(SyncSuite))
}

// SetupSuite seeds the built-in connectors as a router start does, so a directory can bind
// one. Seeding is idempotent, so suites running beside this one see the same rows.
func (s *SyncSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
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
		"llm": "llm-flow", "tts": "en-low-latency", "voice": "aurora",
		"greeting": map[string]any{"text": "Hello.", "mode": "variation"}, "keyterms": []string{"Vision Agents"}, "sandbox": "daytona",
		"tags": map[string]string{"project": "analyst"},
	})

	s.Equal(AgentModeText, result.Config.Mode)
	s.Equal("llm-flow", value(result.Config.Llm))
	s.Equal("aurora", value(result.Config.Voice))
	s.Require().NotNil(result.Config.Greeting)
	s.Equal("Hello.", result.Config.Greeting.Text)
	s.Equal(GreetingModeVariation, value(result.Config.Greeting.Mode))
	s.Equal([]string{"Vision Agents"}, value(result.Config.Keyterms))
	s.Equal(Daytona, value(result.Config.Sandbox))
	s.Equal("analyst", value(result.Config.Tags)["project"])
}

func (s *SyncSuite) TestSyncingAnAgentStoresWhatItLeavesToDispatch() {
	result := s.sync(map[string]any{
		"name": "stream-product", "hash": "v1", "mode": "text",
		"dispatch": map[string]string{"incoming_call": "enabled", "text": "enabled"},
	})

	s.Equal(Enabled, value(value(result.Config.Dispatch).IncomingCall))
	s.Equal(Enabled, value(value(result.Config.Dispatch).Text))
}

func (s *SyncSuite) TestSyncingAnAgentStoresWhetherItsToolsAreOfferedProgressively() {
	result := s.sync(map[string]any{"name": "concierge", "hash": "v1", "tools": map[string]any{"progressive": true}})
	s.True(value(result.Config.Tools.Progressive))

	again := s.sync(map[string]any{"name": "concierge", "hash": "v2"})
	s.True(value(again.Config.Tools.Progressive), "a directory that says nothing leaves what is stored")
}

func (s *SyncSuite) TestSyncingAnAgentStoresHowItsSandboxIsBuilt() {
	result := s.sync(map[string]any{
		"name": "artist", "hash": "v1", "sandbox": "daytona",
		"sandbox_options": map[string]any{"setup": []string{"pip install bpy==5.2.2"}, "timeout_ms": 300000},
	})
	s.Equal([]string{"pip install bpy==5.2.2"}, value(value(result.Config.SandboxOptions).Setup))

	// A directory that stops saying how is not one that wants the build thrown away.
	again := s.sync(map[string]any{"name": "artist", "hash": "v2", "sandbox": "daytona"})
	s.Equal(300000, value(value(again.Config.SandboxOptions).TimeoutMs))
}

func (s *SyncSuite) TestSyncingAnAgentStoresTheMCPServersItNamesByURL() {
	result := s.sync(map[string]any{
		"name": "concierge", "hash": "v1",
		"mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
	})

	s.Equal([]McpServer{{Name: "tablejourney", Url: "https://tablejourney.com/mcp"}}, value(result.Config.McpServers))
}

func (s *SyncSuite) TestSyncingAVoiceAgentStoresItsSubagent() {
	result := s.sync(map[string]any{
		"name": "support", "hash": "v1", "mode": "voice", "subagent": "llm-flow",
	})

	s.Equal("llm-flow", value(result.Config.Subagent))
}

func (s *SyncSuite) TestASyncGivingATextAgentASubagentIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "analyst", "hash": "v1", "mode": "text", "subagent": "llm-flow"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "subagent")
}

func (s *SyncSuite) TestASyncAskingForTooMuchMemoryIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "artist", "hash": "v1", "sandbox": "daytona",
			"sandbox_options": map[string]any{"memory_gb": 1024}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "memory_gb")
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
		"knowledge_urls": []map[string]any{
			{"url": "https://example.com/pricing", "title": "What a call costs", "refresh_hours": 24},
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
	s.Equal(24, value(listed[0].RefreshHours))
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

func (s *SyncSuite) TestASyncRecordsThePointTheDirectoryAndTheAgentAgreed() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})

	changes := s.changes(result.Config.Id)
	s.Empty(changes.Items, "nothing has been changed since the sync")
	s.Require().NotNil(changes.SyncedAt)
	s.False(changes.SyncedAt.IsZero())
}

func (s *SyncSuite) TestTheChangesSinceTheLastSyncAreWhatTheDashboardDid() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"instructions": "Be warm."})

	changes := s.changes(result.Config.Id)

	s.Require().Len(changes.Items, 1)
	s.Equal(AuditSource("dashboard"), changes.Items[0].Source)
	s.Equal("Ada Lovelace", changes.Items[0].ActorName)
	s.Require().NotNil(changes.LastChange)
	s.Equal(changes.Items[0].ID, *changes.LastChange)
}

func (s *SyncSuite) TestASyncAskedToCheckIsRefusedWhenTheSameSettingWasChangedSince() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"instructions": "Be warm."})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be brisk.", "check_changes": true,
	})

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "instructions")
	s.Equal("Be warm.", value(s.configsNamed("support")[0].Instructions),
		"a refused sync writes nothing")
}

func (s *SyncSuite) TestASyncAskedToCheckIsNotRefusedForASettingItsDirectoryDoesNotDeclare() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"llm": "llm-fast"})

	second := s.sync(map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be brisk.", "check_changes": true,
	})

	s.Equal("Be brisk.", value(second.Config.Instructions))
	s.Equal("llm-fast", value(second.Config.Llm),
		"a model the directory says nothing about was never at risk")
}

func (s *SyncSuite) TestASyncIsNotRefusedWhenItsDirectoryAlreadyHoldsTheChange() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"instructions": "Be warm."})

	// What a client does after writing the change into its own files: the directory now
	// says what the dashboard says, so there is nothing left to write over.
	second := s.sync(map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be warm.", "check_changes": true,
	})

	s.Equal("Be warm.", value(second.Config.Instructions))
}

func (s *SyncSuite) TestASyncGoesAheadOnceTheChangeItWouldWriteOverIsAcknowledged() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"instructions": "Be warm."})
	changes := s.changes(result.Config.Id)
	s.Require().NotNil(changes.LastChange)

	second := s.sync(map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be brisk.",
		"check_changes": true, "base_change": *changes.LastChange,
	})

	s.Equal("Be brisk.", value(second.Config.Instructions))
	s.Empty(s.changes(result.Config.Id).Items, "the sync is the newest thing on record again")
}

func (s *SyncSuite) TestASyncThatDoesNotAskToBeCheckedWritesOverTheChange() {
	result := s.sync(map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief."})
	s.patch(result.Config.Id, map[string]any{"instructions": "Be warm."})

	second := s.sync(map[string]any{
		"name": "support", "hash": "v2", "instructions": "Be brisk.",
	})

	s.Equal("Be brisk.", value(second.Config.Instructions),
		"a process syncing on startup is what the default is for")
}

func (s *SyncSuite) TestADocumentChangedSinceTheLastSyncRefusesASyncThatWouldRewriteIt() {
	s.sync(map[string]any{"name": "support", "hash": "v1",
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}})
	s.ingest("support", "faq.md", "Open at eight.")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v2", "check_changes": true,
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}},
	})

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "knowledge faq.md")
}

// The step that has to terminate: a client told what changed writes it into its own files
// and syncs again, and the sync that now says what the dashboard says goes through.
func (s *SyncSuite) TestASyncIsNotRefusedWhenItsDirectoryAlreadyHoldsTheChangedDocument() {
	s.sync(map[string]any{"name": "support", "hash": "v1",
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}})
	s.ingest("support", "faq.md", "Open at eight.")

	second := s.sync(map[string]any{
		"name": "support", "hash": "v2", "check_changes": true,
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at eight."}},
	})

	s.False(second.Unchanged)
	s.Empty(s.changes(second.Config.Id).Items, "the sync is the newest thing on record again")
}

func (s *SyncSuite) TestResyncingAnUnchangedDocumentRecordsNoChange() {
	first := s.sync(map[string]any{"name": "support", "hash": "v1",
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}})
	s.patch(first.Config.Id, map[string]any{"instructions": "Be warm."})

	// A second sync of the same documents under a directory that moved elsewhere.
	second := s.sync(map[string]any{"name": "support", "hash": "v2", "instructions": "Be warm.",
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}})

	for _, entry := range s.auditOf(second.Config.Id) {
		s.NotEqual(AuditResourceType("knowledge"), entry.ResourceType,
			"nobody rewrote the document, so nothing was changed about it")
	}
}

// Editing a config clears its sync_hash, so the next sync of the directory runs. Editing
// what hangs off it -- a document, a page, a skill -- does not, and an untouched directory
// then carries the hash the agent still has. The person at that directory is told anyway:
// theirs is the copy that has gone stale, and the hash is no longer evidence of agreement.
func (s *SyncSuite) TestASyncOfAnUnchangedDirectoryIsStillRefusedOverADocumentChangedSince() {
	declaration := map[string]any{"name": "support", "hash": "v1", "check_changes": true,
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}}
	s.sync(declaration)
	s.ingest("support", "faq.md", "Open at eight.")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", declaration)

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "knowledge faq.md")
}

func (s *SyncSuite) TestAnAcknowledgedSyncOfAnUnchangedDirectoryWritesItsDocumentBack() {
	declaration := map[string]any{"name": "support", "hash": "v1", "check_changes": true,
		"knowledge": []map[string]string{{"source": "faq.md", "text": "Open at nine."}}}
	result := s.sync(declaration)
	s.ingest("support", "faq.md", "Open at eight.")
	changes := s.changes(result.Config.Id)
	s.Require().NotNil(changes.LastChange)

	declaration["base_change"] = *changes.LastChange
	second := s.sync(declaration)

	s.False(second.Unchanged, "the directory had something left to write, untouched as it is")
	s.Equal("Open at nine.", s.documentText(result.Config.Id, "faq.md"))
}

func (s *SyncSuite) TestASyncOfAnUnchangedDirectoryNobodyElseTouchedWritesNothing() {
	declaration := map[string]any{
		"name": "support", "hash": "v1", "instructions": "Be brief.", "check_changes": true,
	}
	s.sync(declaration)

	s.True(s.sync(declaration).Unchanged)
}

func (s *SyncSuite) TestASkillChangedSinceTheLastSyncRefusesASyncThatWouldRewriteIt() {
	declaration := map[string]any{"name": "support", "hash": "v1", "skills": []map[string]string{
		{"config_id": "", "name": "refund", "description": "work out a refund",
			"instructions": "Read the policy."},
	}}
	result := s.sync(declaration)
	skill := s.skillNamed(result.Config.Id, "refund")
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPut, "/v1/agents/skills/"+skill.Id,
		map[string]any{"config_id": result.Config.Id, "name": "refund",
			"description": "work out a refund", "instructions": "Read the new policy."}, nil))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v2", "check_changes": true,
		"skills": []map[string]string{
			{"config_id": "", "name": "refund", "description": "work out a refund",
				"instructions": "Read the policy."},
		},
	})

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "skill refund")
}

// changes are the edits made to an agent since its directory was last synced.
func (s *SyncSuite) changes(configID string) AgentChanges {
	var answered AgentChanges
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+configID+"/changes", nil, &answered))
	return answered
}

// patch is somebody changing a setting in the dashboard.
func (s *SyncSuite) patch(configID string, body map[string]any) {
	s.Require().Equal(http.StatusOK,
		s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+configID, body, nil))
}

// ingest is somebody rewriting a knowledge document in the dashboard.
func (s *SyncSuite) ingest(namespace, source, text string) {
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPost, "/v1/agents/knowledge",
		map[string]any{"namespace": namespace,
			"documents": []map[string]string{{"source": source, "text": text}}}, nil))
}

// documentText is what one of an agent's knowledge documents now says.
func (s *SyncSuite) documentText(configID, source string) string {
	namespace := value(s.configsNamed("support")[0].KnowledgeNamespace)
	s.Require().NotEmpty(namespace, "agent "+configID+" has no knowledge base")
	for _, listed := range s.documents(namespace) {
		if listed.Source != source {
			continue
		}
		var document IndexedKnowledgeDocument
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
			"/v1/agents/knowledge/documents/"+listed.Id, nil, &document))
		return value(document.Text)
	}
	s.Require().Fail("no document called " + source)
	return ""
}

// auditOf is everything on one agent's record, newest first.
func (s *SyncSuite) auditOf(configID string) []AuditEntry {
	var answered AuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/audit/query",
		map[string]any{"filter": map[string]any{"agent_id": configID}}, &answered))
	return answered.Items
}

// dashboard is the backend saying it is the dashboard, with the operator who clicked save.
func (s *SyncSuite) dashboard() *testClient {
	return s.serverClient.from("dashboard", "volt-7", "Ada Lovelace")
}

// skillNamed is one of a config's skills.
func (s *SyncSuite) skillNamed(configID, name string) Skill {
	var listed []Skill
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/skills?config_id="+configID, nil, &listed))
	for _, skill := range listed {
		if skill.Name == name {
			return skill
		}
	}
	s.Require().Fail("no skill called " + name)
	return Skill{}
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

func (s *SyncSuite) TestASyncReplacesTheBindingsStored() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("inbox")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "connectors": []map[string]any{sessionSlack("crm")}})

	bindings := value(second.Config.Connectors)
	s.Require().Len(bindings, 1)
	s.Equal("crm", bindings[0].Name)
	s.Equal(second.Config.Connectors, s.configsNamed("support")[0].Connectors)
}

func (s *SyncSuite) TestASyncThatDeclaresNoBindingsLeavesTheOnesStored() {
	first := s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("inbox")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "instructions": "Be brief."})

	s.Equal(first.Config.Connectors, second.Config.Connectors)
}

func (s *SyncSuite) TestASyncWithNoBindingsClearsThem() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("inbox")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "connectors": []map[string]any{}})

	s.Nil(second.Config.Connectors)
	s.Nil(s.configsNamed("support")[0].Connectors)
}

func (s *SyncSuite) TestASyncBindingAConnectorThatDoesNotExistIsRefusedAndStoresNothing() {
	binding := sessionSlack("crm")
	binding["connector_id"] = "custom_nothing_here"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "custom_nothing_here")
	s.Empty(s.configsNamed("support"))
}

func (s *SyncSuite) TestASyncWithAnAliasHoldingTheToolSeparatorIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("team__inbox")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "__")
	s.Empty(s.configsNamed("support"))
}

func (s *SyncSuite) TestASyncAddingAPluginABindingIsCalledIsRefused() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("linear")}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v2", "plugins": []string{"linear"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "plugin")
	s.Equal("v1", value(s.configsNamed("support")[0].SyncHash))
}
