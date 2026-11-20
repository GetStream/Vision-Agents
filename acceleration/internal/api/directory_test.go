//go:build integration

package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type DirectorySuite struct {
	RouterSuite
}

func TestDirectorySuite(t *testing.T) {
	runSuite(t, new(DirectorySuite))
}

// SetupSuite seeds the built-in connectors as a router start does, so a directory can bind one.
func (s *DirectorySuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{"oauth2_code": namedScheme("oauth2_code")}}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

// SetupTest gives every test an app of its own, because a sync is named by what the agent is
// called.
func (s *DirectorySuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// sentFiles is a directory as a client sends it, with an agent.yaml naming it support unless the
// test gives one.
func sentFiles(held map[string]string) []map[string]string {
	if _, declared := held["agent.yaml"]; !declared {
		held["agent.yaml"] = "name: support\n"
	}
	sent := make([]map[string]string, 0, len(held))
	for _, path := range slices.Sorted(maps.Keys(held)) {
		sent = append(sent, map[string]string{"path": path, "content": held[path]})
	}
	return sent
}

// hashOf is the hash the files request documents, worked out here by hand: every part as
// its length in bytes in decimal, a colon and the part.
func hashOf(parts ...string) string {
	hasher := sha256.New()
	for _, part := range parts {
		fmt.Fprintf(hasher, "%d:%s", len(part), part)
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func (s *DirectorySuite) sync(held map[string]string) SyncAgentResult {
	return s.syncWith(held, nil)
}

// syncWith syncs a directory with the rest of the request beside its files.
func (s *DirectorySuite) syncWith(held map[string]string, beside map[string]any) SyncAgentResult {
	body := map[string]any{"files": sentFiles(held)}
	maps.Copy(body, beside)
	var result SyncAgentResult
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sync/files", body)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &result))
	return result
}

// refused syncs a directory that must be refused, and returns the status and what was said.
func (s *DirectorySuite) refused(held map[string]string, beside map[string]any) (int, ErrorDetail) {
	body := map[string]any{"files": sentFiles(held)}
	maps.Copy(body, beside)
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sync/files", body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

func (s *DirectorySuite) config(id string) AgentConfig {
	var config AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &config))
	return config
}

func (s *DirectorySuite) configsNamed(name string) []AgentConfig {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name="+name, nil, &listed))
	return listed
}

// exported is a config's directory by path.
func (s *DirectorySuite) exported(id string) (map[string]string, AgentDirectory) {
	var directory AgentDirectory
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id+"/directory", nil, &directory))
	held := map[string]string{}
	for _, file := range directory.Files {
		held[file.Path] = file.Content
	}
	return held, directory
}

// connection is a connection of the app's own to a built-in connector, under a label.
func (s *DirectorySuite) connection(connectorID, label string) string {
	ctx := context.Background()
	definition, err := s.store.LatestConnectorDefinition(ctx, s.customerID(), connectorID)
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: definition.ID, DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerApp, AuthScheme: "oauth2_code", Label: label,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(ctx, s.connectors, connection))
	return connection.ID
}

// audited is everything on one agent's record.
func (s *DirectorySuite) audited(configID string) []AuditEntry {
	var answered AuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/audit/query",
		map[string]any{"filter": map[string]any{"agent_id": configID}}, &answered))
	return answered.Items
}

func (s *DirectorySuite) dashboard() *testClient {
	return s.serverClient.from("dashboard", "volt-7", "Ada Lovelace")
}

func (s *DirectorySuite) TestSyncingFilesStoresWhatTheDirectorySays() {
	held := map[string]string{
		"agent.yaml":           "name: support\nmode: text\nllm: llm-flow\ngreeting: {text: Hello., mode: variation}\nkeyterms: [Vision Agents]\ntags: {project: support}\n",
		"instructions.md":      "Be brief.\n",
		"guardrail.md":         "---\ntype: llm\n---\nNo medical advice.\n",
		"skills/refund.md":     "---\ndescription: work out a refund\ndeadline: 45s\n---\nRead the policy.\n",
		"knowledge/pricing.md": "Pro costs twenty dollars.\n",
	}

	synced := s.sync(held)

	s.False(synced.Unchanged)
	stored := s.config(synced.Config.Id)
	s.Equal("support", stored.Name)
	s.Equal(AgentModeText, stored.Mode)
	s.Equal("llm-flow", value(stored.Llm))
	s.Equal("Be brief.", value(stored.Instructions))
	s.Equal("---\ntype: llm\n---\nNo medical advice.", value(stored.Guardrail))
	s.Equal("Hello.", stored.Greeting.Text)
	s.Equal(GreetingModeVariation, value(stored.Greeting.Mode))
	s.Equal([]string{"Vision Agents"}, value(stored.Keyterms))
	s.Equal("support", value(stored.Tags)["project"])
	s.Equal([]string{"refund"}, value(stored.Skills))
	s.Equal("support", value(stored.KnowledgeNamespace))
	skills, err := s.store.CustomerSkills(context.Background(), s.customerID(), stored.Id)
	s.Require().NoError(err)
	s.Require().Len(skills, 1)
	s.Equal("Read the policy.", skills[0].Instructions)
	s.EqualValues(45000, skills[0].DeadlineMs)
}

func (s *DirectorySuite) TestTheHashIsTheRoutersOwnOfTheFiles() {
	synced := s.sync(map[string]string{"agent.yaml": "name: support\n", "instructions.md": "Be brief.\n"})

	s.Equal(hashOf("agent.yaml", "name: support\n", "instructions.md", "Be brief.\n"), value(synced.Config.SyncHash))
}

func (s *DirectorySuite) TestTheSameFilesAgainWriteNothing() {
	held := map[string]string{"instructions.md": "Be brief.\n"}
	first := s.sync(held)
	recorded := len(s.audited(first.Config.Id))

	second := s.sync(held)

	s.True(second.Unchanged)
	s.Equal(first.Config.UpdatedAt, s.config(first.Config.Id).UpdatedAt)
	s.Len(s.audited(first.Config.Id), recorded, "nothing was written, so no sync is recorded")
}

func (s *DirectorySuite) TestAChangedFileSyncsAgain() {
	first := s.sync(map[string]string{"instructions.md": "Be brief.\n"})

	second := s.sync(map[string]string{"instructions.md": "Be even briefer.\n"})

	s.False(second.Unchanged)
	s.Equal(first.Config.Id, second.Config.Id)
	s.Equal("Be even briefer.", value(s.config(first.Config.Id).Instructions))
	s.NotEqual(value(first.Config.SyncHash), value(second.Config.SyncHash))
}

func (s *DirectorySuite) TestASettingAgentYAMLLeavesOutIsLeftAlone() {
	first := s.sync(map[string]string{"agent.yaml": "name: support\nllm: llm-flow\nvoice: aurora\n"})
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+first.Config.Id,
		map[string]any{"voice": "juniper", "search": "search-fast"}, nil))

	s.sync(map[string]string{"agent.yaml": "name: support\nllm: llm-fast\n"})

	stored := s.config(first.Config.Id)
	s.Equal("llm-fast", value(stored.Llm))
	s.Equal("juniper", value(stored.Voice), "a voice the file no longer names is the one chosen since")
	s.Equal("search-fast", value(stored.Search))
}

func (s *DirectorySuite) TestAFilesSyncAskedToCheckIsRefusedOverAChangeItWouldWriteOver() {
	first := s.sync(map[string]string{"agent.yaml": "name: support\nllm: llm-flow\n"})
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+first.Config.Id,
		map[string]any{"llm": "llm-fast"}, nil))

	status, failure := s.refused(map[string]string{"agent.yaml": "name: support\nllm: llm-thinking\n"},
		map[string]any{"check_changes": true})

	s.Equal(http.StatusConflict, status)
	s.Equal(codeUnsyncedChanges, failure.Code)
	s.Contains(failure.Message, "llm")
	s.Equal("llm-fast", value(s.config(first.Config.Id).Llm))
}

func (s *DirectorySuite) TestARefusalOverUnsyncedChangesNamesThemAsDataOnBothSyncs() {
	held := map[string]string{
		"agent.yaml":           "name: support\nllm: llm-flow\n",
		"instructions.md":      "Be brief.\n",
		"skills/refund.md":     "---\ndescription: work out a refund\n---\nRead the policy.\n",
		"knowledge/pricing.md": "Pro costs twenty dollars.\n",
	}
	first := s.sync(held)
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+first.Config.Id,
		map[string]any{"llm": "llm-fast", "instructions": "Be thorough.", "voice": "aurora"}, nil))
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
		"namespace": "support", "documents": []map[string]string{{"source": "added.md", "text": "Added in the dashboard."}},
	}, nil))
	var changes AgentChanges
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+first.Config.Id+"/changes", nil, &changes))
	want := []SyncConflict{
		{ResourceType: "knowledge", Name: "added.md", Removed: true},
		{ResourceType: "agent_config", Name: "instructions"},
		{ResourceType: "agent_config", Name: "llm"},
	}

	held["agent.yaml"], held["instructions.md"] = "name: support\nllm: llm-thinking\n", "Be terse.\n"
	status, files := s.refused(held, map[string]any{"check_changes": true})
	s.Equal(http.StatusConflict, status)
	s.ElementsMatch(want, files.Conflicts, "voice is not one: the directory does not say it")
	s.Equal(value(changes.LastChange), files.LastChange)
	s.Contains(files.Message, "knowledge added.md, which this directory no longer has", "the message still names them")

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v2", "check_changes": true, "llm": "llm-thinking", "instructions": "Be terse.",
		"knowledge": []map[string]string{{"source": "pricing.md", "text": "Pro costs twenty dollars.\n"}},
	})
	var fields ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &fields))
	s.Equal(http.StatusConflict, status)
	s.ElementsMatch(want, fields.Error.Conflicts)
	s.Equal(value(changes.LastChange), fields.Error.LastChange)

	s.False(s.syncWith(held, map[string]any{"check_changes": true, "base_change": files.LastChange}).Unchanged,
		"the last change it named is what lets the sync through")
}

func (s *DirectorySuite) TestWhatAProgramSetsInCodeWinsOverTheFiles() {
	held := map[string]string{
		"agent.yaml":       "name: support\nllm: llm-flow\nvoice: aurora\nsubagent: llm-thinking\ntags: {team: support, tier: free}\n",
		"instructions.md":  "Be brief.\n",
		"guardrail.md":     "---\ntype: llm\n---\nNo medical advice.\n",
		"skills/refund.md": "---\ndescription: work out a refund\n---\nRead the policy.\n",
	}

	synced := s.syncWith(held, map[string]any{
		"instructions": "Be thorough.", "guardrail": "---\ntype: llm\n---\nNo legal advice.",
		"llm": "llm-fast", "subagent": "llm-flow", "sandbox": "daytona", "harness": "default",
		"greeting": map[string]any{"text": "Hello.", "mode": "variation"},
		"tags":     map[string]string{"tier": "paid", "customer": "123"},
		"skills":   []map[string]any{{"config_id": "", "name": "think", "description": "work it out", "instructions": "Reason."}},
	})

	stored := s.config(synced.Config.Id)
	s.Equal("Be thorough.", value(stored.Instructions))
	s.Equal("---\ntype: llm\n---\nNo legal advice.", value(stored.Guardrail))
	s.Equal("llm-fast", value(stored.Llm))
	s.Equal("aurora", value(stored.Voice), "what the code leaves out is the file's")
	s.Equal("llm-flow", value(stored.Subagent))
	s.Equal(Daytona, value(stored.Sandbox))
	s.Equal("Hello.", stored.Greeting.Text)
	s.Equal(map[string]string{"team": "support", "tier": "paid", "customer": "123"}, value(stored.Tags))
	s.Equal([]string{"think"}, value(stored.Skills), "skills written in code take the place of the directory's")
}

func (s *DirectorySuite) TestAnEmptySkillListInCodeNamesNoSkills() {
	held := map[string]string{"skills/refund.md": "---\ndescription: work out a refund\n---\nRead the policy.\n"}

	synced := s.syncWith(held, map[string]any{"skills": []any{}})

	s.Empty(value(synced.Config.Skills))
	s.Equal([]string{"refund"}, value(s.syncWith(held, nil).Config.Skills))
}

func (s *DirectorySuite) TestWhatAProgramSetsInCodeIsPartOfTheHash() {
	held := map[string]string{"agent.yaml": "name: support\n"}
	code := map[string]any{
		"instructions": "Be brief.", "llm": "llm-fast",
		"video":  map[string]any{"max_frames": 4},
		"tags":   map[string]string{"b": "2", "a": "1"},
		"skills": []map[string]any{{"config_id": "", "name": "think", "description": "work it out", "instructions": "Reason.", "deadline_ms": 30000}},
	}

	first := s.syncWith(held, code)

	s.Equal(hashOf("agent.yaml", "name: support\n",
		"instructions", "Be brief.", "llm", "llm-fast", "video.max_frames", "4", "tags.a", "1", "tags.b", "2", "skills", "1",
		"skills.think.description", "work it out", "skills.think.instructions", "Reason.", "skills.think.deadline_ms", "30000",
	), value(first.Config.SyncHash))
	s.True(s.syncWith(held, code).Unchanged, "the same files and the same code write nothing")

	code["instructions"] = "Be thorough."
	changed := s.syncWith(held, code)
	s.False(changed.Unchanged, "a setting changed in code syncs again")
	s.Equal("Be thorough.", value(changed.Config.Instructions))
	s.False(s.syncWith(held, nil).Unchanged, "and so does taking the code away")
}

func (s *DirectorySuite) TestASyncOfFilesNeedsFiles() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync/files", map[string]any{"name": "support"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "files")
	s.Empty(s.configsNamed("support"))
}

func (s *DirectorySuite) TestTheSyncOfFieldsTakesNoFiles() {
	var synced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v1", "instructions": "Be brief.",
		"files": sentFiles(map[string]string{"agent.yaml": "name: support\nllm: llm-fast\n"}),
	}, &synced))

	s.Equal("v1", value(synced.Config.SyncHash))
	s.Nil(synced.Config.Llm, "a field the operation does not have is not read")
}

func (s *DirectorySuite) TestTheNameSentWinsOverAgentYAMLsAndOneOfThemIsNeeded() {
	named := s.syncWith(map[string]string{"agent.yaml": "name: support\n"}, map[string]any{"name": "billing"})
	s.Equal("billing", named.Config.Name)

	fromDirectory := s.syncWith(map[string]string{"agent.yaml": "llm: llm-fast\n"}, map[string]any{"name": "sales"})
	s.Equal("sales", fromDirectory.Config.Name)

	status, failure := s.refused(map[string]string{"agent.yaml": "llm: llm-fast\n"}, nil)
	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidAgentDirectory, failure.Code)
	s.Contains(failure.Message, "the agent has no name")
}

func (s *DirectorySuite) TestADirectoryThatDoesNotReadIsRefusedWithEveryProblemAtItsPlace() {
	status, failure := s.refused(map[string]string{
		"agent.yaml":         "name: support\nllmm: llm-fast\nplugins:\n  - name: github\n",
		"simulations/a.yaml": "- name: lunch\n  scenario: Ask.\n",
	}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(ErrorTypeInvalidRequest, failure.Type)
	s.Equal(codeInvalidAgentDirectory, failure.Code)
	s.Equal([]DirectoryProblem{
		{File: "agent.yaml", Line: 2, Column: 1, Message: `"llmm" is not something agent.yaml says; ` + agentYAMLKeys + ` are`},
		{File: "agent.yaml", Line: 4, Column: 3, Message: "plugins: no longer read: plugins were replaced by connectors, so name the provider under connectors"},
		{File: "simulations/a.yaml", Line: 1, Column: 3, Message: `simulation "lunch" needs an assertion`},
	}, failure.Problems)
	s.Contains(failure.Message, "agent.yaml:2:1: ")
	s.Empty(s.configsNamed("support"))
}

const agentYAMLKeys = "name, description, mode, region, stt, tts, sts, voice, llm, harness, role, subagent, search, image, " +
	"greeting, sandbox, sandbox_options, connectors, tools, channels, chat_channels, scheduling, calls, floor_control, " +
	"agents, keyterms, tags, video, dispatch, ui, app"

func (s *DirectorySuite) TestAValueTheSchemaRefusesIsRefusedWhereAgentYAMLSaysIt() {
	status, failure := s.refused(map[string]string{
		"agent.yaml": "name: support\nmode: sung\ngreeting:\n  text: Hi.\n  mode: shouted\nchat_channels:\n  - type: team\n    answer: sometimes\n",
	}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidAgentDirectory, failure.Code)
	at := map[string][2]int{}
	for _, problem := range failure.Problems {
		s.Equal("agent.yaml", problem.File)
		setting, _, _ := strings.Cut(problem.Message, ": ")
		at[setting] = [2]int{problem.Line, problem.Column}
	}
	s.Equal(map[string][2]int{"mode": {2, 7}, "greeting.mode": {5, 9}, "chat_channels[0].answer": {8, 13}}, at)
}

func (s *DirectorySuite) TestASimulationTheSchemaRefusesIsRefusedInItsFolder() {
	status, failure := s.refused(map[string]string{
		"simulations/a.yaml": "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n  variations: 99\n",
	}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Require().Len(failure.Problems, 1)
	s.Equal("simulations", failure.Problems[0].File)
	s.Contains(failure.Problems[0].Message, "simulations[0].variations")
}

func (s *DirectorySuite) TestAFileOverTheSizeAFileMayHaveIsRefused() {
	status, failure := s.refused(map[string]string{"knowledge/big.md": strings.Repeat("a", 1<<20+1)}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidAgentDirectory, failure.Code)
	s.Equal("knowledge/big.md", failure.Problems[0].File)
	s.Contains(failure.Message, "a file holds at most 1048576")
	s.Empty(s.configsNamed("support"))
}

func (s *DirectorySuite) TestADirectoryOverTheSizeOneMayHaveIsRefused() {
	held := map[string]string{}
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"} {
		held["knowledge/"+name+".md"] = strings.Repeat("a", 1<<20)
	}

	status, failure := s.refused(held, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "one holds at most 8388608")
}

func (s *DirectorySuite) TestMoreFilesThanARequestMaySendAreRefused() {
	held := map[string]string{}
	for index := range 1024 {
		held[fmt.Sprintf("src/%04d.go", index)] = "x"
	}

	status, failure := s.refused(held, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeValidationFailed, failure.Code)
	s.Contains(failure.Message, "files")
}

func (s *DirectorySuite) TestFilesLeftOutAreWarnedOfOnceWithACount() {
	synced := s.sync(map[string]string{"a.txt": "x", "b.txt": "x", "c.txt": "x", "d.txt": "x", "e.txt": "x"})

	s.Equal([]string{"5 files sent are not part of an agent directory, so they were left out: a.txt, b.txt, c.txt and 2 more"}, synced.Warnings)
}

func (s *DirectorySuite) TestARefusalNamesAtMostAHundredProblemsAndSaysHowManyMore() {
	var declaration strings.Builder
	declaration.WriteString("name: support\n")
	for index := range 150 {
		fmt.Fprintf(&declaration, "unknown_%03d: x\n", index)
	}

	status, failure := s.refused(map[string]string{"agent.yaml": declaration.String()}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Require().Len(failure.Problems, 100)
	s.Equal(50, failure.ProblemsOmitted)
	for _, problem := range failure.Problems {
		s.Positive(problem.Line, "every entry is a problem at a place, none a count of the rest: %v", problem)
	}
	s.Equal(5, strings.Count(failure.Message, "agent.yaml:"), "the message says the first few")
	s.Contains(failure.Message, "; and 145 more")
	s.Less(len(failure.Message), 4096)
}

func (s *DirectorySuite) TestARefusalThatNamesEveryProblemOmitsNone() {
	_, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sync/files",
		map[string]any{"files": sentFiles(map[string]string{"agent.yaml": "name: support\nllmm: x\n"})})

	s.NotContains(string(payload), "problems_omitted")
}

func (s *DirectorySuite) TestWhatTheRouterCannotRunIsSaidTogetherWhereAgentYAMLSaysIt() {
	status, failure := s.refused(map[string]string{
		"agent.yaml":          "name: support\nmode: text\nsubagent: llm-thinking\nfloor_control:\n  model: stream/chorus-asr\n",
		"skills/pdf/SKILL.md": "---\ndescription: Fill a form\n---\nRun it.\n",
		"skills/pdf/fill.py":  "print()\n",
	}, map[string]any{"dry_run": true})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeInvalidAgentDirectory, failure.Code)
	at := map[int]string{}
	for _, problem := range failure.Problems {
		s.Equal("agent.yaml", problem.File)
		at[problem.Line] = problem.Message
	}
	s.Len(failure.Problems, 3, "%v", failure.Problems)
	s.Contains(at[3], "subagent", "a text agent names no subagent")
	s.Contains(at[4], "chorus", "this router has no chorus-asr server")
	s.Contains(at[1], "sandbox", "a skill's script has nowhere to run, said at the top since the file names no sandbox")
}

func (s *DirectorySuite) TestFilesSentBesideTheDirectoryAreNoPartOfItsHashOrItsLimits() {
	held := map[string]string{"agent.yaml": "name: support\n", "instructions.md": "Be brief.\n"}
	alone := s.sync(held)

	among := s.sync(map[string]string{
		"agent.yaml": "name: support\n", "instructions.md": "Be brief.\n",
		"README.md": "# Support\n", "src/main.go": "package main\n", "build/app.js": strings.Repeat("a", 1<<20+1),
	})

	s.True(among.Unchanged, "a client that sends the whole folder syncs what one that sends the directory alone did")
	s.Equal(value(alone.Config.SyncHash), value(among.Config.SyncHash))
}

func (s *DirectorySuite) TestAUIFileOutsideTheUIFolderIsReadWhereAgentYAMLNamesIt() {
	synced := s.sync(map[string]string{
		"agent.yaml":          "name: support\nui:\n  blueprints: design/recipes.yaml\n",
		"design/recipes.yaml": "- name: booking\n  when: booking a table\n  order: [CardHeader]\n",
	})

	s.Contains(value(synced.Config.UiInstructions), "**booking**")
	s.Empty(synced.Warnings)
}

func (s *DirectorySuite) TestWhatDoesNotReadAndWhatTheSchemaRefusesAreSaidTogether() {
	status, failure := s.refused(map[string]string{"agent.yaml": "name: support\nmode: banana\nllmm: llm-fast\n"}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Require().Len(failure.Problems, 2, "%v", failure.Problems)
	s.Equal([]int{3, 1}, []int{failure.Problems[0].Line, failure.Problems[0].Column})
	s.Contains(failure.Problems[0].Message, `"llmm" is not something agent.yaml says`)
	s.Equal([]int{2, 7}, []int{failure.Problems[1].Line, failure.Problems[1].Column})
	s.Contains(failure.Problems[1].Message, "mode: ")
}

func (s *DirectorySuite) TestTwoSkillsOneNameAreRefusedBeforeAnythingIsWritten() {
	held := map[string]string{
		"skills/a.md": "---\nname: refund\ndescription: One\n---\nRead the policy.\n",
		"skills/b.md": "---\nname: refund\ndescription: Two\n---\nRead the policy.\n",
	}

	dry, failure := s.refused(held, map[string]any{"dry_run": true})
	wet, _ := s.refused(held, nil)

	s.Equal(http.StatusBadRequest, dry)
	s.Equal(http.StatusBadRequest, wet)
	s.Contains(failure.Message, `skill "refund" is also declared in skills/a.md`)
	s.Empty(s.configsNamed("support"))
}

func (s *DirectorySuite) TestTwoSkillsOneNameSentAsFieldsAreRefusedBeforeAnythingIsWritten() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "support", "hash": "v1",
		"skills": []map[string]string{
			{"config_id": "", "name": "refund", "description": "one", "instructions": "Read the policy."},
			{"config_id": "", "name": "refund", "description": "two", "instructions": "Read the policy."},
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `two skills are called "refund"`)
	s.Empty(s.configsNamed("support"), "the config is not left stored with half its skills")
}

func (s *DirectorySuite) TestADryRunRefusesAPageTheSyncWouldRefuse() {
	held := map[string]string{"knowledge/urls.yaml": "- https://\n"}

	dry, failure := s.refused(held, map[string]any{"dry_run": true})
	wet, _ := s.refused(held, nil)

	s.Equal(http.StatusBadRequest, dry)
	s.Equal(wet, dry)
	s.Contains(failure.Message, "names no host")
	s.Empty(s.configsNamed("support"))
}

func (s *DirectorySuite) TestADryRunOfANewAgentAnswersTheModeItWouldBeStoredWith() {
	held := map[string]string{"agent.yaml": "name: support\n"}

	preview := s.syncWith(held, map[string]any{"dry_run": true})

	s.Equal(AgentModeVoice, preview.Config.Mode)
	s.Equal(preview.Config.Mode, s.sync(held).Config.Mode)
}

func (s *DirectorySuite) TestAFileThatIsNoPartOfTheFormatIsLeftOutWithAWarning() {
	synced := s.sync(map[string]string{"instruction.md": "Be brief.\n", ".agent_sync": "{}"})

	s.Equal([]string{"1 files sent are not part of an agent directory, so they were left out: instruction.md"}, synced.Warnings)
	s.Empty(value(synced.Config.Instructions))
}

func (s *DirectorySuite) TestAConnectorIsBoundToTheConnectionItsLabelOrIdNames() {
	labelled := s.connection("slack", "Support workspace")
	other := s.connection("slack", "Sales workspace")

	synced := s.sync(map[string]string{"agent.yaml": `name: support
connectors:
  - github
  - name: inbox
    connector: slack
    tools: [search]
    connection: Support workspace
    required: true
  - name: sales
    connector: slack
    connection: ` + other + "\n"})

	bound := value(synced.Config.Connectors)
	s.Require().Len(bound, 3)
	s.Equal(AgentConnectorSelectionTypeSession, bound[0].Connection.Type)
	s.Equal([]string{"*"}, value(bound[0].ToolPatterns))
	s.Equal("slack", bound[1].ConnectorId)
	s.Equal(labelled, value(bound[1].Connection.ConnectionId))
	s.Equal([]string{"search"}, value(bound[1].ToolPatterns))
	s.True(value(bound[1].Required))
	s.Equal(other, value(bound[2].Connection.ConnectionId))
	s.Empty(synced.Warnings)
}

func (s *DirectorySuite) TestAnIdWinsOverALabelThatSaysTheSame() {
	named := s.connection("slack", "placeholder")
	s.connection("slack", named)

	synced := s.sync(map[string]string{"agent.yaml": "name: support\nconnectors:\n  - {name: slack, connection: " + named + "}\n"})

	s.Equal(named, value(value(synced.Config.Connectors)[0].Connection.ConnectionId))
}

func (s *DirectorySuite) TestAConnectorWhoseConnectionTheAppLacksIsLeftOutUntilItHasOne() {
	held := map[string]string{"agent.yaml": "name: support\nllm: llm-flow\nconnectors:\n  - github\n  - {name: inbox, connector: slack, connection: Support workspace}\n"}

	first := s.sync(held)

	s.Len(value(first.Config.Connectors), 1, "the rest of the agent still syncs")
	s.Equal("llm-flow", value(first.Config.Llm))
	s.Equal([]LeftOutConnector{{Name: "inbox", ConnectorId: "slack", Connection: "Support workspace"}}, value(first.Config.LeftOutConnectors))
	s.Equal([]string{`connector "inbox" was left out: the app has no connection with id or label "Support workspace"`}, first.Warnings)
	s.True(s.sync(held).Unchanged, "the same files with the connection still missing write nothing")

	connected := s.connection("slack", "Support workspace")
	found := s.sync(held)

	s.False(found.Unchanged, "the sync that finds the connection is not taken for the one that left it out")
	s.Require().Len(value(found.Config.Connectors), 2)
	s.Equal(connected, value(value(found.Config.Connectors)[1].Connection.ConnectionId))
	s.Nil(found.Config.LeftOutConnectors)
}

func (s *DirectorySuite) TestARequiredConnectorWhoseConnectionTheAppLacksIsRefusedWhereItIsNamed() {
	status, failure := s.refused(map[string]string{
		"agent.yaml": "name: support\nconnectors:\n  - github\n  - name: inbox\n    connector: slack\n    connection: Support workspace\n    required: true\n",
	}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal([]DirectoryProblem{{File: "agent.yaml", Line: 4, Column: 5,
		Message: `connector "inbox": no app connection with id or label "Support workspace": add the connection, or drop required`}}, failure.Problems)
	s.Empty(s.configsNamed("support"))
}

func (s *DirectorySuite) TestALabelTwoConnectionsShareIsRefused() {
	s.connection("slack", "workspace")
	s.connection("slack", "workspace")

	status, failure := s.refused(map[string]string{
		"agent.yaml": "name: support\nconnectors:\n  - {name: slack, connection: workspace}\n",
	}, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, `connector "slack": "workspace" matches 2 app connections`)
}

func (s *DirectorySuite) TestALabelOfAnotherConnectorsConnectionIsNotFound() {
	s.connection("github", "workspace")

	synced := s.sync(map[string]string{"agent.yaml": "name: support\nconnectors:\n  - {name: slack, connection: workspace}\n"})

	s.Empty(value(synced.Config.Connectors))
	s.Len(value(synced.Config.LeftOutConnectors), 1)
}

func (s *DirectorySuite) TestTheUIKeyIsCompiledByTheRouter() {
	synced := s.sync(map[string]string{
		"agent.yaml":         "name: support\nui:\n  keep: [CardHeader]\n  rules: [Be brief.]\n",
		"ui/blueprints.yaml": "- name: booking\n  when: booking a table\n  order: [CardHeader]\n",
	})

	instructions := value(synced.Config.UiInstructions)
	s.Contains(instructions, "## Interactive UI Generation")
	s.Contains(instructions, "- **booking** (when booking a table): `Card([CardHeader])`")
	s.Contains(instructions, "- Be brief.")
	s.NotContains(instructions, "BarChart(")

	without := s.sync(map[string]string{"agent.yaml": "name: support\n"})
	s.Nil(without.Config.UiInstructions, "a directory with no ui key teaches none")
}

func (s *DirectorySuite) TestSimulationFilesAreTheWholeOfTheAgentsSimulationsAndNoFileLeavesThemAlone() {
	first := s.sync(map[string]string{
		"simulations/a.yaml": "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n- name: refund\n  scenario: Ask.\n  assertion: Refunded.\n",
	})
	s.Len(s.simulations(first.Config.Id), 2)

	s.sync(map[string]string{"instructions.md": "Be brief.\n"})
	s.Len(s.simulations(first.Config.Id), 2, "a directory with no simulations file leaves them alone")

	s.sync(map[string]string{"simulations/a.yaml": "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n"})
	s.Equal([]string{"lunch"}, s.simulations(first.Config.Id))

	s.sync(map[string]string{"simulations/a.yaml": "[]\n"})
	s.Empty(s.simulations(first.Config.Id), "a file that declares none deletes them")
}

// simulations are the names of one config's simulations, in order.
func (s *DirectorySuite) simulations(configID string) []string {
	all, err := s.store.CustomerSimulations(context.Background(), s.customerID())
	s.Require().NoError(err)
	var names []string
	for _, simulation := range all {
		if simulation.ConfigID == configID {
			names = append(names, simulation.Name)
		}
	}
	slices.Sort(names)
	return names
}

func (s *DirectorySuite) TestASyncSaysWhatItDoesToSkillsKnowledgePagesAndSimulations() {
	first := map[string]string{
		"skills/refund.md":     "---\ndescription: work out a refund\n---\nRead the policy.\n",
		"skills/think.md":      "---\ndescription: think\n---\nReason.\n",
		"knowledge/pricing.md": "Pro costs twenty dollars.\n",
		"knowledge/hours.md":   "Open nine to five.\n",
		"knowledge/urls.yaml":  "- https://example.com/pricing\n",
		"simulations/a.yaml":   "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n- name: refund\n  scenario: Ask.\n  assertion: Refunded.\n",
	}
	created := s.sync(first)
	s.ElementsMatch([]SyncChange{
		{ResourceType: "skill", Name: "refund", Action: "created"}, {ResourceType: "skill", Name: "think", Action: "created"},
		{ResourceType: "knowledge", Name: "pricing.md", Action: "created"}, {ResourceType: "knowledge", Name: "hours.md", Action: "created"},
		{ResourceType: "knowledge_url", Name: "https://example.com/pricing", Action: "created"},
		{ResourceType: "simulation", Name: "lunch", Action: "created"}, {ResourceType: "simulation", Name: "refund", Action: "created"},
	}, created.Changes)

	second := map[string]string{
		"skills/refund.md":     "---\ndescription: work out a refund\n---\nRead the new policy.\n",
		"skills/think.md":      "---\ndescription: think\n---\nReason.\n",
		"knowledge/pricing.md": "Pro costs thirty dollars.\n",
		"knowledge/urls.yaml":  "- url: https://example.com/pricing\n  title: Pricing\n",
		"simulations/a.yaml":   "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n- name: hours\n  scenario: Ask.\n  assertion: Told.\n",
	}
	want := []SyncChange{
		{ResourceType: "skill", Name: "refund", Action: "updated"},
		{ResourceType: "knowledge", Name: "pricing.md", Action: "updated"}, {ResourceType: "knowledge", Name: "hours.md", Action: "deleted"},
		{ResourceType: "knowledge_url", Name: "https://example.com/pricing", Action: "updated"},
		{ResourceType: "simulation", Name: "hours", Action: "created"}, {ResourceType: "simulation", Name: "refund", Action: "deleted"},
	}

	preview := s.syncWith(second, map[string]any{"dry_run": true})
	s.ElementsMatch(want, preview.Changes, "a dry run says what the sync would do, and what it leaves as it is is not listed")
	s.Equal([]string{"lunch", "refund"}, s.simulations(created.Config.Id), "and does none of it")

	s.ElementsMatch(want, s.sync(second).Changes)
	s.Equal([]string{"hours", "lunch"}, s.simulations(created.Config.Id))
	s.Empty(s.sync(second).Changes)

	none := s.syncWith(map[string]string{"instructions.md": "Be brief.\n"}, map[string]any{"dry_run": true})
	s.ElementsMatch([]SyncChange{
		{ResourceType: "knowledge", Name: "pricing.md", Action: "deleted"},
	}, none.Changes, "with no knowledge folder the agent's own base is emptied, and with no simulations file they are left alone")
}

func (s *DirectorySuite) TestADryRunAnswersWhatWouldBeStoredAndWritesNothing() {
	created := s.syncWith(map[string]string{
		"agent.yaml":           "name: support\nllm: llm-flow\n",
		"instructions.md":      "Be brief.\n",
		"skills/refund.md":     "---\ndescription: work out a refund\n---\nRead the policy.\n",
		"knowledge/pricing.md": "Pro costs twenty dollars.\n",
		"simulations/a.yaml":   "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n",
	}, map[string]any{"dry_run": true})

	s.False(created.Unchanged)
	s.Empty(created.Config.Id, "a config a dry run would create has no id")
	s.Equal("llm-flow", value(created.Config.Llm))
	s.Equal("Be brief.", value(created.Config.Instructions))
	s.Equal([]string{"refund"}, value(created.Config.Skills))
	s.Equal("support", value(created.Config.KnowledgeNamespace))
	s.Empty(s.configsNamed("support"), "nothing was stored")
	documents, err := s.store.CustomerKnowledgeDocuments(context.Background(), s.customerID(), "support")
	s.Require().NoError(err)
	s.Empty(documents)
}

func (s *DirectorySuite) TestADryRunOverAStoredAgentLeavesItAsItIs() {
	first := s.sync(map[string]string{"agent.yaml": "name: support\nllm: llm-flow\n", "instructions.md": "Be brief.\n"})
	before := s.config(first.Config.Id)
	recorded := len(s.audited(first.Config.Id))

	preview := s.syncWith(map[string]string{"agent.yaml": "name: support\nllm: llm-fast\n", "instructions.md": "Be thorough.\n"},
		map[string]any{"dry_run": true})

	s.Equal(first.Config.Id, preview.Config.Id)
	s.Equal("llm-fast", value(preview.Config.Llm))
	s.Equal("Be thorough.", value(preview.Config.Instructions))
	s.Equal(before, s.config(first.Config.Id))
	s.Len(s.audited(first.Config.Id), recorded, "a dry run records no sync")

	same := s.syncWith(map[string]string{"agent.yaml": "name: support\nllm: llm-flow\n", "instructions.md": "Be brief.\n"},
		map[string]any{"dry_run": true})
	s.True(same.Unchanged, "a dry run of the files already synced says a sync would do nothing")
}

func (s *DirectorySuite) TestADryRunIsRefusedAsTheSyncWouldBe() {
	first := s.sync(map[string]string{"agent.yaml": "name: support\nllm: llm-flow\n"})
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+first.Config.Id,
		map[string]any{"llm": "llm-fast"}, nil))

	conflict, failure := s.refused(map[string]string{"agent.yaml": "name: support\nllm: llm-thinking\n"},
		map[string]any{"dry_run": true, "check_changes": true})
	s.Equal(http.StatusConflict, conflict)
	s.Equal(codeUnsyncedChanges, failure.Code)

	invalid, _ := s.refused(map[string]string{"agent.yaml": "name: support\nsandbox: nowhere\n"}, map[string]any{"dry_run": true})
	s.Equal(http.StatusBadRequest, invalid)

	oversized, failure := s.refused(map[string]string{
		"agent.yaml":          "name: support\nsandbox: daytona\n",
		"skills/big/SKILL.md": "---\ndescription: big\n---\nRun it.\n",
		"skills/big/a.txt":    strings.Repeat("a", 1<<20),
		"skills/big/b.txt":    "one byte too many",
	}, map[string]any{"dry_run": true})
	s.Equal(http.StatusBadRequest, oversized)
	s.Contains(failure.Message, "a skill's files may hold at most")
}

// frontDesk is a directory written the way an export writes one, setting most of what
// agent.yaml can say.
func (s *DirectorySuite) frontDesk() map[string]string {
	connection := s.connection("slack", "Support workspace")
	return map[string]string{
		"agent.yaml": `name: front-desk
mode: voice
tts: tts-fast
voice: aurora
llm: llm-flow
subagent: llm-thinking
greeting:
  text: Hi, how can I help?
  mode: variation
sandbox: daytona
sandbox_options:
  setup:
    - pip install pandas
  timeout: 1m30s
  cpu: 2
connectors:
  - github
  - name: inbox
    connector: slack
    tools:
      - search
      - post_*
    connection: ` + connection + `
    required: true
    require_approval:
      - post_*
tools:
  - name: booking
    loading: summary
    tools:
      - lookup_booking
    output_retention: 30d
chat_channels:
  - type: team
    answer: mentioned
scheduling:
  enabled: true
  max_tasks_per_user: 10
  default_timezone: Europe/Amsterdam
calls:
  end_call:
    enabled: true
    when: The caller says goodbye.
  press:
    enabled: false
  transfer:
    destinations:
      - name: billing
        number: "+15551234567"
        when: 'Questions about an invoice: refunds, too.'
        warm: true
keyterms:
  - Stream
  - 10DLC
tags:
  project: front-desk
video:
  source: screen
  max_frames: 4
dispatch:
  incoming_call: enabled
ui:
  keep:
    - CardHeader
  rules:
    - Be brief.
`,
		"instructions.md":              "You answer the phone.\n",
		"guardrail.md":                 "---\ntype: llm\n---\nNo medical advice.\n",
		"skills/refunds/SKILL.md":      "---\ndescription: Work out a refund\ndeadline: 2m\n---\nRead the policy.\n",
		"skills/refunds/reference.md":  "# Refund policy\n",
		"knowledge/plans/pro.md":       "Pro costs twenty dollars.\n",
		"simulations/simulations.yaml": "- name: lunch\n  scenario: Order a turkey club.\n  assertion: One turkey club.\n  variations: 3\n",
		"ui/blueprints.yaml":           "- name: booking\n  when: booking a table\n  order: [CardHeader]\n",
	}
}

func (s *DirectorySuite) TestAnAgentIsExportedAsTheDirectoryItWasSyncedFrom() {
	written := s.frontDesk()
	synced := s.sync(written)

	held, directory := s.exported(synced.Config.Id)

	s.Equal("front-desk", directory.Name)
	s.Equal(written, held)
	s.Empty(directory.NotExported)
	var paths []string
	for _, file := range directory.Files {
		paths = append(paths, file.Path)
	}
	s.IsIncreasing(paths)
}

func (s *DirectorySuite) TestAnExportLeavesOutWhatASyncFillsInOnItsOwn() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "plain", "mode": "voice"}, &created))

	held, directory := s.exported(created.Id)

	s.Equal(map[string]string{"agent.yaml": "name: plain\nmode: voice\n"}, held)
	s.Empty(directory.NotExported)
}

func (s *DirectorySuite) TestSyncingAnExportBackChangesNothing() {
	synced := s.sync(s.frontDesk())
	before := s.config(synced.Config.Id)
	held, _ := s.exported(synced.Config.Id)

	preview := s.syncWith(held, map[string]any{"dry_run": true})
	again := s.sync(held)

	after := s.config(synced.Config.Id)
	s.True(preview.Unchanged)
	s.True(again.Unchanged, "an export of what was synced is the files that were synced")
	s.Equal(before, after)
}

func (s *DirectorySuite) TestSyncingTheExportOfAnAgentMadeInTheDashboardChangesNothing() {
	connection := s.connection("slack", "")
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "desk", "mode": "voice", "llm": "llm-flow", "voice": "aurora", "instructions": "Be brief.",
		"greeting": map[string]any{"text": "Hello."}, "keyterms": []string{"Stream"},
		"llm_fallbacks": []string{"llm-fast"}, "visible_tools": []string{"search"},
		"connectors": []map[string]any{
			{"name": "inbox", "connector_id": "slack", "connection": map[string]any{"type": "fixed", "connection_id": connection},
				"tools": []any{}, "tool_patterns": []string{"search"}, "required": true},
		},
		"calls": map[string]any{"end_call": map[string]any{"enabled": true, "when": "They say goodbye."}},
	}, &created))
	before := s.config(created.Id)
	held, directory := s.exported(created.Id)
	s.Empty(directory.NotExported)

	synced := s.sync(held)

	s.False(synced.Unchanged, "the config was never synced, so the files are new to it")
	after := s.config(created.Id)
	before.UpdatedAt, before.SyncHash = after.UpdatedAt, after.SyncHash
	s.Equal(before, after, "what agent.yaml cannot say, such as llm_fallbacks and visible_tools, is left as it was")
}

func (s *DirectorySuite) TestAnExportSaysWhatItsDirectoryCannot() {
	connection := s.connection("slack", "")
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "desk", "mode": "voice", "ui_instructions": "Answer with cards.",
		"connectors": []map[string]any{
			fixedSlack("inbox", connection),
			{"name": "idle", "connector_id": "github", "connection": map[string]any{"type": "session"}, "tools": []any{}},
		},
	}, &created))

	held, directory := s.exported(created.Id)

	s.Equal("name: desk\nmode: voice\nconnectors:\n  - name: inbox\n    connector: slack\n    tools:\n      - search\n    connection: "+
		connection+"\n    required: true\n", held["agent.yaml"])
	s.Len(directory.NotExported, 4, "%v", directory.NotExported)
	said := strings.Join(directory.NotExported, "\n")
	s.Contains(said, "The connector inbox grants its tools by name and schema digest")
	s.Contains(said, "The connector inbox has a timeout, events or a policy")
	s.Contains(said, "The connector idle grants no tools")
	s.Contains(said, "UI instructions were not compiled from a directory's files")
}

func (s *DirectorySuite) TestAConnectorLeftOutIsExportedAsAgentYAMLNamedIt() {
	synced := s.sync(map[string]string{"agent.yaml": "name: support\nconnectors:\n  - {name: inbox, connector: slack, connection: Support workspace}\n"})

	held, _ := s.exported(synced.Config.Id)

	s.Equal("name: support\nmode: voice\nconnectors:\n  - name: inbox\n    connector: slack\n    connection: Support workspace\n", held["agent.yaml"])
}

func (s *DirectorySuite) TestUIInstructionsWrittenSinceTheSyncAreNotExportedAsTheFilesTheyReplaced() {
	held := map[string]string{"agent.yaml": "name: support\nui:\n  rules: [Be brief.]\n"}
	edits := map[string]func(id string) int{
		"patched": func(id string) int {
			return s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+id, map[string]any{"ui_instructions": "Answer with cards."}, nil)
		},
		"replaced": func(id string) int {
			return s.dashboard().do(http.MethodPut, "/v1/agents/configs/"+id,
				map[string]any{"name": "support", "mode": "voice", "ui_instructions": "Answer with cards."}, nil)
		},
	}
	for how, edit := range edits {
		s.useApp(s.data.createApp())
		synced := s.sync(held)
		s.Require().Equal(http.StatusOK, edit(synced.Config.Id), how)

		exported, directory := s.exported(synced.Config.Id)

		s.NotContains(exported["agent.yaml"], "ui:", how)
		s.Contains(strings.Join(directory.NotExported, "\n"), "UI instructions were not compiled from a directory's files", how)
	}
}

func (s *DirectorySuite) TestAnEditThatLeavesTheUIInstructionsAloneKeepsTheFilesTheyCameFrom() {
	synced := s.sync(map[string]string{"agent.yaml": "name: support\nui:\n  rules: [Be brief.]\n"})
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+synced.Config.Id,
		map[string]any{"voice": "aurora"}, nil))

	exported, directory := s.exported(synced.Config.Id)

	s.Contains(exported["agent.yaml"], "ui:\n  rules:\n    - Be brief.\n")
	s.Empty(directory.NotExported)
}

func (s *DirectorySuite) TestAnExportLeavesOutKnowledgeASyncWouldNotTakeBack() {
	synced := s.sync(map[string]string{"knowledge/small.md": "Pro costs twenty dollars.\n"})
	var ingested any
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
		"namespace": "support", "documents": []map[string]string{{"source": "big.md", "text": strings.Repeat("word ", 1<<18)}},
	}, &ingested))

	held, directory := s.exported(synced.Config.Id)

	s.Contains(held, "knowledge/small.md")
	s.NotContains(held, "knowledge/big.md")
	s.Contains(strings.Join(directory.NotExported, "\n"), `The knowledge document "big.md" is more than a directory may hold`)
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync/files",
		map[string]any{"files": sentFiles(held), "dry_run": true}, nil), "what is exported is something a sync takes")
}

func (s *DirectorySuite) TestASkillIsExportedAsASyncReadsItBack() {
	held := map[string]string{
		"agent.yaml":            "name: support\nmode: voice\n",
		"skills/steps/SKILL.md": "---\ndescription: Say \"hello\"\n---\n- First.\n- Second.\n",
		"skills/rule/SKILL.md":  "---\ndescription: \"'quoted'\"\n---\n---\n\nAfter a rule.\n",
	}
	synced := s.sync(held)

	exported, directory := s.exported(synced.Config.Id)

	s.Equal(held["skills/steps/SKILL.md"], exported["skills/steps/SKILL.md"])
	s.Equal(held["skills/rule/SKILL.md"], exported["skills/rule/SKILL.md"])
	s.Empty(directory.NotExported)
	s.True(s.sync(exported).Unchanged)
}

func (s *DirectorySuite) TestAnExportOfSkillFilesAFolderCannotHoldStillSyncsBack() {
	synced := s.sync(map[string]string{"agent.yaml": "name: support\nmode: voice\n"})
	for _, skill := range []map[string]any{
		{"name": "Alpha", "files": map[string]string{"SKILL.md": "another", ".hidden.md": "x", "notes.md": "kept"}},
		{"name": "alpha"},
	} {
		skill["config_id"], skill["description"], skill["instructions"] = synced.Config.Id, "d", "i"
		s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost, "/v1/agents/skills", skill, nil))
	}
	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch, "/v1/agents/configs/"+synced.Config.Id,
		map[string]any{"skills": []string{"Alpha", "alpha"}}, nil))

	held, directory := s.exported(synced.Config.Id)

	s.Len(directory.NotExported, 2, "%v", directory.NotExported)
	s.Equal("kept", held["skills/Alpha/notes.md"])
	s.Contains(held, "skills/skill-1/SKILL.md", "alpha has a folder Alpha does not share where case folds")
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync/files", map[string]any{"files": sentFiles(held)}, nil),
		"what is exported is a directory a sync takes")
}

func (s *DirectorySuite) TestAnExportSaysWhenASyncWouldTrimTheInstructionsOrDropTheKnowledgeBase() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "desk", "mode": "voice", "instructions": "\nBe brief.\n\n", "knowledge_namespace": "handbook",
	}, &created))

	_, directory := s.exported(created.Id)

	said := strings.Join(directory.NotExported, "\n")
	s.Contains(said, "instructions.md starts or ends with blank space")
	s.Contains(said, `None of the agent's knowledge base "handbook" is in these files`)
}

func (s *DirectorySuite) TestAnExportCountsEverythingItWritesAgainstWhatADirectoryMayHold() {
	held := map[string]string{"simulations/a.yaml": "- name: lunch\n  scenario: " + strings.Repeat("Ask. ", 190_000) + "\n  assertion: Answered.\n"}
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		held["knowledge/"+name+".md"] = strings.Repeat(name+" ", 1<<19)[:1<<20-16]
	}
	synced := s.sync(held)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
		"namespace": "support", "documents": []map[string]string{{"source": "h.md", "text": strings.Repeat("h ", 1<<18)}},
	}, nil))

	exported, directory := s.exported(synced.Config.Id)

	total := 0
	for _, content := range exported {
		total += len(content)
	}
	s.LessOrEqual(total, 8<<20)
	s.Contains(strings.Join(directory.NotExported, "\n"), `The knowledge document "h.md" is more than a directory may hold`)
}

func (s *DirectorySuite) TestOnlyTheAppsOwnBackendMayReadOrSendADirectory() {
	synced := s.sync(map[string]string{"instructions.md": "Be brief.\n"})

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id+"/directory", nil, nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sync/files", map[string]any{
			"files": sentFiles(map[string]string{"agent.yaml": "name: agent-" + s.utils.uuid() + "\n"}), "dry_run": true,
		}, nil)
	})
}

func (s *DirectorySuite) TestAnotherAppsAgentHasNoDirectoryToRead() {
	synced := s.sync(map[string]string{"instructions.md": "Be brief.\n"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id+"/directory", nil, nil)
	})
}
