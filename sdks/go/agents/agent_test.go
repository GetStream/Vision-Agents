package agents

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// backend is a stand-in for the acceleration router's configuration paths: enough of them
// to see what an agent stores and how it edits what is already there.
type backend struct {
	*httptest.Server

	mu      sync.Mutex
	configs []acceleration.AgentConfig
	skills  []acceleration.Skill
	syncs   []acceleration.SyncAgentRequest
	updates []string
}

func newBackend(t *testing.T) *backend {
	t.Helper()

	router := &backend{}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /v1/agents/configs", func(w http.ResponseWriter, _ *http.Request) {
		router.mu.Lock()
		defer router.mu.Unlock()
		reply(w, http.StatusOK, router.configs)
	})
	mux.HandleFunc("POST /v1/agents/configs", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.AgentConfigRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		stored := acceleration.AgentConfig{
			Id: "config-1", Name: request.Name, Instructions: request.Instructions,
			KnowledgeNamespace: request.KnowledgeNamespace, Skills: request.Skills,
			ThinkingLlm: request.ThinkingLlm, Tags: request.Tags,
			CreatedAt: time.Now(), UpdatedAt: time.Now(),
		}
		router.configs = append(router.configs, stored)
		reply(w, http.StatusCreated, stored)
	})
	mux.HandleFunc("PUT /v1/agents/configs/{id}", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.AgentConfigRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.updates = append(router.updates, r.PathValue("id"))
		for index, config := range router.configs {
			if config.Id != r.PathValue("id") {
				continue
			}
			router.configs[index].Instructions = request.Instructions
			reply(w, http.StatusOK, router.configs[index])
			return
		}
		reply(w, http.StatusNotFound, acceleration.Error{Error: "no such config"})
	})

	mux.HandleFunc("GET /v1/agents/skills", func(w http.ResponseWriter, _ *http.Request) {
		router.mu.Lock()
		defer router.mu.Unlock()
		reply(w, http.StatusOK, router.skills)
	})
	mux.HandleFunc("POST /v1/agents/skills", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SkillRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		stored := acceleration.Skill{
			Id: "skill-" + request.Name, Name: request.Name, Description: request.Description,
			Instructions: request.Instructions, DeadlineMs: request.DeadlineMs,
			CreatedAt: time.Now(), UpdatedAt: time.Now(),
		}
		router.skills = append(router.skills, stored)
		reply(w, http.StatusCreated, stored)
	})
	mux.HandleFunc("PUT /v1/agents/skills/{id}", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SkillRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.updates = append(router.updates, r.PathValue("id"))
		reply(w, http.StatusOK, acceleration.Skill{
			Id: r.PathValue("id"), Name: request.Name, Description: request.Description,
			Instructions: request.Instructions, CreatedAt: time.Now(), UpdatedAt: time.Now(),
		})
	})

	mux.HandleFunc("POST /v1/agents/sync", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SyncAgentRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.syncs = append(router.syncs, request)
		stored := acceleration.AgentConfig{
			Id: "config-" + request.Name, Name: request.Name, Instructions: request.Instructions,
			ThinkingLlm: request.ThinkingLlm, Llm: request.Llm, Tags: request.Tags,
			CreatedAt: time.Now(), UpdatedAt: time.Now(),
		}
		if request.Knowledge != nil || request.KnowledgeUrls != nil {
			stored.KnowledgeNamespace = &request.Name
		}
		if request.Skills != nil {
			named := []string{}
			for _, skill := range *request.Skills {
				named = append(named, skill.Name)
			}
			stored.Skills = &named
		}
		router.configs = append(router.configs[:0:0], stored)
		reply(w, http.StatusOK, acceleration.SyncAgentResult{Config: stored})
	})

	router.Server = httptest.NewServer(mux)
	t.Cleanup(router.Close)
	return router
}

func reply(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// agentOn builds an agent talking to a stand-in router.
func agentOn(t *testing.T, router *backend, options Options) *Agent {
	t.Helper()
	if options.LLM == nil {
		options.LLM = stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
		})
	}

	agent, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

func TestAnAgentNeedsAName(t *testing.T) {
	if _, err := New(Options{LLM: stream.Accelerated(stream.Config{})}); err == nil {
		t.Error("an agent has to be called something")
	}
}

func TestANamedAgentRunsItsStoredConfigWithItsTools(t *testing.T) {
	agent, err := New(Options{Name: "jean", Tools: []tools.Tool{lookupOrder{}}})
	if err != nil {
		t.Fatal(err)
	}

	if agent.LLM() == nil {
		t.Fatal("an agent given only a name has nothing answering")
	}
	if listed := agent.Tools().List(); len(listed) != 1 || listed[0].Name != "lookup_order" {
		t.Errorf("the model is offered %+v", listed)
	}
}

func TestANamedAgentReadsItsFolderUnderAgents(t *testing.T) {
	root := t.TempDir()
	write(t, root, "agents/jean/agent.yaml", "name: jean\n")
	write(t, root, "agents/jean/instructions.md", "You are Jean.\n")
	t.Chdir(root)

	agent, err := New(Options{Name: "jean"})
	if err != nil {
		t.Fatal(err)
	}
	if agent.Folder() == nil || agent.Instructions() != "You are Jean." {
		t.Errorf("agents/jean was not read: instructions are %q", agent.Instructions())
	}
}

type lookupOrder struct {
	OrderID string `json:"order_id"`
}

func (lookupOrder) Name() string                     { return "lookup_order" }
func (lookupOrder) Description() string              { return "Look up an order by its number" }
func (lookupOrder) Run(context.Context) (any, error) { return "shipped", nil }

func TestAnAgentJoinsUnderAUserIDDerivedFromItsName(t *testing.T) {
	router := newBackend(t)
	agent := agentOn(t, router, Options{Name: "Jean Le Bot"})

	if agent.options.UserID != "jean-le-bot" {
		t.Errorf("the agent joins as %q", agent.options.UserID)
	}
}

func TestAChatIsAskedThroughItsResponsesLikeASessionOpenedByName(t *testing.T) {
	router := newWorked(t, nil)
	agent, err := New(Options{
		Name: "jean",
		LLM: stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	answer, err := session.Responses.Create(t.Context(), "Where is order 1042?")
	if err != nil {
		t.Fatal(err)
	}
	if answer.ID() != "response-1" || answer.Created.SessionId != session.ID() {
		t.Errorf("the answer is %+v, want response-1 in %s", answer.Created, session.ID())
	}
	if asked := router.questions(); len(asked) != 1 || asked[0] != session.ID()+": Where is order 1042?" {
		t.Errorf("the router was asked %v", asked)
	}
}

func TestASessionCanChangeWhatTheAgentWasConfiguredWith(t *testing.T) {
	router := newWorked(t, nil)
	agent, err := New(Options{
		Name:         "jean",
		Instructions: "You are Jean.",
		CostTracking: map[string]string{"team": "support", "tier": "free"},
		LLM: stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{
		Instructions: "You are Jean, and brief.",
		CostTracking: map[string]string{"tier": "pro"},
		Title:        "Order 1042",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	router.mu.Lock()
	defer router.mu.Unlock()
	opened := router.opened[0]
	if opened.Instructions == nil || *opened.Instructions != "You are Jean, and brief." {
		t.Errorf("the session was opened with instructions %v", opened.Instructions)
	}
	if opened.Tags == nil || (*opened.Tags)["team"] != "support" || (*opened.Tags)["tier"] != "pro" {
		t.Errorf("the session was labelled %v", opened.Tags)
	}
	if opened.Title == nil || *opened.Title != "Order 1042" {
		t.Errorf("the session was titled %v", opened.Title)
	}
	if agent.options.CostTracking["tier"] != "free" {
		t.Errorf("the session wrote its labels through to the agent: %v", agent.options.CostTracking)
	}
}

func TestSyncStoresTheAgentAndEditsItTheSecondTime(t *testing.T) {
	router := newBackend(t)
	agent := agentOn(t, router, Options{Name: "jean", Instructions: "Be brief."})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if stored.Name != "jean" || *stored.Instructions != "Be brief." {
		t.Fatalf("the config was stored as %+v", stored)
	}

	agent.options.Instructions = "Be briefer."
	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.configs) != 1 {
		t.Errorf("syncing twice stored %d configs", len(router.configs))
	}
	if *router.configs[0].Instructions != "Be briefer." {
		t.Errorf("the stored config still says %q", *router.configs[0].Instructions)
	}
}

func TestSyncPushesADirectorysSkillsAndKnowledge(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "llm: openai/gpt-5.6\ntags:\n  team: support\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")
	write(t, root, "knowledge/urls.yaml", "- url: https://example.com/plans\n  title: Plans\n  refresh_hours: 24\n")

	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()

	if len(router.syncs) != 1 {
		t.Fatalf("the directory was synced in %d requests", len(router.syncs))
	}
	synced := router.syncs[0]
	if synced.Hash != agent.Folder().Hash() {
		t.Errorf("the directory was sent as %q, but hashes to %q", synced.Hash, agent.Folder().Hash())
	}
	if synced.Skills == nil || (*synced.Skills)[0].Name != "think" {
		t.Errorf("the skills sent are %+v", synced.Skills)
	}
	if synced.Knowledge == nil || (*synced.Knowledge)[0].Source != "pricing.md" {
		t.Errorf("the knowledge sent is %+v", synced.Knowledge)
	}
	if synced.KnowledgeUrls == nil || (*synced.KnowledgeUrls)[0].Url != "https://example.com/plans" ||
		*(*synced.KnowledgeUrls)[0].Title != "Plans" || (*synced.KnowledgeUrls)[0].Description != nil ||
		*(*synced.KnowledgeUrls)[0].RefreshHours != 24 {
		t.Errorf("the pages sent are %+v", synced.KnowledgeUrls)
	}
	if synced.Llm == nil || *synced.Llm != "openai/gpt-5.6" || (*synced.Tags)["team"] != "support" {
		t.Errorf("what agent.yaml declares did not go with it: %+v", synced)
	}
	if synced.Stt != nil || synced.Mode != nil {
		t.Errorf("settings the declaration never named were sent: %+v", synced)
	}
	if stored.KnowledgeNamespace == nil || *stored.KnowledgeNamespace != "jean" {
		t.Errorf("the config does not point at the knowledge: %+v", stored)
	}
	if ReadStamp(root) != synced.Hash {
		t.Errorf("%s records %q", AgentStamp, ReadStamp(root))
	}
}

func TestSyncSendsHowTheSandboxIsBuilt(t *testing.T) {
	root := filepath.Join(t.TempDir(), "artist")
	write(t, root, "agent.yaml", `sandbox: daytona
sandbox_options:
  setup: [pip install bpy==5.2.2]
  timeout: 5m
  memory_gb: 4
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	options := router.syncs[0].SandboxOptions
	if options == nil || (*options.Setup)[0] != "pip install bpy==5.2.2" || *options.TimeoutMs != 300000 || *options.MemoryGb != 4 {
		t.Errorf("how the sandbox is built went as %+v", options)
	}
}

func TestSyncSendsTheMCPServersNamedByURL(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", `mcp_servers:
  - name: tablejourney
    url: https://tablejourney.com/mcp
    tools: [search_*]
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	servers := router.syncs[0].McpServers
	if servers == nil || len(*servers) != 1 || (*servers)[0].Name != "tablejourney" || (*servers)[0].Url != "https://tablejourney.com/mcp" ||
		(*servers)[0].Tools == nil || strings.Join(*(*servers)[0].Tools, ",") != "search_*" {
		t.Errorf("the MCP servers went as %+v", servers)
	}
}

func TestSyncSendsHowEachPluginIsReached(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", `user_plugins: [linear, calcom]
plugin_options:
  - plugin: linear
    readonly: true
    scopes: [read]
  - plugin: calcom
    toolsets: [bookings, availability]
    tools: [get_bookings, get_availability]
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	options := router.syncs[0].PluginOptions
	if options == nil || len(*options) != 2 || (*options)[0].Plugin != "linear" ||
		(*options)[0].Readonly == nil || !*(*options)[0].Readonly ||
		(*options)[0].Scopes == nil || len(*(*options)[0].Scopes) != 1 || (*(*options)[0].Scopes)[0] != "read" {
		t.Fatalf("the plugin options went as %+v", options)
	}
	if toolsets := (*options)[1].Toolsets; toolsets == nil || strings.Join(*toolsets, ",") != "bookings,availability" {
		t.Errorf("calcom's toolsets went as %+v", toolsets)
	}
	if tools := (*options)[1].Tools; tools == nil || strings.Join(*tools, ",") != "get_bookings,get_availability" {
		t.Errorf("calcom's tools went as %+v", tools)
	}
}

func TestSyncSaysNothingOfTheSandboxWhenTheDeclarationDoesNot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "analyst")
	write(t, root, "agent.yaml", "sandbox: daytona\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if router.syncs[0].SandboxOptions != nil {
		t.Errorf("options nobody declared were sent: %+v", router.syncs[0].SandboxOptions)
	}
}

func TestAnUnchangedDirectoryIsNotSyncedAgain(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	router := newBackend(t)

	for range 2 {
		stored, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if stored == nil || stored.Name != "jean" {
			t.Fatalf("the sync answered %+v", stored)
		}
	}
	write(t, root, "instructions.md", "You are Jean, and brief.\n")
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 {
		t.Errorf("three syncs, one of them of an edited directory, sent %d requests", len(router.syncs))
	}
}

func TestWhatTheCodeSetsIsPartOfWhatIsSynced(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	router := newBackend(t)

	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}
	written := Options{Dir: root, Instructions: "You are somebody else."}
	if _, err := agentOn(t, router, written).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 || *router.syncs[1].Instructions != "You are somebody else." {
		t.Errorf("instructions written in code were not synced: %+v", router.syncs)
	}
}

func TestADirectorysSkillsAreWhatTheAgentJoinsWith(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")

	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	skills := router.syncs[0].Skills
	if skills == nil || len(*skills) != 1 || (*skills)[0].Name != "think" {
		t.Errorf("the config would be stored with %+v", skills)
	}
}

func TestADirectoryDoesNotWriteThroughToAHarnessSharedWithAnotherAgent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")

	router := newBackend(t)
	shared := DefaultHarness()
	agentOn(t, router, Options{Dir: root, Harness: shared})

	if len(shared.Skills) != 0 {
		t.Errorf("the caller's harness now holds %+v", shared.Skills)
	}
}

func TestAMemoryFilterSaysWhoTheMemoriesAreAboutAndWhatNarrowsThem(t *testing.T) {
	memory := memoryOf(map[string]string{"user_id": "123", "tenant": "acme"})

	if memory.UserId == nil || *memory.UserId != "123" {
		t.Fatalf("the memories are about %+v", memory.UserId)
	}
	if memory.Filter == nil || (*memory.Filter)["tenant"] != "acme" {
		t.Errorf("the filter is %+v", memory.Filter)
	}
	if _, leaked := (*memory.Filter)["user_id"]; leaked {
		t.Error("who the memories are about is not also a label")
	}
	if memoryOf(nil) != nil {
		t.Error("without a filter nothing is recalled and nothing is sent")
	}
}

func TestAgentYAMLNamesTheHarnessAndSandboxTheConfigIsStoredWith(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nharness: default\nsandbox: daytona\n")

	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root, Harness: &Harness{
		Subagents: map[string]string{"default": "openai/gpt-5.6-sol"},
	}})
	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	synced := router.syncs[0]
	if synced.Harness == nil || *synced.Harness != acceleration.Default {
		t.Errorf("the harness was stored as %v", synced.Harness)
	}
	if synced.Sandbox == nil || *synced.Sandbox != "daytona" {
		t.Errorf("the sandbox was stored as %v", synced.Sandbox)
	}
	if synced.ThinkingLlm == nil || *synced.ThinkingLlm != "openai/gpt-5.6-sol" {
		t.Errorf("the thinking llm was stored as %v", synced.ThinkingLlm)
	}
}

func TestTheDefaultHarnessLeavesTheBuiltInSkillsAlone(t *testing.T) {
	router := newBackend(t)
	stored, err := agentOn(t, router, Options{Name: "jean", Harness: DefaultHarness()}).Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	if stored.Skills != nil {
		t.Errorf("the built-in set was replaced by %+v", *stored.Skills)
	}
}

func TestAHarnessThatDoesNotExistIsRefused(t *testing.T) {
	if _, err := New(Options{Name: "jean", Harness: &Harness{Name: "fancy"}}); err == nil {
		t.Fatal("the backend has only the default harness, so another name means nothing")
	}
}

func TestSeveralSubagentsWithNoDefaultIsRefused(t *testing.T) {
	harness := &Harness{Subagents: map[string]string{"fast": "a", "slow": "b"}}

	if err := harness.Validate(); err == nil {
		t.Fatal("which one runs skills would be decided by map iteration order")
	}
}
