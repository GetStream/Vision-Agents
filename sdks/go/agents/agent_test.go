package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
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
	syncRaw [][]byte
	updates []string
	// connections answers GET /v1/agents/connections. Nil answers an empty page.
	connections       func(*http.Request) acceleration.ConnectionPage
	connectionQueries []string
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
			Subagent: request.Subagent, Tags: request.Tags,
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
		reply(w, http.StatusNotFound, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: "no such config"}})
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

	mux.HandleFunc("GET /v1/agents/connections", func(w http.ResponseWriter, r *http.Request) {
		router.mu.Lock()
		router.connectionQueries = append(router.connectionQueries, r.URL.RawQuery)
		answer := router.connections
		router.mu.Unlock()
		if answer == nil {
			reply(w, http.StatusOK, acceleration.ConnectionPage{})
			return
		}
		reply(w, http.StatusOK, answer(r))
	})

	mux.HandleFunc("POST /v1/agents/sync", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var request acceleration.SyncAgentRequest
		_ = json.Unmarshal(raw, &request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.syncRaw = append(router.syncRaw, raw)
		router.syncs = append(router.syncs, request)
		stored := acceleration.AgentConfig{
			Id: "config-" + request.Name, Name: request.Name, Instructions: request.Instructions,
			Subagent: request.Subagent, Llm: request.Llm, Tags: request.Tags,
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

func (b *backend) answerConnections(answer func(*http.Request) acceleration.ConnectionPage) {
	b.mu.Lock()
	b.connections = answer
	b.mu.Unlock()
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

func TestASyncTheRouterRefusesCarriesWhatItSaid(t *testing.T) {
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(stream.RequestIDHeader, "request-1")
		reply(w, http.StatusUnauthorized, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
			Message: "the token has expired", Type: acceleration.ErrorTypeAuthentication,
			Code: "unauthenticated", DocUrl: "https://getstream.io/agents/docs/api/errors/#unauthenticated",
		}})
	}))
	t.Cleanup(router.Close)
	agent, err := New(Options{Name: "jean", LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
	})})
	if err != nil {
		t.Fatal(err)
	}

	_, err = agent.Sync(t.Context())

	var refused *stream.RouterError
	if !errors.As(err, &refused) {
		t.Fatalf("the refusal came back as %v", err)
	}
	if refused.Status != http.StatusUnauthorized || refused.Type != "authentication" ||
		refused.Code != "unauthenticated" || refused.RequestID != "request-1" ||
		refused.DocURL != "https://getstream.io/agents/docs/api/errors/#unauthenticated" {
		t.Errorf("the refusal was read as %+v", *refused)
	}
	if err.Error() != "agents: the token has expired" {
		t.Errorf("the refusal says %q", err)
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

func TestSyncSendsWhatTheSandboxMayReach(t *testing.T) {
	root := filepath.Join(t.TempDir(), "researcher")
	write(t, root, "agent.yaml", `sandbox: daytona
sandbox_options:
  network: true
  allowed_domains: [api.example.com]
  network_allow_list: [10.0.0.0/8]
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	options := router.syncs[0].SandboxOptions
	if options == nil || !*options.Network || (*options.AllowedDomains)[0] != "api.example.com" ||
		(*options.NetworkAllowList)[0] != "10.0.0.0/8" {
		t.Errorf("what the sandbox may reach went as %+v", options)
	}
}

func TestSyncSendsTheChannelsTheAgentAnswersOn(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", `channels:
  whatsapp:
    number: "+15556325550"
  sms:
    number: "+12187021098"
  identity: link
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	named := router.syncs[0].Channels
	if named == nil || named.Whatsapp == nil || named.Whatsapp.Number != "+15556325550" ||
		named.Sms == nil || named.Sms.Number != "+12187021098" || named.Imessage != nil ||
		named.Identity == nil || *named.Identity != acceleration.ChannelIdentityLink {
		t.Errorf("the channels went as %+v", named)
	}
}

// A file saying nothing about channels leaves the agent reachable in Stream Chat alone.
func TestSyncSendsNoChannelsWhenTheFileNamesNone(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "llm: llm-fast\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if router.syncs[0].Channels != nil {
		t.Errorf("the channels went as %+v", router.syncs[0].Channels)
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

func TestSyncSendsWhetherToolsAreOfferedProgressively(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "tools:\n  progressive: true\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if sent := router.syncs[0].Tools; sent == nil || sent.Progressive == nil || !*sent.Progressive {
		t.Errorf("tools went as %+v", sent)
	}
}

func TestSyncSendsWhoLogsIntoEachMCPServer(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", `mcp_servers:
  - name: crm
    url: https://crm.example.com/mcp
    scopes: [contacts.read]
  - name: notes
    url: https://notes.example.com/mcp
    user: true
`)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	servers := router.syncs[0].McpServers
	if servers == nil || len(*servers) != 2 {
		t.Fatalf("the MCP servers went as %+v", servers)
	}
	crm, notes := (*servers)[0], (*servers)[1]
	if crm.Scopes == nil || strings.Join(*crm.Scopes, ",") != "contacts.read" || crm.User != nil {
		t.Errorf("the app's server went as %+v", crm)
	}
	if notes.User == nil || !*notes.User || notes.Scopes != nil {
		t.Errorf("each user's server went as %+v", notes)
	}
}

func TestSyncSendsHowEachPluginIsReached(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", `plugins:
  - sentry
  - name: linear
    user: true
    readonly: true
    scopes: [read]
  - name: calcom
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
	named := router.syncs[0].Plugins
	if named == nil || len(*named) != 3 {
		t.Fatalf("the plugins went as %+v", named)
	}
	if sentry, err := (*named)[0].AsPluginEntry0(); err != nil || sentry != "sentry" {
		t.Errorf("a plugin with nothing said about it went as %q (%v), not its id", sentry, err)
	}
	linear, err := (*named)[1].AsPluginWithOptions()
	if err != nil || linear.Name != "linear" || linear.User == nil || !*linear.User ||
		linear.Readonly == nil || !*linear.Readonly ||
		linear.Scopes == nil || strings.Join(*linear.Scopes, ",") != "read" {
		t.Errorf("linear went as %+v (%v)", linear, err)
	}
	calcom, err := (*named)[2].AsPluginWithOptions()
	if err != nil || calcom.User != nil || calcom.Toolsets == nil || strings.Join(*calcom.Toolsets, ",") != "bookings,availability" ||
		calcom.Tools == nil || strings.Join(*calcom.Tools, ",") != "get_bookings,get_availability" {
		t.Errorf("calcom went as %+v (%v)", calcom, err)
	}
}

func TestSyncWarnsThatPluginsAreDeprecatedAndStillSendsThem(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", "plugins: [sentry]\nplugin_events:\n  - plugin: sentry\n    event: issue.created\n")
	router := newBackend(t)
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(logged.String(), `level=WARN msg="agent.yaml plugins are deprecated: rename plugins to connectors" agent=triage plugins=[sentry] plugin_events=1`) {
		t.Errorf("no deprecation warning in %q", logged.String())
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0]
	if sent.Plugins == nil || len(*sent.Plugins) != 1 || sent.PluginEvents == nil || len(*sent.PluginEvents) != 1 {
		t.Errorf("the plugins went as %+v and their events as %+v, not as declared", sent.Plugins, sent.PluginEvents)
	}
}

func TestSyncWithoutPluginsWarnsOfNone(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", "llm: openai/gpt-4o\n")
	router := newBackend(t)
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	if strings.Contains(logged.String(), "deprecated") {
		t.Errorf("an agent naming no plugins was warned: %q", logged.String())
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
	if synced.Subagent == nil || *synced.Subagent != "openai/gpt-5.6-sol" {
		t.Errorf("the subagent was stored as %v", synced.Subagent)
	}
}

func TestAgentYAMLNamesTheRoleTheConfigIsStoredWith(t *testing.T) {
	root := filepath.Join(t.TempDir(), "nora")
	write(t, root, "agent.yaml", "name: nora\nrole: ai_moderator\n")

	router := newBackend(t)
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if role := router.syncs[0].Role; role == nil || *role != acceleration.AiModerator {
		t.Errorf("the role was stored as %v", role)
	}
}

// A declaration that says nothing about the role sends none, which leaves the stored one.
func TestAgentYAMLWithoutARoleSendsNone(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")

	router := newBackend(t)
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if role := router.syncs[0].Role; role != nil {
		t.Errorf("a declaration without a role sent %v", *role)
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

// controlYAML is an agent.yaml with no connectors key. Its sync body was probed on
// 80773902 (connectors/tool-patterns), before connectors were read.
const controlYAML = "name: concierge\nllm: openai/gpt-5.6\nplugins:\n  - sentry\nmcp_servers:\n  - name: notes\n    url: https://notes.example/mcp\n"

func TestSyncWithoutConnectorsSendsTheBodyItSentBefore(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", controlYAML)
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	const probed = `{"hash":"9d57177d6eb6a640963fc6c6775fa120","llm":"openai/gpt-5.6","mcp_servers":[{"name":"notes","url":"https://notes.example/mcp"}],"name":"concierge","plugins":["sentry"]}`
	if len(router.syncRaw) != 1 || string(router.syncRaw[0]) != probed {
		t.Errorf("body = %s", router.syncRaw)
	}
	if len(router.connectionQueries) != 0 {
		t.Errorf("connections were listed: %v", router.connectionQueries)
	}
}

func TestSyncSendsAConnectorNamedAloneAsASessionBinding(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 {
		t.Fatalf("connectors went as %+v", sent)
	}
	linear := (*sent)[0]
	if linear.Name != "linear" || linear.ConnectorId != "linear" ||
		linear.Connection.Type != acceleration.AgentConnectorSelectionTypeSession || linear.Connection.ConnectionId != nil ||
		linear.ToolPatterns == nil || strings.Join(*linear.ToolPatterns, ",") != "*" || len(linear.Tools) != 0 || linear.Required != nil {
		t.Errorf("linear went as %+v", linear)
	}
	if strings.Contains(string(router.syncRaw[0]), `"tools":null`) {
		t.Errorf("body = %s", router.syncRaw[0])
	}
	if len(router.connectionQueries) != 0 {
		t.Errorf("a session binding listed connections: %v", router.connectionQueries)
	}
}

func TestSyncSendsConnectorToolsAsPatterns(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    tools: [list_*, get_issue]\n    required: true\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 {
		t.Fatalf("connectors went as %+v", sent)
	}
	github := (*sent)[0]
	if github.Name != "github" || github.ConnectorId != "github" ||
		github.Connection.Type != acceleration.AgentConnectorSelectionTypeSession ||
		github.ToolPatterns == nil || strings.Join(*github.ToolPatterns, ",") != "list_*,get_issue" ||
		github.Required == nil || !*github.Required || len(github.Tools) != 0 {
		t.Errorf("github went as %+v", github)
	}
}

func TestSyncSendsEveryToolWhenToolsAreNull(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    tools: null\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].ToolPatterns == nil || strings.Join(*(*sent)[0].ToolPatterns, ",") != "*" {
		t.Errorf("null tools went as %+v", sent)
	}
}

func TestSyncSendsRequiredFalse(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    required: false\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Required == nil || *(*sent)[0].Required ||
		(*sent)[0].ToolPatterns == nil || strings.Join(*(*sent)[0].ToolPatterns, ",") != "*" {
		t.Errorf("required went as %+v", sent)
	}
	if !strings.Contains(string(router.syncRaw[0]), `"required":false`) {
		t.Errorf("body = %s", router.syncRaw[0])
	}
}

func TestSyncOfAnEmptyConnectorListClearsThem(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors: []\n")
	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if router.syncs[0].Connectors == nil || len(*router.syncs[0].Connectors) != 0 {
		t.Errorf("connectors went as %+v", router.syncs[0].Connectors)
	}
	if !strings.Contains(string(router.syncRaw[0]), `"connectors":[]`) {
		t.Errorf("body = %s", router.syncRaw[0])
	}
}

func TestSyncResolvesAConnectionIdBeforeItsLabel(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connector: linear\n    connection: acct\n")
	router := newBackend(t)
	router.answerConnections(func(r *http.Request) acceleration.ConnectionPage {
		query := r.URL.Query()
		if query.Get("owner_type") != "app" || query.Get("connector_id") != "linear" {
			t.Errorf("listed %s", r.URL.RawQuery)
		}
		if query.Get("cursor") == "" {
			return connectionPage("2", listedConnection("wrong", "acct"))
		}
		return connectionPage("", listedConnection("acct", "other"))
	})
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.connectionQueries) != 2 {
		t.Fatalf("queries = %v", router.connectionQueries)
	}
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 {
		t.Fatalf("connectors went as %+v", sent)
	}
	got := (*sent)[0]
	if got.Name != "issues" || got.ConnectorId != "linear" ||
		got.Connection.Type != acceleration.AgentConnectorSelectionTypeFixed ||
		got.Connection.ConnectionId == nil || *got.Connection.ConnectionId != "acct" {
		t.Errorf("issues went as %+v", got)
	}
}

func TestSyncResolvesTheOneConnectionWithThatLabel(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connector: linear\n    connection: work\n")
	router := newBackend(t)
	router.answerConnections(func(r *http.Request) acceleration.ConnectionPage {
		if r.URL.Query().Get("connector_id") != "linear" {
			t.Errorf("listed connector %s", r.URL.Query().Get("connector_id"))
		}
		if r.URL.Query().Get("cursor") == "" {
			return connectionPage("2", listedConnection("other", "nope"))
		}
		return connectionPage("", listedConnection("the-one", "work"))
	})
	agent := agentOn(t, router, Options{Dir: root})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Connection.ConnectionId == nil || *(*sent)[0].Connection.ConnectionId != "the-one" {
		t.Errorf("issues went as %+v, queries %v", sent, router.connectionQueries)
	}
	if len(router.connectionQueries) != 2 {
		t.Errorf("queries = %v", router.connectionQueries)
	}
}

// resolvingYAML names a session binding and a connection the app has. Its sync body was
// probed on a5721c90, before a missing connection was left out.
const resolvingYAML = "name: concierge\nconnectors:\n  - linear\n  - name: code\n    connector: github\n    connection: company\n"

func TestSyncOfConnectionsThatAllResolveSendsTheBodyItSentBefore(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", resolvingYAML)
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("conn-1", "company"))
	})
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	const probed = `{"connectors":[{"connection":{"type":"session"},"connector_id":"linear","name":"linear","tool_patterns":["*"],"tools":[]},{"connection":{"connection_id":"conn-1","type":"fixed"},"connector_id":"github","name":"code","tool_patterns":["*"],"tools":[]}],"hash":"84cee98b6c473b082438ee7b52cea7c6","name":"concierge"}`
	if len(router.syncRaw) != 1 || string(router.syncRaw[0]) != probed {
		t.Errorf("body = %s", router.syncRaw)
	}
	if stamp := ReadStamp(root); stamp != router.syncs[0].Hash {
		t.Errorf("stamped %q after sending %q", stamp, router.syncs[0].Hash)
	}
	if strings.Contains(logged.String(), "left out") {
		t.Errorf("warned of a connection the app has: %q", logged.String())
	}
}

func TestSyncLeavesOutAConnectorWhoseConnectionIsMissing(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n  - name: issues\n    connection: missing\n    required: false\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	if stored == nil || stored.Name != "concierge" {
		t.Errorf("the sync answered %+v", stored)
	}
	if !strings.Contains(logged.String(), `level=WARN msg="agent.yaml connector left out: the app has no such connection" agent=concierge connector=issues connection=missing`) {
		t.Errorf("no warning in %q", logged.String())
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 1 {
		t.Fatalf("%d syncs were sent", len(router.syncs))
	}
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Name != "linear" {
		t.Errorf("connectors went as %+v", sent)
	}
}

// The router is told what was left out, so it can show it. The hash is not the one 7f4c3e16
// sent for this directory (87c8a2808cd085a316aa2942c6b827a8), before the list was sent: the
// router stored that one without the list, and would answer unchanged to it.
func TestSyncSendsTheConnectorsItLeftOut(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n  - name: issues\n    connection: company\n  - name: code\n    connector: github\n    connection: team\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})

	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	const want = `{"connectors":[{"connection":{"type":"session"},"connector_id":"linear","name":"linear","tool_patterns":["*"],"tools":[]}],"hash":"370694e1e9f5b6495455c9b89fce3218",` +
		`"left_out_connectors":[{"connection":"company","connector_id":"issues","name":"issues"},{"connection":"team","connector_id":"github","name":"code"}],"name":"concierge"}`
	if len(router.syncRaw) != 1 || string(router.syncRaw[0]) != want {
		t.Errorf("body = %s", router.syncRaw)
	}
}

// An entry over the router's caps is warned about but not sent in the list, so the router
// does not refuse the whole sync over it. The hash still covers it.
func TestSyncLeavesAnOverLongLeftOutEntryOutOfTheList(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	label := strings.Repeat("c", 121)
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n  - name: issues\n    connection: "+label+
		"\n  - name: "+strings.Repeat("a", 64)+"\n    connection: team\n  - name: tracker\n    connector: "+strings.Repeat("b", 65)+"\n    connection: team\n  - name: code\n    connector: github\n    connection: team\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(logged.String(), "connector=issues connection="+label) {
		t.Errorf("no warning for the long label in %q", logged.String())
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].LeftOutConnectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Name != "code" {
		t.Errorf("left_out_connectors went as %+v", sent)
	}
}

// The caps are the router's: an entry exactly at them is sent. The label is multibyte, over
// the cap in bytes and at it in runes, which is what the router counts.
func TestSyncSendsALeftOutEntryExactlyAtTheCaps(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	alias, connector, label := strings.Repeat("a", 63), strings.Repeat("b", 64), strings.Repeat("é", 120)
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n  - name: "+alias+"\n    connector: "+connector+"\n    connection: "+label+"\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})

	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	want := []acceleration.LeftOutConnector{{Name: alias, ConnectorId: connector, Connection: label}}
	if got := router.syncs[0].LeftOutConnectors; got == nil || !reflect.DeepEqual(*got, want) {
		t.Errorf("left_out_connectors went as %+v", got)
	}
}

// The same holds for a multibyte alias and connector id, whose bytes are over the caps and
// whose runes are at them.
func TestSyncCountsRunesNotBytesInTheNameAndConnectorOfALeftOutEntry(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	alias, connector := strings.Repeat("é", 63), strings.Repeat("é", 64)
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n  - name: "+alias+"\n    connector: "+connector+"\n    connection: team\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})

	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if got := router.syncs[0].LeftOutConnectors; got == nil || len(*got) != 1 {
		t.Errorf("left_out_connectors went as %+v", got)
	}
}

// The router takes at most 64 entries. More are left out of the list, so the sync is not
// refused where it synced before, and each is still warned about.
func TestSyncSendsAtMost64LeftOutEntries(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	yaml := "name: concierge\nconnectors:\n"
	for i := range 65 {
		yaml += fmt.Sprintf("  - name: c%d\n    connector: github\n    connection: team\n", i)
	}
	write(t, root, "agent.yaml", yaml)
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	var logged bytes.Buffer
	agent := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))})

	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	sent := router.syncs[0].LeftOutConnectors
	if sent == nil || len(*sent) != 64 || (*sent)[63].Name != "c63" {
		t.Errorf("left_out_connectors went as %+v", sent)
	}
	if !strings.Contains(logged.String(), "connector=c64 ") {
		t.Errorf("no warning for the 65th in %q", logged.String())
	}
}

// Two entries left out: each is warned about, and the hash covers both, so finding one of the
// connections later still changes it.
func TestSyncOfTwoMissingConnectionsWarnsOfBothAndAddsOneOnceItExists(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connection: company\n  - name: chat\n    connection: team\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	var logged bytes.Buffer
	if _, err := agentOn(t, router, Options{Dir: root, Logger: slog.New(slog.NewTextHandler(&logged, nil))}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"connector=issues connection=company", "connector=chat connection=team"} {
		if !strings.Contains(logged.String(), want) {
			t.Errorf("no warning with %q in %q", want, logged.String())
		}
	}

	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("conn-1", "company"))
	})
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 {
		t.Fatalf("%d syncs were sent", len(router.syncs))
	}
	if router.syncs[0].Hash == router.syncs[1].Hash {
		t.Errorf("both syncs sent hash %q, so the router keeps the first", router.syncs[0].Hash)
	}
	sent := router.syncs[1].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Name != "issues" {
		t.Errorf("connectors went as %+v", sent)
	}
}

func TestSyncRefusesARequiredConnectorWhoseConnectionIsMissing(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connection: missing\n    required: true\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	agent := agentOn(t, router, Options{Dir: root})

	_, err := agent.Sync(t.Context())
	if err == nil || !strings.Contains(err.Error(), `connector "issues"`) || !strings.Contains(err.Error(), `"missing"`) ||
		!strings.Contains(err.Error(), "add the connection, or drop required") {
		t.Fatalf("sync error = %v", err)
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncRaw) != 0 {
		t.Errorf("a sync was sent: %s", router.syncRaw[0])
	}
}

func TestSyncFailsWhenConnectionsCannotBeListed(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connection: work\n")
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	agent, err := New(Options{Dir: root, LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: closed.URL, CustomerID: "acme"},
	})})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := agent.Sync(t.Context()); err == nil || !strings.Contains(err.Error(), "listing connections") {
		t.Fatalf("sync error = %v", err)
	}
}

// The router does nothing for a hash it already stored (acceleration/internal/api/sync.go:180),
// so the sync that finds the connection has to send another one.
func TestASyncThatLeftAConnectorOutAddsItOnceTheConnectionExists(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connection: company\n")
	router := newBackend(t)
	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("other", "work"))
	})
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.answerConnections(func(*http.Request) acceleration.ConnectionPage {
		return connectionPage("", listedConnection("conn-1", "company"))
	})
	for range 2 {
		if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 {
		t.Fatalf("three syncs, the last of an unchanged directory, sent %d requests", len(router.syncs))
	}
	if router.syncs[0].Hash == router.syncs[1].Hash {
		t.Errorf("both syncs sent hash %q, so the router keeps the first", router.syncs[0].Hash)
	}
	sent := router.syncs[1].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Connection.ConnectionId == nil || *(*sent)[0].Connection.ConnectionId != "conn-1" {
		t.Errorf("connectors went as %+v", sent)
	}
}

func TestSyncStopsWhenAConnectionPageRepeatsItsCursor(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connector: linear\n    connection: work\n")
	router := newBackend(t)
	router.answerConnections(func(r *http.Request) acceleration.ConnectionPage {
		if r.URL.Query().Get("cursor") == "" {
			return connectionPage("again", listedConnection("other", "nope"))
		}
		return connectionPage("again", listedConnection("the-one", "work"))
	})
	agent := agentOn(t, router, Options{Dir: root})

	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if _, err := agent.Sync(ctx); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.connectionQueries) != 2 {
		t.Fatalf("queries = %v", router.connectionQueries)
	}
	sent := router.syncs[0].Connectors
	if sent == nil || len(*sent) != 1 || (*sent)[0].Connection.ConnectionId == nil || *(*sent)[0].Connection.ConnectionId != "the-one" {
		t.Errorf("issues went as %+v", sent)
	}
}

func TestSyncRefusesALabelSharedByTwoConnections(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connection: work\n")
	router := newBackend(t)
	router.answerConnections(func(r *http.Request) acceleration.ConnectionPage {
		if r.URL.Query().Get("cursor") == "" {
			return connectionPage("2", listedConnection("a", "work"))
		}
		return connectionPage("", listedConnection("b", "work"))
	})
	agent := agentOn(t, router, Options{Dir: root})

	_, err := agent.Sync(t.Context())
	if err == nil || !strings.Contains(err.Error(), `connector "issues"`) || !strings.Contains(err.Error(), "matches 2") {
		t.Fatalf("sync error = %v", err)
	}
}

func listedConnection(id, label string) acceleration.Connection {
	return acceleration.Connection{Id: &id, Label: &label}
}

func connectionPage(next string, items ...acceleration.Connection) acceleration.ConnectionPage {
	copied := append([]acceleration.Connection(nil), items...)
	page := acceleration.ConnectionPage{Items: &copied}
	if next != "" {
		page.NextCursor = &next
	}
	return page
}
