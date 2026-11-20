package agents

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
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
	syncs   []acceleration.SyncAgentFilesRequest
	syncRaw [][]byte
	updates []string
	// refuseSkills, when set, is the reason POST /v1/agents/skills answers 400.
	refuseSkills string
	// refuseSkillEdits, when set, is the reason PUT /v1/agents/skills/{id} answers 400.
	refuseSkillEdits string
	// leavesOut, when set, is a connector every sync leaves out, as the router does for one
	// whose connection the app does not have.
	leavesOut *acceleration.LeftOutConnector
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
			Id: "config-" + request.Name, Name: request.Name, Instructions: request.Instructions,
			KnowledgeNamespace: request.KnowledgeNamespace, Skills: request.Skills,
			Subagent: request.Subagent, Tags: request.Tags, Sandbox: request.Sandbox,
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
			// As the router's storedSkillSandboxComplaint: a config left without a sandbox
			// is refused while a skill it names, as stored, still has scripts.
			if deref(request.Sandbox) == "" {
				for _, skill := range router.skills {
					if skill.ConfigId == config.Id && slices.Contains(deref(request.Skills), skill.Name) &&
						len(scripts(deref(skill.FilePaths))) > 0 {
						reply(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
							Message: fmt.Sprintf("skill %q runs scripts, so the agent needs a sandbox", skill.Name),
						}})
						return
					}
				}
			}
			router.configs[index].Instructions = request.Instructions
			router.configs[index].Sandbox = request.Sandbox
			router.configs[index].Skills = request.Skills
			reply(w, http.StatusOK, router.configs[index])
			return
		}
		reply(w, http.StatusNotFound, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: "no such config"}})
	})

	mux.HandleFunc("GET /v1/agents/skills", func(w http.ResponseWriter, r *http.Request) {
		router.mu.Lock()
		defer router.mu.Unlock()
		listed := []acceleration.Skill{}
		for _, skill := range router.skills {
			if configID := r.URL.Query().Get("config_id"); configID == "" || skill.ConfigId == configID {
				listed = append(listed, skill)
			}
		}
		reply(w, http.StatusOK, listed)
	})
	// A skill belongs to a config that exists, as the router's createSkill and updateSkill
	// insist, and its name is unique among that config's own skills.
	mux.HandleFunc("POST /v1/agents/skills", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SkillRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		if !router.skillAllowed(w, request) {
			return
		}
		for _, skill := range router.skills {
			if skill.ConfigId == request.ConfigId && skill.Name == request.Name {
				reply(w, http.StatusConflict, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
					Message: "this agent already has a skill with this name",
				}})
				return
			}
		}
		stored := acceleration.Skill{
			Id: "skill-" + request.ConfigId + "-" + request.Name, ConfigId: request.ConfigId,
			Name: request.Name, Description: request.Description,
			Instructions: request.Instructions, DeadlineMs: request.DeadlineMs,
			FilePaths: filePaths(request.Files), CreatedAt: time.Now(), UpdatedAt: time.Now(),
		}
		router.skills = append(router.skills, stored)
		reply(w, http.StatusCreated, stored)
	})
	mux.HandleFunc("PUT /v1/agents/skills/{id}", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SkillRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		if router.refuseSkillEdits != "" {
			reply(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: router.refuseSkillEdits}})
			return
		}
		if !router.skillAllowed(w, request) {
			return
		}
		router.updates = append(router.updates, r.PathValue("id"))
		for index, skill := range router.skills {
			if skill.Id != r.PathValue("id") {
				continue
			}
			router.skills[index].Description = request.Description
			router.skills[index].Instructions = request.Instructions
			router.skills[index].FilePaths = filePaths(request.Files)
			router.skills[index].UpdatedAt = time.Now()
			reply(w, http.StatusOK, router.skills[index])
			return
		}
		reply(w, http.StatusNotFound, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: "no such skill"}})
	})

	mux.HandleFunc("POST /v1/agents/sync/files", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var request acceleration.SyncAgentFilesRequest
		_ = json.Unmarshal(raw, &request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.syncRaw = append(router.syncRaw, raw)
		router.syncs = append(router.syncs, request)
		// Whatever the router makes of the request: the same one hashes the same.
		hashed := sha256.Sum256(raw)
		hash := hex.EncodeToString(hashed[:])
		result := acceleration.SyncAgentResult{}
		name := deref(request.Name)
		result.Config = acceleration.AgentConfig{
			Id: "config-" + name, Name: name, SyncHash: &hash, CreatedAt: time.Now(), UpdatedAt: time.Now(),
		}
		if router.leavesOut != nil {
			result.Config.LeftOutConnectors = &[]acceleration.LeftOutConnector{*router.leavesOut}
			result.Warnings = &[]string{"connector " + router.leavesOut.Name + " was left out"}
		}
		router.configs = append(router.configs[:0:0], result.Config)
		reply(w, http.StatusOK, result)
	})

	router.Server = httptest.NewServer(mux)
	t.Cleanup(router.Close)
	return router
}

// skillAllowed answers a skill the router would refuse, with the router's own words, and
// says whether it was fine. The caller holds the lock.
func (b *backend) skillAllowed(w http.ResponseWriter, request acceleration.SkillRequest) bool {
	if strings.TrimSpace(request.ConfigId) == "" {
		reply(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
			Message: "a skill belongs to an agent config, so one has to be named",
		}})
		return false
	}
	if b.refuseSkills != "" {
		reply(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: b.refuseSkills}})
		return false
	}
	for _, config := range b.configs {
		if config.Id != request.ConfigId {
			continue
		}
		// As the router's skillSandboxComplaint: scripts need the config's sandbox.
		if paths := filePaths(request.Files); deref(config.Sandbox) == "" && len(scripts(deref(paths))) > 0 {
			reply(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
				Message: fmt.Sprintf("skill %q runs scripts, so the agent needs a sandbox", request.Name),
			}})
			return false
		}
		return true
	}
	reply(w, http.StatusNotFound, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{Message: "no such agent config"}})
	return false
}

// filePaths are the paths of a skill's files, as the router lists them.
func filePaths(files *map[string]string) *[]string {
	if files == nil {
		return nil
	}
	paths := slices.Sorted(maps.Keys(*files))
	return &paths
}

// scripts are the paths the router's harness.Scripts runs rather than reads: any file that
// is not .md, .mdx or .txt.
func scripts(paths []string) []string {
	var run []string
	for _, file := range paths {
		if ext := strings.ToLower(filepath.Ext(file)); ext != ".md" && ext != ".mdx" && ext != ".txt" {
			run = append(run, file)
		}
	}
	return run
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

// zoneOf is the timezone the one session opened on router was asked for, or "" when it was
// asked for none.
func zoneOf(t *testing.T, router *worked) string {
	t.Helper()
	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.opened) != 1 {
		t.Fatalf("the router was asked to open %d sessions", len(router.opened))
	}
	zone := router.opened[0].Timezone
	if zone == nil {
		return ""
	}
	if *zone == "" {
		t.Fatal("the session was sent an empty timezone rather than none")
	}
	return *zone
}

func TestASessionThatNamesNoTimezoneIsInTheAgents(t *testing.T) {
	router := newWorked(t, nil)
	agent, err := New(Options{
		Name:     "jean",
		Timezone: "Europe/Lisbon",
		LLM: stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := agent.Chat(t.Context(), SessionOptions{Title: "Order 1042"})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if zone := zoneOf(t, router); zone != "Europe/Lisbon" {
		t.Errorf("the session was opened in %q", zone)
	}
}

func TestASessionsOwnTimezoneWinsOverTheAgents(t *testing.T) {
	router := newWorked(t, nil)
	agent, err := New(Options{
		Name:     "jean",
		Timezone: "Europe/Lisbon",
		LLM: stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{Timezone: "America/New_York"})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if zone := zoneOf(t, router); zone != "America/New_York" {
		t.Errorf("the session was opened in %q", zone)
	}
	if agent.options.Timezone != "Europe/Lisbon" {
		t.Errorf("the session wrote its zone through to the agent: %q", agent.options.Timezone)
	}
}

func TestASessionWithNoTimezoneAnywhereSendsNone(t *testing.T) {
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

	session, err := agent.Join(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if zone := zoneOf(t, router); zone != "" {
		t.Errorf("a session nobody named a zone for was opened in %q", zone)
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

func TestSyncStoresTheSkillsUnderTheConfigAndEditsThemTheSecondTime(t *testing.T) {
	router := newBackend(t)
	agent := agentOn(t, router, Options{Name: "jean", Harness: &Harness{Skills: []Skill{
		{Name: "think", Description: "Work it out", Instructions: "Reason it through."},
	}}})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if stored.Skills == nil || len(*stored.Skills) != 1 || (*stored.Skills)[0] != "think" {
		t.Fatalf("the config names %+v", stored.Skills)
	}

	agent.options.Harness.Skills[0].Instructions = "Reason it through twice."
	again, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if again.Id != stored.Id {
		t.Errorf("syncing twice stored a second config: %s then %s", stored.Id, again.Id)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.configs) != 1 {
		t.Fatalf("syncing twice left %d configs: %+v", len(router.configs), router.configs)
	}
	if len(router.skills) != 1 {
		t.Fatalf("syncing twice left %d skills: %+v", len(router.skills), router.skills)
	}
	if router.skills[0].ConfigId != stored.Id || router.skills[0].Instructions != "Reason it through twice." {
		t.Errorf("the skill is %+v, where the config is %s", router.skills[0], stored.Id)
	}
}

func TestSyncEditsOnlyItsOwnSkillWhenAnotherAgentHasOneByTheSameName(t *testing.T) {
	router := newBackend(t)
	think := func(instructions string) *Harness {
		return &Harness{Skills: []Skill{{Name: "think", Description: "Work it out", Instructions: instructions}}}
	}
	anna := agentOn(t, router, Options{Name: "anna", Harness: think("Anna's way.")})
	jean := agentOn(t, router, Options{Name: "jean", Harness: think("Jean's way.")})
	for _, agent := range []*Agent{anna, jean} {
		if _, err := agent.Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	jean.options.Harness.Skills[0].Instructions = "Jean's other way."
	if _, err := jean.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.skills) != 2 {
		t.Fatalf("the skills stored are %+v", router.skills)
	}
	if router.skills[0].ConfigId != "config-anna" || router.skills[0].Instructions != "Anna's way." {
		t.Errorf("another agent's skill was changed: %+v", router.skills[0])
	}
	if router.skills[1].ConfigId != "config-jean" || router.skills[1].Instructions != "Jean's other way." {
		t.Errorf("the agent's own skill is %+v", router.skills[1])
	}
}

func TestSyncSaysWhyTheRouterRefusedASkill(t *testing.T) {
	router := newBackend(t)
	router.refuseSkills = "a skill may bring at most 64 files"
	agent := agentOn(t, router, Options{Name: "jean", Harness: &Harness{Skills: []Skill{
		{Name: "think", Description: "Work it out", Instructions: "Reason it through."},
	}}})

	_, err := agent.Sync(t.Context())

	if err == nil || !strings.Contains(err.Error(), "at most 64 files") {
		t.Errorf("a skill the router refused came back as %v", err)
	}
}

func TestSyncSaysWhyTheRouterRefusedASkillEdit(t *testing.T) {
	// With a sandbox stored the skills are edited before the config, without one after it.
	for name, sandbox := range map[string]*Sandbox{"without a sandbox": nil, "with a sandbox": Daytona()} {
		t.Run(name, func(t *testing.T) {
			router := newBackend(t)
			agent := agentOn(t, router, Options{Name: "jean", Harness: &Harness{Sandbox: sandbox, Skills: []Skill{
				{Name: "think", Description: "Work it out", Instructions: "Reason it through."},
			}}})
			if _, err := agent.Sync(t.Context()); err != nil {
				t.Fatal(err)
			}
			router.mu.Lock()
			router.refuseSkillEdits = "a skill may bring at most 64 files"
			router.mu.Unlock()
			_, err := agent.Sync(t.Context())
			if err == nil || !strings.Contains(err.Error(), "at most 64 files") {
				t.Errorf("an edit the router refused came back as %v", err)
			}
		})
	}
}

func TestOneSyncDropsTheSandboxAndASkillsScriptsTogether(t *testing.T) {
	router := newBackend(t)
	harness := &Harness{Sandbox: Daytona(), Skills: []Skill{{
		Name: "forms", Description: "Fill a form", Instructions: "Fill it in.",
		Files: map[string]string{"scripts/fill.py": "print(1)", "reference.md": "# Forms"},
	}}}
	agent := agentOn(t, router, Options{Name: "jean", Harness: harness})
	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	harness.Sandbox = nil
	harness.Skills[0].Files = map[string]string{"reference.md": "# Forms"}
	synced, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatalf("the sync that drops the sandbox with the scripts was refused: %v", err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if synced.Sandbox != nil || router.configs[0].Sandbox != nil {
		t.Errorf("the config kept its sandbox: %v", router.configs[0].Sandbox)
	}
	if paths := deref(router.skills[0].FilePaths); !slices.Equal(paths, []string{"reference.md"}) {
		t.Errorf("the skill holds %v", paths)
	}
	// Each written once: the skill while the sandbox is still stored, then the config.
	if want := []string{"skill-config-jean-forms", "config-jean"}; !slices.Equal(router.updates, want) {
		t.Errorf("the edits were %v, not %v", router.updates, want)
	}
}

func TestOneSyncAddsTheSandboxAndASkillsScriptsTogether(t *testing.T) {
	router := newBackend(t)
	harness := &Harness{Skills: []Skill{{
		Name: "forms", Description: "Fill a form", Instructions: "Fill it in.",
		Files: map[string]string{"reference.md": "# Forms"},
	}}}
	agent := agentOn(t, router, Options{Name: "jean", Harness: harness})
	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	harness.Sandbox = Daytona()
	harness.Skills[0].Files = map[string]string{"scripts/fill.py": "print(1)", "reference.md": "# Forms"}
	synced, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatalf("the sync that adds the sandbox with the scripts was refused: %v", err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if deref(synced.Sandbox) != acceleration.Daytona {
		t.Errorf("the config's sandbox is %v", synced.Sandbox)
	}
	if paths := deref(router.skills[0].FilePaths); !slices.Equal(paths, []string{"reference.md", "scripts/fill.py"}) {
		t.Errorf("the skill holds %v", paths)
	}
}

func TestASyncThatDropsTheSandboxButKeepsScriptsIsRefusedAndKeepsTheSandbox(t *testing.T) {
	router := newBackend(t)
	harness := &Harness{Sandbox: Daytona(), Skills: []Skill{{
		Name: "forms", Description: "Fill a form", Instructions: "Fill it in.",
		Files: map[string]string{"scripts/fill.py": "print(1)"},
	}}}
	agent := agentOn(t, router, Options{Name: "jean", Harness: harness})
	if _, err := agent.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	harness.Sandbox = nil
	_, err := agent.Sync(t.Context())

	if err == nil || !strings.Contains(err.Error(), "needs a sandbox") {
		t.Errorf("a config left with scripts and no sandbox came back as %v", err)
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if deref(router.configs[0].Sandbox) != acceleration.Daytona {
		t.Errorf("the refused sync left the sandbox as %v", router.configs[0].Sandbox)
	}
}

func TestDefineSkillsStoresThemUnderTheConfigNamed(t *testing.T) {
	router := newBackend(t)
	backend := stream.Backend{URL: router.URL, CustomerID: "acme"}
	config, err := DefineAgent(t.Context(), backend, acceleration.AgentConfigRequest{Name: "jean"})
	if err != nil {
		t.Fatal(err)
	}
	skills := []Skill{{Name: "think", Description: "Work it out", Instructions: "Reason it through."}}

	if err := DefineSkills(t.Context(), backend, config.Id, skills); err != nil {
		t.Fatal(err)
	}
	if err := DefineSkills(t.Context(), backend, "nobody", skills); err == nil {
		t.Error("skills were stored for a config that does not exist")
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.skills) != 1 || router.skills[0].ConfigId != config.Id {
		t.Errorf("the skills stored are %+v", router.skills)
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

func TestSyncSendsADirectorysFilesAsTheyAre(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	held := map[string]string{
		"agent.yaml":             "llm: openai/gpt-5.6\ntags:\n  team: support\n",
		"instructions.md":        "You are Jean.\n",
		"guardrail.md":           "---\ntype: llm\n---\nNo medical advice.\n",
		"skills/think.md":        "---\ndescription: Work it out\n---\nReason it through.\n",
		"skills/pdf/SKILL.md":    "---\ndescription: Fill a form\n---\nRun the script.\n",
		"skills/pdf/fill.py":     "print('filled')\n",
		"knowledge/pricing.md":   "# Pricing\n\nA call costs a penny.\n",
		"knowledge/urls.yaml":    "- url: https://example.com/plans\n  title: Plans\n",
		"simulations/lunch.yaml": "- name: lunch\n  scenario: Order.\n  assertion: Ordered.\n",
		"ui/components.json":     "[]\n",
	}
	for path, content := range held {
		write(t, root, path, content)
	}

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
	if deref(synced.Name) != "jean" {
		t.Errorf("the directory was synced as %q, not after its folder", deref(synced.Name))
	}
	sent := map[string]string{}
	for _, file := range synced.Files {
		sent[file.Path] = file.Content
	}
	if !maps.Equal(sent, held) {
		t.Errorf("the files sent are %+v", sent)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(router.syncRaw[0], &body); err != nil {
		t.Fatal(err)
	}
	if len(body) != 2 {
		t.Errorf("the request says more than the name and the files, which is the router's to read: %s", router.syncRaw[0])
	}
	if stamp := readStamp(root); stamp.SyncHash != deref(stored.SyncHash) || stamp.Hash == "" {
		t.Errorf("%s records %+v, and the router answered %q", AgentStamp, stamp, deref(stored.SyncHash))
	}
}

func TestAStampedDirectoryIsSyncedAgainOnceTheStoredConfigIsNoLongerWhatWasSynced(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	router := newBackend(t)

	for range 2 {
		if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	// Somebody imports another directory in the dashboard, or edits the config, which clears it.
	router.mu.Lock()
	edited := "imported"
	router.configs[0].SyncHash = &edited
	router.mu.Unlock()
	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 {
		t.Errorf("the directory was sent %d times: once, not again while the config was as synced, and again once it was not", len(router.syncs))
	}
}

func TestWhatTheCodeSetsBesideADirectoryGoesWithItsFilesAndWinsOverThem(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	router := newBackend(t)

	if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
		t.Fatal(err)
	}
	written := Options{
		Dir: root, Instructions: "You are somebody else.", Guardrail: "Refuse medical advice.",
		CostTracking: map[string]string{"team": "support", "customer": "123"},
		Harness: &Harness{
			Subagents: map[string]string{"default": "openai/gpt-5.6-sol"}, Sandbox: Daytona(),
			Skills: []Skill{
				{Name: "think", Description: "Work it out", Instructions: "Reason.", Deadline: 30 * time.Second},
				{Name: "fill", Description: "Fill a form", Instructions: "Run it.", Files: map[string]string{"fill.py": "print()\n"}},
			},
		},
	}
	for range 2 {
		if _, err := agentOn(t, router, written).Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 2 {
		t.Fatalf("what the code set was synced %d times, not once after the directory's own", len(router.syncs)-1)
	}
	plain, synced := router.syncs[0], router.syncs[1]
	if plain.Instructions != nil || plain.Guardrail != nil || plain.Tags != nil || plain.Skills != nil || plain.Subagent != nil {
		t.Errorf("a directory with nothing set in code was sent with %+v", plain)
	}
	if deref(synced.Instructions) != "You are somebody else." || deref(synced.Guardrail) != "Refuse medical advice." {
		t.Errorf("the instructions and guardrail written in code went as %v and %v", synced.Instructions, synced.Guardrail)
	}
	if deref(synced.Subagent) != "openai/gpt-5.6-sol" || deref(synced.Sandbox) != "daytona" || synced.Harness != nil {
		t.Errorf("the harness went as %v, %v and %v", synced.Harness, synced.Subagent, synced.Sandbox)
	}
	if !maps.Equal(deref(synced.Tags), written.CostTracking) {
		t.Errorf("the cost labels went as %v", synced.Tags)
	}
	if skills := deref(synced.Skills); len(skills) != 2 || skills[1].Name != "fill" || deref(skills[1].Files)["fill.py"] != "print()\n" ||
		skills[0].Name != "think" || deref(skills[0].DeadlineMs) != 30000 {
		t.Errorf("the skills written in code went as %+v", synced.Skills)
	}
	if len(synced.Files) != 2 || synced.Files[1].Content != "You are Jean.\n" {
		t.Errorf("the directory's own files went as %+v", synced.Files)
	}
}

func TestADirectoryThatLeftAConnectorOutIsSentAgainUntilItsConnectionIsFound(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nconnectors:\n  - {name: linear, connection: company}\n")
	router := newBackend(t)
	router.leavesOut = &acceleration.LeftOutConnector{Name: "linear", ConnectorId: "linear", Connection: "company"}

	for range 2 {
		if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	router.mu.Lock()
	router.leavesOut = nil
	router.mu.Unlock()
	for range 2 {
		if _, err := agentOn(t, router, Options{Dir: root}).Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.syncs) != 3 {
		t.Errorf("the directory was sent %d times: twice while a connector was left out, once to bind it, and not again", len(router.syncs))
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
// 80773902 (connectors/tool-patterns), before connectors were read, again on
// 259da758b once it named no plugins, and on 1af941985 once it named no MCP servers.
// The body now also carries calls, which a sync always sends; an older router ignores it
// (Huma's AllowAdditionalPropertiesByDefault).
const controlYAML = "name: concierge\nllm: openai/gpt-5.6\n"

func pointer[T any](value T) *T { return &value }
