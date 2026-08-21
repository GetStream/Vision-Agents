package agents

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
	"github.com/GetStream/Vision-Agents/agents-core-go/stream"
)

// backend is a stand-in for the acceleration router's configuration paths: enough of them
// to see what an agent stores and how it edits what is already there.
type backend struct {
	*httptest.Server

	mu        sync.Mutex
	configs   []acceleration.AgentConfig
	skills    []acceleration.Skill
	knowledge []acceleration.IngestKnowledgeRequest
	updates   []string
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

	mux.HandleFunc("POST /v1/agents/knowledge", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.IngestKnowledgeRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		defer router.mu.Unlock()
		router.knowledge = append(router.knowledge, request)
		reply(w, http.StatusOK, acceleration.IngestedKnowledge{
			Namespace: request.Namespace, Documents: len(request.Documents),
			Passages: len(request.Documents),
		})
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

func TestAnAgentNeedsAnLLMAndAName(t *testing.T) {
	if _, err := New(Options{Name: "jean"}); err == nil {
		t.Error("an agent with nothing answering is not an agent")
	}
	if _, err := New(Options{LLM: stream.Accelerated(stream.Config{})}); err == nil {
		t.Error("an agent has to be called something")
	}
}

func TestAnAgentJoinsUnderAUserIDDerivedFromItsName(t *testing.T) {
	router := newBackend(t)
	agent := agentOn(t, router, Options{Name: "Jean Le Bot"})

	if agent.options.UserID != "jean-le-bot" {
		t.Errorf("the agent joins as %q", agent.options.UserID)
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
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")

	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()

	if len(router.skills) != 1 || router.skills[0].Name != "think" {
		t.Errorf("the skills stored are %+v", router.skills)
	}
	if len(router.knowledge) != 1 {
		t.Fatalf("the knowledge posted is %+v", router.knowledge)
	}
	if router.knowledge[0].Namespace != "jean" {
		t.Errorf("the knowledge went to %q", router.knowledge[0].Namespace)
	}
	if stored.KnowledgeNamespace == nil || *stored.KnowledgeNamespace != "jean" {
		t.Errorf("the config does not point at the knowledge: %+v", stored)
	}
	if stored.Skills == nil || (*stored.Skills)[0] != "think" {
		t.Errorf("the config does not name the skill: %+v", stored.Skills)
	}
}

func TestADirectorysSkillsAreWhatTheAgentJoinsWith(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")

	router := newBackend(t)
	agent := agentOn(t, router, Options{Dir: root})

	var call stream.Call
	agent.options.Harness.apply(&call)

	if call.Skills == nil || len(*call.Skills) != 1 || (*call.Skills)[0].Name != "think" {
		t.Errorf("the session would be created with %+v", call.Skills)
	}
}

func TestADirectoryDoesNotWriteThroughToAHarnessSharedWithAnotherAgent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
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

func TestAHarnessRendersIntoTheCallItConfigures(t *testing.T) {
	harness := &Harness{
		UseSkills: true,
		Subagents: map[string]string{"default": "openai/gpt-5.6-sol"},
		VM:        Daytona(),
		Tasks:     3,
		Skills: []Skill{{
			Name: "think", Description: "Work it out",
			Instructions: "Reason it through.", Deadline: 30 * time.Second,
		}},
	}

	var call stream.Call
	harness.apply(&call)

	if call.Subagent != "openai/gpt-5.6-sol" || call.Tasks != 3 || call.Sandbox != "daytona" {
		t.Errorf("the call was configured as %+v", call)
	}
	if call.Skills == nil || len(*call.Skills) != 1 {
		t.Fatalf("the skills are %+v", call.Skills)
	}
	skill := (*call.Skills)[0]
	if skill.Name != "think" || skill.DeadlineMs == nil || *skill.DeadlineMs != 30000 {
		t.Errorf("the skill went over as %+v", skill)
	}
}

func TestTheDefaultHarnessLeavesTheBuiltInSkillsAlone(t *testing.T) {
	var call stream.Call
	DefaultHarness().apply(&call)

	if call.Skills != nil {
		t.Errorf("the built-in set was replaced by %+v", call.Skills)
	}
}

func TestAHarnessAskingForNoSkillsTurnsDelegationOff(t *testing.T) {
	var call stream.Call
	(&Harness{UseSkills: false}).apply(&call)

	if call.Skills == nil || len(*call.Skills) != 0 {
		t.Errorf("the skills are %+v, want an empty list rather than none at all", call.Skills)
	}
}

func TestSeveralSubagentsWithNoDefaultIsRefused(t *testing.T) {
	harness := &Harness{Subagents: map[string]string{"fast": "a", "slow": "b"}}

	if err := harness.Validate(); err == nil {
		t.Fatal("which one runs skills would be decided by map iteration order")
	}
}
