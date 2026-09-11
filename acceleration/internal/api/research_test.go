package api

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestSandboxProfileMapping(t *testing.T) {
	name := "support"
	config := storedConfig(AgentConfigRequest{Name: "support", SandboxProfile: &name}, "acme")
	if config.SandboxProfile != name || agentConfigOf(config).SandboxProfile == nil {
		t.Fatal("profile lost in config conversion")
	}
	spec := specOf(CreateSessionRequest{}, "acme", &config)
	if spec.SandboxProfile != name {
		t.Fatal("stored profile not used")
	}
	other := "other"
	spec = specOf(CreateSessionRequest{SandboxProfile: &other}, "acme", &config)
	if spec.SandboxProfile != other {
		t.Fatal("session override lost")
	}
	if got := session.FromConfig(store.AgentConfig{Sandbox: "daytona"}); got.Sandbox != "daytona" || got.SandboxProfile != "" {
		t.Fatal("legacy sandbox changed")
	}
}
func TestResearchProgressFrame(t *testing.T) {
	f, ok := frameOf(session.ResearchProgress{ToolCallID: "call", Phase: "answered", ElapsedMS: 20, VerifiedCitations: 1})
	if !ok || f["type"] != "research_progress" || f["verified_citations"] != 1 {
		t.Fatal(f)
	}
}

func TestPendingResponseFrame(t *testing.T) {
	f, ok := frameOf(agent.Responded{PendingWork: true, Text: "Looking it up"})
	if !ok || f["pending_work"] != true {
		t.Fatal(f)
	}
}
