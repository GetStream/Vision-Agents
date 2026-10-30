package dispatch

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

var investigate = Tool{Name: "investigate_sdk", Description: "Read SDK source"}

func TestAHostedToolIsOfferedOnlyForItsAgentAndCustomer(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}

	if tools, timeout := pool.HostedTools("acme", "support"); len(tools) != 1 || timeout != time.Minute {
		t.Fatalf("offered %+v for %s", tools, timeout)
	}
	if tools, _ := pool.HostedTools("acme", "sales"); len(tools) != 0 {
		t.Errorf("another agent is offered %+v", tools)
	}
	if tools, _ := pool.HostedTools("globex", "support"); len(tools) != 0 {
		t.Errorf("another customer is offered %+v", tools)
	}
}

func TestAWorkerHostsNoMoreToolsForAnAgentThanASessionOffers(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	tools := make([]Tool, maxHostedTools+1)
	for i := range tools {
		tools[i] = Tool{Name: fmt.Sprintf("tool_%d", i), Description: "Does one thing"}
	}

	if err := pool.Host(worker, "support", tools, time.Minute); err == nil {
		t.Fatalf("hosted %d tools for one agent", len(tools))
	}
	if err := pool.Host(worker, "support", tools[:maxHostedTools], time.Minute); err != nil {
		t.Fatalf("refused %d tools: %v", maxHostedTools, err)
	}
}

func TestAWorkerHostsToolsForABoundedNumberOfAgents(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	for i := range maxHostedAgents {
		if err := pool.Host(worker, fmt.Sprintf("agent-%d", i), []Tool{investigate}, time.Minute); err != nil {
			t.Fatalf("agent %d: %v", i, err)
		}
	}

	if err := pool.Host(worker, "one-more", []Tool{investigate}, time.Minute); err == nil {
		t.Fatalf("a worker hosted tools for %d agents", maxHostedAgents+1)
	}
	if err := pool.Host(worker, "agent-0", []Tool{investigate}, time.Minute); err != nil {
		t.Errorf("offering again for an agent it hosts was refused: %v", err)
	}
	other, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(other, "one-more", []Tool{investigate}, time.Minute); err != nil {
		t.Errorf("another worker was held to the first one's agents: %v", err)
	}
}

func TestAHostedCallIsAnsweredByTheWorkerRunningIt(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}

	go func() {
		call := <-worker.ToolCalls()
		worker.Resolve(call.ID, ToolResult{Output: llm.TextParts("targetSdkVersion 35 in " + call.Arguments)})
	}()
	output, err := pool.RunHosted(context.Background(), "acme", "support",
		ToolCall{ID: "call-1", SessionID: "s", Name: "investigate_sdk", Arguments: `{"sdk":"android"}`})
	if err != nil || llm.TextOf(output) != `targetSdkVersion 35 in {"sdk":"android"}` {
		t.Fatalf("answered %q, %v", llm.TextOf(output), err)
	}
}

func TestAHostedCallGoesToWhicheverHostHasTheLeastToAnswer(t *testing.T) {
	// Taking turns alone would send this to the first host, which is already investigating
	// four things and would answer this one after all of them.
	pool := NewPool()
	busy, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(busy, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}
	held, stop := context.WithCancel(context.Background())
	defer stop()
	for _, id := range []string{"held-1", "held-2", "held-3", "held-4"} {
		go pool.RunHosted(held, "acme", "support", ToolCall{ID: id, Name: "investigate_sdk"})
	}
	// Reading them off is what proves all four are registered and still unanswered.
	for range 4 {
		<-busy.ToolCalls()
	}

	idle, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(idle, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}
	go func() {
		call := <-idle.ToolCalls()
		idle.Resolve(call.ID, ToolResult{Output: llm.TextParts("read android")})
	}()
	output, err := pool.RunHosted(context.Background(), "acme", "support",
		ToolCall{ID: "call-1", Name: "investigate_sdk"})

	if err != nil || llm.TextOf(output) != "read android" {
		t.Fatalf("answered %q, %v; the idle host should have taken it", llm.TextOf(output), err)
	}
}

func TestAHostedCallFailsWhenItsWorkerGoesAway(t *testing.T) {
	pool := NewPool()
	worker, release := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}

	go func() {
		<-worker.ToolCalls()
		release()
	}()
	if _, err := pool.RunHosted(context.Background(), "acme", "support", ToolCall{ID: "call-1", Name: "investigate_sdk"}); err == nil {
		t.Fatal("a call whose worker left was answered")
	}
	if tools, _ := pool.HostedTools("acme", "support"); len(tools) != 0 {
		t.Errorf("a released worker still hosts %+v", tools)
	}
}

func TestAHostedCallNobodyRunsIsRefusedAtOnce(t *testing.T) {
	pool := NewPool()
	if _, err := pool.RunHosted(context.Background(), "acme", "support", ToolCall{ID: "call-1", Name: "investigate_sdk"}); err == nil {
		t.Fatal("a tool nobody hosts was answered")
	}
}

func TestAHostedCallIsBoundedByTheWorkersTimeout(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, 20*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.RunHosted(context.Background(), "acme", "support", ToolCall{ID: "call-1", Name: "investigate_sdk"}); err == nil {
		t.Fatal("a call nobody answered did not time out")
	}
	if worker.Resolve("call-1", ToolResult{Output: llm.TextParts("late")}) {
		t.Error("an answer after the timeout was taken")
	}
}

// progressHeard is a conversation that records what a hosted call told it.
type progressHeard struct {
	held    []string
	updates []string
	holds   bool
}

func (p *progressHeard) HoldTool(id string) bool {
	p.held = append(p.held, id)
	return p.holds
}

func (p *progressHeard) ToolUpdate(id, text string, verbatim bool) {
	p.updates = append(p.updates, id+": "+text)
}

func TestAWorkerRunningAHostedCallReachesItsConversation(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}
	heard := &progressHeard{holds: true}

	go func() {
		call := <-worker.ToolCalls()
		worker.Update(call.ID, "reading the gradle files", false)
		if _, held := worker.Hold(call.ID); !held {
			t.Error("a call its conversation holds was not held")
		}
		worker.Resolve(call.ID, ToolResult{Output: llm.TextParts("35")})
	}()
	if _, err := pool.RunHosted(context.Background(), "acme", "support",
		ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk", Progress: heard}); err != nil {
		t.Fatal(err)
	}

	if len(heard.updates) != 1 || heard.updates[0] != "call-1: reading the gradle files" || len(heard.held) != 1 {
		t.Fatalf("the conversation heard %+v and held %+v", heard.updates, heard.held)
	}
	if _, held := worker.Hold("call-1"); held {
		t.Error("an answered call was held")
	}
	worker.Update("call-1", "late", false)
	if len(heard.updates) != 1 {
		t.Errorf("an answered call's update reached the conversation: %+v", heard.updates)
	}
}
