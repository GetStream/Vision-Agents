package dispatch

import (
	"context"
	"testing"
	"time"
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

func TestAHostedCallIsAnsweredByTheWorkerRunningIt(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", Registration{Capacity: 1})
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}

	go func() {
		call := <-worker.ToolCalls()
		worker.Resolve(call.ID, ToolResult{Output: "targetSdkVersion 35 in " + call.Arguments})
	}()
	output, err := pool.RunHosted(context.Background(), "acme", "support",
		ToolCall{ID: "call-1", SessionID: "s", Name: "investigate_sdk", Arguments: `{"sdk":"android"}`})
	if err != nil || output != `targetSdkVersion 35 in {"sdk":"android"}` {
		t.Fatalf("answered %q, %v", output, err)
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
		idle.Resolve(call.ID, ToolResult{Output: "read android"})
	}()
	output, err := pool.RunHosted(context.Background(), "acme", "support",
		ToolCall{ID: "call-1", Name: "investigate_sdk"})

	if err != nil || output != "read android" {
		t.Fatalf("answered %q, %v; the idle host should have taken it", output, err)
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
	if worker.Resolve("call-1", ToolResult{Output: "late"}) {
		t.Error("an answer after the timeout was taken")
	}
}
