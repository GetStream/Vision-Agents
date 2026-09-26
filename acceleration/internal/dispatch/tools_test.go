package dispatch

import (
	"context"
	"testing"
	"time"
)

var investigate = Tool{Name: "investigate_sdk", Description: "Read SDK source"}

func TestAHostedToolIsOfferedOnlyForItsConfigAndCustomer(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", 1)
	if err := pool.Host(worker, "support", []Tool{investigate}, time.Minute); err != nil {
		t.Fatal(err)
	}

	if tools, timeout := pool.HostedTools("acme", "support"); len(tools) != 1 || timeout != time.Minute {
		t.Fatalf("offered %+v for %s", tools, timeout)
	}
	if tools, _ := pool.HostedTools("acme", "sales"); len(tools) != 0 {
		t.Errorf("another config is offered %+v", tools)
	}
	if tools, _ := pool.HostedTools("globex", "support"); len(tools) != 0 {
		t.Errorf("another customer is offered %+v", tools)
	}
}

func TestAHostedCallIsAnsweredByTheWorkerRunningIt(t *testing.T) {
	pool := NewPool()
	worker, _ := pool.Register("acme", 1)
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

func TestAHostedCallFailsWhenItsWorkerGoesAway(t *testing.T) {
	pool := NewPool()
	worker, release := pool.Register("acme", 1)
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
	worker, _ := pool.Register("acme", 1)
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
