//go:build e2e

// Package agentscorego's end-to-end suite drives a running acceleration router through the
// SDK, rather than through a stand-in for one.
//
// It is behind a tag because it needs a router, which needs credentials and a database. Point
// it at one and run it:
//
//	STREAM_ACCELERATION_URL=http://localhost:8099 \
//	STREAM_ACCELERATION_CUSTOMER_ID=e2e \
//	go test -tags e2e -v ./...
//
// Anything the deployment cannot do is skipped rather than failed: a router with no
// knowledge provider is a valid router, and so is one with no telephony.
package agentscorego

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/agents-core-go/agents"
	"github.com/GetStream/Vision-Agents/agents-core-go/stream"
)

// router is where the acceleration backend under test is, or the suite skips.
func router(t *testing.T) stream.Backend {
	t.Helper()
	if os.Getenv(stream.URLEnv) == "" {
		t.Skipf("%s is not set, so there is no router to test against", stream.URLEnv)
	}

	backend, err := stream.Backend{}.Resolve()
	if err != nil {
		t.Skip(err)
	}
	return backend
}

// chatting builds an agent that holds its conversation in writing, which needs no Stream
// credentials and no call.
func chatting(t *testing.T, options agents.Options) *agents.Agent {
	t.Helper()

	if options.LLM == nil {
		options.LLM = stream.Accelerated(stream.Config{Backend: router(t)})
	}
	if options.Name == "" {
		options.Name = "e2e-jean"
	}

	agent, err := agents.New(options)
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

// waitFor reads the session until one of the kinds arrives, or the test runs out of patience.
func waitFor(t *testing.T, session *agents.Session, kinds ...string) stream.Event {
	t.Helper()

	wanted := map[string]bool{}
	for _, kind := range kinds {
		wanted[kind] = true
	}

	deadline := time.After(90 * time.Second)
	for {
		select {
		case event, open := <-session.Events():
			if !open {
				t.Fatalf("the conversation ended before any of %v", kinds)
			}
			t.Logf("event %s %s", event.Kind, event.Text)
			if event.Kind == "error" && !wanted["error"] {
				t.Fatalf("the backend reported: %s", event.Error)
			}
			if wanted[event.Kind] {
				return event
			}
		case <-deadline:
			t.Fatalf("none of %v arrived", kinds)
		}
	}
}

func TestTheRouterIsReachableAndSaysWhatItCanDo(t *testing.T) {
	backend := router(t)

	client, err := backend.Client()
	if err != nil {
		t.Fatal(err)
	}
	health, err := client.GetHealthWithResponse(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if health.JSON200 == nil {
		t.Fatalf("the router answered %s: %s", health.Status(), health.Body)
	}
	t.Logf("the router reports %v", health.JSON200.Dependencies)
}

func TestAConversationInWritingIsAnsweredByAModel(t *testing.T) {
	agent := chatting(t, agents.Options{
		Instructions: "You are Jean. Answer in one short sentence.",
	})

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if err := session.Respond("What is the capital of France?"); err != nil {
		t.Fatal(err)
	}

	answered := waitFor(t, session, "responded")
	if answered.Text == "" {
		t.Fatal("the model answered with nothing")
	}
	if !strings.Contains(strings.ToLower(answered.Text), "paris") {
		t.Errorf("the model said %q, which does not answer the question", answered.Text)
	}
}

func TestAFunctionRegisteredHereIsRunHereWhenTheModelAsksForIt(t *testing.T) {
	llm := stream.Accelerated(stream.Config{Backend: router(t)})

	asked := make(chan string, 4)
	err := agents.RegisterFunction(llm, "get_weather",
		"Get the current weather for a location. Always use this rather than guessing.",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
		}) (any, error) {
			asked <- in.Location
			return "It is 20 degrees and raining sideways in " + in.Location + ".", nil
		})
	if err != nil {
		t.Fatal(err)
	}

	agent := chatting(t, agents.Options{
		LLM:          llm,
		Instructions: "You are Jean. Use your tools to answer, and keep replies to one sentence.",
	})

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if err := session.Respond("What is the weather in Boulder, CO right now?"); err != nil {
		t.Fatal(err)
	}

	// The turn the tool was asked for in ends with nothing said: the backend folds a
	// successful result into the history and stays quiet rather than answering out of it,
	// which is what a tool that pressed a menu key wants. So the result is read on the
	// next turn.
	waitFor(t, session, "tool_ran")

	select {
	case location := <-asked:
		t.Logf("the model asked about %q", location)
		if !strings.Contains(strings.ToLower(location), "boulder") {
			t.Errorf("the model filled in %q", location)
		}
	default:
		t.Fatal("the function was never run")
	}

	if err := session.Respond("So what is it like there?"); err != nil {
		t.Fatal(err)
	}
	answered := waitFor(t, session, "responded")
	if !strings.Contains(strings.ToLower(answered.Text), "rain") &&
		!strings.Contains(answered.Text, "20") {
		t.Errorf("the answer %q does not use what the function returned", answered.Text)
	}
}

func TestSayingSomethingSkipsTheModelEntirely(t *testing.T) {
	agent := chatting(t, agents.Options{Instructions: "You are Jean."})

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if err := session.Say("We close in five minutes."); err != nil {
		t.Fatal(err)
	}

	said := waitFor(t, session, "responded", "spoke")
	if !strings.Contains(said.Text, "five minutes") {
		t.Errorf("what went out was %q", said.Text)
	}
}

func TestASessionIsRecordedAndCanBeReadBack(t *testing.T) {
	backend := router(t)
	agent := chatting(t, agents.Options{Instructions: "You are Jean. Be brief."})

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	client, err := backend.Client()
	if err != nil {
		t.Fatal(err)
	}
	listed, err := client.ListSessionsWithResponse(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if listed.JSON200 == nil {
		t.Fatalf("the router answered %s: %s", listed.Status(), listed.Body)
	}

	for _, running := range *listed.JSON200 {
		if running.Id == session.ID() {
			return
		}
	}
	t.Errorf("%s is not among the sessions the router is holding", session.ID())
}

func TestSyncStoresTheAgentAndEditsItTheSecondTime(t *testing.T) {
	agent := chatting(t, agents.Options{
		Name:         "e2e-sync",
		Instructions: "You are Jean, the first time.",
		Harness: &agents.Harness{
			UseSkills: true,
			Skills: []agents.Skill{{
				Name:         "e2e-think",
				Description:  "Work something out before answering",
				Instructions: "Take your time and reason it through.",
				Deadline:     30 * time.Second,
			}},
		},
	})

	stored, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if stored.Name != "e2e-sync" {
		t.Fatalf("the config was stored as %+v", stored)
	}
	if stored.Skills == nil || (*stored.Skills)[0] != "e2e-think" {
		t.Errorf("the config does not name the skill: %+v", stored.Skills)
	}

	again, err := agent.Sync(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if again.Id != stored.Id {
		t.Errorf("syncing twice stored a second config: %s then %s", stored.Id, again.Id)
	}
	if !again.UpdatedAt.After(stored.CreatedAt) && again.UpdatedAt != stored.UpdatedAt {
		t.Logf("stored at %s, updated at %s", stored.CreatedAt, again.UpdatedAt)
	}
}

func TestAStoredConfigIsWhatASessionStartsFrom(t *testing.T) {
	backend := router(t)

	configured := chatting(t, agents.Options{
		Name:         "e2e-config",
		Instructions: "You are Bernard. Whatever you are asked, reply with exactly: BERNARD HERE.",
	})
	if _, err := configured.Sync(t.Context()); err != nil {
		t.Fatal(err)
	}

	// A second agent that is told nothing but the config's name, so what it answers as can
	// only have come from what was stored.
	byName := stream.Accelerated(stream.Config{Agent: "e2e-config", Backend: backend})
	agent, err := agents.New(agents.Options{Name: "e2e-config-user", LLM: byName})
	if err != nil {
		t.Fatal(err)
	}

	session, err := agent.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if err := session.Respond("Hello, who is this?"); err != nil {
		t.Fatal(err)
	}

	answered := waitFor(t, session, "responded")
	if !strings.Contains(strings.ToUpper(answered.Text), "BERNARD") {
		t.Errorf("the session answered %q, so it did not start from the stored config", answered.Text)
	}
}

func TestADirectoryIsPushedAndLookedUpAgain(t *testing.T) {
	backend := router(t)

	root := t.TempDir() + "/e2e-knowledge"
	writeFile(t, root+"/instructions.md",
		"You are Jean. Look things up rather than guessing, and answer in one sentence.")
	writeFile(t, root+"/knowledge/refunds.md",
		"# Refunds\n\nAcme refunds any order within 47 days of delivery. "+
			"The window is 47 days, not 30, because Acme's founder was superstitious.\n")

	agent, err := agents.New(agents.Options{
		Dir: root,
		LLM: stream.Accelerated(stream.Config{Backend: backend}),
	})
	if err != nil {
		t.Fatal(err)
	}

	stored, err := agent.Sync(t.Context())
	if err != nil {
		if strings.Contains(err.Error(), "no provider configured") {
			t.Skip("this deployment has no knowledge provider, so there is nothing to fill")
		}
		t.Fatal(err)
	}
	if stored.KnowledgeNamespace == nil || *stored.KnowledgeNamespace != "e2e-knowledge" {
		t.Fatalf("the config does not point at the knowledge: %+v", stored)
	}
	t.Logf("filled %s and stored config %s", *stored.KnowledgeNamespace, stored.Id)

	// A session started from that config can look the document up, which is the only way
	// it could know a window nothing in its instructions mentions.
	reader := stream.Accelerated(stream.Config{Agent: "e2e-knowledge", Backend: backend})
	asking, err := agents.New(agents.Options{Name: "e2e-knowledge-reader", LLM: reader})
	if err != nil {
		t.Fatal(err)
	}

	session, err := asking.Chat(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))

	if err := session.Respond("How many days do I have to ask Acme for a refund?"); err != nil {
		t.Fatal(err)
	}

	// The lookup is a tool like any other, so the turn that reaches for it says nothing and
	// the answer comes on the next one.
	waitFor(t, session, "looked_up")
	if err := session.Respond("So how many days is it?"); err != nil {
		t.Fatal(err)
	}

	answered := waitFor(t, session, "responded")
	if !strings.Contains(answered.Text, "47") {
		t.Errorf("the agent said %q, so it did not read what was written down", answered.Text)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func dir(path string) string {
	if index := strings.LastIndex(path, "/"); index > 0 {
		return path[:index]
	}
	return "."
}
