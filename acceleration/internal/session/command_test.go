package session

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	getstream "github.com/GetStream/getstream-go/v5"
)

// gatedLLM answers only as often as the test says, which is what a command being stopped
// mid-answer needs: the model is still generating while the stop is decided.
type gatedLLM struct {
	entered chan string
	allowed chan struct{}
}

func newGatedLLM() *gatedLLM {
	return &gatedLLM{entered: make(chan string, 8), allowed: make(chan struct{}, 8)}
}

func (g *gatedLLM) Start(context.Context) error { return nil }

func (g *gatedLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	asked := ""
	for _, message := range params.Input {
		if message.Role == llm.User {
			asked = message.Content
		}
	}
	select {
	case g.entered <- asked:
	default:
	}
	select {
	case <-g.allowed:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: g.Provider(), Model: g.Model()})
	script.OutputText("Answer to " + asked)
	script.Done()
	return script.Stream(), nil
}

func (g *gatedLLM) Provider() string               { return "stub" }
func (g *gatedLLM) Model() string                  { return "stub-llm" }
func (g *gatedLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (g *gatedLLM) Close() error                   { return nil }

// answers lets that many held replies through.
func (g *gatedLLM) answers(count int) {
	for range count {
		g.allowed <- struct{}{}
	}
}

// chatStub is Stream Chat with nothing behind it: enough for a persistent conversation to
// open a channel, write to it and read its own history back.
type chatStub struct {
	mu       sync.Mutex
	channels map[string]map[string]any
	messages map[string]map[string]any
	order    []string
}

func newChatStub(s *SessionSuite) *getstream.Stream {
	db := &chatStub{channels: map[string]map[string]any{}, messages: map[string]map[string]any{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		db.mu.Lock()
		defer db.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		parts := strings.Split(r.URL.Path, "/")
		result := map[string]any{}
		switch {
		case strings.HasSuffix(r.URL.Path, "/query"):
			id := parts[len(parts)-2]
			if data, ok := body["data"].(map[string]any); ok {
				db.channels[id] = data
			}
			result["channel"] = db.channels[id]
			var all []map[string]any
			for _, mid := range db.order {
				if db.messages[mid]["cid"] == "agent:"+id {
					all = append(all, db.messages[mid])
				}
			}
			result["messages"] = all
		case strings.HasSuffix(r.URL.Path, "/message"):
			m := body["message"].(map[string]any)
			id := m["id"].(string)
			if _, exists := db.messages[id]; !exists {
				db.order = append(db.order, id)
				m["cid"] = "agent:" + parts[len(parts)-2]
				db.messages[id] = m
			}
			result["message"] = db.messages[id]
		case strings.Contains(r.URL.Path, "/messages/"):
			id := parts[len(parts)-1]
			if id != "ephemeral" && r.Method == "PUT" {
				for key, value := range body["set"].(map[string]any) {
					if key == "text" || key == "attachments" {
						db.messages[id][key] = value
					} else {
						db.messages[id]["custom"].(map[string]any)[key] = value
					}
				}
			}
			if id == "ephemeral" {
				id = parts[len(parts)-2]
			}
			result["message"] = db.messages[id]
		}
		_ = json.NewEncoder(w).Encode(result)
	}))
	s.T().Cleanup(server.Close)
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(server.URL))
	s.Require().NoError(err)
	return client
}

// persists prepares a manager whose text sessions keep a durable command ledger, answered
// by a model whose timing the test decides.
func (s *SessionSuite) persists() {
	service, err := persistent.NewForChat(s.T().TempDir(), newChatStub(s))
	s.Require().NoError(err)
	s.T().Cleanup(service.Close)
	s.conversations = service
	s.gated = newGatedLLM()
	s.manages()
}

// commands opens the text session an employee submits durable commands to.
func (s *SessionSuite) commands() *Session {
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID:          "acme",
		Text:                true,
		PersistConversation: true,
		LLMTarget:           "en-low-latency",
		Caller:              routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = created.Close() })
	return created
}

// asked waits for the question the model was handed.
func (s *SessionSuite) asked() string {
	select {
	case question := <-s.gated.entered:
		return question
	case <-time.After(settleFor):
		s.Require().Fail("the model was never asked anything")
		return ""
	}
}

// stored reads the conversation back the way a reconnecting client would.
func (s *SessionSuite) stored(running *Session) []persistent.Message {
	spec := running.Spec()
	page, err := s.conversations.HistoryForCaller(s.ctx, spec.CustomerID, spec.AgentID, spec.ConversationID, "", spec.Caller.UserID)
	s.Require().NoError(err)
	return page.Messages
}

func (s *SessionSuite) TestStoppingACommandLeavesALaterOneAnsweringItsOwnQuestion() {
	s.persists()
	running := s.commands()

	first, err := running.RespondCommand(s.ctx, "command-a", "First question")
	s.Require().NoError(err)
	s.Equal("First question", s.asked())

	stopped, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)
	s.Equal(first.AssistantMessageID, stopped.AssistantMessageID)

	second, err := running.RespondCommand(s.ctx, "command-b", "Second question")
	s.Require().NoError(err)
	s.NotEqual(first.AssistantMessageID, second.AssistantMessageID)
	s.Equal("Second question", s.asked())

	// Both generations are released together, so the stopped command's output races the
	// live one exactly as it would when a model keeps writing through an interruption.
	s.gated.answers(2)

	s.eventually(func() bool {
		receipt, err := running.Command("command-b")
		return err == nil && receipt.State == "completed"
	}, "the second command should answer its own question")

	dead, err := running.Command("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", dead.State, "a stopped command is never revived by its own late output")

	s.eventually(func() bool {
		saved := s.stored(running)
		return len(saved) == 4 && saved[3].Text == "Answer to Second question"
	}, "the durable reply should hold only the second command's answer")
	saved := s.stored(running)
	s.Equal("cancelled", saved[1].State)
	s.NotContains(saved[3].Text, "First question")
}

func (s *SessionSuite) TestStoppingACommandInTheAcceptanceGapLeavesNothingToRun() {
	s.persists()
	running := s.commands()

	// No wait for the model here: the stop is decided in the gap between the command
	// being accepted and its execution producing anything.
	accepted, err := running.RespondCommand(s.ctx, "command-a", "First question")
	s.Require().NoError(err)
	s.Equal("thinking", accepted.State)

	stopped, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)

	// Letting the model through afterwards must not restart an abandoned reply.
	s.gated.answers(1)
	s.Never(func() bool {
		receipt, err := running.Command("command-a")
		return err != nil || receipt.State != "cancelled"
	}, time.Second, 20*time.Millisecond)

	saved := s.stored(running)
	s.Require().Len(saved, 2)
	s.Empty(saved[1].Text, "a command stopped before it wrote anything leaves no reply text")
}

func (s *SessionSuite) TestRepeatedStopsAndFinishedCommandsConvergeOnOneReceipt() {
	s.persists()
	running := s.commands()
	s.gated.answers(1)

	finished, err := running.RespondCommand(s.ctx, "command-a", "First question")
	s.Require().NoError(err)
	s.Equal("First question", s.asked())
	s.eventually(func() bool {
		receipt, err := running.Command("command-a")
		return err == nil && receipt.State == "completed"
	}, "the first command should finish on its own")

	// Stopping a command that already answered replays its terminal receipt rather than
	// interrupting whatever the session is doing now.
	replayed, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal("completed", replayed.State)
	s.Equal(finished.AssistantMessageID, replayed.AssistantMessageID)

	live, err := running.RespondCommand(s.ctx, "command-b", "Second question")
	s.Require().NoError(err)
	s.Equal("Second question", s.asked())

	again, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal(replayed, again, "a terminal command replays the same receipt while another is running")

	current, err := running.Command("command-b")
	s.Require().NoError(err)
	s.Equal("thinking", current.State, "replaying a finished stop must not touch the running command")

	stopped, err := running.InterruptCommand("command-b")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)
	s.Equal(live.AssistantMessageID, stopped.AssistantMessageID)
	repeated, err := running.InterruptCommand("command-b")
	s.Require().NoError(err)
	s.Equal(stopped, repeated)

	_, err = running.InterruptCommand("command-never-submitted")
	s.Require().ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = running.Command("command-never-submitted")
	s.Require().ErrorIs(err, persistent.ErrCommandNotFound)
}

func (s *SessionSuite) TestConcurrentStopsAndSubmissionsKeepEachCommandSeparate() {
	s.persists()
	running := s.commands()
	s.gated.answers(8)

	first, err := running.RespondCommand(s.ctx, "command-a", "First question")
	s.Require().NoError(err)

	var workers sync.WaitGroup
	stops := make(chan persistent.CommandReceipt, 8)
	failures := make(chan error, 8)
	for range 8 {
		workers.Go(func() {
			receipt, err := running.InterruptCommand("command-a")
			failures <- err
			stops <- receipt
		})
	}
	var second persistent.CommandReceipt
	s.eventually(func() bool {
		second, err = running.RespondCommand(s.ctx, "command-b", "Second question")
		return err == nil
	}, "a new command should be accepted once the stopped one is terminal")
	workers.Wait()
	close(stops)
	close(failures)

	for err := range failures {
		s.Require().NoError(err)
	}
	for receipt := range stops {
		s.Equal(first.AssistantMessageID, receipt.AssistantMessageID, "a stop only ever answers for its own command")
		s.NotEqual(second.AssistantMessageID, receipt.AssistantMessageID)
	}
	live, err := running.Command("command-b")
	s.Require().NoError(err)
	s.Equal(second.AssistantMessageID, live.AssistantMessageID)
}
