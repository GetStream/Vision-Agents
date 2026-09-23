package session

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
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

// persists prepares a manager whose text sessions keep a durable command ledger, answered
// by a model whose timing the test decides.
func (s *SessionSuite) persists() {
	s.outbox = s.T().TempDir()
	service, err := persistent.NewForChat(s.outbox, chattest.Client(s.T()))
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

func (s *SessionSuite) TestPersistentToolResultsStayBoundToTheirCommandAndTurn() {
	s.persists()
	running := s.commands()
	receipt, err := running.persisted.BeginCommand("command-a", "Question")
	s.Require().NoError(err)
	running.persisted.BindTurn(receipt.CommandID, "turn-a")
	events, detach := running.Watch()
	defer detach()

	resolved := make(chan error, 1)
	go func() {
		_, err := running.tools.Run(context.Background(), llm.ToolCall{
			ID: "call-a", TurnID: "turn-a", Name: "lookup_record",
		})
		resolved <- err
	}()
	asked := awaitToolCall(events)
	s.Require().NotNil(asked)
	s.Equal("command-a", asked.CommandID)
	s.Equal("turn-a", asked.TurnID)
	s.False(running.ResolveTool("call-a", "legacy result", ""))
	s.False(running.ResolveCommandTool("call-a", "command-b", "turn-a", llm.TextParts("wrong command"), ""))
	s.False(running.ResolveCommandTool("call-a", "command-a", "turn-b", llm.TextParts("wrong turn"), ""))
	s.True(running.ResolveCommandTool("call-a", "command-a", "turn-a", llm.TextParts("authorized"), ""))
	s.False(running.ResolveCommandTool("call-a", "command-a", "turn-a", llm.TextParts("duplicate"), ""))
	s.Require().NoError(<-resolved)
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

func (s *SessionSuite) TestAStopThatCouldNotBeRecordedIsNotReportedAsStopped() {
	s.persists()
	running := s.commands()

	accepted, err := running.RespondCommand(s.ctx, "command-a", "First question")
	s.Require().NoError(err)
	s.Equal("First question", s.asked())

	// The conversation's own record is made unwritable, which is the case where the stop
	// may have happened but cannot be known to have happened.
	state := filepath.Join(s.outbox, strings.TrimPrefix(running.Spec().ConversationID, "agent:"), "state.json")
	s.Require().NoError(os.Remove(state))
	s.Require().NoError(os.Mkdir(state, 0700))

	_, err = running.InterruptCommand("command-a")
	s.Require().ErrorContains(err, "persistence outcome unknown")

	// Retrying the same stop once the record is writable again settles it.
	s.Require().NoError(os.Remove(state))
	stopped, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)
	s.Equal(accepted.AssistantMessageID, stopped.AssistantMessageID)
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

func (s *SessionSuite) TestPersistentConversationsStillRequireTextMode() {
	s.manages()
	_, err := s.manager.Create(s.ctx, Spec{
		CallID:              "call-1",
		CustomerID:          "acme",
		PersistConversation: true,
		LLMTarget:           "en-low-latency",
		STTTarget:           "en-low-latency",
		TTSTarget:           "en-low-latency",
	})
	s.ErrorContains(err, "persistent conversations require text mode")
}

func (s *SessionSuite) TestSharedConversationHandsOffAfterWatcherDetachAndRejectsRemovedMemberCommands() {
	client := chattest.Client(s.T())
	service, err := persistent.NewForChat(s.T().TempDir(), client)
	s.Require().NoError(err)
	s.T().Cleanup(service.Close)
	s.conversations = service
	s.manages()
	const channelID = "support-12345678-1234-1234-1234-123456789abc"
	const agentID = "test-agent"
	setMembers := func(members ...string) {
		entries := []getstream.ChannelMemberRequest{{UserID: agentID}}
		for _, member := range members {
			entries = append(entries, getstream.ChannelMemberRequest{UserID: member})
		}
		creator := agentID
		// Provision the in-memory Chat fixture; this is not a live permission test.
		_, err := client.Chat().GetOrCreateChannel(s.ctx, "agent", channelID, &getstream.GetOrCreateChannelRequest{
			Data: &getstream.ChannelInput{CreatedByID: &creator, Members: entries, Custom: map[string]any{
				"support_customer_id": "acme", "support_agent_id": agentID,
				"support_owner_id": "alice", "support_access": "members",
				persistent.TriggerField: persistent.SessionCommandTrigger,
			}},
		})
		s.Require().NoError(err)
	}
	setMembers("alice", "bob")
	spec := Spec{CustomerID: "acme", Text: true, PersistConversation: true,
		AgentID: agentID, ConversationID: "agent:" + channelID,
		LLMTarget: "en-low-latency", Caller: routing.Caller{UserID: "alice"}}
	alice, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	_, detachAlice := alice.Watch()
	defer detachAlice()
	_, err = alice.RespondCommand(s.ctx, "alice-command", "The team codeword is TEAM_CANVAS_42")
	s.Require().NoError(err)
	s.eventually(func() bool {
		messages := s.stored(alice)
		return len(messages) == 2 && messages[1].State == "completed" && messages[1].Saved
	}, "Alice's command should be saved before handoff")
	spec.Caller.UserID = "bob"
	_, err = s.manager.Create(s.ctx, spec)
	s.ErrorContains(err, "already open")
	detachAlice()
	s.eventually(func() bool { return alice.State() == Ended }, "detaching the tool host should release Alice's session")
	bob, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	_, detachBob := bob.Watch()
	defer detachBob()
	s.Equal("bob", bob.Spec().Caller.UserID)
	restored := bob.voiceAgent.History()
	s.Require().Len(restored, 3)
	s.Equal(llm.System, restored[0].Role)
	s.Contains(restored[0].Content, "conversational attribution")
	s.Equal(llm.Message{Role: llm.User, Content: `{"author":{"user_id":""},"text":"The team codeword is TEAM_CANVAS_42"}`}, restored[1])
	s.Equal(llm.Message{Role: llm.Assistant, Content: "Hello."}, restored[2])
	_, err = bob.InterruptCommand("alice-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	// Hold Bob's durable command before model output, exposing a late callback
	// that incorrectly interrupts the reused conversation from Alice's old session.
	_, err = bob.persisted.BeginCommand("bob-pending", "Still answering")
	s.Require().NoError(err)
	detachAlice()
	s.Require().Never(func() bool {
		receipt, err := bob.persisted.Command("bob-pending")
		return err != nil || receipt.State != "thinking"
	}, 150*time.Millisecond, 5*time.Millisecond, "Alice's late detach must not cancel Bob's command")
	_, err = bob.InterruptCommand("bob-pending")
	s.Require().NoError(err)
	_, err = bob.RespondCommand(s.ctx, "bob-command", "What is our codeword?")
	s.Require().NoError(err)
	setMembers("alice")
	_, err = bob.RespondCommand(s.ctx, "removed-command", "This must not run")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = bob.Command("bob-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = bob.InterruptCommand("bob-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = alice.RespondCommand(s.ctx, "stale-session-command", "An old session must not act as Bob")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	detachBob()
	s.eventually(func() bool { return bob.State() == Ended }, "Bob's detached session should close")
	_, err = s.manager.Create(s.ctx, spec)
	s.Require().Error(err, "a removed member cannot reopen the shared channel")
}

func (s *SessionSuite) TestAVoiceSessionRestoresChatHistoryWithoutPersisting() {
	s.outbox = s.T().TempDir()
	service, err := persistent.NewForChat(s.outbox, chattest.Client(s.T()))
	s.Require().NoError(err)
	s.T().Cleanup(service.Close)
	s.conversations = service
	s.manages()

	written, err := s.manager.Create(s.ctx, Spec{
		CustomerID:          "acme",
		Text:                true,
		PersistConversation: true,
		AgentID:             "test-agent",
		LLMTarget:           "en-low-latency",
		Caller:              routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	_, err = written.RespondCommand(s.ctx, "seed-1", "The project name is Nimbus")
	s.Require().NoError(err)
	cid := written.Spec().ConversationID
	s.eventually(func() bool {
		page, err := s.conversations.HistoryForCaller(s.ctx, "acme", "test-agent", cid, "", "employee-1")
		if err != nil {
			return false
		}
		completed := 0
		for _, message := range page.Messages {
			if message.State == "completed" && message.Text != "" {
				completed++
			}
		}
		return completed >= 2
	}, "chat never stored the text turn")
	s.Require().NoError(written.Close())

	voice := s.joins(Spec{
		CallID:         "athv-nimbus",
		ConversationID: cid,
		AgentID:        "test-agent",
		Caller:         routing.Caller{UserID: "employee-1"},
	})
	s.Nil(voice.persisted)
	s.Equal([]llm.Message{
		{Role: llm.User, Content: "The project name is Nimbus"},
		{Role: llm.Assistant, Content: "Hello."},
	}, voice.voiceAgent.History())
}
