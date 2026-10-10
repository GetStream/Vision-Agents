package session

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
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
	service := persistent.NewForChat(chattest.Client(s.T()))
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
	receipt, err := running.persisted.BeginCommand("command-a", "Question", "")
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
	s.False(running.DecideCommandTool("call-a", "command-b", "turn-a", true, ""), "an approval is bound to its command")
	s.False(running.DecideCommandTool("call-a", "command-a", "turn-b", true, ""), "and to its turn")
	s.True(running.DecideCommandTool("call-a", "command-a", "turn-a", true, ""))
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

func (s *SessionSuite) TestACommandNamesTheResponseItIsRecordedAs() {
	s.persists()
	running := s.commands()
	held := &heldRecorder{}
	running.records = held

	_, responseID, err := running.RespondCommand(s.ctx, "command-a", "First question", "")
	s.Require().NoError(err)
	s.Require().NotEmpty(responseID, "a caller following this command needs the turn to read back")

	s.gated.answers(1)
	s.eventually(func() bool {
		receipt, err := running.Command("command-a")
		return err == nil && receipt.State == "completed"
	}, "the command should finish")
	held.mu.Lock()
	defer held.mu.Unlock()
	s.Require().Len(held.responses, 1, "the turn's own event does not record a second response")
	s.Equal(responseID, held.responses[0].ID)
	s.Equal("First question", held.responses[0].Said)
}

func (s *SessionSuite) TestAFinishedLoginCarriesOnWithWhatItWasAskedFor() {
	s.persists()
	running := s.commands()
	running.persisted.AcceptLogins([]string{"slack"})
	_, _, err := running.RespondCommand(s.ctx, "command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	s.Equal("Tell Nash a joke on Slack", s.asked())
	slack, ok := plugins.Lookup("slack")
	s.Require().True(ok)
	running.persisted.Observe(agent.ToolStarted{ID: "list", Tool: "slack__list_tools", StartedAt: time.Now().UTC()})
	running.persisted.Observe(agent.ToolRan{ID: "list", Tool: "slack__list_tools",
		Result: plugins.AuthorizationResult(slack, "https://slack.com/oauth/v2_user/authorize?state=s1", "")})
	s.gated.answers(1)
	s.eventually(func() bool {
		receipt, err := running.Command("command-a")
		return err == nil && receipt.State == "completed"
	}, "the reply asking for the login should finish")

	s.manager.LoginFinished("s1")
	s.Equal("Slack is connected now. Carry on with what I asked for before you needed it.", s.asked())
	s.gated.answers(1)
	s.eventually(func() bool {
		saved := s.stored(running)
		return len(saved) == 3 && saved[2].State == "completed"
	}, "the conversation should carry on with a reply of its own")
	saved := s.stored(running)
	s.Equal([]string{"user", "assistant", "assistant"}, []string{saved[0].Role, saved[1].Role, saved[2].Role},
		"nobody is shown having asked again")
	s.Equal(plugins.AuthorizationConnected, saved[1].Authorizations[0].Status)
}

func (s *SessionSuite) TestALoginFinishedAfterTheWatcherLeftStillCarriesOn() {
	s.persists()
	running := s.commands()
	_, detach := running.Watch()
	running.persisted.AcceptLogins([]string{"slack"})
	_, _, err := running.RespondCommand(s.ctx, "command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	s.Equal("Tell Nash a joke on Slack", s.asked())
	slack, ok := plugins.Lookup("slack")
	s.Require().True(ok)
	running.persisted.Observe(agent.ToolStarted{ID: "list", Tool: "slack__list_tools", StartedAt: time.Now().UTC()})
	running.persisted.Observe(agent.ToolRan{ID: "list", Tool: "slack__list_tools",
		Result: plugins.AuthorizationResult(slack, "https://slack.com/oauth/v2_user/authorize?state=s1", "")})
	s.gated.answers(1)
	s.eventually(func() bool {
		receipt, err := running.Command("command-a")
		return err == nil && receipt.State == "completed"
	}, "the reply asking for the login should finish")

	// The page is left while the login is made in another tab.
	detach()
	s.manager.LoginFinished("s1")
	s.Equal("Slack is connected now. Carry on with what I asked for before you needed it.", s.asked())
	s.gated.answers(1)
	s.eventually(func() bool {
		saved := s.stored(running)
		return len(saved) == 3 && saved[2].State == "completed"
	}, "the conversation should carry on with nobody watching")
	s.Equal(Live, running.State())
}

func (s *SessionSuite) TestAWatcherComingBackKeepsTheConversationOpen() {
	s.grace = 50 * time.Millisecond
	s.persists()
	running := s.commands()
	_, detach := running.Watch()
	detach()
	_, detachAgain := running.Watch()
	defer detachAgain()

	s.Never(func() bool { return running.State() == Ended },
		150*time.Millisecond, 5*time.Millisecond, "a conversation somebody watches again must not end")
}

func (s *SessionSuite) TestAConversationNobodyWatchesEndsOnceItsGraceIsOver() {
	s.grace = 20 * time.Millisecond
	s.persists()
	running := s.commands()
	_, detach := running.Watch()
	detach()

	s.eventually(func() bool { return running.State() == Ended }, "nobody came back, so the session should end")
}

func (s *SessionSuite) TestReopeningAConversationNobodyWatchesDoesNotWaitForItsGrace() {
	s.persists()
	running := s.commands()
	_, detach := running.Watch()
	detach()

	reopened, err := s.manager.Create(s.ctx, Spec{
		CustomerID:          "acme",
		Text:                true,
		PersistConversation: true,
		ConversationID:      running.Spec().ConversationID,
		LLMTarget:           "en-low-latency",
		Caller:              routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = reopened.Close() })
	s.Equal(Ended, running.State(), "the session nobody watched should hand the conversation over")
	s.Equal(running.Spec().ConversationID, reopened.Spec().ConversationID)
}

func (s *SessionSuite) TestStoppingACommandLeavesALaterOneAnsweringItsOwnQuestion() {
	s.persists()
	running := s.commands()

	first, _, err := running.RespondCommand(s.ctx, "command-a", "First question", "")
	s.Require().NoError(err)
	s.Equal("First question", s.asked())

	stopped, err := running.InterruptCommand("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)
	s.Equal(first.AssistantMessageID, stopped.AssistantMessageID)

	second, _, err := running.RespondCommand(s.ctx, "command-b", "Second question", "")
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
	accepted, _, err := running.RespondCommand(s.ctx, "command-a", "First question", "")
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

	finished, _, err := running.RespondCommand(s.ctx, "command-a", "First question", "")
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

	live, _, err := running.RespondCommand(s.ctx, "command-b", "Second question", "")
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

	first, _, err := running.RespondCommand(s.ctx, "command-a", "First question", "")
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
		second, _, err = running.RespondCommand(s.ctx, "command-b", "Second question", "")
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

func (s *SessionSuite) TestStartingVoiceJoinsTheCallNamedAfterTheSessionWithItsHistory() {
	s.manages()
	recalled := []llm.Message{
		{Role: llm.User, Content: "Where is order 4471?"},
		{Role: llm.Assistant, Content: "It ships on Friday."},
	}
	created := s.writes(Spec{Recall: &Recall{Messages: recalled}})

	s.Require().NoError(created.StartVoice(s.ctx))

	s.True(created.Voicing())
	s.Equal(created.ID(), created.Spec().CallID)
	s.Equal("agent", created.Spec().CallType)
	s.Require().Len(s.edges, 1, "the call is joined once voice starts")
	s.Equal(recalled, created.current().History())
}

func (s *SessionSuite) TestStoppingVoiceLeavesTheCallAndCarriesOnInWriting() {
	s.manages()
	recalled := []llm.Message{
		{Role: llm.User, Content: "Where is order 4471?"},
		{Role: llm.Assistant, Content: "It ships on Friday."},
	}
	created := s.writes(Spec{Recall: &Recall{Messages: recalled}})
	s.Require().NoError(created.StartVoice(s.ctx))

	s.Require().NoError(created.StopVoice(s.ctx))

	s.False(created.Voicing())
	s.Empty(created.Spec().CallID)
	s.True(s.edges[0].gone(), "the agent left the call")
	s.Equal(recalled, created.current().History())
	s.Equal(Live, created.State())
}

func (s *SessionSuite) TestStartingVoiceTwiceJoinsTheCallOnce() {
	s.manages()
	created := s.writes(Spec{})

	s.Require().NoError(created.StartVoice(s.ctx))
	s.Require().NoError(created.StartVoice(s.ctx))

	s.Len(s.edges, 1)
}

func (s *SessionSuite) TestAnEndedSessionCannotStartVoice() {
	s.manages()
	created := s.writes(Spec{})
	created.Close()

	s.Error(created.StartVoice(s.ctx))
	s.Empty(s.edges)
}

func (s *SessionSuite) TestSharedConversationHandsOffAfterWatcherDetachAndRejectsRemovedMemberCommands() {
	client := chattest.Client(s.T())
	service := persistent.NewForChat(client)
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
	_, _, err = alice.RespondCommand(s.ctx, "alice-command", "The team codeword is TEAM_CANVAS_42", "")
	s.Require().NoError(err)
	s.eventually(func() bool {
		messages := s.stored(alice)
		return len(messages) == 2 && messages[1].State == "completed" && messages[1].Saved
	}, "Alice's command should be saved before handoff")
	spec.Caller.UserID = "bob"
	_, err = s.manager.Create(s.ctx, spec)
	s.ErrorContains(err, "already open")
	detachAlice()
	bob, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	s.Equal(Ended, alice.State(), "Bob reopening the channel should end the session Alice stopped watching")
	_, detachBob := bob.Watch()
	defer detachBob()
	s.Equal("bob", bob.Spec().Caller.UserID)
	restored := bob.current().History()
	s.Require().Len(restored, 3)
	s.Equal(llm.System, restored[0].Role)
	s.Contains(restored[0].Content, "conversational attribution")
	s.Equal(llm.Message{Role: llm.User, Content: `{"author":{"user_id":"alice","display_name":"alice"},"text":"The team codeword is TEAM_CANVAS_42"}`}, restored[1])
	s.Equal(llm.Message{Role: llm.Assistant, Content: "Hello."}, restored[2])
	_, err = bob.InterruptCommand("alice-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	// Hold Bob's durable command before model output, exposing a late callback
	// that incorrectly interrupts the reused conversation from Alice's old session.
	_, err = bob.persisted.BeginCommand("bob-pending", "Still answering", "")
	s.Require().NoError(err)
	detachAlice()
	s.Require().Never(func() bool {
		receipt, err := bob.persisted.Command("bob-pending")
		return err != nil || receipt.State != "thinking"
	}, 150*time.Millisecond, 5*time.Millisecond, "Alice's late detach must not cancel Bob's command")
	_, err = bob.InterruptCommand("bob-pending")
	s.Require().NoError(err)
	_, _, err = bob.RespondCommand(s.ctx, "bob-command", "What is our codeword?", "")
	s.Require().NoError(err)
	setMembers("alice")
	_, _, err = bob.RespondCommand(s.ctx, "removed-command", "This must not run", "")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = bob.Command("bob-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, err = bob.InterruptCommand("bob-command")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	_, _, err = alice.RespondCommand(s.ctx, "stale-session-command", "An old session must not act as Bob", "")
	s.ErrorIs(err, persistent.ErrCommandNotFound)
	detachBob()
	_, err = s.manager.Create(s.ctx, spec)
	s.Require().Error(err, "a removed member cannot reopen the shared channel")
}

func (s *SessionSuite) TestAVoiceSessionRestoresChatHistoryWithoutPersisting() {
	service := persistent.NewForChat(chattest.Client(s.T()))
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
	_, _, err = written.RespondCommand(s.ctx, "seed-1", "The project name is Nimbus", "")
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
	}, voice.current().History())
}
