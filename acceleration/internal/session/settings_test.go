package session

import (
	"sync"
	"time"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SetSettings moves a call onto the models, voice and overwrites it names and leaves the
// rest of the spec as it was; a session in writing has no speech models to move. The
// values are those the change was made against.
func (s *SessionSuite) TestSettingsMoveTheSessionsModelsAndNothingElse() {
	s.manages()
	created := s.joins(Spec{Custom: map[string]any{"ticket": "T-1"}})
	target, voice, thinking, verbosity := "llm-flow", "alloy", "high", "low"
	temperature, tokens := 0.3, 256

	s.Require().NoError(created.SetSettings(s.ctx, Settings{
		LLM: &target, STT: &target, TTS: &target, Voice: &voice, Thinking: &thinking,
		Verbosity: &verbosity, Temperature: &temperature, MaxOutputTokens: &tokens,
	}))

	spec := created.Spec()
	s.Equal([]string{"llm-flow", "llm-flow", "llm-flow", "", "alloy"},
		[]string{spec.LLMTarget, spec.STTTarget, spec.TTSTarget, spec.STSTarget, spec.Voice})
	s.Equal("high", spec.ModelOverwrites.Thinking)
	s.Equal("low", spec.ModelOverwrites.Verbosity)
	s.Equal(0.3, *spec.ModelOverwrites.Temperature)
	s.Equal(256, *spec.ModelOverwrites.MaxOutputTokens)
	s.Equal(map[string]any{"ticket": "T-1"}, spec.Custom)
	s.Equal([]string{"acme", "call-1", "cascade"}, []string{spec.CustomerID, spec.CallID, created.Mode()})
	asked, used := created.Voice()
	s.Equal([]string{"alloy", "alloy"}, []string{asked, used})

	err := s.writes(Spec{}).SetSettings(s.ctx, Settings{Voice: &voice})
	s.ErrorContains(err, "a text session has no voice")
}

// A label set while SetSettings waits for the reply in flight is kept: SetSettings writes
// back only the models it moved, not the spec it read before waiting.
func (s *SessionSuite) TestALabelSetWhileTheModelsMoveIsKept() {
	s.gated = newGatedLLM()
	s.manages()
	created := s.joins(Spec{})
	s.says(created, "hello")
	select {
	case <-s.gated.entered:
	case <-time.After(settleFor):
		s.FailNow("the model was never asked")
	}
	opened := s.openedModels()

	target := "llm-flow"
	moved := make(chan error, 1)
	go func() { moved <- created.SetSettings(s.ctx, Settings{LLM: &target}) }()
	// The new model is opened before SetSettings waits for the reply, after it read the spec.
	s.eventually(func() bool { return s.openedModels() > opened }, "the new model was never opened")
	custom := map[string]any{"ticket": "T-1"}
	created.Describe(s.ctx, Labels{Custom: &custom})
	s.gated.allowed <- struct{}{}

	select {
	case err := <-moved:
		s.Require().NoError(err)
	case <-time.After(settleFor):
		s.FailNow("the models never moved")
	}
	s.Equal(target, created.Spec().LLMTarget)
	s.Equal(custom, created.Spec().Custom, "the label set while the models moved is kept")
}

// The spec is read off the session's goroutine -- by its log rows, its fan-out and the API --
// while SetSettings and Describe change it. Run with -race, which reports a read without the
// lock.
func (s *SessionSuite) TestTheSpecIsReadWhileItChanges() {
	s.manages()
	logs := make(chan store.AgentLog, 4096)
	s.manager.logs = &logRecorder{queue: logs, store: &store.Store{}}
	created := s.joins(Spec{})
	events, detach := created.Watch()
	defer detach()
	go func() {
		for range events {
		}
	}()

	targets := []string{"llm-flow", "en-low-latency"}
	s.readsWhile(func() {
		for i := range 20 {
			s.NoError(created.SetSettings(s.ctx, Settings{LLM: &targets[i%2]}))
			custom := map[string]any{"round": i}
			created.Describe(s.ctx, Labels{Custom: &custom})
			_, err := created.Respond(s.ctx, "hello", nil, options.LLM{})
			s.NoError(err)
		}
	},
		func() { created.Voice() },
		func() { created.Mode() },
		func() { _, gone := created.Watch(); gone() },
		func() { created.placed(ToolCall{ID: "unknown"}) },
	)

	s.Equal("en-low-latency", created.Spec().LLMTarget)
	s.Equal(map[string]any{"round": 19}, created.Spec().Custom)
	s.Equal("cascade", created.Mode())
	s.eventually(func() bool { return len(logs) > 0 }, "no log row was written")
	s.Equal("acme", (<-logs).CustomerID, "a log row names the session's customer")
}

// The spec's call fields are read off the session's goroutine -- by its log rows, its
// fan-out, tool results and the API -- while voice starts and stops. Run with -race, which
// reports a read without the lock.
func (s *SessionSuite) TestTheSpecIsReadWhileVoiceStartsAndStops() {
	service := persistent.NewForChat(chattest.Client(s.T()))
	s.T().Cleanup(service.Close)
	s.conversations = service
	s.manages()
	logs := make(chan store.AgentLog, 4096)
	s.manager.logs = &logRecorder{queue: logs, store: &store.Store{}}
	created := s.writes(Spec{PersistConversation: true})
	events, detach := created.Watch()
	defer detach()
	go func() {
		for range events {
		}
	}()

	s.readsWhile(func() {
		for range 5 {
			s.NoError(created.StartVoice(s.ctx, Call{}))
			s.NoError(created.StopVoice(s.ctx))
			// A reply still being written in the conversation refuses the next one.
			if _, err := created.Respond(s.ctx, "hello", nil, options.LLM{}); err != nil {
				s.ErrorIs(err, persistent.ErrResponseRunning)
			}
		}
	},
		func() { created.Mode() },
		func() { _, gone := created.WatchPendingVoiceTools(); gone() },
		func() { created.ResolveToolParts("unknown", llm.TextParts("output"), "") },
		func() { created.ResolveTurnTool("unknown", "turn-1", llm.TextParts("output"), "") },
		func() { _ = created.askTool(ToolCall{ID: "unknown", Name: "unknown", TurnID: "turn-1"}) },
	)

	s.False(created.Voicing())
	s.Equal("text", created.Mode())
	s.eventually(func() bool { return len(logs) > 0 }, "no log row was written")
	s.Equal("acme", (<-logs).CustomerID, "a log row names the session's customer")
}

// readsWhile runs each reader over and over on a goroutine of its own until change returns.
func (s *SessionSuite) readsWhile(change func(), readers ...func()) {
	changed := make(chan struct{})
	var reading sync.WaitGroup
	for _, read := range readers {
		reading.Go(func() {
			for {
				select {
				case <-changed:
					return
				default:
					read()
				}
			}
		})
	}
	change()
	close(changed)
	reading.Wait()
}

// openedModels is how many models the agent opened after its first.
func (s *SessionSuite) openedModels() int {
	s.othersMu.Lock()
	defer s.othersMu.Unlock()
	return len(s.others)
}
