package session

import (
	"context"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
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

// A settings change made while voice stops reaches the agent that holds the session once
// it has stopped. Made on the agent being left, the change would answer as made while the
// agent built from the spec read before it went on without it.
func (s *SessionSuite) TestASettingsChangeDuringAVoiceStopReachesTheAgentLeftHoldingTheSession() {
	s.manages()
	created := s.joins(Spec{})
	build := created.build
	building, release := make(chan struct{}), make(chan struct{})
	created.build = func(ctx context.Context, next Spec) (*agent.Agent, error) {
		close(building)
		<-release
		return build(ctx, next)
	}
	stopped := make(chan error, 1)
	go func() { stopped <- created.StopVoice(s.ctx) }()
	select {
	case <-building:
	case <-time.After(settleFor):
		s.FailNow("voice never began to stop")
	}

	temperature := 0.3
	moved := make(chan error, 1)
	go func() { moved <- created.SetSettings(s.ctx, Settings{Temperature: &temperature}) }()
	// The change either finished against the agent being left or waits for the swap; the
	// swap is let go only once it is one or the other.
	s.eventually(func() bool { return len(moved) > 0 || waitsForLock("(*Session).SetSettings") },
		"the settings change neither finished nor waited")
	close(release)
	for _, done := range []chan error{stopped, moved} {
		select {
		case err := <-done:
			s.Require().NoError(err)
		case <-time.After(settleFor):
			s.FailNow("voice never stopped or the settings never moved")
		}
	}

	s.Equal("text", created.Mode())
	s.Equal(0.3, *created.Spec().ModelOverwrites.Temperature)
	s.says(created, "how warm is it")
	s.eventually(func() bool { return len(s.askedAbout("how warm is it")) > 0 }, "the model was never asked")
	for _, asked := range s.askedAbout("how warm is it") {
		s.Require().NotNil(asked.Temperature, "the agent holding the session answers at the temperature set")
		s.Equal(0.3, *asked.Temperature)
	}
}

// An interrupt is not held back by a settings change waiting for the reply in flight, nor by
// a voice stop waiting behind that change: both wait on that reply, and the interrupt is what
// ends it.
func (s *SessionSuite) TestAnInterruptEndsTheReplyASettingsChangeWaitsOn() {
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
	// The new model is opened before SetSettings waits for the reply.
	s.eventually(func() bool { return s.openedModels() > opened }, "the new model was never opened")
	stopped := make(chan error, 1)
	go func() { stopped <- created.StopVoice(s.ctx) }()
	s.eventually(func() bool { return waitsForLock("(*Session).swap") }, "voice did not wait to stop")

	interrupted := make(chan struct{})
	go func() {
		created.Interrupt()
		close(interrupted)
	}()
	select {
	case <-interrupted:
	case <-time.After(settleFor):
		s.gated.answers(1)
		s.FailNow("the interrupt waited for the settings change")
	}
	select {
	case err := <-moved:
		s.Require().NoError(err)
	case <-time.After(settleFor):
		s.gated.answers(1)
		s.FailNow("the settings never moved once the reply was interrupted")
	}
	select {
	case err := <-stopped:
		s.Require().NoError(err)
	case <-time.After(settleFor):
		s.FailNow("voice never stopped once the settings moved")
	}
	s.Equal("text", created.Mode())
	s.Equal(target, created.Spec().LLMTarget)
}

// askedAbout is every request a model opened after the first was asked to answer text in.
func (s *SessionSuite) askedAbout(text string) []llm.ResponseParams {
	s.othersMu.Lock()
	defer s.othersMu.Unlock()
	var found []llm.ResponseParams
	for _, model := range s.others {
		model.mu.Lock()
		for _, params := range model.asked {
			for _, message := range params.Input {
				if message.Role == llm.User && message.Content == text {
					found = append(found, params)
				}
			}
		}
		model.mu.Unlock()
	}
	return found
}

// waitsForLock reports whether a goroutine is waiting on a mutex inside function.
func waitsForLock(function string) bool {
	buffer := make([]byte, 1<<20)
	for _, goroutine := range strings.Split(string(buffer[:runtime.Stack(buffer, true)]), "\n\n") {
		if strings.Contains(goroutine, "[sync.Mutex.Lock") && strings.Contains(goroutine, function) {
			return true
		}
	}
	return false
}

// The spec is read off the session's goroutine -- by its log rows, its fan-out and the API --
// while SetSettings and Describe change it. Run with -race, which reports a read without the
// lock.
func (s *SessionSuite) TestTheSpecIsReadWhileItChanges() {
	s.manages()
	s.manager.records = &heldRecorder{}
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
		func() { _ = created.Rewind(s.ctx, &recordedTurns{}, "unknown") },
		func() {
			title := "renamed"
			created.Describe(s.ctx, Labels{Title: &title})
		},
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
		}
	},
		func() {
			// A reply still being written in the conversation refuses the next one.
			if _, err := created.Respond(s.ctx, "hello", nil, options.LLM{}); err != nil {
				s.ErrorIs(err, persistent.ErrResponseRunning)
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
