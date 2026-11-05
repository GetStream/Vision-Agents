package session

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Every reason router_session_ends_total documents, and whether it is a failure. Written
// out rather than read from endReasons, so a reason added, dropped or reclassed there fails
// here.
var documentedEnds = map[string]string{
	"agent_hangup":  "false",
	"transferred":   "false",
	"hangup":        "false",
	"start":         "true",
	"unanswered":    "false",
	"voicemail":     "false",
	"closed":        "false",
	"shutdown":      "false",
	"create_failed": "true",
	"agent_failed":  "true",
	"model_failed":  "true",
}

// series is the value of the default registry's series name with labels, and whether it
// exists. Read from the registry /metrics serves, so a metric registered elsewhere, or not
// at all, is not read.
func (s *SessionSuite) series(name string, labels map[string]string) (float64, bool) {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			got := map[string]string{}
			for _, pair := range metric.GetLabel() {
				got[pair.GetName()] = pair.GetValue()
			}
			if len(got) != len(labels) {
				continue
			}
			matches := true
			for key, value := range labels {
				matches = matches && got[key] == value
			}
			if !matches {
				continue
			}
			if metric.GetGauge() != nil {
				return metric.GetGauge().GetValue(), true
			}
			return metric.GetCounter().GetValue(), true
		}
	}
	return 0, false
}

// active is router_sessions_active.
func (s *SessionSuite) active() float64 {
	value, ok := s.series("router_sessions_active", map[string]string{})
	s.Require().True(ok, "router_sessions_active is not served")
	return value
}

// ends is router_session_ends_total for reason, under the failed value the documented set
// gives it.
func (s *SessionSuite) ends(reason string) float64 {
	failed, documented := documentedEnds[reason]
	s.Require().True(documented, "%s is not a documented reason", reason)
	value, ok := s.series("router_session_ends_total", map[string]string{"reason": reason, "failed": failed})
	s.Require().True(ok, "no series for %s with failed=%s", reason, failed)
	return value
}

// allEnds is every documented reason's count.
func (s *SessionSuite) allEnds() map[string]float64 {
	counts := map[string]float64{}
	for reason := range documentedEnds {
		counts[reason] = s.ends(reason)
	}
	return counts
}

// endedOnce checks that, since before, the session ends counted are one under reason and
// none under any other.
func (s *SessionSuite) endedOnce(before map[string]float64, reason string) {
	s.eventually(func() bool { return s.ends(reason) == before[reason]+1 }, "the end was not counted under "+reason)
	after := s.allEnds()
	for other, count := range after {
		want := before[other]
		if other == reason {
			want++
		}
		s.Equal(want, count, "ends counted under %s", other)
	}
}

func (s *SessionSuite) TestASessionClosedByItsCallerIsActiveUntilThenAndEndsOnceAsClosed() {
	// Catches a session the gauge never counts, or a close that leaves it counted or counts
	// its end twice when it is closed again.
	s.manages()
	active, before := s.active(), s.allEnds()

	created := s.joins(Spec{})
	s.Equal(active+1, s.active(), "a held session is not counted")

	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)
	s.Require().NoError(created.Close())

	s.Equal(active, s.active(), "a closed session is still counted")
	s.endedOnce(before, "closed")
	s.Empty(created.endReason(), "the row's end reason changed")
}

func (s *SessionSuite) TestAPhoneSessionNobodyJoinedEndsAsAFailedStart() {
	// Catches a timeout's end left uncounted, or counted as a normal end.
	s.manages()
	active, before := s.active(), s.allEnds()

	created := s.rings(outbound())
	s.timedOut(created)

	s.Equal(active, s.active(), "a timed out session is still counted")
	s.endedOnce(before, "start")
}

func (s *SessionSuite) TestAPhoneSessionTheCallerLeftEndsAsAHangup() {
	s.manages()
	active, before := s.active(), s.allEnds()

	created := s.rings(inbound(), caller)
	s.settles(created)
	s.call().leaves(caller)
	s.timedOut(created)

	s.Equal(active, s.active())
	s.endedOnce(before, "hangup")
}

func (s *SessionSuite) TestASessionItsAgentHungUpEndsAsTheAgentsHangup() {
	// Catches the hang-up path, which closes off the agent's goroutine, missing its decrement.
	s.hangingUp()
	active, before := s.active(), s.allEnds()

	created := s.joins(Spec{ID: "session-1", CallID: "their-call", ConfigID: "cfg-1", Calls: endsCalls()})
	s.hangsUp(created)

	s.eventually(func() bool { return s.active() == active }, "a hung up session is still counted")
	s.endedOnce(before, "agent_hangup")
}

func (s *SessionSuite) TestASessionThatLeftAVoicemailEndsAsVoicemail() {
	s.manages()
	active, before := s.active(), s.allEnds()

	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))
	s.endedForVoicemail(created)

	s.eventually(func() bool { return s.active() == active }, "a session that left a voicemail is still counted")
	s.endedOnce(before, "voicemail")
}

func (s *SessionSuite) TestAPlacedCallNobodyAnsweredEndsAsUnansweredAndNoFailure() {
	s.manages()
	active, before := s.active(), s.allEnds()

	created := s.rings(outbound())
	s.settles(created)
	s.True(s.manager.EndUnanswered("acme", created.ID()))

	s.Equal(Ended, created.State())
	s.Equal(active, s.active())
	s.endedOnce(before, "unanswered")
}

func (s *SessionSuite) TestTheSessionAnAgentLeftBehindEndsAsClosedAndItsSuccessorIsCounted() {
	// Catches a session taken off the map without closing (supersede) left counted, which a
	// restarted agent would leave behind on every restart.
	s.manages()
	active, before := s.active(), s.allEnds()

	left := s.joins(Spec{CallID: "call-1"})
	s.joins(Spec{CallID: "call-1"})

	s.Equal(Ended, left.State())
	s.Equal(active+1, s.active(), "both sessions are counted, or neither")
	s.endedOnce(before, "closed")
}

func (s *SessionSuite) TestShutdownEndsEverySessionAsShutdown() {
	s.manages()
	active, before := s.active(), s.allEnds()
	s.joins(Spec{CallID: "call-1"})
	s.joins(Spec{CallID: "call-2", CustomerID: "other"})

	s.Require().NoError(s.manager.Shutdown())

	s.Equal(active, s.active(), "a session the shutdown ended is still counted")
	s.Equal(before["shutdown"]+2, s.ends("shutdown"))
	s.Equal(before["closed"], s.ends("closed"), "a shutdown was counted as a caller's close")
}

func (s *SessionSuite) TestASessionWhoseCallRefusedTheAgentEndsOnceAsAFailedCreate() {
	// Catches a create that fails after the session existed going uncounted, or counted a
	// second time by its own close.
	s.manages()
	active, before := s.active(), s.allEnds()

	_, err := s.manager.Create(s.ctx, Spec{
		CallID: "call-1", CustomerID: "acme", Edge: refusedEdge{newQuietEdge()},
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
	})
	s.Require().ErrorContains(err, "the call refused the agent")

	s.Equal(active, s.active(), "a session never held is counted")
	s.endedOnce(before, "create_failed")
}

func (s *SessionSuite) TestARefusedSpecIsNoSessionEnd() {
	// Catches a request refused before any session existed counted as a session that failed.
	s.manages()
	before := s.allEnds()

	_, err := s.manager.Create(s.ctx, Spec{CallID: "call-1", CustomerID: "acme", STSTarget: "en-low-latency"})
	s.Require().Error(err)

	s.Equal(before, s.allEnds())
}

func (s *SessionSuite) TestMovingASessionOntoACallAndOffItEndsNothing() {
	// Catches the agent swap, which closes one agent and builds another, counted as a session
	// end, or as the agent dropping out.
	s.manages()
	active, before := s.active(), s.allEnds()
	created := s.writes(Spec{})

	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	s.Require().NoError(created.StopVoice(s.ctx))

	s.Equal(active+1, s.active())
	s.Equal(before, s.allEnds(), "moving the session counted an end")
	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)
	s.endedOnce(before, "closed")
}

func (s *SessionSuite) TestASessionWhoseAgentDroppedOutEndsAsAFailedAgent() {
	// Catches an agent failure that ends up counted as whatever closed the session after it.
	speech := s.dropsOut()
	s.manages()
	active, before := s.active(), s.allEnds()
	created := s.joins(Spec{STSTarget: "en-low-latency"})
	s.eventually(func() bool { return speech.opened() }, "the speech-to-speech model was never opened")

	speech.fail()
	s.eventually(func() bool { return s.call().gone() }, "the agent did not leave the call")
	s.Equal(Live, created.State(), "the session ended with its agent")
	s.Equal(active+1, s.active())

	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)

	s.Equal(active, s.active())
	s.endedOnce(before, "agent_failed")
	s.Empty(created.endReason(), "the row's end reason changed")
}

func (s *SessionSuite) TestASessionWhoseAgentDroppedOutBeforeItTimedOutEndsAsAFailedAgent() {
	// Catches the timeout that ends a session after its agent dropped out counted in place of
	// the agent's failure, and the failure written over the reason the row keeps.
	speech := s.dropsOut()
	s.manages()
	before := s.allEnds()
	created := s.joins(Spec{STSTarget: "en-low-latency"})
	s.eventually(func() bool { return speech.opened() }, "the speech-to-speech model was never opened")
	speech.fail()
	s.eventually(func() bool { return s.call().gone() }, "the agent did not leave the call")

	created.SetTimeout(testTimeout, TimeoutHangup)
	s.timedOut(created)

	s.endedOnce(before, "agent_failed")
	s.Equal(TimeoutHangup, created.endReason(), "the row's end reason changed")
}

func (s *SessionSuite) TestASessionWhoseModelRefusedEveryTurnEndsAsAFailedModel() {
	// Catches a cascade agent whose model refuses every request, which keeps the call going
	// with nothing said, counted as a normal close.
	s.manages()
	active, before := s.active(), s.allEnds()
	created := s.joins(Spec{})
	s.model.mu.Lock()
	s.model.refuses = errors.New("401 Unauthorized: invalid api key")
	s.model.mu.Unlock()

	s.refusedTo(created, "Hi, is anyone there?")
	s.refusedTo(created, "I want to move my appointment.")
	s.Equal(Live, created.State(), "a failed turn ended the session")

	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)

	s.Equal(active, s.active())
	s.endedOnce(before, "model_failed")
	s.Empty(created.endReason(), "the row's end reason changed")
}

func (s *SessionSuite) TestASessionThatLostOneTurnAfterAnsweringEndsAsClosed() {
	// Catches a long call counted as failed over one turn its model lost.
	s.manages()
	before := s.allEnds()
	created := s.joins(Spec{})
	s.hears(created)
	s.model.mu.Lock()
	s.model.refuses = errors.New("the provider is down")
	s.model.mu.Unlock()
	s.refusedTo(created, "And can you send me a reminder?")

	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)

	s.endedOnce(before, "closed")
}

func (s *SessionSuite) TestTheEndOfASessionWhoseModelAnsweredNothing() {
	// Catches the reason that ends a session after its model failed every turn counted in
	// place of the failure, the lost reply the agent says taken for an answer, a greeting
	// taken for one, and a turn the injection screen refused or a voice failing taken for
	// a failure of the model.
	observed := func(events ...Event) *Session {
		session := &Session{}
		lost := ""
		for _, event := range events {
			session.observeTurn(event, &lost)
		}
		return session
	}
	ended := func(events ...Event) string { return observed(events...).endLabel() }
	call := func(turn string, success bool, code string) Event {
		return agent.ModelCall{CallTiming: llm.CallTiming{Purpose: "reply", TurnID: turn, Success: success, ErrorCode: code}}
	}
	refused := agent.Error{Context: "llm", Err: errors.New("401 Unauthorized")}
	lostTurn := func(turn string) []Event {
		return []Event{call(turn, false, "create_failed"), refused, agent.Responded{TurnID: turn, Text: "Sorry, could you ask me that again?"}}
	}

	s.Equal(endModelFailed, ended(lostTurn("turn-1")...), "a lost reply")

	hungUp := observed(append([]Event{agent.Said{TurnID: "say-1", Text: "Hi, how can I help?"}}, lostTurn("turn-1")...)...)
	hungUp.noteEnd(TimeoutHangup)
	s.Equal(endModelFailed, hungUp.endLabel(), "the caller hanging up on a silent agent")

	greeted := append([]Event{agent.Responded{TurnID: "say-1", Text: "Hi there.", Purpose: agent.PurposeGreeting}}, lostTurn("turn-1")...)
	s.Equal(endModelFailed, ended(greeted...), "a greeting written in a text session")

	s.Equal(endModelFailed, ended(agent.Error{Context: "sts", Err: errors.New("the model went away")}), "a speech-to-speech model failing its turn")

	answered := append([]Event{call("turn-1", true, ""), agent.Responded{TurnID: "turn-1", Text: "Friday at ten."}}, lostTurn("turn-2")...)
	s.Equal(endClosed, ended(answered...), "a lost turn after an answered one")

	retried := append([]Event{
		call("turn-1", false, "upstream_error"), call("turn-1", true, ""),
		agent.Responded{TurnID: "turn-1", Text: "Friday at ten."},
	}, lostTurn("turn-2")...)
	s.Equal(endClosed, ended(retried...), "a turn a second provider answered")

	interrupted := append([]Event{
		call("turn-1", false, string(llm.StatusCancelled)), agent.Responded{TurnID: "turn-1", Text: "Friday at"},
	}, lostTurn("turn-2")...)
	s.Equal(endClosed, ended(interrupted...), "a reply the caller cut short")

	s.Equal(endClosed, ended(agent.Error{Context: "llm", Err: policy.ErrPromptInjection, TurnID: "turn-1"}), "a turn the screen refused")
	s.Equal(endClosed, ended(agent.Error{Context: "tts", Err: errors.New("the voice is down")}), "a voice failing, which is no model's turn")
}

func (s *SessionSuite) TestACreateTurnedAwayIsNoFailedCreate() {
	// Catches a redelivered hook that lost the race for its session's id, or a create a
	// router shutting down turned away, counted as a session that failed to open: an alert
	// on failed="true" would fire on every redelivery and deploy.
	s.manages()
	before := s.allEnds()

	first := newJoiningEdge()
	lost := make(chan error, 1)
	go func() {
		_, err := s.manager.Create(s.ctx, Spec{
			ID: "session-dup", CallID: "call-1", CustomerID: "acme", Edge: first,
			LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
		})
		lost <- err
	}()
	<-first.entered
	s.joins(Spec{ID: "session-dup", CallID: "call-2"})
	close(first.allowed)
	s.Require().ErrorIs(<-lost, ErrSessionExists)
	s.Equal(before, s.allEnds(), "the create that lost the race was counted")

	late := newJoiningEdge()
	go func() {
		_, err := s.manager.Create(s.ctx, Spec{
			CallID: "call-3", CustomerID: "acme", Edge: late,
			LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
		})
		lost <- err
	}()
	<-late.entered
	s.Require().NoError(s.manager.Shutdown())
	close(late.allowed)
	s.Require().ErrorContains(<-lost, "the manager is shut down")
	s.Equal(before["create_failed"], s.ends("create_failed"), "the create the shutdown turned away was counted")
	s.Equal(before["shutdown"]+1, s.ends("shutdown"))
}

// attendedJoiningEdge is a joiningEdge whose call has a participant in it.
type attendedJoiningEdge struct{ joiningEdge }

func (e attendedJoiningEdge) Attendance() <-chan agent.Attendance { return e.attending }

func (s *SessionSuite) TestACreateTurnedAwayIsNotCountedSilent() {
	// Catches a create the shutdown turned away, whose agent had joined a call with a caller
	// in it, counted no_inbound_audio: the session was never held, so no call was judged.
	s.manages()
	before := s.silences()

	late := attendedJoiningEdge{joiningEdge{quietEdge: newQuietEdge(visitor), entered: make(chan struct{}, 1), allowed: make(chan struct{})}}
	late.attended = true
	lost := make(chan error, 1)
	go func() {
		_, err := s.manager.Create(s.ctx, Spec{
			CallID: "call-3", CustomerID: "acme", Edge: late,
			LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
		})
		lost <- err
	}()
	<-late.entered
	s.Require().NoError(s.manager.Shutdown())
	close(late.allowed)
	s.Require().ErrorContains(<-lost, "the manager is shut down")

	s.Equal(before, s.silences())
}

func (s *SessionSuite) TestEveryEndSeriesIsADocumentedReason() {
	// Catches a reason outside the documented set, or one whose failed value moved, which a
	// dashboard or an alert would miss.
	s.manages()
	s.joins(Spec{CallID: "call-1"})
	s.Require().NoError(s.manager.Shutdown())

	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	seen := map[string]string{}
	for _, family := range families {
		if family.GetName() != "router_session_ends_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			s.Len(labels, 2, "the labels are reason and failed")
			seen[labels["reason"]] = labels["failed"]
		}
	}
	s.Equal(documentedEnds, seen)
}

// joiningEdge is a call that the agent joins only once allowed, saying on entered that it
// is joining.
type joiningEdge struct {
	*quietEdge
	entered chan struct{}
	allowed chan struct{}
}

func newJoiningEdge() joiningEdge {
	return joiningEdge{quietEdge: newQuietEdge(), entered: make(chan struct{}, 1), allowed: make(chan struct{})}
}

func (e joiningEdge) Join(ctx context.Context) error {
	e.entered <- struct{}{}
	<-e.allowed
	return e.quietEdge.Join(ctx)
}

// refusedTo has somebody say text to a session on a call whose model refuses them: the
// model is asked, maybe more than once, and the turn settles.
func (s *SessionSuite) refusedTo(created *Session, text string) {
	s.eventually(func() bool { return len(s.edges) > 0 }, "the session never joined its call")
	alice := stt.Participant{ID: "alice", Name: "Alice"}
	asked := len(s.model.requests())
	pcm := audio.PcmData{Samples: make([]int16, 320), SampleRate: 16_000, Channels: 1}
	s.edges[len(s.edges)-1].inbound <- agent.InboundAudio{Participant: alice, Audio: pcm}
	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: text, Language: "en", Confidence: 1})
	s.eventually(func() bool { return len(s.model.requests()) > asked }, "the model was never asked")
	s.eventually(func() bool { return !created.Busy() }, "the refused turn never settled")
}

// refusedEdge is a call that will not have the agent.
type refusedEdge struct{ *quietEdge }

func (refusedEdge) Join(context.Context) error { return errors.New("the call refused the agent") }

// droppingSpeech is a speech-to-speech model that fails once told to (fail), after which it
// cannot be opened again, so the agent leaves the call.
type droppingSpeech struct {
	mu      sync.Mutex
	current *sts.Emitter
	gone    atomic.Bool
}

// dropsOut routes native sessions to a droppingSpeech.
func (s *SessionSuite) dropsOut() *droppingSpeech {
	speech := &droppingSpeech{}
	s.speech = func() (sts.STS, error) {
		if speech.gone.Load() {
			return nil, errors.New("the model is unavailable")
		}
		speech.mu.Lock()
		defer speech.mu.Unlock()
		speech.current = sts.NewEmitter(sts.EmitterBuffer)
		return &openSpeech{emitter: speech.current}, nil
	}
	return speech
}

func (d *droppingSpeech) opened() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.current != nil
}

func (d *droppingSpeech) fail() {
	d.gone.Store(true)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.current.Send(sts.Error{Provider: "stub", Model: "stub-sts", Err: errors.New("the model went away"), Fatal: true})
}

// openSpeech is one session of a droppingSpeech: it hears and says nothing.
type openSpeech struct{ emitter *sts.Emitter }

func (o *openSpeech) Start(context.Context) error                     { return nil }
func (o *openSpeech) ProcessAudio(sts.PcmData, sts.Participant) error { return nil }
func (o *openSpeech) SendText(string, sts.Participant) error          { return nil }
func (o *openSpeech) SendFrame(llm.ImagePart) error                   { return sts.ErrNoImages }
func (o *openSpeech) SetInstructions(string) error                    { return nil }
func (o *openSpeech) SetTools([]llm.Tool) error                       { return nil }
func (o *openSpeech) Answer(string, string, error) error              { return nil }
func (o *openSpeech) Prompt(string) error                             { return nil }
func (o *openSpeech) Interrupt(int) error                             { return nil }
func (o *openSpeech) Events() <-chan sts.Event                        { return o.emitter.Events() }
func (o *openSpeech) Close() error                                    { o.emitter.Close(); return nil }
func (o *openSpeech) Provider() string                                { return "stub" }
func (o *openSpeech) Model() string                                   { return "stub-sts" }
func (o *openSpeech) SampleRate() int                                 { return 24_000 }
func (o *openSpeech) Capabilities() sts.Capabilities {
	return sts.Capabilities{
		Text: true, InputTranscript: true, OutputTranscript: true,
	}
}
