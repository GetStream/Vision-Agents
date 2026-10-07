package agent

import (
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// chunkAt is 20 ms of 16 kHz audio whose RMS is the level given: a square wave, so every
// sample is as far from zero as the level says.
func chunkAt(level int16) audio.PcmData {
	samples := make([]int16, 320)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = level
		} else {
			samples[i] = -level
		}
	}
	return audio.PcmData{Samples: samples, SampleRate: 16_000, Channels: 1}
}

// hearsSilence has a participant's line carry the given length of silence, a chunk at a time.
func (v *voiceActivity) hearsSilence(participantID string, length time.Duration, at time.Time) {
	for range int(length / chunkDuration) {
		v.observe(participantID, chunkAt(0), at)
	}
}

// gateFixture is an agent that can only be asked whether the first frame of a reply may go
// out, with a caller named alice whose reply is turn-1. The reply waits for the caller to have
// been quiet for the window, and for no longer than the longest hold.
func gateFixture(t *testing.T, window, longest time.Duration) (*Agent, *pipeline) {
	t.Helper()
	p := newPipeline(context.Background(), false)
	t.Cleanup(p.cancel)
	return &Agent{
		logger:          slog.New(slog.DiscardHandler),
		voiced:          newVoiceActivity(),
		replySilence:    window,
		replySilenceMax: longest,
	}, p
}

func (a *Agent) holdReplyFor(participantID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.gated = heldReply{turn: "turn-1", participant: stt.Participant{ID: participantID}}
}

func TestVoiceActivityHearsALoudChunkAndNotSilenceOrHiss(t *testing.T) {
	v := newVoiceActivity()
	now := time.Now()

	v.observe("alice", chunkAt(0), now)
	v.observe("alice", chunkAt(100), now)
	require.True(t, v.lastVoiced("alice").IsZero(), "neither silence nor the hiss of a line is a voice")

	v.observe("alice", chunkAt(3000), now.Add(time.Second))
	require.Equal(t, now.Add(time.Second), v.lastVoiced("alice"))
	require.True(t, v.lastVoiced("bob").IsZero(), "somebody else's voice is theirs")
}

func TestVoiceActivityHearsACallThatOpensWithSpeech(t *testing.T) {
	v := newVoiceActivity()
	now := time.Now()

	v.observe("alice", chunkAt(3000), now)

	require.Equal(t, now, v.lastVoiced("alice"))
}

func TestVoiceActivityTakesASteadyNoiseForBackground(t *testing.T) {
	v := newVoiceActivity()
	start := time.Now()

	for i := range 200 {
		v.observe("alice", chunkAt(400), start.Add(time.Duration(i)*20*time.Millisecond))
	}
	require.True(t, v.lastVoiced("alice").IsZero(), "a noisy room is not somebody talking")

	v.observe("alice", chunkAt(3000), start.Add(5*time.Second))
	require.Equal(t, start.Add(5*time.Second), v.lastVoiced("alice"), "a voice over the noise is heard")
}

func TestVoiceActivityStillHearsAVoiceAtTheEndOfALongTurn(t *testing.T) {
	v := newVoiceActivity()
	start := time.Now()

	var last time.Time
	for i := range 1000 {
		level := int16(3000)
		if i%5 == 0 {
			level = 600
		}
		at := start.Add(time.Duration(i) * 20 * time.Millisecond)
		v.observe("alice", chunkAt(level), at)
		if level == 3000 {
			last = at
		}
	}

	require.Equal(t, last, v.lastVoiced("alice"), "twenty seconds of speaking raised the bar above the voice")
}

func TestVoiceActivityForgetsSomebodyWhoLeft(t *testing.T) {
	v := newVoiceActivity()
	v.observe("alice", chunkAt(3000), time.Now())

	v.forget("alice")

	require.True(t, v.lastVoiced("alice").IsZero())
}

func TestAReplyIsLetOutAtOnceWhenTheCallerHasBeenQuietLongEnough(t *testing.T) {
	a, p := gateFixture(t, 500*time.Millisecond, 5*time.Second)
	a.voiced.observe("alice", chunkAt(3000), time.Now().Add(-time.Second))
	a.holdReplyFor("alice")

	start := time.Now()
	admitted := a.admitFirstFrame(p, context.Background(), "turn-1")

	require.True(t, admitted)
	require.Less(t, time.Since(start), 100*time.Millisecond, "a caller who has been quiet is not waited for")
}

func TestAReplyIsLetOutWhenNothingHasEverBeenHeardFromTheCaller(t *testing.T) {
	a, p := gateFixture(t, 500*time.Millisecond, 5*time.Second)
	a.voiced.observe("alice", chunkAt(0), time.Now())
	a.holdReplyFor("alice")

	start := time.Now()

	require.True(t, a.admitFirstFrame(p, context.Background(), "turn-1"))
	require.Less(t, time.Since(start), 100*time.Millisecond)
}

func TestAReplyIsHeldUntilTheCallerHasBeenQuietForTheWindow(t *testing.T) {
	window := 300 * time.Millisecond
	a, p := gateFixture(t, window, 5*time.Second)
	voicedAt := time.Now().Add(-100 * time.Millisecond)
	a.voiced.observe("alice", chunkAt(3000), voicedAt)
	a.holdReplyFor("alice")

	admitted := a.admitFirstFrame(p, context.Background(), "turn-1")

	require.True(t, admitted)
	require.GreaterOrEqual(t, time.Since(voicedAt), window, "the reply was let out before the silence was confirmed")
}

func TestAReplyIsLetOutWhenTheSilenceIsConfirmedBeforeTheLongestHold(t *testing.T) {
	window, longest := 200*time.Millisecond, time.Second
	a, p := gateFixture(t, window, longest)
	voicedAt := time.Now()
	a.voiced.observe("alice", chunkAt(3000), voicedAt)
	a.holdReplyFor("alice")

	admitted := a.admitFirstFrame(p, context.Background(), "turn-1")

	held := time.Since(voicedAt)
	require.True(t, admitted)
	require.GreaterOrEqual(t, held, window, "the reply was let out before the silence was confirmed")
	require.Less(t, held, longest-200*time.Millisecond, "the reply waited for the longest hold when the silence was already confirmed")
}

func TestAReplyIsLetOutAtTheLongestHoldWhileTheLineNeverGoesQuiet(t *testing.T) {
	longest := 300 * time.Millisecond
	for name, breath := range map[string]time.Duration{
		"voice all the way through":  0,
		"breaths between the voices": 100 * time.Millisecond,
	} {
		t.Run(name, func(t *testing.T) {
			a, p := gateFixture(t, 5*time.Second, longest)
			a.voiced.observe("alice", chunkAt(3000), time.Now())
			a.holdReplyFor("alice")
			stop, done := make(chan struct{}), make(chan struct{})
			go func() {
				defer close(done)
				ticker := time.NewTicker(5 * time.Millisecond)
				defer ticker.Stop()
				for {
					select {
					case <-stop:
						return
					case <-ticker.C:
						a.voiced.observe("alice", chunkAt(3000), time.Now())
						a.voiced.hearsSilence("alice", breath, time.Now())
					}
				}
			}()

			start := time.Now()
			admitted := a.admitFirstFrame(p, context.Background(), "turn-1")
			held := time.Since(start)
			close(stop)
			<-done

			require.True(t, admitted, "a line that never went quiet is not a caller starting again")
			require.GreaterOrEqual(t, held, longest, "the reply was let out before the longest hold")
			require.Less(t, held, longest+300*time.Millisecond, "the reply was held past the longest hold")
		})
	}
}

func TestAVoiceWhileAReplyIsHeldOnlyRestartsTheCountOfQuiet(t *testing.T) {
	v := newVoiceActivity()
	start := time.Now()
	window, longest := 700*time.Millisecond, 1500*time.Millisecond
	readyAt := start.Add(100 * time.Millisecond)
	v.observe("alice", chunkAt(3000), start)

	left, capped := v.holdFor("alice", window, longest, readyAt, readyAt)
	require.Equal(t, 600*time.Millisecond, left, "the silence is counted from the caller's last voice")
	require.False(t, capped)

	// A cough after a long quiet stretch, which is what used to have the reply dropped.
	coughAt := start.Add(500 * time.Millisecond)
	v.observe("alice", chunkAt(3000), coughAt)
	left, capped = v.holdFor("alice", window, longest, readyAt, coughAt.Add(20*time.Millisecond))
	require.Equal(t, 680*time.Millisecond, left, "the cough restarts the count and nothing else")
	require.False(t, capped)

	left, capped = v.holdFor("alice", window, longest, readyAt, coughAt.Add(window))
	require.Zero(t, left, "the reply is let out once the caller has been quiet since the cough")
	require.False(t, capped)
}

func TestTheLongestHoldReleasesAReplyWhateverTheLineSounds(t *testing.T) {
	v := newVoiceActivity()
	start := time.Now()
	window, longest := 700*time.Millisecond, time.Second
	v.observe("alice", chunkAt(3000), start.Add(300*time.Millisecond))

	left, capped := v.holdFor("alice", window, longest, start, start.Add(longest-100*time.Millisecond))
	require.Equal(t, 100*time.Millisecond, left, "the wait is shortened to what is left of the longest hold")
	require.False(t, capped)

	// The line is still voiced when the longest hold has passed.
	voicedAt := start.Add(longest)
	v.observe("alice", chunkAt(3000), voicedAt)
	left, capped = v.holdFor("alice", window, longest, start, voicedAt)
	require.Zero(t, left)
	require.True(t, capped)
}

func TestAReplyIsHeldForNobodyNeverHeardToVoiceAnything(t *testing.T) {
	v := newVoiceActivity()
	now := time.Now()
	v.observe("alice", chunkAt(0), now)

	left, capped := v.holdFor("alice", time.Second, time.Second, now, now)
	require.Zero(t, left)
	require.False(t, capped)
	left, _ = v.holdFor("bob", time.Second, time.Second, now, now)
	require.Zero(t, left)
}

func TestACoughWhileAReplyIsHeldDelaysItAndDoesNotDropIt(t *testing.T) {
	window := 300 * time.Millisecond
	a, p := gateFixture(t, window, 5*time.Second)
	a.voiced.observe("alice", chunkAt(3000), time.Now().Add(-100*time.Millisecond))
	a.holdReplyFor("alice")
	var coughAt time.Time
	coughed := make(chan struct{})
	time.AfterFunc(50*time.Millisecond, func() {
		defer close(coughed)
		coughAt = time.Now()
		a.voiced.observe("alice", chunkAt(3000), coughAt)
	})

	admitted := a.admitFirstFrame(p, context.Background(), "turn-1")

	<-coughed
	require.True(t, admitted, "a cough dropped the reply")
	require.GreaterOrEqual(t, time.Since(coughAt), window, "the silence was counted from before the cough")
}

func TestAHeldReplyIsLetGoOfWhenItIsAbandonedOrThePipelineStops(t *testing.T) {
	a, p := gateFixture(t, 5*time.Second, 10*time.Second)
	a.voiced.observe("alice", chunkAt(3000), time.Now())

	abandoned, abandon := context.WithCancel(context.Background())
	a.holdReplyFor("alice")
	time.AfterFunc(50*time.Millisecond, abandon)
	require.False(t, a.admitFirstFrame(p, abandoned, "turn-1"), "an abandoned reply is not let out")

	a.holdReplyFor("alice")
	time.AfterFunc(50*time.Millisecond, p.cancel)
	require.False(t, a.admitFirstFrame(p, context.Background(), "turn-1"), "a stopped pipeline lets nothing out")
}

func TestATurnThatIsNotAReplyToTheCallerIsNeverHeld(t *testing.T) {
	a, p := gateFixture(t, 5*time.Second, 10*time.Second)
	a.voiced.observe("alice", chunkAt(3000), time.Now())
	a.holdReplyFor("alice")

	start := time.Now()

	require.True(t, a.admitFirstFrame(p, context.Background(), "say-1"), "a greeting is not waiting on anybody")
	require.Less(t, time.Since(start), 100*time.Millisecond)
	a.mu.Lock()
	require.Equal(t, "turn-1", a.gated.turn, "the reply that is waiting is still the one that was named")
	a.mu.Unlock()
}

func TestHearingAndAdmittingAudioDoNotAllocate(t *testing.T) {
	window, longest := 40*time.Millisecond, 10*time.Millisecond
	a, p := gateFixture(t, window, longest)
	chunk := chunkAt(3000)
	alice := stt.Participant{ID: "alice"}
	a.voiced.observe("alice", chunk, time.Now())
	ctx := context.Background()

	require.Zero(t, testing.AllocsPerRun(100, func() {
		a.voiced.observe("alice", chunk, time.Now())
	}), "hearing a chunk of audio")

	require.Zero(t, testing.AllocsPerRun(100, func() {
		a.admitFirstFrame(p, ctx, "say-1")
	}), "a turn nobody is waiting on")

	require.Zero(t, testing.AllocsPerRun(100, func() {
		a.voiced.mu.Lock()
		a.voiced.speakers["alice"].last = time.Now().Add(-time.Second)
		a.voiced.mu.Unlock()
		a.gated = heldReply{turn: "turn-1", participant: alice}
		a.admitFirstFrame(p, ctx, "turn-1")
	}), "a reply to a caller who has been quiet")

	require.Zero(t, testing.AllocsPerRun(20, func() {
		a.voiced.mu.Lock()
		a.voiced.speakers["alice"].last = time.Now().Add(2*time.Millisecond - window)
		a.voiced.mu.Unlock()
		a.gated = heldReply{turn: "turn-1", participant: alice}
		a.admitFirstFrame(p, ctx, "turn-1")
	}), "a reply held for a moment")

	require.Zero(t, testing.AllocsPerRun(20, func() {
		a.voiced.observe("alice", chunk, time.Now())
		a.gated = heldReply{turn: "turn-1", participant: alice}
		if !a.admitFirstFrame(p, ctx, "turn-1") {
			t.Error("a reply held to the longest hold was dropped")
		}
	}), "a reply held to the longest hold by a line that does not go quiet")
}

func TestTheReplySilenceDefaultsToSevenHundredMillisecondsAndCanBeTurnedOff(t *testing.T) {
	options := func(silence *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, ReplySilence: silence,
		}
	}

	left, err := New(options(nil))
	require.NoError(t, err)
	require.Equal(t, 700*time.Millisecond, left.replySilence)
	require.NotNil(t, left.voiced)

	off := time.Duration(0)
	disabled, err := New(options(&off))
	require.NoError(t, err)
	require.Zero(t, disabled.replySilence)
	require.Nil(t, disabled.voiced, "nothing is listened to for a gate that is off")

	negative := -time.Millisecond
	_, err = New(options(&negative))
	require.Error(t, err)
}

func TestAHeldReplyIsLetOutAfterAtMostASecondAndTheLimitCannotBeLeftOut(t *testing.T) {
	options := func(silence, longest *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{},
			ReplySilence: silence, ReplySilenceMax: longest,
		}
	}
	duration := func(d time.Duration) *time.Duration { return &d }

	left, err := New(options(nil, nil))
	require.NoError(t, err)
	require.Equal(t, time.Second, left.replySilenceMax)

	set, err := New(options(nil, duration(1500*time.Millisecond)))
	require.NoError(t, err)
	require.Equal(t, 1500*time.Millisecond, set.replySilenceMax)

	_, err = New(options(nil, duration(0)))
	require.Error(t, err, "no limit would hold a reply for as long as the line is noisy")
	_, err = New(options(duration(500*time.Millisecond), duration(0)))
	require.Error(t, err)
	_, err = New(options(nil, duration(-time.Millisecond)))
	require.Error(t, err)

	off, err := New(options(duration(0), duration(0)))
	require.NoError(t, err, "with the wait for silence off there is nothing to limit")
	require.Nil(t, off.voiced)
}

// speakAloud pushes a chunk of a participant's audio that carries a voice into the call, waits
// for the agent to have heard it, and returns when it did.
func (s *AgentSuite) speakAloud(participant stt.Participant) time.Time {
	before := s.agent.voiced.lastVoiced(participant.ID)
	s.edge.inbound <- InboundAudio{Participant: participant, Audio: chunkAt(4000)}
	s.eventually(func() bool { return s.agent.voiced.lastVoiced(participant.ID).After(before) },
		"the voice was never heard")
	return s.agent.voiced.lastVoiced(participant.ID)
}

// keepsTalking has a participant's voice arrive every few milliseconds until the test ends.
func (s *AgentSuite) keepsTalking(participant stt.Participant) {
	stop, done := make(chan struct{}), make(chan struct{})
	var once sync.Once
	finish := func() { once.Do(func() { close(stop); <-done }) }
	go func() {
		defer close(done)
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				select {
				case s.edge.inbound <- InboundAudio{Participant: participant, Audio: chunkAt(4000)}:
				default:
				}
			}
		}
	}()
	// Registered after the agent's own cleanup, so it runs before the edge is closed.
	s.T().Cleanup(finish)
}

// replyIsHeld waits until the first frame of a reply has been asked about and is waiting on the
// caller's silence.
func (s *AgentSuite) replyIsHeld() {
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return !s.agent.gated.readyAt.IsZero() && s.agent.speakingTurn != ""
	}, "the reply never reached the first frame")
}

func (s *AgentSuite) TestTheFirstFrameOfAReplyWaitsForTheCallerToHaveBeenQuiet() {
	window := 600 * time.Millisecond
	s.replySilence = &window
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 30 * time.Millisecond}
		return edge
	}
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	heardAt := s.speakAloud(alice)

	s.says(alice, "please find a table")

	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")
	s.Never(func() bool { return len(s.edge.heard()) > 0 }, 150*time.Millisecond, 10*time.Millisecond,
		"the reply was let out before the caller had been quiet long enough")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never let out")
	s.Zero(countOf[Interrupted](s.reported()))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Positive(edge.markedWrites())
	queuedAt := turn.StartedAt.Add(time.Duration(turn.FirstFrameQueuedMs * float64(time.Millisecond)))
	s.GreaterOrEqual(queuedAt.Sub(heardAt), window-time.Millisecond,
		"the turn says the first frame was queued before the wait was over")
	s.Greater(turn.FirstAudibleFrameMs, turn.FirstFrameQueuedMs)
	s.False(turn.Interrupted)
}

func (s *AgentSuite) TestACoughWhileTheReplyIsHeldDelaysItAndDoesNotDropIt() {
	window := 400 * time.Millisecond
	s.replySilence = &window
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 30 * time.Millisecond}
		return edge
	}
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.replyIsHeld()

	coughedAt := s.speakAloud(alice)

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "a cough cost the caller the whole reply")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	s.Zero(countOf[Interrupted](s.reported()), "a cough dropped the reply")
	turn, _ := firstOf[Turn](s.reported())
	s.False(turn.Interrupted)
	queuedAt := turn.StartedAt.Add(time.Duration(turn.FirstFrameQueuedMs * float64(time.Millisecond)))
	s.GreaterOrEqual(queuedAt.Sub(coughedAt), window-time.Millisecond,
		"the reply was let out before the caller had been quiet since the cough")
}

func (s *AgentSuite) TestNewWordsWhileTheReplyIsHeldCancelItUnheardAndLeaveNothingInTheHistory() {
	window, longest := time.Minute, time.Minute
	s.replySilence = &window
	s.replySilenceMax = &longest
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 30 * time.Millisecond}
		return edge
	}
	s.join(true)
	// The words that follow are answered by a reply that never gets going, so any audio there
	// is can only be the one that was held.
	s.model.mu.Lock()
	s.model.then = []string{}
	s.model.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.replyIsHeld()
	first, _ := firstOf[Responding](s.reported())
	s.Require().Len(s.agent.History(), 2, "the finished reply is in the history while it is held")

	s.says(alice, "actually make it for four")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the new words did not cancel the held reply")
	interrupted, _ := firstOf[Interrupted](s.reported())
	s.Equal(first.TurnID, interrupted.TurnID)
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(first.TurnID, turn.TurnID)
	s.True(turn.Interrupted)
	s.Zero(turn.FirstFrameQueuedMs, "no frame of the reply was ever queued")
	s.never(func() bool { return len(s.edge.heard()) > 0 || edge.markedWrites() > 0 },
		"a frame of the cancelled reply was emitted")
	s.Zero(countOf[Spoke](s.reported()), "a reply nobody heard was reported as spoken")
	var said []string
	for _, message := range s.agent.History() {
		s.NotEqual(llm.Assistant, message.Role, "a reply nobody heard was kept as something the caller was told")
		said = append(said, message.Content)
	}
	s.Equal([]string{"please find a table", "actually make it for four"}, said)
	s.False(s.interruptionNotePending(), "there is no reply that may have been heard in part")
}

func (s *AgentSuite) TestFramesAfterTheFirstAreNotHeld() {
	window := 500 * time.Millisecond
	s.replySilence = &window
	s.join(true)
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")
	turnID := s.voice.spoken()[0].ID

	s.synthesises(turnID)
	s.replyIsHeld()
	s.Empty(s.edge.heard(), "the first frame went out while the caller had not been quiet long enough")
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the first frame was never let out")

	// The caller makes a sound in the middle of the reply. That is barge-in's to decide, not
	// the gate's, so the rest of the reply is not held to it.
	s.speakAloud(alice)
	s.synthesises(turnID)
	s.synthesises(turnID)

	s.eventually(func() bool { return len(s.edge.heard()) == 3 }, "a frame after the first was held")
	s.Zero(countOf[Interrupted](s.reported()))
}

func (s *AgentSuite) TestAReplyIsNotHeldWhenTheReplySilenceIsOff() {
	off := time.Duration(0)
	s.replySilence = &off
	s.join(true)
	s.Nil(s.agent.voiced)
	alice := stt.Participant{ID: "alice"}
	s.keepsTalking(alice)

	s.says(alice, "please find a table")

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "a reply with no gate was not spoken over a voice")
	s.Zero(countOf[Interrupted](s.reported()))
}

func (s *AgentSuite) TestAReplyIsLetOutAtTheLongestHoldWhileTheCallerNeverStopsTalking() {
	window, longest := time.Minute, 300*time.Millisecond
	s.replySilence = &window
	s.replySilenceMax = &longest
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.keepsTalking(alice)
	s.eventually(func() bool { return !s.agent.voiced.lastVoiced(alice.ID).IsZero() }, "the voice was never heard")

	saidAt := time.Now()
	s.says(alice, "please find a table")

	s.eventually(func() bool { return len(s.edge.heard()) > 0 },
		"a reply held for a line that never went quiet was never let out")
	s.GreaterOrEqual(time.Since(saidAt), longest, "the reply was let out before the longest hold")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.False(turn.Interrupted, "a line that never went quiet is not a caller starting again")
	s.Zero(countOf[Interrupted](s.reported()))
}

func (s *AgentSuite) TestAGreetingIsNotHeldToTheCallersSilence() {
	window := time.Minute
	s.replySilence = &window
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)

	s.Require().NoError(s.agent.Say(s.ctx, "Welcome"))

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the greeting waited on a caller it was not answering")
}
