package agent

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
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

func TestAReplyIsNotHeldWhenTheCallerHasBeenQuietLongEnough(t *testing.T) {
	a, p := gateFixture(t, 500*time.Millisecond, 5*time.Second)
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now.Add(-time.Second))
	a.holdReplyFor("alice")

	hold, publish := a.holdFirstFrame(p, context.Background(), "turn-1", now)

	require.Nil(t, hold, "a caller who has been quiet is not waited for")
	require.True(t, publish)
}

func TestAReplyIsNotHeldWhenNothingHasEverBeenHeardFromTheCaller(t *testing.T) {
	a, p := gateFixture(t, 500*time.Millisecond, 5*time.Second)
	now := time.Now()
	a.voiced.observe("alice", chunkAt(0), now)
	a.holdReplyFor("alice")

	hold, publish := a.holdFirstFrame(p, context.Background(), "turn-1", now)

	require.Nil(t, hold)
	require.True(t, publish)
}

func TestAReplyIsHeldUntilTheCallerHasBeenQuietForTheWindow(t *testing.T) {
	window := 300 * time.Millisecond
	a, p := gateFixture(t, window, 5*time.Second)
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now.Add(-100*time.Millisecond))
	a.holdReplyFor("alice")

	hold, publish := a.holdFirstFrame(p, context.Background(), "turn-1", now)

	require.NotNil(t, hold)
	require.False(t, publish)
	require.Equal(t, window, hold.window)
	left, capped := a.holdLeft(hold, now)
	require.Equal(t, 200*time.Millisecond, left, "the silence is counted from the caller's last voice")
	require.False(t, capped)
	left, _ = a.holdLeft(hold, now.Add(200*time.Millisecond))
	require.Zero(t, left, "the reply is let out when the silence is confirmed")
	a.mu.Lock()
	require.Equal(t, now, a.gated.readyAt, "the reply says when its hold began")
	require.Equal(t, "turn-1", a.gated.turn, "it stays a reply nobody has heard until some of it is let out")
	a.mu.Unlock()
}

func TestAReplyIsLetOutAtTheLongestHoldWhileTheLineNeverGoesQuiet(t *testing.T) {
	longest := 300 * time.Millisecond
	for name, breath := range map[string]time.Duration{
		"voice all the way through":  0,
		"breaths between the voices": 100 * time.Millisecond,
	} {
		t.Run(name, func(t *testing.T) {
			a, p := gateFixture(t, 5*time.Second, longest)
			start := time.Now()
			a.voiced.observe("alice", chunkAt(3000), start)
			a.holdReplyFor("alice")
			hold, _ := a.holdFirstFrame(p, context.Background(), "turn-1", start)
			require.NotNil(t, hold)

			// The line is voiced every few milliseconds, with a breath after each voice.
			var letOutAt time.Duration
			for at := 5 * time.Millisecond; at <= 2*longest; at += 5 * time.Millisecond {
				now := start.Add(at)
				a.voiced.observe("alice", chunkAt(3000), now)
				a.voiced.hearsSilence("alice", breath, now)
				if left, capped := a.holdLeft(hold, now); left <= 0 {
					require.True(t, capped, "the line never went quiet, so the longest hold ended it")
					letOutAt = at
					break
				}
			}

			require.Equal(t, longest, letOutAt, "the reply was not let out at the longest hold")
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
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now.Add(-100*time.Millisecond))
	a.holdReplyFor("alice")
	hold, _ := a.holdFirstFrame(p, context.Background(), "turn-1", now)
	require.NotNil(t, hold)

	coughAt := now.Add(50 * time.Millisecond)
	a.voiced.observe("alice", chunkAt(3000), coughAt)

	left, capped := a.holdLeft(hold, coughAt)
	require.Equal(t, window, left, "the cough restarted the count of quiet")
	require.False(t, capped)
	require.False(t, hold.ctx.Err() != nil, "the cough did not abandon the reply")
	left, _ = a.holdLeft(hold, coughAt.Add(window))
	require.Zero(t, left, "the reply is let out once the caller has been quiet since the cough")
}

func TestAConfidentEndingIsHeldForTheShorterOfTheTwoSilences(t *testing.T) {
	a := &Agent{replySilence: 700 * time.Millisecond, replySilenceConfident: 300 * time.Millisecond}

	require.Equal(t, 700*time.Millisecond, a.silenceFor(heldReply{}))
	require.Equal(t, 300*time.Millisecond, a.silenceFor(heldReply{confident: true}))

	a.replySilenceConfident = 2 * time.Second
	require.Equal(t, 700*time.Millisecond, a.silenceFor(heldReply{confident: true}),
		"a sure ending is never waited on for longer than one that is in doubt")
	a.replySilenceConfident = 0
	require.Zero(t, a.silenceFor(heldReply{confident: true}), "zero lets a sure ending out at once")
}

func TestAReplyToAConfidentEndingIsHeldForTheShorterSilence(t *testing.T) {
	a, p := gateFixture(t, 5*time.Second, 10*time.Second)
	a.replySilenceConfident = 100 * time.Millisecond
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now)
	a.gated = heldReply{turn: "turn-1", participant: stt.Participant{ID: "alice"}, confident: true}

	hold, _ := a.holdFirstFrame(p, context.Background(), "turn-1", now)

	require.NotNil(t, hold)
	require.Equal(t, 100*time.Millisecond, hold.window)
	left, _ := a.holdLeft(hold, now.Add(100*time.Millisecond))
	require.Zero(t, left, "a sure ending was waited on for the reply silence")
}

func TestAReplyThatIsAbandonedOrWhosePipelineHasStoppedIsLetOutNoMore(t *testing.T) {
	a, p := gateFixture(t, 5*time.Second, 10*time.Second)
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now)

	abandoned, abandon := context.WithCancel(context.Background())
	abandon()
	a.holdReplyFor("alice")
	hold, publish := a.holdFirstFrame(p, abandoned, "turn-1", now)
	require.Nil(t, hold)
	require.False(t, publish, "an abandoned reply is not let out")

	p.cancel()
	a.holdReplyFor("alice")
	hold, publish = a.holdFirstFrame(p, context.Background(), "turn-1", now)
	require.Nil(t, hold)
	require.False(t, publish, "a stopped pipeline lets nothing out")
}

func TestATurnThatIsNotAReplyToTheCallerIsNeverHeld(t *testing.T) {
	a, p := gateFixture(t, 5*time.Second, 10*time.Second)
	now := time.Now()
	a.voiced.observe("alice", chunkAt(3000), now)
	a.holdReplyFor("alice")

	hold, publish := a.holdFirstFrame(p, context.Background(), "say-1", now)

	require.Nil(t, hold, "a greeting is not waiting on anybody")
	require.True(t, publish)
	a.mu.Lock()
	require.Equal(t, "turn-1", a.gated.turn, "the reply that is waiting is still the one that was named")
	a.mu.Unlock()
}

func TestHearingAudioAndDecidingWhetherToHoldItDoNotAllocate(t *testing.T) {
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
		a.holdFirstFrame(p, ctx, "say-1", time.Now())
	}), "a turn nobody is waiting on")

	require.Zero(t, testing.AllocsPerRun(100, func() {
		now := time.Now()
		a.voiced.mu.Lock()
		a.voiced.speakers["alice"].last = now.Add(-time.Second)
		a.voiced.mu.Unlock()
		a.gated = heldReply{turn: "turn-1", participant: alice}
		a.holdFirstFrame(p, ctx, "turn-1", now)
	}), "a reply to a caller who has been quiet")

	now := time.Now()
	a.voiced.observe("alice", chunk, now)
	a.gated = heldReply{turn: "turn-1", participant: alice}
	hold, _ := a.holdFirstFrame(p, ctx, "turn-1", now)
	require.NotNil(t, hold)
	require.Zero(t, testing.AllocsPerRun(100, func() {
		a.holdLeft(hold, now.Add(2*time.Millisecond))
	}), "asking how much longer a reply is held for")
	require.Zero(t, testing.AllocsPerRun(100, func() {
		a.holdLeft(hold, now.Add(time.Second))
	}), "asking about a reply that has been held to the longest hold by a line that does not go quiet")
}

func TestAConfidentEndingIsHeldForThreeHundredMillisecondsUnlessTheScoreAndTheSilenceAreSet(t *testing.T) {
	options := func(silence *time.Duration, score *float64) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{},
			ReplySilenceConfident: silence, ReplyConfidentScore: score,
		}
	}
	silence := func(d time.Duration) *time.Duration { return &d }
	score := func(f float64) *float64 { return &f }

	left, err := New(options(nil, nil))
	require.NoError(t, err)
	require.Equal(t, 300*time.Millisecond, left.replySilenceConfident)
	require.Equal(t, 0.9, left.replyConfidentScore)

	set, err := New(options(silence(0), score(0)))
	require.NoError(t, err)
	require.Zero(t, set.replySilenceConfident)
	require.Zero(t, set.replyConfidentScore)

	for _, invalid := range []Options{
		options(silence(-time.Millisecond), nil), options(nil, score(-0.1)),
		options(nil, score(1.1)), options(nil, score(math.NaN())),
	} {
		_, err = New(invalid)
		require.Error(t, err)
	}
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
	require.NotNil(t, disabled.voiced, "the quiet a reply is started on is still listened for")

	unheard := options(&off)
	unheard.PreviewQuiet = &off
	deaf, err := New(unheard)
	require.NoError(t, err)
	require.Nil(t, deaf.voiced, "nothing is listened to when neither needs it")

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

	_, err = New(options(duration(0), duration(0)))
	require.NoError(t, err, "with the wait for silence off there is nothing to limit")
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

// answersAfterAScoreOf has a caller speak, and the acoustic end-of-turn score of their words be
// the given one, and returns when the first frame of the reply to them was queued, counted from
// when the caller was last heard to voice anything.
func (s *AgentSuite) answersAfterAScoreOf(score float64) time.Duration {
	var requests atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, score, &requests))
	longest := 5 * time.Second
	s.replySilenceMax = &longest
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 30 * time.Millisecond}
		return edge
	}
	s.join(false)
	alice := stt.Participant{ID: "alice"}
	heardAt := s.speakAloud(alice)

	s.primaryCandidate(alice, "please find a table")

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never let out")
	s.flowOnlyBesideTheReply("the acoustic score decided the turn")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	queuedAt := turn.StartedAt.Add(time.Duration(turn.FirstFrameQueuedMs * float64(time.Millisecond)))
	return queuedAt.Sub(heardAt)
}

func (s *AgentSuite) TestAReplyToAnEndingTheAcousticScoreWasSureOfIsHeldForTheShorterSilence() {
	window, confident := 1500*time.Millisecond, 50*time.Millisecond
	s.replySilence = &window
	s.replySilenceConfident = &confident

	queued := s.answersAfterAScoreOf(0.95)

	s.Less(queued, window-300*time.Millisecond, "a sure ending was waited on for the reply silence")
}

func (s *AgentSuite) TestAReplyToAnEndingTheAcousticScoreWasNotSureOfKeepsTheReplySilence() {
	window, confident := 1500*time.Millisecond, 50*time.Millisecond
	s.replySilence = &window
	s.replySilenceConfident = &confident

	queued := s.answersAfterAScoreOf(0.7)

	s.GreaterOrEqual(queued, window-time.Millisecond, "an ending that was in doubt was let out before the silence")
}

func (s *AgentSuite) TestTheShorterSilenceCanBeTurnedOffByTheScoreItNeeds() {
	window, confident, never := 1500*time.Millisecond, 50*time.Millisecond, 0.0
	s.replySilence = &window
	s.replySilenceConfident = &confident
	s.replyConfidentScore = &never

	queued := s.answersAfterAScoreOf(0.99)

	s.GreaterOrEqual(queued, window-time.Millisecond, "a score of zero turned the shorter silence on for every ending")
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
	s.Positive(turn.AudioDroppedMs, "the audio that was held and never let out was lost from the turn")
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

// saysWhileTheReplyIsHeld has a caller say words while the reply to their first words is held,
// and the flow controller rule on those with the floor given. The first reply asks for the name
// on a reservation and the reply to what comes after it has it, so hearing both one after the
// other is the caller being asked for what they have just said.
func (s *AgentSuite) saysWhileTheReplyIsHeld(floor, words string) (first, second string) {
	window, longest := 600*time.Millisecond, 5*time.Second
	s.replySilence = &window
	s.replySilenceMax = &longest
	s.join(true)
	s.flow.reply = []string{`{"disposition":"respond","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"` + floor + `"}`}
	s.model.reply = []string{"What name is on the reservation?"}
	s.model.then = []string{"I have the last name."}
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.replyIsHeld()
	held, _ := firstOf[Responding](s.reported())

	s.says(alice, words)

	s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 }, "the new words were never answered")
	var answered []Responding
	for _, event := range s.reported() {
		if responding, ok := event.(Responding); ok {
			answered = append(answered, responding)
		}
	}
	return held.TurnID, answered[1].TurnID
}

// spokenTurns are the turns reported as spoken, in order.
func (s *AgentSuite) spokenTurns() []string {
	var spoken []string
	for _, event := range s.reported() {
		if said, ok := event.(Spoke); ok {
			spoken = append(spoken, said.TurnID)
		}
	}
	return spoken
}

func (s *AgentSuite) TestWordsAddedWhileTheReplyIsHeldReplaceItWhateverTheControllerMadeOfTheFloor() {
	for _, floor := range []string{"continue", "shorten", "stop"} {
		s.Run(floor, func() {
			s.SetupTest()
			first, second := s.saysWhileTheReplyIsHeld(floor, "my last name is Gonzalez")

			s.eventually(func() bool { return len(s.spokenTurns()) == 1 }, "the reply to the new words was never spoken")
			s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
				"the reply nobody had heard was not cancelled")
			interrupted, _ := firstOf[Interrupted](s.reported())
			s.Equal(first, interrupted.TurnID)
			s.never(func() bool { return len(s.spokenTurns()) > 1 }, "two replies were spoken, one after the other")
			s.Equal([]string{second}, s.spokenTurns())
			s.Len(s.edge.heard(), 1, "audio of the replaced reply was let out")
		})
	}
}

func (s *AgentSuite) TestAMurmurWhileTheReplyIsHeldLeavesItToBeHeardBeforeTheAnswerToTheMurmur() {
	first, second := s.saysWhileTheReplyIsHeld("continue", "mm hmm")

	s.eventually(func() bool { return len(s.spokenTurns()) == 2 }, "both replies were not spoken")
	s.Equal([]string{first, second}, s.spokenTurns())
	s.Zero(countOf[Interrupted](s.reported()), "a murmur took the floor")
}

func (s *AgentSuite) TestAHeldReplyDoesNotKeepTheRestOfWhatTheVoiceSaysWaiting() {
	window, longest := time.Second, 5*time.Second
	s.replySilence = &window
	s.replySilenceMax = &longest
	s.join(true)
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")
	replyID := s.voice.spoken()[0].ID
	s.synthesises(replyID)
	s.replyIsHeld()
	// What the voice goes on to say of the held reply waits behind its first audio.
	s.voice.emitter.Send(tts.SynthesisComplete{
		SynthesisID: replyID, AudioDurationMs: 10, TimeToFirstByteMs: 5,
	})

	s.Require().NoError(s.agent.Say(s.ctx, "Welcome"))
	spoken := s.voice.spoken()
	s.synthesises(spoken[len(spoken)-1].ID)

	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "a held reply kept another turn's audio from going out")
	s.Zero(countOf[Spoke](s.reported()), "the held reply was reported as spoken before any of it was let out")
	s.Len(s.edge.heard(), 1)

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the held reply was never let out")
	s.Len(s.edge.heard(), 2, "the reply was reported as spoken before its audio was published")
	spokeEvent, _ := firstOf[Spoke](s.reported())
	s.Equal(replyID, spokeEvent.TurnID)
}

func (s *AgentSuite) TestTheHoldOfAReplyIsReportedOnItsTurn() {
	window := 500 * time.Millisecond
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

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Positive(turn.ReplyHoldMs, "the reply waited for the caller to have been quiet")
	s.LessOrEqual(turn.ReplyHoldMs, window.Seconds()*1000, "it waited no longer than the silence")
	s.LessOrEqual(turn.ReplyHoldMs, turn.TTSToAudioMs, "the hold is inside the leg from the text to the audio")
	s.LessOrEqual(turn.ReplyHoldMs, turn.RoundtripMs, "and inside the whole wait")
}

func (s *AgentSuite) TestAReplyThatWasNotHeldReportsNoHold() {
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.says(alice, "please find a table")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Zero(turn.ReplyHoldMs, "a caller who had been quiet was waited for")
}

func (s *AgentSuite) TestClosingTheAgentWhileAReplyIsHeldPublishesNothingOfIt() {
	window, longest := time.Minute, time.Minute
	s.replySilence = &window
	s.replySilenceMax = &longest
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.replyIsHeld()

	s.Require().NoError(s.agent.Close())

	s.Empty(s.edge.heard(), "a reply that was still held was published when the agent closed")
	s.Zero(countOf[Spoke](s.reported()))
}

// droppedMs is how much of a turn's speech has been counted as dropped, or the held speech a
// closed turn would count, which is the same to whoever reads the turn.
func (s *AgentSuite) droppedMs(turnID string) float64 {
	s.agent.turns.mu.Lock()
	defer s.agent.turns.mu.Unlock()
	if current := s.agent.turns.open[turnID]; current != nil {
		return current.audioDroppedMs + current.audioHeldMs
	}
	return 0
}

func (s *AgentSuite) TestLaterFramesOfAReplyThatWasGivenUpWhileHeldAreNeverLetOut() {
	window, longest := time.Minute, time.Minute
	s.replySilence = &window
	s.replySilenceMax = &longest
	s.join(true)
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")
	replyID := s.voice.spoken()[0].ID
	s.synthesises(replyID)
	s.replyIsHeld()

	// The pipeline stops while the reply is held, which gives it up.
	s.agent.mu.Lock()
	stopping := s.agent.pipe
	s.agent.mu.Unlock()
	stopping.cancel()
	s.eventually(func() bool { return s.droppedMs(turnOf(replyID)) >= 10 }, "the held audio was not given up")

	// Nothing says any more that this reply is waiting, as is the case once a newer one has taken
	// its place, and yet none of what follows it is let out.
	s.agent.mu.Lock()
	s.agent.gated = heldReply{turn: "reply-newer", participant: alice}
	s.agent.mu.Unlock()
	s.synthesises(replyID)

	s.eventually(func() bool { return s.droppedMs(turnOf(replyID)) >= 20 }, "a later frame was not counted as dropped")
	s.Empty(s.edge.heard(), "a later frame of a reply that was given up was let out")
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
	s.Zero(s.agent.replySilence)
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

// resumeFixture is an agent that can only be asked whether a sentence that follows a pause in a
// reply is held, in the reply turn-1 to a caller named alice. The sentence waits for her to have
// been quiet for the window, for no longer than the longest hold in all.
func resumeFixture(t *testing.T, gap, window, longest time.Duration) *Agent {
	t.Helper()
	a := &Agent{
		logger:                slog.New(slog.DiscardHandler),
		voiced:                newVoiceActivity(),
		turns:                 newTurnTracker(func(Turn) {}),
		replySilence:          700 * time.Millisecond,
		replySilenceMax:       longest,
		replySilenceConfident: window,
		replyResumeGap:        gap,
	}
	a.turns.begin("turn-1", stt.Participant{ID: "alice"}, time.Now(), time.Time{}, 0)
	return a
}

func TestAReplyThatSpeaksAgainAfterAPauseOfItsOwnIsResuming(t *testing.T) {
	var out outgoing
	start := time.Now()
	out.letOut(true, 100*time.Millisecond, start)
	pauseBegan := start.Add(100 * time.Millisecond)

	require.False(t, out.resumes(true, 200*time.Millisecond, pauseBegan.Add(199*time.Millisecond)),
		"a gap shorter than the resume gap is how sentences follow one another")
	require.True(t, out.resumes(true, 200*time.Millisecond, pauseBegan.Add(200*time.Millisecond)))
	require.False(t, out.resumes(false, 200*time.Millisecond, pauseBegan.Add(time.Second)),
		"a chunk with no voice in it is not the agent speaking again")
}

func TestAReplyThatHasNotCarriedAVoiceHasNothingToResume(t *testing.T) {
	var out outgoing
	now := time.Now()

	require.False(t, out.resumes(true, 200*time.Millisecond, now.Add(time.Hour)),
		"the first sound of a reply is the first-frame gate's, however long ago the turn began")

	out.letOut(false, 100*time.Millisecond, now)
	require.False(t, out.resumes(true, 200*time.Millisecond, now.Add(time.Hour)),
		"silence let out first is not a voice that stopped")
}

func TestSpeechQueuedAheadOfWhatIsStillPlayingIsCarryingOnNotResuming(t *testing.T) {
	var out outgoing
	start := time.Now()
	// A voice sends a sentence far faster than it is spoken, so the next one can arrive while
	// this is still playing, however long ago the last chunk was let out.
	out.letOut(true, 3*time.Second, start)

	require.False(t, out.resumes(true, 200*time.Millisecond, start.Add(2*time.Second)), "it is still playing")
	require.False(t, out.resumes(true, 200*time.Millisecond, start.Add(3100*time.Millisecond)))
	require.True(t, out.resumes(true, 200*time.Millisecond, start.Add(3200*time.Millisecond)),
		"it has been silent for the gap since it ended")
}

func TestSilenceLetOutInsideAReplyCountsTowardsThePause(t *testing.T) {
	var out outgoing
	start := time.Now()
	out.letOut(true, 100*time.Millisecond, start)
	// A voice that spells out a pause as silence instead of leaving one.
	out.letOut(false, 150*time.Millisecond, start.Add(100*time.Millisecond))

	require.False(t, out.resumes(true, 200*time.Millisecond, start.Add(290*time.Millisecond)))
	require.True(t, out.resumes(true, 200*time.Millisecond, start.Add(300*time.Millisecond)),
		"150 ms of silence let out and 50 ms of nothing is a 200 ms pause")
}

func TestASentenceAfterAPauseIsHeldWhileTheCallerWhoStartedTalkingInItHasNotBeenQuiet(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, 5*time.Second)
	start := time.Now()
	out := outgoing{turn: "turn-1"}
	out.letOut(true, 20*time.Millisecond, start)
	a.voiced.observe("alice", chunkAt(3000), start.Add(250*time.Millisecond))
	now := start.Add(300 * time.Millisecond)

	hold := a.holdAfterPause(&out, context.Background(), "turn-1", true, now)

	require.NotNil(t, hold)
	require.True(t, hold.resume)
	require.Equal(t, "alice", hold.participant)
	require.Equal(t, 300*time.Millisecond, hold.window, "the silence of a turn that was sure to have ended")
	require.Equal(t, 5*time.Second, hold.longest)
	left, capped := a.holdLeft(hold, now)
	require.Equal(t, 250*time.Millisecond, left, "the silence is counted from the caller's last voice")
	require.False(t, capped)
	left, _ = a.holdLeft(hold, now.Add(250*time.Millisecond))
	require.Zero(t, left, "the sentence is let out once the caller has been quiet for the window")
}

func TestASentenceAfterAPauseIsNotHeldForACallerWhoIsNotVoiced(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, 5*time.Second)
	start := time.Now()
	out := outgoing{turn: "turn-1"}
	out.letOut(true, 20*time.Millisecond, start)
	now := start.Add(300 * time.Millisecond)

	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", true, now),
		"nothing has been heard from the caller")

	a.voiced.observe("alice", chunkAt(0), now)
	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", true, now),
		"neither silence nor the hiss of a line is a voice")

	a.voiced.observe("alice", chunkAt(3000), start)
	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", true, now),
		"a caller who has been quiet for the window is not waited for")
}

func TestASentenceThatFollowsWithoutAPauseIsNotHeldForAVoicedCaller(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, 5*time.Second)
	start := time.Now()
	out := outgoing{turn: "turn-1"}
	out.letOut(true, 20*time.Millisecond, start)
	a.voiced.observe("alice", chunkAt(3000), start.Add(100*time.Millisecond))

	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", true, start.Add(150*time.Millisecond)),
		"the reply carried on after 130 ms, which is not a pause")
	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", false, start.Add(time.Second)),
		"a chunk with no voice in it is not a sentence starting")
}

func TestASentenceIsHeldNoLongerThanWhatIsLeftOfTheLongestHoldForTheTurn(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, time.Second)
	start := time.Now()
	out := outgoing{turn: "turn-1", held: 900 * time.Millisecond}
	out.letOut(true, 20*time.Millisecond, start)
	a.voiced.observe("alice", chunkAt(3000), start.Add(300*time.Millisecond))
	now := start.Add(300 * time.Millisecond)

	hold := a.holdAfterPause(&out, context.Background(), "turn-1", true, now)

	require.NotNil(t, hold)
	require.Equal(t, 100*time.Millisecond, hold.longest, "the holds of a turn together last no longer than the longest hold")
	left, capped := a.holdLeft(hold, now.Add(100*time.Millisecond))
	require.Zero(t, left)
	require.True(t, capped, "a line that is never quiet is let out by the longest hold")

	out.held = time.Second
	require.Nil(t, a.holdAfterPause(&out, context.Background(), "turn-1", true, now),
		"once the longest hold is spent later sentences are let out as they come")
}

func TestASentenceOfATurnNobodyIsAnsweringIsNeverHeld(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, 5*time.Second)
	start := time.Now()
	out := outgoing{turn: "say-1"}
	out.letOut(true, 20*time.Millisecond, start)
	a.voiced.observe("alice", chunkAt(3000), start.Add(300*time.Millisecond))

	require.Nil(t, a.holdAfterPause(&out, context.Background(), "say-1", true, start.Add(300*time.Millisecond)),
		"a greeting is not waiting on anybody")
}

func TestASentenceIsWaitedOnForTheShorterOfTheTwoSilences(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, 5*time.Second)
	a.replySilence = 100 * time.Millisecond
	start := time.Now()
	out := outgoing{turn: "turn-1"}
	out.letOut(true, 20*time.Millisecond, start)
	a.voiced.observe("alice", chunkAt(3000), start.Add(300*time.Millisecond))

	hold := a.holdAfterPause(&out, context.Background(), "turn-1", true, start.Add(300*time.Millisecond))

	require.NotNil(t, hold)
	require.Equal(t, 100*time.Millisecond, hold.window,
		"a sentence is never waited on for longer than the first sound of a reply in doubt")
}

func TestTheSentencesThatFollowAPauseAreHeldOnlyWhenEverythingTheyNeedIsOn(t *testing.T) {
	on := func() *Agent {
		return resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, time.Second)
	}
	require.True(t, on().holdsLaterSentences())

	off := on()
	off.replyResumeGap = 0
	require.False(t, off.holdsLaterSentences(), "a resume gap of zero turns them off")

	off = on()
	off.replySilence = 0
	require.False(t, off.holdsLaterSentences(), "the reply silence of zero turns off the whole gate")

	off = on()
	off.replySilenceConfident = 0
	require.False(t, off.holdsLaterSentences(), "there is nothing to wait for with a silence of zero")

	off = on()
	off.voiced = nil
	require.False(t, off.holdsLaterSentences(), "there is no telling without the caller's audio")
}

func TestDecidingWhetherASentenceAfterAPauseIsHeldDoesNotAllocate(t *testing.T) {
	a := resumeFixture(t, 200*time.Millisecond, 300*time.Millisecond, time.Second)
	chunk := chunkAt(3000)
	ctx := context.Background()
	now := time.Now()
	out := outgoing{turn: "turn-1"}

	require.Zero(t, testing.AllocsPerRun(100, func() {
		out.letOut(levelOf(chunk.Samples) >= voicedFloor, lengthOf(chunk), now)
	}), "measuring a chunk of the reply and recording that it was let out")

	require.Zero(t, testing.AllocsPerRun(100, func() {
		out.playsUntil, out.voicedUntil = now, now
		a.holdAfterPause(&out, ctx, "turn-1", true, now.Add(50*time.Millisecond))
	}), "a sentence that follows without a pause")

	a.voiced.observe("alice", chunk, now.Add(-time.Second))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		out.playsUntil, out.voicedUntil = now, now
		a.holdAfterPause(&out, ctx, "turn-1", true, now.Add(300*time.Millisecond))
	}), "a sentence after a pause, for a caller who has been quiet")

	a.voiced.observe("alice", chunk, now.Add(300*time.Millisecond))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		out.playsUntil, out.voicedUntil, out.held = now, now, time.Second
		a.holdAfterPause(&out, ctx, "turn-1", true, now.Add(300*time.Millisecond))
	}), "a sentence after a pause, once the longest hold is spent")
}

func TestTheGapThatMakesAReplyResumeDefaultsToTwoHundredMillisecondsAndCanBeTurnedOff(t *testing.T) {
	options := func(gap *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, ReplyResumeGap: gap,
		}
	}

	left, err := New(options(nil))
	require.NoError(t, err)
	require.Equal(t, 200*time.Millisecond, left.replyResumeGap)

	off := time.Duration(0)
	disabled, err := New(options(&off))
	require.NoError(t, err)
	require.Zero(t, disabled.replyResumeGap)

	negative := -time.Millisecond
	_, err = New(options(&negative))
	require.Error(t, err)
}

// movableClock is the time as the test says it is.
type movableClock struct {
	mu sync.Mutex
	at time.Time
}

func newMovableClock() *movableClock {
	return &movableClock{at: time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)}
}

func (c *movableClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

func (c *movableClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.at.Add(d)
}

// countingEdge is a loopback edge that remembers how much it was handed, which dropping its
// speech does not make it forget.
type countingEdge struct {
	*loopbackEdge

	mu     sync.Mutex
	chunks []int
}

func (e *countingEdge) PublishAudio(pcm audio.PcmData) error {
	e.mu.Lock()
	e.chunks = append(e.chunks, len(pcm.Samples))
	e.mu.Unlock()
	return e.loopbackEdge.PublishAudio(pcm)
}

func (e *countingEdge) published() []int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]int(nil), e.chunks...)
}

// speechChunk is a chunk of speech that carries a voice and plays for the given number of 20 ms.
func speechChunk(frames int) audio.PcmData {
	chunk := chunkAt(3000)
	samples := make([]int16, 0, frames*len(chunk.Samples))
	for range frames {
		samples = append(samples, chunk.Samples...)
	}
	chunk.Samples = samples
	return chunk
}

// synthesisesSpeech has the voice send a chunk of speech that carries a voice for a turn.
func (s *AgentSuite) synthesisesSpeech(turnID string, frames int) {
	s.voice.emitter.Send(tts.AudioChunk{SynthesisID: turnID, Audio: speechChunk(frames)})
}

// looksAtTheHolds has the voice send an event that says nothing, which has the goroutine that
// reads the voice look again at the replies it is holding, against a clock that has moved.
func (s *AgentSuite) looksAtTheHolds() {
	s.voice.emitter.Send(tts.Connected{Provider: "stub"})
}

// heldMs is how much of a turn's speech is waiting in a hold for the caller to have been quiet.
func (s *AgentSuite) heldMs(turnID string) float64 {
	s.agent.turns.mu.Lock()
	defer s.agent.turns.mu.Unlock()
	if current := s.agent.turns.open[turnID]; current != nil {
		return current.audioHeldMs
	}
	return 0
}

// pausesMidReply joins an agent whose reply to a caller has let out its first sentence, 20 ms of
// speech, and then goes quiet for the given time on a clock the test moves. The voice has nothing
// more to say until the test has it say more, and the caller has not been heard.
func (s *AgentSuite) pausesMidReply(pause time.Duration) (clock *movableClock, caller stt.Participant, replyID string) {
	clock = newMovableClock()
	s.join(true)
	s.agent.mu.Lock()
	s.agent.clock = clock.now
	s.agent.mu.Unlock()
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	caller = stt.Participant{ID: "caller"}
	s.speak(caller)

	replyID, err := s.agent.RespondTo(s.ctx, "please find a table", nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")
	s.synthesisesSpeech(replyID, 1)
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the first sentence was never let out")
	clock.advance(pause)
	return clock, caller, replyID
}

func (s *AgentSuite) TestASentenceAfterAPauseWaitsForACallerWhoStartedTalkingInItAndIsPlayedInOrder() {
	clock, caller, replyID := s.pausesMidReply(300 * time.Millisecond)
	s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now())
	clock.advance(50 * time.Millisecond)

	// The next sentence, and the rest of what the voice says of the reply, arrive while the caller
	// is talking.
	s.synthesisesSpeech(replyID, 2)
	s.eventually(func() bool { return s.heldMs(replyID) >= 40 }, "the sentence after the pause was not held")
	s.synthesisesSpeech(replyID, 3)
	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: replyID, AudioDurationMs: 120, TimeToFirstByteMs: 5})
	s.eventually(func() bool { return s.heldMs(replyID) >= 100 }, "what followed it was not held behind it")
	s.Len(s.edge.heard(), 1, "the sentence was let out into the caller's voice")
	s.Zero(countOf[Spoke](s.reported()), "the end of the reply was reported before its last sentences were let out")

	// The caller has been quiet for the confident silence, 300 ms, by now.
	clock.advance(250 * time.Millisecond)
	s.looksAtTheHolds()

	s.eventually(func() bool { return len(s.edge.heard()) == 3 }, "the held sentences were never let out")
	var played []int
	for _, chunk := range s.edge.heard() {
		played = append(played, len(chunk.Samples))
	}
	s.Equal([]int{320, 640, 960}, played, "the sentences were not played in the order they came in")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply was never reported as spoken")
	s.Zero(s.heldMs(replyID))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.InDelta(250, turn.ReplyHoldMs, 0.001, "the wait before the sentence is the turn's reply hold")
	s.False(turn.Interrupted)
	s.Zero(countOf[Interrupted](s.reported()), "the hold only delays the sentence and never drops it")
}

func (s *AgentSuite) TestASentenceAfterAPauseIsNotHeldWhenTheCallerIsNotVoiced() {
	clock, caller, replyID := s.pausesMidReply(300 * time.Millisecond)

	s.synthesisesSpeech(replyID, 2)
	s.eventually(func() bool { return len(s.edge.heard()) == 2 }, "a sentence was held for a caller who was not talking")

	// Somebody who last spoke before the reply took its pause has been quiet for long enough.
	s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now().Add(-300*time.Millisecond))
	clock.advance(300 * time.Millisecond)
	s.synthesisesSpeech(replyID, 3)

	s.eventually(func() bool { return len(s.edge.heard()) == 3 }, "a sentence was held for a caller who had been quiet")
	s.Zero(s.heldMs(replyID))
}

func (s *AgentSuite) TestASentenceThatFollowsWithoutAPauseIsNotHeldWhileTheCallerTalksOverTheReply() {
	clock, caller, replyID := s.pausesMidReply(100 * time.Millisecond)
	s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now())

	s.synthesisesSpeech(replyID, 2)

	s.eventually(func() bool { return len(s.edge.heard()) == 2 },
		"a sentence that followed the last without a pause was held")
}

func (s *AgentSuite) TestSentencesAreHeldNoMoreOnceTheTurnHasBeenHeldForTheLongestHold() {
	longest := 300 * time.Millisecond
	s.replySilenceMax = &longest
	clock, caller, replyID := s.pausesMidReply(300 * time.Millisecond)
	talks := func() { s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now()) }
	talks()
	s.synthesisesSpeech(replyID, 2)
	s.eventually(func() bool { return s.heldMs(replyID) >= 40 }, "the sentence after the pause was not held")

	// The caller never goes quiet, so the longest hold lets the sentence out.
	clock.advance(150 * time.Millisecond)
	talks()
	clock.advance(150 * time.Millisecond)
	talks()
	s.looksAtTheHolds()
	s.eventually(func() bool { return len(s.edge.heard()) == 2 }, "the longest hold did not let the sentence out")

	// After another pause the caller is still talking, and what has been spent is all there is.
	clock.advance(300 * time.Millisecond)
	talks()
	s.synthesisesSpeech(replyID, 3)

	s.eventually(func() bool { return len(s.edge.heard()) == 3 }, "a sentence was held after the longest hold was spent")
	s.Zero(s.heldMs(replyID))
}

func (s *AgentSuite) TestNoSentenceIsHeldAfterAPauseWhenTheResumeGapIsOff() {
	off := time.Duration(0)
	s.replyResumeGap = &off
	clock, caller, replyID := s.pausesMidReply(300 * time.Millisecond)
	s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now())

	s.synthesisesSpeech(replyID, 2)

	s.eventually(func() bool { return len(s.edge.heard()) == 2 }, "a sentence was held with the resume gap off")
}

func (s *AgentSuite) TestNewWordsWhileASentenceAfterAPauseIsHeldCancelItAndTheRestOfTheReply() {
	var edge *countingEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &countingEdge{loopbackEdge: base}
		return edge
	}
	clock, caller, replyID := s.pausesMidReply(300 * time.Millisecond)
	s.agent.voiced.observe(caller.ID, chunkAt(4000), clock.now())
	s.synthesisesSpeech(replyID, 2)
	s.eventually(func() bool { return s.heldMs(replyID) >= 40 }, "the sentence after the pause was not held")
	s.synthesisesSpeech(replyID, 3)
	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: replyID, AudioDurationMs: 120, TimeToFirstByteMs: 5})
	s.eventually(func() bool { return s.heldMs(replyID) >= 100 }, "what followed it was not held behind it")
	// The words that follow are answered by a reply that never gets going, so any audio there is
	// can only be the one that was held.
	s.model.mu.Lock()
	s.model.then = []string{}
	s.model.mu.Unlock()

	s.says(caller, "actually make it for four")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the new words did not cancel the reply while a sentence of it was held")
	interrupted, _ := firstOf[Interrupted](s.reported())
	s.Equal(replyID, interrupted.TurnID)
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(replyID, turn.TurnID)
	s.True(turn.Interrupted)
	s.InDelta(100, turn.AudioDroppedMs, 0.001, "the speech that was held and never let out was lost from the turn")
	s.never(func() bool { return len(edge.published()) > 1 }, "a sentence of the cancelled reply was let out")
	s.Equal([]int{320}, edge.published(), "only the sentence that was played was published")
	s.Zero(countOf[Spoke](s.reported()), "a reply cut short was reported as spoken")

	// What was generated stays in the history, as it does for any reply that was interrupted, and
	// the next turn is told that the reply may not have been heard in full.
	said := "Hello there. How are you?"
	history := s.agent.History()
	s.Equal(1, countMessagesWithContent(history, said), "the interrupted reply is kept once")
	s.Equal(llm.User, history[len(history)-1].Role)
	s.Equal("actually make it for four", history[len(history)-1].Content)
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the new words were never answered")
	s.Contains(s.model.requests()[1].Instructions, interruptedReplyNote)
}
