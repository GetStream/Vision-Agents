package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/audioturn"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// barrierPlayoutEdge makes the last edge write observable without introducing another
// publisher goroutine or changing the production edge contract. In held mode, a write
// that already passed admission waits at the commit point and checks its epoch again.
type barrierPlayoutEdge struct {
	*loopbackEdge
	entered chan struct{}
	release chan struct{}
	dropped chan struct{}

	writeOnce   sync.Once
	dropOnce    sync.Once
	releaseOnce sync.Once
	armed       atomic.Bool
	held        bool
}

func newBarrierPlayoutEdge(base *loopbackEdge, held, armed bool) *barrierPlayoutEdge {
	e := &barrierPlayoutEdge{
		loopbackEdge: base,
		entered:      make(chan struct{}),
		release:      make(chan struct{}),
		dropped:      make(chan struct{}),
		held:         held,
	}
	e.armed.Store(armed)
	return e
}

func (e *barrierPlayoutEdge) PublishAudioContext(ctx context.Context, pcm audio.PcmData) error {
	if !e.armed.Load() {
		return e.loopbackEdge.PublishAudio(pcm)
	}
	e.writeOnce.Do(func() { close(e.entered) })
	if e.held {
		<-e.release
	} else {
		select {
		case <-e.release:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return e.loopbackEdge.PublishAudio(pcm)
}

func (e *barrierPlayoutEdge) arm() { e.armed.Store(true) }

func (e *barrierPlayoutEdge) DropSpeech() {
	e.loopbackEdge.DropSpeech()
	e.dropOnce.Do(func() { close(e.dropped) })
}

func (e *barrierPlayoutEdge) releaseWrite() {
	e.releaseOnce.Do(func() { close(e.release) })
}

func setStubVoiceSilent(voice *stubTTS) {
	voice.mu.Lock()
	voice.silent = true
	voice.mu.Unlock()
}

func (s *AgentSuite) TestPrimaryPartialDropsQueuedAndInFlightAudioBeforeProviderInterruptReturns() {
	secondScoreEntered := make(chan struct{})
	allowSecondScore := make(chan struct{})
	var releaseScore sync.Once
	s.T().Cleanup(func() { releaseScore.Do(func() { close(allowSecondScore) }) })
	var secondScoreOnce sync.Once
	var scoreRequests atomic.Int32
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read acoustic frame: %v", err)
			return
		}
		request := scoreRequests.Add(1)
		if request == 2 {
			secondScoreOnce.Do(func() { close(secondScoreEntered) })
		}
		if request >= 2 {
			select {
			case <-allowSecondScore:
			case <-r.Context().Done():
				return
			}
		}
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.9)
	}))
	var barrier *barrierPlayoutEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		barrier = newBarrierPlayoutEdge(base, true, false)
		return barrier
	}
	s.join(true)
	s.T().Cleanup(barrier.releaseWrite)
	setStubVoiceSilent(s.voice)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "Tell me about the dinner menu")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "the initial high acoustic score did not start a reply")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the first synthesis was not registered")
	firstTurn := s.model.requests()[0].ID
	s.Require().NotEmpty(firstTurn)
	s.synthesises(firstTurn)
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the first audio chunk did not enter the playout queue")
	firstEpoch, active := s.agent.synthesisContext(firstTurn)
	s.Require().True(active)
	oldTailID := firstTurn + "/sentence/held-tail"
	s.Require().True(s.agent.registerSynthesis(firstTurn, oldTailID, true))
	oldTailEpoch, active := s.agent.synthesisContext(oldTailID)
	s.Require().True(active)

	// The second provider chunk passes admission but is held just before the edge commit.
	// The first chunk is already queued, so interruption must clear both kinds of audio.
	barrier.arm()
	s.synthesises(firstTurn)
	select {
	case <-barrier.entered:
	case <-time.After(settleFor):
		s.FailNow("the second synthesis chunk never reached the edge commit barrier")
	}

	flowRequests := len(s.flow.requests())
	s.flow.mu.Lock()
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.flow.mu.Unlock()
	s.mutters(participant, "okay")
	s.eventually(func() bool { return len(s.flow.requests()) > flowRequests }, "a brief acknowledgement did not remain on the semantic overlap path")
	s.eventually(func() bool {
		s.agent.converse.mu.Lock()
		defer s.agent.converse.mu.Unlock()
		seen, ok := s.agent.converse.overlapping[participant.ID]
		return ok && seen.inflight == ""
	}, "the acknowledgement's semantic overlap ruling did not settle")
	afterAcknowledgement := len(s.flow.requests())
	voiceRequestsBeforeFinal := len(s.voice.spoken())
	respondingBeforeFinal := countOf[Responding](s.reported())
	respondedBeforeFinal := countOf[Responded](s.reported())
	s.True(s.agent.speaking(firstTurn), "an acknowledgement must not fast-stop the current reply")
	s.Len(s.edge.heard(), 1, "an acknowledgement must leave queued speech alone")
	s.Zero(countOf[Interrupted](s.reported()))

	interruptEntered := make(chan struct{})
	interruptRelease := make(chan struct{})
	var releaseInterrupt sync.Once
	s.T().Cleanup(func() { releaseInterrupt.Do(func() { close(interruptRelease) }) })
	s.voice.mu.Lock()
	s.voice.interruptEntered = interruptEntered
	s.voice.interruptRelease = interruptRelease
	s.voice.mu.Unlock()

	s.speak(participant)
	s.mutters(participant, "actually stop and show me the other options")
	select {
	case <-interruptEntered:
	case <-time.After(settleFor):
		s.FailNow("the substantive primary partial never reached provider cleanup")
	}
	select {
	case <-barrier.dropped:
	case <-time.After(settleFor):
		s.FailNow("the local edge queue was not dropped before provider cleanup")
	}

	s.Empty(s.edge.heard(), "queued speech must be silent while the provider interrupt is held")
	s.agent.mu.Lock()
	s.Equal("", s.agent.speakingTurn, "the caller takes the local floor before TTS cleanup returns")
	s.NotNil(s.agent.interruptDone, "the provider cleanup remains in progress behind its barrier")
	s.agent.mu.Unlock()
	s.ErrorIs(firstEpoch.Err(), context.Canceled, "the active synthesis epoch must be invalidated at local stop")
	s.ErrorIs(oldTailEpoch.Err(), context.Canceled, "every synthesis ID from the old epoch must be invalidated")
	s.Zero(countOf[Interrupted](s.reported()), "the terminal event follows provider cleanup")
	s.Len(s.model.requests(), 1, "provisional words are not answered")

	// Queue the settled text while STT is still waiting for provider cleanup. It must not
	// be answered until its own primary EOT request returns.
	s.says(participant, "Actually stop and show me the other options")
	releaseInterrupt.Do(func() { close(interruptRelease) })
	barrier.releaseWrite()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the interrupted turn was not reported after cleanup")
	select {
	case <-secondScoreEntered:
	case <-time.After(settleFor):
		s.FailNow("the settled replacement words were not scored")
	}
	s.Equal(respondingBeforeFinal, countOf[Responding](s.reported()), "the settled text cannot start a reply before its score")
	s.Equal(respondedBeforeFinal, countOf[Responded](s.reported()), "the settled text cannot finish a reply before its score")
	s.Len(s.voice.spoken(), voiceRequestsBeforeFinal, "speculative model work must not start TTS before the score")
	s.Empty(s.edge.heard(), "no replacement audio can be published before the score")
	s.Len(s.flow.requests(), afterAcknowledgement, "primary mode must not route this candidate through semantic flow")

	releaseScore.Do(func() { close(allowSecondScore) })
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 }, "the final answer did not start after EOT")
	var responding []Responding
	for _, event := range s.reported() {
		if typed, ok := event.(Responding); ok {
			responding = append(responding, typed)
		}
	}
	s.Require().GreaterOrEqual(len(responding), 2)
	secondTurn := responding[1].TurnID
	s.Require().NotEmpty(secondTurn)
	s.eventually(func() bool {
		_, ok := s.agent.synthesisContext(secondTurn)
		return ok
	}, "the next synthesis did not begin in a fresh epoch")

	// An abandoned ID remains tied to the canceled epoch. Its late chunk and completion
	// may neither publish nor settle the newer synthesis.
	newCtx, ok := s.agent.synthesisContext(secondTurn)
	s.Require().True(ok)
	beforeUtterances := func() int {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.utterances
	}()
	s.synthesises(firstTurn)
	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: firstTurn, AudioDurationMs: 10})
	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: oldTailID,
		Audio:       audio.PcmData{Samples: []int16{3, 4}, SampleRate: 16_000, Channels: 1},
	})
	s.voice.emitter.Send(tts.Error{Provider: "stub", SynthesisID: oldTailID, Err: errors.New("late old synthesis error")})
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		_, firstRetained := s.agent.synthesisCtx[firstTurn]
		_, tailRetained := s.agent.synthesisCtx[oldTailID]
		return !firstRetained && !tailRetained
	}, "the old completion did not retire its canceled ID")
	s.agent.mu.Lock()
	gotUtterances := s.agent.utterances
	currentTurn := s.agent.speakingTurn
	s.agent.mu.Unlock()
	s.Equal(beforeUtterances, gotUtterances, "an old completion cannot settle a new epoch's utterance")
	s.Equal(secondTurn, currentTurn)
	s.Empty(s.edge.heard(), "late audio from the abandoned ID cannot reappear in the new epoch")
	s.NoError(newCtx.Err())

	// A delayed semantic interrupt naming the old turn is stale even while a tool is
	// registered for the new one.
	toolCtx, cancelTool := context.WithCancel(context.Background())
	defer cancelTool()
	s.agent.mu.Lock()
	if s.agent.toolCancels == nil {
		s.agent.toolCancels = make(map[string]context.CancelFunc)
	}
	s.agent.toolCancels[secondTurn] = cancelTool
	s.agent.mu.Unlock()
	s.agent.perform(Action{Kind: ActInterrupt, TurnID: firstTurn, Participant: participant})
	s.True(s.agent.speaking(secondTurn), "an old interrupt must not stop the current reply")
	s.NoError(toolCtx.Err(), "a stale action must not cancel current-turn work")

	s.synthesises(secondTurn)
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the new epoch's audio did not reach playout")
	s.Equal(1, countOf[Interrupted](s.reported()), "the stale interrupt must not report another interruption")
	barrier.releaseWrite()
}

func (s *AgentSuite) TestPrimaryPartialGuardRejectsNoiseEchoOtherSpeakerAndBackchannel() {
	participant := stt.Participant{ID: "caller"}
	state := floor{Speaking: "turn-active", Reply: "The menu has three courses and a seasonal special."}
	partial := func(text, speaker string, mode stt.Mode) stt.Transcript {
		return stt.Transcript{Participant: participant, Mode: mode, Speaker: speaker, Text: text}
	}
	for _, tc := range []struct {
		name       string
		transcript stt.Transcript
		state      floor
		mode       EOTMode
		eot        bool
	}{
		{name: "acknowledgement", transcript: partial("okay", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "noise", transcript: partial("cough", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "marked noise", transcript: partial("(background noise)", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "hesitations", transcript: partial("um um", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "exact word-boundary echo", transcript: partial("menu has three", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "different speaker", transcript: partial("actually stop the answer", "another-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary, eot: true},
		{name: "backchannel", transcript: partial("actually stop the answer", "caller-voice", stt.ModeReplacement), state: floor{Speaking: backchannelPrefix + "murmur", Reply: state.Reply}, mode: EOTModePrimary, eot: true},
		{name: "quiet floor", transcript: partial("actually stop the answer", "caller-voice", stt.ModeReplacement), state: floor{Quiet: true, Reply: state.Reply}, mode: EOTModePrimary, eot: true},
		{name: "final transcript", transcript: partial("actually stop the answer", "caller-voice", stt.ModeFinal), state: state, mode: EOTModePrimary, eot: true},
		{name: "nonprimary mode", transcript: partial("actually stop the answer", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModeGate, eot: true},
		{name: "EOT disabled", transcript: partial("actually stop the answer", "caller-voice", stt.ModeReplacement), state: state, mode: EOTModePrimary},
	} {
		s.Run(tc.name, func() {
			options := Options{EOTMode: tc.mode}
			if tc.eot {
				options.EOT = &audioturn.Client{}
			}
			agent := &Agent{
				options:      options,
				voices:       map[string]string{participant.ID: "caller-voice"},
				speakingTurn: state.Speaking,
			}
			s.False(agent.primaryPartialInterrupt(tc.transcript, tc.state), "this transcript must not bypass semantic overlap handling")
		})
	}
	options := Options{EOT: &audioturn.Client{}, EOTMode: EOTModePrimary}
	agent := &Agent{options: options, voices: map[string]string{participant.ID: "caller-voice"}, speakingTurn: state.Speaking}
	s.True(agent.primaryPartialInterrupt(partial("stop", "caller-voice", stt.ModeReplacement), state), "a single explicit stop word takes the floor")
	s.True(agent.primaryPartialInterrupt(partial("actually stop and give me another option", "caller-voice", stt.ModeReplacement), state),
		"substantive speech from the known caller can take the local floor immediately")
	s.True(agent.primaryPartialInterrupt(partial("menus are different now", "caller-voice", stt.ModeReplacement), state),
		"a longer word that contains an echoed token is not an exact word-sequence echo")
}

func (s *AgentSuite) TestQueuedOnlyPlayoutTailKeepsTheFloorActiveAndOlderAudioSurvivesTurnRollover() {
	s.join(true)
	setStubVoiceSilent(s.voice)
	oldTurn, oldSynthesis := "turn-old", "turn-old/sentence/0"
	s.agent.mu.Lock()
	s.agent.speakingTurn = oldTurn
	s.agent.generating = true
	s.agent.mu.Unlock()
	s.Require().True(s.agent.registerSynthesis(oldTurn, oldSynthesis, true))

	// The old synthesis remains valid while a newer speakingTurn is admitted. Epoch
	// cancellation, not the mutable current-turn pointer, controls its queued tail.
	s.agent.mu.Lock()
	s.agent.speakingTurn = "turn-new"
	s.agent.generating = false
	s.agent.utterances = 0
	s.agent.mu.Unlock()
	s.edge.holdSpeech(true)
	state := s.agent.floor()
	s.False(state.Quiet, "an edge-only queued tail keeps the floor active after generation ends")
	s.Equal("turn-new", state.Speaking)

	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: oldSynthesis,
		Audio:       audio.PcmData{Samples: []int16{1, 2}, SampleRate: 16_000, Channels: 1},
	})
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "an already-admitted older tail was dropped on ordinary turn rollover")
	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: oldSynthesis, AudioDurationMs: 1})
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		_, ok := s.agent.synthesisCtx[oldSynthesis]
		return !ok
	}, "the older tail did not settle its exact synthesis ID")
	s.True(s.agent.speaking("turn-new"), "settling the old tail cannot clear the current turn")
	s.False(s.agent.floor().Quiet, "the playout queue remains active until the edge hears it")
}

func (s *AgentSuite) TestCloseCancelsAContextBoundPublisherWaitingAtTheEdge() {
	var barrier *barrierPlayoutEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		barrier = newBarrierPlayoutEdge(base, false, true)
		return barrier
	}
	s.join(true)
	s.T().Cleanup(barrier.releaseWrite)
	setStubVoiceSilent(s.voice)
	participant := stt.Participant{ID: "caller"}

	// Start the normal cascade answer path, then hold the first audio write in the
	// edge. Close must cancel its publication epoch before waiting for pipeline readers.
	s.Require().NoError(s.agent.respondTurn("turn-close", participant, "hello", heard{}, "", nil, options.LLM{}))
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the close test did not start synthesis")
	turn := s.model.requests()[0].ID
	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: turn,
		Audio:       audio.PcmData{Samples: []int16{1, 2}, SampleRate: 16_000, Channels: 1},
	})
	select {
	case <-barrier.entered:
	case <-time.After(settleFor):
		s.FailNow("audio did not block at the context-aware edge")
	}

	closed := make(chan error, 1)
	go func() { closed <- s.agent.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
	case <-time.After(settleFor):
		s.FailNow("Close left a context-bound edge publisher blocked")
	}
	s.Empty(s.edge.heard(), "a write canceled by Close must not be published afterward")
	barrier.releaseWrite()
}

func (s *AgentSuite) TestFailedSynthesisContinuationCannotRecreateARetiredID() {
	s.join(true)
	turnID, synthesisID := "turn-failed", "turn-failed/sentence/0"
	s.agent.mu.Lock()
	s.agent.speakingTurn = turnID
	s.agent.mu.Unlock()
	s.True(s.agent.registerSynthesis(turnID, synthesisID, true))
	s.True(s.agent.finishSynthesis(synthesisID))
	s.False(s.agent.registerSynthesis(turnID, synthesisID, false), "a late continuation cannot resurrect a terminal synthesis")
	s.agent.mu.Lock()
	_, retained := s.agent.synthesisCtx[synthesisID]
	utterances := s.agent.utterances
	s.agent.mu.Unlock()
	s.False(retained)
	s.Zero(utterances)
}
