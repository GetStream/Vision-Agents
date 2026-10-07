package agent

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func TestSpeechlessIsOnlyWhatNobodySaid(t *testing.T) {
	for _, text := range []string{
		"", "   ", ".", "...", "?!", "♪", "(coughs)", "[cough]", "*coughs*", "<noise>", "{laughter}",
		"[BLANK_AUDIO]", "(clears throat)", "[people talking in the background]", "(coughs) (laughs)",
		"( coughs ).", "uh", "Um.", "um... uh", "Er, erm", "eh", "Ah.", "Ahem", "hm", "Hmm.", "mm", "mmm",
		"(coughs) um",
	} {
		require.True(t, speechless(text), "%q has no words in it", text)
	}
	for _, text := range []string{
		"hello", "yes", "no", "okay", "uh huh", "Mm hmm.", "mm hmm", "hmm yes", "um yes please",
		"cough", "I have a cough", "I need (two) tables", "five (", "(I would like a table",
		"(I would like a table for four people tonight at seven)", "table (coughs) for two",
		"7:30", "42", "ok (laughs)",
	} {
		require.False(t, speechless(text), "%q is something somebody said", text)
	}
}

func TestANoteAboutASoundIsOnlyOneThatIsClosed(t *testing.T) {
	require.Equal(t, " ", withoutMarkers("(coughs)"))
	require.Equal(t, "table   for two", withoutMarkers("table (coughs) for two"))
	require.Equal(t, "(coughs", withoutMarkers("(coughs"), "a note that never closes was cut off, so it is words")
	require.Equal(t, "naïve   café", withoutMarkers("naïve [x] café"), "multibyte text is kept whole")
}

func TestNoiseAndSpeechlessTextNeverTakeTheFloorFromAReply(t *testing.T) {
	for _, text := range []string{"(coughs)", "[background noise]", "um um", "*sighs*", "uh"} {
		require.False(t, substantiveBargeIn(text, "The menu has three courses."), "%q took the floor", text)
	}
	require.True(t, substantiveBargeIn("stop", "The menu has three courses."))
	require.True(t, substantiveBargeIn("actually make it for four", "The menu has three courses."))
}

func TestWithdrawingAReplyOnlyTakesBackWhatNobodyHeardAndNothingItDid(t *testing.T) {
	user := llm.Message{Role: llm.User, Content: "words"}
	reply := llm.Message{Role: llm.Assistant, Content: "an answer"}
	earlier := []llm.Message{{Role: llm.User, Content: "before"}, {Role: llm.Assistant, Content: "then"}}
	history := func(rest ...llm.Message) []llm.Message {
		return append(append([]llm.Message(nil), earlier...), rest...)
	}
	held := func(committed int) heldReply {
		return heldReply{turn: "turn-1", asked: len(earlier) + 1, committed: committed}
	}
	cases := []struct {
		name    string
		history []llm.Message
		held    heldReply
		pending int
		want    int
	}{
		{"a reply still being written", history(user), held(0), 0, len(earlier)},
		{"a finished reply nobody has heard", history(user, reply), held(len(earlier) + 2), 0, len(earlier)},
		{"a reply that asked for a tool",
			history(user, llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{Name: "lookup"}}}),
			held(len(earlier) + 2), 0, -1},
		{"a tool still running", history(user), held(0), 1, -1},
		{"something added after the reply", history(user, reply, user), held(len(earlier) + 2), 0, -1},
		{"a reply that has begun to be heard", history(user, reply), heldReply{asked: len(earlier) + 1}, 0, -1},
		{"another turn's reply", history(user, reply), heldReply{turn: "turn-2", asked: len(earlier) + 1}, 0, -1},
		{"words that were never recorded", history(user, reply), heldReply{turn: "turn-1"}, 0, -1},
		{"a history that was rewritten", []llm.Message{user}, held(0), 0, -1},
		{"a history whose words are not the caller's", history(reply, reply), held(0), 0, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := &Agent{history: tc.history, gated: tc.held, pendingTools: tc.pending}
			require.Equal(t, tc.want, a.withdrawableLocked("turn-1"))
		})
	}
}

// flowOnlyBesideTheReply says that the flow controller was asked at most once about a turn the
// acoustic score decided, which is beside the reply and never before it. Whether it was asked at all
// depends on whether the reply had been let out by the time the turn was carried out, which a test
// whose reply is not held does not control.
func (s *AgentSuite) flowOnlyBesideTheReply(message string) {
	s.T().Helper()
	s.LessOrEqual(len(s.flow.requests()), 1, message)
}

// overlapAsks is how many times the flow controller was asked about a caller talking over the agent.
func (s *AgentSuite) overlapAsks() int {
	var asks int
	for _, request := range s.flow.requests() {
		if strings.HasPrefix(request.ID, overlapPrefix) {
			asks++
		}
	}
	return asks
}

// decisionsOf are the kinds of judgement the agent reported, in order.
func (s *AgentSuite) decisionsOf(kind ActionKind) []Decided {
	var found []Decided
	for _, event := range s.reported() {
		if decided, ok := event.(Decided); ok && decided.Kind == string(kind) {
			found = append(found, decided)
		}
	}
	return found
}

// heldPrimaryReply joins an agent in primary mode whose replies wait for the caller to have been
// quiet for the window, and for no longer than the longest hold, and whose acoustic score for any
// words is the one given.
func (s *AgentSuite) heldPrimaryReply(window, longest time.Duration, score float64) (*atomic.Int64, *markedEdge) {
	var scores atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, score, &scores))
	s.replySilence, s.replySilenceMax = &window, &longest
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 30 * time.Millisecond}
		return edge
	}
	s.join(true)
	return &scores, edge
}

// endsClearly has a participant who has been heard to talk say words whose end the acoustic score
// is sure of.
func (s *AgentSuite) endsClearly(participant stt.Participant, text string) {
	s.speakAloud(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	s.says(participant, text)
}

func (s *AgentSuite) TestWordsThatAreOnlyASoundStartNoReplyWhateverTheScore() {
	for _, text := range []string{"...", "(coughs)", "[BLANK_AUDIO]", "uh", "hmm", "(laughs) ah"} {
		s.Run(text, func() {
			s.SetupTest()
			var scores atomic.Int64
			s.primaryEOTServer(primaryScoreHandler(s, 0.99, &scores))
			s.join(true)
			alice := stt.Participant{ID: "alice"}
			s.endsClearly(alice, text)

			if words(text) != "" {
				// Punctuation alone never becomes a turn to be set aside.
				s.eventually(func() bool { return len(s.decisionsOf(ActIgnore)) == 1 },
					"the sound was never set aside")
			}
			s.never(func() bool {
				return len(s.model.requests()) > 0 || len(s.voice.spoken()) > 0 ||
					countOf[Responding](s.reported()) > 0 || scores.Load() > 0 || len(s.flow.requests()) > 0
			}, "words that are only a sound were answered, scored or put to the flow controller")
			s.Empty(s.agent.History())
			s.Empty(s.edge.heard())

			// What comes after is heard as it always was.
			s.says(alice, "please find a table")
			s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
				"the caller was not answered after the sound")
			s.EqualValues(1, scores.Load(), "only the words were scored")
			s.Equal("please find a table", s.agent.History()[0].Content)
		})
	}
}

func (s *AgentSuite) TestAHesitationRevisedIntoTheWordsAfterItIsStillAnswered() {
	// A transcriber that finalizes "Um." and then revises the same utterance into what came
	// after must not be taken for repeating something that was already answered.
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "Um.", Utterance: 5, Language: "en", Confidence: 1})
	s.eventually(func() bool { return len(s.decisionsOf(ActIgnore)) == 1 }, "the hesitation was never set aside")

	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "I would like a table.", Utterance: 5, Language: "en", Confidence: 1})

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words after the hesitation were never answered")
	s.Equal("I would like a table.", s.agent.History()[0].Content)
}

func (s *AgentSuite) TestAFlowRulingThatTheWordsWereNotForTheAgentWithdrawsTheHeldReplyUnheard() {
	_, edge := s.heldPrimaryReply(time.Minute, time.Minute, 0.95)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "my brother is asking about dinner")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the ruling did not withdraw the reply that nobody had heard")
	responding, _ := firstOf[Responding](s.reported())
	interrupted, _ := firstOf[Interrupted](s.reported())
	s.Equal(responding.TurnID, interrupted.TurnID)
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.True(turn.Interrupted)
	s.Zero(turn.FirstFrameQueuedMs, "a frame of the withdrawn reply was queued")
	s.never(func() bool { return len(s.edge.heard()) > 0 || edge.markedWrites() > 0 },
		"a frame of the withdrawn reply was emitted")
	s.Zero(countOf[Spoke](s.reported()), "a reply nobody heard was reported as spoken")
	s.Empty(s.agent.History(), "words that were not for the agent entered the conversation")
	s.False(s.interruptionNotePending(), "there is no reply that may have been heard in part")
	ignored := s.decisionsOf(ActIgnore)
	s.Require().Len(ignored, 1)
	s.Equal(responding.TurnID, ignored[0].TurnID)
	s.Contains(ignored[0].Reason, "not meant for the agent")
	s.Empty(s.agent.addressedTurns(), "the ruling was not used up")

	// The caller who does talk to the agent is answered, as if the words had never been said.
	s.endsClearly(alice, "please find a table")
	s.eventually(func() bool { return len(s.agent.History()) == 2 }, "the caller was not answered")
	s.Equal("please find a table", s.agent.History()[0].Content)
	last := s.model.requests()[len(s.model.requests())-1]
	s.Zero(countMessagesWithContent(last.Input, "my brother is asking about dinner"),
		"the model was shown words that were ruled not to be for the agent")
}

func (s *AgentSuite) TestTheWordsAreAskedAboutAsTheyWereBeforeAnAcousticScoreDecidedThem() {
	s.heldPrimaryReply(time.Minute, time.Minute, 0.95)
	s.flow.reply = []string{`{"disposition":"respond","floor":"continue"}`}
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "please find a table")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller was not answered")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was not asked beside the reply")

	question := s.flow.requests()[0].Input[0].Content
	s.Contains(question, "(nothing said yet)", "the question carried the words it was about as history")
	s.Contains(question, "The agent is not speaking.")
	s.Contains(question, `has just said: "please find a table"`)
	s.NotContains(question, "different voice")
	s.never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "a ruling to respond took the reply back")
	s.Len(s.agent.History(), 2)
}

func (s *AgentSuite) TestAnythingButIgnoreLeavesTheReplyAsItIs() {
	for _, ruling := range []string{
		`{"disposition":"respond","floor":"stop"}`,
		`{"disposition":"clarify","floor":"continue"}`,
		`{"disposition":"wait","floor":"continue"}`,
	} {
		s.Run(ruling, func() {
			s.SetupTest()
			s.heldPrimaryReply(time.Minute, time.Minute, 0.95)
			s.flow.reply = []string{ruling}
			alice := stt.Participant{ID: "alice"}

			s.endsClearly(alice, "please find a table")

			s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was not asked")
			s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller was not answered")
			s.never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "the reply was taken back")
			s.Len(s.agent.History(), 2)
		})
	}
}

func (s *AgentSuite) TestARulingAfterTheReplyHasBeenLetOutDoesNotCutIt() {
	hold := make(chan struct{})
	s.T().Cleanup(func() {
		select {
		case <-hold:
		default:
			close(hold)
		}
	})
	_, edge := s.heldPrimaryReply(300*time.Millisecond, 400*time.Millisecond, 0.95)
	s.flow.mu.Lock()
	s.flow.holdCreate = hold
	s.flow.mu.Unlock()
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "my brother is asking about dinner")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was not asked")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was never let out")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.Empty(s.agent.addressedTurns(), "the controller was still being waited on once the reply had begun to be heard")

	// Whatever the controller goes on to say, the reply has been heard, and what it said is not
	// taken back. The ruling is delivered the way the harness delivers one.
	turnID := respondingTurn(s.reported(), 0)
	s.agent.mu.Lock()
	s.agent.addressed[turnID] = candidate{ID: turnID, Participant: alice, Text: "my brother is asking about dinner"}
	s.agent.mu.Unlock()
	s.True(s.agent.addresseeRuled(harness.Decided{
		CandidateID: turnID, Disposition: harness.Ignore, Floor: harness.Continue,
	}))
	close(hold)

	s.never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "a ruling that came too late cut the reply")
	s.Positive(edge.markedWrites())
	s.Len(s.agent.History(), 2, "what was heard was taken out of the conversation")
}

func (s *AgentSuite) TestASlowFlowControllerNeverDelaysTheReply() {
	hold := make(chan struct{})
	s.T().Cleanup(func() {
		select {
		case <-hold:
		default:
			close(hold)
		}
	})
	_, edge := s.heldPrimaryReply(200*time.Millisecond, 250*time.Millisecond, 0.95)
	s.flow.mu.Lock()
	s.flow.holdCreate = hold
	s.flow.mu.Unlock()
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "please find a table")

	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was not asked")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 && edge.markedWrites() > 0 },
		"the reply waited for a flow controller that never answered")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply was not spoken")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Len(s.agent.History(), 2)
}

func (s *AgentSuite) TestAFlowControllerThatFailsLeavesTheReplyAlone() {
	_, edge := s.heldPrimaryReply(200*time.Millisecond, 250*time.Millisecond, 0.95)
	s.flow.mu.Lock()
	s.flow.refuses = errors.New("the controller is down")
	s.flow.mu.Unlock()
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "please find a table")

	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the flow controller was not asked")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 && edge.markedWrites() > 0 },
		"a controller that failed cost the caller their reply")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply was not spoken")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Len(s.agent.History(), 2)
}

func (s *AgentSuite) TestNothingIsAskedWhenNoReplyIsHeldForTheCaller() {
	zero := time.Duration(0)
	s.replySilence = &zero
	var scores atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, 0.95, &scores))
	s.join(true)
	alice := stt.Participant{ID: "alice"}

	s.endsClearly(alice, "please find a table")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller was not answered")
	s.never(func() bool { return len(s.flow.requests()) > 0 },
		"a ruling was asked for that could not arrive before the reply was let out")
}

func (s *AgentSuite) TestWordsInAnotherVoiceAreNotAnsweredOnTheScoreAlone() {
	scores, _ := s.heldPrimaryReply(time.Minute, time.Minute, 0.99)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	alice := stt.Participant{ID: "alice"}
	s.speakAloud(alice)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(alice.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	// The first voice heard on the track is the caller's.
	s.agent.mu.Lock()
	s.agent.voices[alice.ID] = "caller-voice"
	s.agent.mu.Unlock()

	s.saysInVoice(alice, "mom where is my backpack", "another-voice")

	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the other voice was never put to the controller")
	s.Contains(s.flow.requests()[0].Input[0].Content, "in a different voice")
	s.never(func() bool {
		return len(s.model.requests()) > 0 || len(s.voice.spoken()) > 0 || countOf[Responding](s.reported()) > 0
	}, "a reply was started for somebody else in the room")
	s.Zero(scores.Load(), "the acoustic score decided words that were not the caller's")
	s.Empty(s.agent.History())

	s.saysInVoice(alice, "what time is my flight", "caller-voice")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller was not answered")
	s.EqualValues(1, scores.Load(), "the caller's own words are decided by the score")
	s.Equal("what time is my flight", s.agent.History()[0].Content)
}

// addressedTurns are the turns whose flow ruling on who they were for is still awaited.
func (a *Agent) addressedTurns() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	turns := make([]string, 0, len(a.addressed))
	for id := range a.addressed {
		turns = append(turns, id)
	}
	return turns
}

func (s *ConverseSuite) TestWordsThatAreOnlyASoundAreNotPutToAnybody() {
	for _, text := range []string{"(coughs)", "[BLANK_AUDIO]", "*sighs*", "uh", "Hmm."} {
		s.Run(text, func() {
			s.build(DuplexOptions{})
			s.converse.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: text}, s.quiet())

			action := s.converse.Settled(s.held(), s.quiet())

			s.Equal(ActIgnore, action.Kind, "words that are only a sound were put to the controller")
			s.converse.mu.Lock()
			s.Empty(s.converse.candidates, "the sound was kept as a turn to be ruled on")
			s.converse.mu.Unlock()
			s.eventually(func() bool { return len(s.decisions()) == 1 }, "the sound was not reported as set aside")
			s.Equal(string(ActIgnore), s.decisions()[0].Kind)

			// The next words are a turn of their own, and are put as usual.
			next := s.converse.Settled(s.heldAfter("book a table"), s.quiet())
			s.Equal(ActAsk, next.Kind)
			s.Equal("book a table", next.Candidate.Text)
		})
	}
}

// heldAfter says something and waits for the words to hold still.
func (s *ConverseSuite) heldAfter(text string) candidate {
	s.converse.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: text}, s.quiet())
	return s.held()
}

func (s *ConverseSuite) TestASoundOverTheReplyIsNotAskedAbout() {
	for _, text := range []string{"(coughs)", "[background noise]", "um um", "*sighs*"} {
		s.Run(text, func() {
			s.build(DuplexOptions{})
			s.Empty(s.overheard(text, s.talking()), "the controller was asked whether a sound should stop the reply")
		})
	}
	s.build(DuplexOptions{})
	s.Len(s.overheard("actually make it for four", s.talking()), 1, "words that are more than a sound are asked about")
}
