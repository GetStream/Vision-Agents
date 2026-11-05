package audiofloor

import (
	"fmt"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// asking is an agent that has its transcriber ask about each pause, ending turns by hearer's
// choice of a point.
type asking struct {
	hearer   *Hearer
	mu       sync.Mutex
	speaking bool
	asked    int
	ruled    []Ruled
	// overheard are the rulings over the agent handed over at once.
	overheard []Ruled
}

func (a *asking) Hearer() *Hearer { return a.hearer }

func (a *asking) Speaking() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.speaking
}

func (a *asking) Overheard(_ stt.Participant, ruled Ruled) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.overheard = append(a.overheard, ruled)
}

func (a *asking) overheardSoFar() []Ruled {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]Ruled(nil), a.overheard...)
}

// speaks has the agent talking, or not.
func (a *asking) speaks(speaking bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.speaking = speaking
}

func (a *asking) Moment(stt.Participant) harness.FlowTurn {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.asked++
	return harness.FlowTurn{Instructions: "You book ferry crossings.", Speaking: a.speaking, Reply: "Your crossing is"}
}

func (a *asking) Ruled(_ stt.Participant, ruled Ruled) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.ruled = append(a.ruled, ruled)
}

func (a *asking) rulings() []Ruled {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]Ruled(nil), a.ruled...)
}

// PausesSuite is a transcriber whose agent has it ask about each pause.
type PausesSuite struct {
	transcribing
	agent *asking
}

func TestPausesSuite(t *testing.T) {
	suite.Run(t, new(PausesSuite))
}

func (s *PausesSuite) SetupTest() {
	s.transcribing.SetupTest()
	s.agent = &asking{hearer: NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))}
	s.transcriber.AskOnPauses(s.agent)
	s.ending(0.999)
}

// ending has the server answer p_eot, with the floor free, and a ruling to respond.
func (s *PausesSuite) ending(pEOT float64) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.decision = fmt.Sprintf(`{"answer":"respond","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`, pEOT, pEOT)
}

// counts are the server's decisions, those that wrote words too, and its transcripts.
func (s *PausesSuite) counts() (decisions, withWords, transcripts int) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return len(s.server.turns), s.server.withWords, len(s.server.audio) - len(s.server.turns)
}

// The first check of a pause, at the policy's first check counted from the voice's last
// frame, is the one request the pause costs: the decision and the words of the whole
// utterance in one pass. A turn it ends goes out as one final, its ruling handed over first.
func (s *PausesSuite) TestAPauseIsOneRequestAndItsWordsTheFinal() {
	s.say(1000, true)
	s.say(160, false)
	decisions, _, _ := s.counts()
	s.Zero(decisions, "not before the first check")
	s.say(60, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	rulings := s.agent.rulings()
	s.Require().Len(rulings, 1, "handed over before the final")
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue}, rulings[0].Ruling)
	s.Equal("Two tickets to Bilbao, please.", rulings[0].Words)
	s.InDelta(float64(200), float64(rulings[0].Paused.Milliseconds()), 40, "asked 200 ms after the voice")
	s.True(rulings[0].Ended)
	decisions, withWords, transcripts := s.counts()
	s.Equal([3]int{1, 1, 0}, [3]int{decisions, withWords, transcripts})
	s.InDelta(float64(300+1000+200)*SampleRate/1000-104, float64(s.requests()[0]), float64(2*frameSamples), "the whole utterance")
	s.Equal("en,pt-BR", s.server.registered[0].Languages[0]+","+s.server.registered[0].Languages[1], "the callers' languages")

	s.say(4000, false)
	s.none()
	decisions, _, _ = s.counts()
	s.Equal(1, decisions, "an ended utterance is asked about no more")
}

// Over the agent, a caller voiced for overAfter is asked about without waiting for a pause,
// the decision and the words in one request, then every interval while they talk on and the
// agent speaks: a ruling to stop is handed over at once with the words (Overheard), the
// words are shown as an interim, and the utterance goes on, to end at a pause as ever. With
// the agent quiet nothing is asked until a pause.
func (s *PausesSuite) TestACallerTalkingOverTheAgentIsAskedAboutWithoutAPause() {
	s.transcriber.interims = true
	s.say(1000, true)
	first := next[stt.Transcript](&s.transcribing)
	s.True(first.Interim)
	decisions, _, _ := s.counts()
	s.Zero(decisions, "nothing asked while the agent is quiet and the caller talks on")

	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.words = "No, no, stop."
	s.server.mu.Unlock()
	s.say(120, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 1 }, 5*time.Second, time.Millisecond,
		"asked within overPoll of the agent speaking: the caller has been voiced for overAfter")
	ruled := s.agent.overheardSoFar()[0]
	s.True(ruled.Overheard)
	s.True(ruled.Speaking)
	s.Equal(harness.Stop, ruled.Ruling.Floor)
	s.Equal("No, no, stop.", ruled.Words)
	s.InDelta(float64(300+1100), ruled.AudioMs, 40, "the utterance so far")
	shown := next[stt.Transcript](&s.transcribing)
	s.True(shown.Interim)
	s.Equal("No, no, stop.", shown.Text)
	decisions, withWords, transcripts := s.counts()
	s.Equal([3]int{1, 1, 1}, [3]int{decisions, withWords, transcripts}, "the decision and the words in one request; the interim before the agent spoke")

	s.say(220, true)
	s.none()
	s.Len(s.agent.overheardSoFar(), 1, "not before the interval")
	s.say(100, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 2 }, 5*time.Second, time.Millisecond, "asked again after the interval")
	s.True(next[stt.Transcript](&s.transcribing).Interim, "each check's words are shown")
	s.say(300, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 3 }, 5*time.Second, time.Millisecond)
	s.True(next[stt.Transcript](&s.transcribing).Interim)

	s.agent.speaks(false)
	s.ending(0.999)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance, "the utterance went on through the checks over the agent, and ended at its pause")
	s.Len(s.agent.rulings(), 1)
}

// A turn a check ended is over whatever comes next. Speech that resumes before its words are
// written, the caller going on or somebody else in the room, is the next utterance's: the
// final goes out with the words when they come, rather than being thrown away with the pause
// as if the model had never ruled, which left a turn open for as long as the room talked.
func (s *PausesSuite) TestATurnACheckEndedIsOverThoughSpeechResumesBeforeItsWords() {
	s.server.mu.Lock()
	s.server.wordsAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.say(200, true)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.Len(s.agent.rulings(), 1, "the ruling handed over")

	s.say(800, true)
	s.say(220, false)
	again := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, again.Mode)
	s.Equal(int64(2), again.Utterance, "the speech that resumed is the next utterance")
	decisions, withWords, _ := s.counts()
	s.Equal([2]int{2, 2}, [2]int{decisions, withWords})
}

// A pause the model waits on is asked about again every interval, for the decision alone,
// and the turn it ends goes out with the first check's words.
func (s *PausesSuite) TestAPauseTheModelWaitsOnIsAskedAgainForTheDecisionAlone() {
	s.ending(0.5)
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.none()
	s.ending(0.999)
	s.say(200, false)
	s.none()
	s.say(120, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.InDelta(500, float64(s.agent.rulings()[0].Paused.Milliseconds()), 40, "the second check, 300 ms after the first")
	decisions, withWords, transcripts := s.counts()
	s.Equal([3]int{2, 1, 0}, [3]int{decisions, withWords, transcripts})
}

// The session's choice of a point reads each check: a point that asks for two checks in a
// row ends the turn on the second, 300 ms after the first, with the first check's words.
func (s *PausesSuite) TestTheSessionsPointReadsEachCheck() {
	s.agent.hearer = NewHearer(s.client, EndOfTurn{MaxWaitMs: 1400}, slog.New(slog.DiscardHandler))
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.none()
	s.say(320, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	ruled := s.agent.rulings()[0]
	s.Equal("max_wait_ms:1400", ruled.Ending.Name)
	s.Equal(2, ruled.InARow)
	s.InDelta(500, float64(ruled.Paused.Milliseconds()), 40)
	decisions, withWords, _ := s.counts()
	s.Equal([2]int{2, 1}, [2]int{decisions, withWords})
}

// A check is asked once the one before has answered, so a slow answer makes the next check
// late. It is still the one after the last: a point of two checks in a row ends the turn on
// it, rather than at the cap.
func (s *PausesSuite) TestACheckAfterASlowAnswerIsStillTheNextInTheRow() {
	s.agent.hearer = NewHearer(s.client, EndOfTurn{MaxWaitMs: 1400}, slog.New(slog.DiscardHandler))
	s.server.mu.Lock()
	s.server.decideAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	// The caller stays quiet for twice the interval while the first check is answered.
	s.say(600, false)
	s.answered()
	s.server.mu.Lock()
	s.server.decideAfter = 0
	s.server.mu.Unlock()
	s.say(40, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	ruled := s.agent.rulings()[0]
	s.Equal(2, ruled.InARow)
	s.True(ruled.Ended)
	s.InDelta(860, float64(ruled.Paused.Milliseconds()), 40, "the second check, as soon as the first had answered")
}

// Speech after a pause the model waited on begins another pause, whose first check writes
// the whole utterance again; the check of the pause before answers no one.
func (s *PausesSuite) TestSpeechAfterAPauseBeginsAnother() {
	s.ending(0.5)
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.say(600, true)
	s.ending(0.999)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(int64(1), end.Utterance)
	_, withWords, _ := s.counts()
	s.Equal(2, withWords)
	requests := s.requests()
	s.Greater(requests[1], requests[0]+500*SampleRate/1000, "the whole utterance, the second time too")
	s.Len(s.agent.rulings(), 1)
}

// The utterance a check ended is over: the next speech is the next utterance, with none of
// its audio.
func (s *PausesSuite) TestTheNextUtteranceBeginsAfterTheOneACheckEnded() {
	s.say(1000, true)
	s.say(220, false)
	s.Equal(int64(1), next[stt.Transcript](&s.transcribing).Utterance)
	s.say(100, false)
	s.say(600, true)
	s.say(220, false)
	again := next[stt.Transcript](&s.transcribing)
	s.Equal(int64(2), again.Utterance)
	s.InDelta(float64(100+40+600+200), again.AudioDurationMs, 60, "from where the last ended")
}

// Over the agent the model's ruling is the turn's end, at the first check.
func (s *PausesSuite) TestOverTheAgentTheFirstCheckRules() {
	s.agent.mu.Lock()
	s.agent.speaking = true
	s.agent.mu.Unlock()
	s.server.mu.Lock()
	s.server.decision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"stop"}}`
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.Equal(stt.ModeFinal, next[stt.Transcript](&s.transcribing).Mode)
	ruled := s.agent.rulings()[0]
	s.True(ruled.Speaking)
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Stop}, ruled.Ruling)
	s.True(s.server.turns[0].Speaking)
}

// A noise the model hears no words in ends nothing it sends: no final, no ruling.
func (s *PausesSuite) TestNoWordsAreNothingHeard() {
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	s.say(400, true)
	s.say(3000, false)
	s.none()
	s.Empty(s.agent.rulings())
	_, withWords, transcripts := s.counts()
	s.Equal([2]int{1, 0}, [2]int{withWords, transcripts})
}

// A turn a check ended waits for its words however long they take, past the transcriber's
// own end: a long utterance takes seconds to write, and asking for its words again would
// only take longer.
func (s *PausesSuite) TestATurnACheckEndedWaitsForItsWords() {
	s.server.mu.Lock()
	s.server.wordsAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.say(int((endAfter+checkGrace)/time.Millisecond)+200, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Len(s.agent.rulings(), 1)
	_, withWords, transcripts := s.counts()
	s.Equal([2]int{1, 0}, [2]int{withWords, transcripts})
}

// Words asked for alone, at a pause no check could rule on, are of an utterance that is
// over once a later check has ended it: they are dropped rather than sent after its final,
// where they would be taken for the next utterance's and answered again.
func (s *PausesSuite) TestWordsAskedForAloneAreDroppedOnceACheckEndedTheUtterance() {
	gate := make(chan struct{})
	s.server.mu.Lock()
	s.server.failingDecides = 1
	s.server.transcribeGate = gate
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	// The check failed, so the pause's words are asked for alone, of a server that is slow.
	s.say(20, false)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond)

	// The caller goes on, and the next pause's check ends the utterance on its own words.
	s.say(600, true)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Len(s.agent.rulings(), 1)

	// The slow words come back: nothing of the utterance follows its final.
	close(gate)
	s.none()
}

// A pause the server can't rule on is left to the words alone, as with no agent.
func (s *PausesSuite) TestAPauseTheServerCantRuleOnIsLeftToTheWordsAlone() {
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.say(int(endAfter/time.Millisecond), false)
	for _, mode := range []stt.Mode{stt.ModeReplacement, stt.ModeFinal} {
		s.Equal(mode, next[stt.Transcript](&s.transcribing).Mode)
	}
	s.Empty(s.agent.rulings())
}

// AloneSuite is a transcriber no agent asks on behalf of: transcription alone, or an agent
// that acts on nothing the model says. It asks about each pause for an agent of its own.
type AloneSuite struct{ transcribing }

func TestAloneSuite(t *testing.T) {
	suite.Run(t, new(AloneSuite))
}

// ending has the server answer p_eot, with the floor free, and a ruling to respond.
func (s *AloneSuite) ending(pEOT float64) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.decision = fmt.Sprintf(`{"answer":"respond","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`, pEOT, pEOT)
}

// With no agent asking, the transcriber asks about each pause for an agent of its own, a
// plain voice assistant with no tools, registered once, and the utterance ends when the
// model says so, by the server's default preset: one final at the first check, its words
// from the same pass, rather than at the silence cap.
func (s *AloneSuite) TestWithNoAgentTheModelEndsTheUtteranceUnderTheTranscribersOwn() {
	s.ending(0.999)
	s.say(1000, true)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.InDelta(float64(300+1000+200), end.AudioDurationMs, 40, "the whole utterance, 200 ms into the pause")

	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.Require().Len(s.server.registered, 1)
	s.Equal(neutralInstructions, s.server.registered[0].Instructions)
	s.Empty(s.server.registered[0].Tools)
	s.Equal([]string{"en", "pt-BR"}, s.server.registered[0].Languages)
	s.Len(s.server.turns, 1)
	s.False(s.server.turns[0].Speaking)
	s.Empty(s.server.turns[0].Conversation)
	s.Equal(1, s.server.withWords)
	s.Len(s.server.audio, 1, "one request: the decision and the words in one pass")
}

// A pause the model waits on is checked again every interval, and the utterance ends at the
// cap whatever the model says, on the first check's words: never later than the silence cap
// it ended at before, and no words are asked for twice.
func (s *AloneSuite) TestWithNoAgentAPauseTheModelWaitsOnEndsAtTheCap() {
	s.ending(0.1)
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.none()
	for quiet := 220; quiet < int(endAfter/time.Millisecond); quiet += 300 {
		s.say(300, false)
		s.answered()
	}
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.Equal(1, s.server.withWords, "the words once, with the first check")
	s.Len(s.server.audio, len(s.server.turns), "no words asked for alone")
	s.GreaterOrEqual(len(s.server.turns), 8, "checked every interval to the cap")
}

// A server with no policy for its model can't rule on a pause: it is asked once, and from
// then on the session's utterances end on the words alone, at the silence cap, as they did
// before the model was asked.
func (s *AloneSuite) TestAServerWithNoPolicyIsAskedOnceAndTheWordsComeAlone() {
	s.server.mu.Lock()
	s.server.noPolicy = true
	s.server.mu.Unlock()
	for utterance := int64(1); utterance <= 2; utterance++ {
		s.say(1000, true)
		s.say(220, false)
		s.answered()
		s.say(40, false)
		said := next[stt.Transcript](&s.transcribing)
		s.Equal(stt.ModeReplacement, said.Mode)
		s.Equal(utterance, said.Utterance)
		s.say(int(endAfter/time.Millisecond), false)
		end := next[stt.Transcript](&s.transcribing)
		s.Equal(stt.ModeFinal, end.Mode)
		s.Equal(utterance, end.Utterance)
	}
	s.Equal(1, s.infos(), "asked for its policy once")
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.Empty(s.server.registered)
	s.Empty(s.server.turns)
	s.Len(s.server.audio, 2, "the words of each utterance, once")
}

// A check the server fails to answer leaves that pause to the words alone, and the next
// utterance is asked about again: a server that is busy or restarting costs a request, not
// the session's turn-taking.
func (s *AloneSuite) TestACheckThatFailsLeavesThePauseToTheWordsAloneAndTheNextAsksAgain() {
	s.ending(0.999)
	s.server.mu.Lock()
	s.server.failingDecides = 1
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	s.answered()
	s.say(40, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](&s.transcribing).Mode)
	s.say(int(endAfter/time.Millisecond), false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)

	s.say(1000, true)
	s.say(220, false)
	again := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, again.Mode)
	s.Equal(int64(2), again.Utterance)
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.Len(s.server.turns, 1, "the second utterance's check, answered")
	s.Equal(1, s.server.withWords)
}
