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
}

func (a *asking) Hearer() *Hearer { return a.hearer }

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

// answered waits for the check asked to have answered.
func (s *PausesSuite) answered() {
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return !s.transcriber.pause.inFlight
	}, 5*time.Second, time.Millisecond)
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
