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

// FloorSuite is a floor that rules alone (NewFloor): the server is asked about each pause and
// about the caller talking over the agent, and another transcriber writes the words, handed
// over as it writes them.
type FloorSuite struct {
	transcribing
	agent *asking

	mu sync.Mutex
	// settles says the other transcriber can be asked for its final, and asked counts the
	// times it was.
	settles bool
	asked   int
}

func TestFloorSuite(t *testing.T) {
	suite.Run(t, new(FloorSuite))
}

func (s *FloorSuite) SetupTest() {
	s.settles, s.asked = true, 0
	s.rulesAlone = true
	s.finalize = func() (bool, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.settles {
			s.asked++
		}
		return s.settles, nil
	}
	s.transcribing.SetupTest()
	s.agent = &asking{hearer: NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))}
	s.transcriber.AskOnPauses(s.agent)
	s.ending(0.999)
}

// ending has the server answer p_eot, with the floor free, and a ruling to respond.
func (s *FloorSuite) ending(pEOT float64) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.decision = fmt.Sprintf(`{"answer":"respond","answer_probability":0.99,"p_eot":%v,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`, pEOT)
}

// writes hands the floor what the other transcriber wrote of run, and says whether it took it.
func (s *FloorSuite) writes(run int64, mode stt.Mode, text string) bool {
	return s.transcriber.Heard(stt.Transcript{Participant: stt.Participant{ID: "p"}, Mode: mode, Utterance: run, Text: text,
		Provider: "stub", Model: "stub-1"})
}

// askedForFinals is how many times the other transcriber was asked for its final.
func (s *FloorSuite) askedForFinals() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.asked
}

// counts are the server's decisions, those that wrote words too, and its transcripts.
func (s *FloorSuite) counts() (decisions, withWords, transcripts int) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return len(s.server.turns), s.server.withWords, len(s.server.audio) - len(s.server.turns)
}

// final is the next final, past the words shown meanwhile.
func (s *FloorSuite) final() stt.Transcript {
	s.T().Helper()
	for {
		if said := next[stt.Transcript](&s.transcribing); said.Mode == stt.ModeFinal {
			return said
		}
	}
}

// noFinal is none for a while: the words shown may come, a final not.
func (s *FloorSuite) noFinal() {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			said, ok := event.(stt.Transcript)
			s.Require().False(ok && !said.Interim, "a final: %+v", event)
		case <-time.After(150 * time.Millisecond):
			return
		}
	}
}

// The server is asked about the pause alone, without the words, which are the other
// transcriber's: shown as they come, whatever kind they are, and its final since the voice
// stopped is the utterance's, ended by the check's ruling and timed by the floor's clock.
func (s *FloorSuite) TestAPauseIsAskedAboutAloneAndEndsOnTheOtherTranscribersWords() {
	s.say(1000, true)
	s.True(s.writes(1, stt.ModeReplacement, "Two tickets"))
	shown := next[stt.Transcript](&s.transcribing)
	s.True(shown.Interim, "a revision is shown, not acted on: %+v", shown)
	s.True(shown.Floor)
	s.Equal("Two tickets", shown.Text)
	s.Equal(int64(1), shown.Utterance)
	s.Equal("stub", shown.Provider)
	s.True(s.writes(1, stt.ModeFinal, "Two tickets to Bilbao, please."))
	s.say(160, false)
	decisions, _, _ := s.counts()
	s.Zero(decisions, "not before the first check")
	s.say(60, false)

	end := s.final()
	s.Equal(int64(1), end.Utterance)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
	s.True(end.Floor, "the floor timed it")
	s.Equal("stub", end.Provider, "the words are the other transcriber's")
	s.Equal("stub-1", end.Model)
	s.InDelta(float64(300+1000+200), end.AudioDurationMs, 40, "the utterance's audio, by the floor's clock")
	rulings := s.agent.rulings()
	s.Require().Len(rulings, 1, "handed over before the final")
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue}, rulings[0].Ruling)
	s.Equal("Two tickets to Bilbao, please.", rulings[0].Words)
	s.True(rulings[0].Ended)
	decisions, withWords, transcripts := s.counts()
	s.Equal([3]int{1, 0, 0}, [3]int{decisions, withWords, transcripts}, "one decision, asked alone, and no words asked for")
	s.Zero(s.askedForFinals(), "its final had come since the voice stopped")

	s.say(4000, false)
	s.none()
}

// A run of speech the other transcriber ends on its own while the caller talks on, as it may at
// a breath, is a run: its final is no turn, and the utterance's words are its finals since the
// floor last ended one, the one it was asked for at the floor's end included.
func (s *FloorSuite) TestARunTheOtherTranscriberEndsMidSentenceIsNoTurn() {
	s.say(800, true)
	s.writes(1, stt.ModeReplacement, "A table for two at 7:30")
	s.writes(1, stt.ModeFinal, "A table for two at 7:30, patio.")
	s.say(600, true)
	s.noFinal()
	s.writes(2, stt.ModeReplacement, "peanut allergy")
	s.say(220, false)
	s.Eventually(func() bool { return s.askedForFinals() == 1 }, 5*time.Second, time.Millisecond,
		"asked for the final of the run it had open")
	s.noFinal()
	decisions, _, _ := s.counts()
	s.Equal(1, decisions)
	s.Len(s.agent.rulings(), 0, "the ruling waits for the words")

	s.writes(2, stt.ModeFinal, "Peanut allergy.")
	end := s.final()
	s.Equal("A table for two at 7:30, patio. Peanut allergy.", end.Text)
	s.Equal(int64(1), end.Utterance)
	rulings := s.agent.rulings()
	s.Require().Len(rulings, 1)
	s.Equal(end.Text, rulings[0].Words)
}

// A run the other transcriber settled and then carries on, restated whole, as one that takes the
// end of a run back does, or whose final comes twice, is its newest words, not two runs.
func (s *FloorSuite) TestARunRestatedAfterItsFinalIsItsNewestWords() {
	s.say(1000, true)
	s.writes(1, stt.ModeFinal, "A table for two")
	s.writes(1, stt.ModeReplacement, "A table for two at seven")
	s.writes(1, stt.ModeFinal, "A table for two at 7.")
	s.writes(1, stt.ModeFinal, "A table for two at 7.")
	s.say(220, false)
	s.Equal("A table for two at 7.", s.final().Text)
}

// A final that comes after the floor ended the utterance, correcting the words it had open, is
// waited for: the utterance's words are the final's, not the revision's.
func (s *FloorSuite) TestALateFinalThatCorrectsTheWordsIsWaitedFor() {
	s.say(1000, true)
	s.writes(1, stt.ModeReplacement, "a table at seven thirty")
	s.say(220, false)
	s.Eventually(func() bool { return s.askedForFinals() == 1 }, 5*time.Second, time.Millisecond)
	time.Sleep(300 * time.Millisecond)
	s.noFinal()

	s.writes(1, stt.ModeFinal, "A table at 7:13.")
	s.Equal("A table at 7:13.", s.final().Text)
}

// A transcriber that can't be told to settle its words, and whose final doesn't come within
// finalWait, has the utterance end on its words as they stood. What more it writes of that run
// is the ended utterance's, which is over, and its next run is the next utterance's.
func (s *FloorSuite) TestAFinalThatNeverComesEndsTheUtteranceOnTheWordsAsTheyStood() {
	s.settles = false
	s.say(1000, true)
	s.writes(1, stt.ModeReplacement, "a table for two")
	pause := time.Now()
	s.say(220, false)
	end := s.final()
	waited := time.Since(pause)
	s.Equal("a table for two", end.Text)
	s.GreaterOrEqual(waited, finalWait-50*time.Millisecond, "the final was waited for")
	s.Less(waited, finalWait+time.Second, "for no longer than finalWait")
	s.Zero(s.askedForFinals())

	s.writes(1, stt.ModeFinal, "A table for two.")
	s.none()
	s.writes(2, stt.ModeReplacement, "by the window")
	shown := next[stt.Transcript](&s.transcribing)
	s.True(shown.Interim)
	s.Equal(int64(2), shown.Utterance, "the next run's words are the next utterance's")
	s.Equal("by the window", shown.Text)
}

// Words of a run the other transcriber begins before the one the floor ended its utterance in
// has settled end that one, on its words as they stand, and are the next utterance's alone.
func (s *FloorSuite) TestWordsOfALaterRunAreTheNextUtterances() {
	s.say(1000, true)
	s.writes(1, stt.ModeReplacement, "I want the vegan menu")
	s.say(220, false)
	s.Eventually(func() bool { return s.askedForFinals() == 1 }, 5*time.Second, time.Millisecond)
	s.writes(2, stt.ModeReplacement, "and a glass")
	end := s.final()
	s.Equal("I want the vegan menu", end.Text)
	s.Equal(int64(1), end.Utterance)

	s.say(800, true)
	s.writes(2, stt.ModeFinal, "And a glass of water.")
	s.say(220, false)
	end = s.final()
	s.Equal("And a glass of water.", end.Text)
	s.Equal(int64(2), end.Utterance)
}

// Over the agent the caller is asked about without a pause, the decision alone, and a ruling to
// stop is handed over with the other transcriber's words so far, which the agent tells its own
// echo by.
func (s *FloorSuite) TestOverTheAgentAStopCarriesTheOtherTranscribersWords() {
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.unfinishedDecision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.mu.Unlock()
	s.say(200, true)
	s.writes(1, stt.ModeReplacement, "No, no, stop.")
	s.say(300, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) >= 1 }, 5*time.Second, time.Millisecond)
	ruled := s.agent.overheardSoFar()[0]
	s.True(ruled.Overheard)
	s.Equal(harness.Stop, ruled.Ruling.Floor)
	s.Equal("No, no, stop.", ruled.Words)
	decisions, withWords, transcripts := s.counts()
	s.GreaterOrEqual(decisions, 1)
	s.Equal([2]int{0, 0}, [2]int{withWords, transcripts}, "no words asked of the server")
}

// A pause no check can rule on, the server failing, ends at the other transcriber's own end of
// its run once the voice has paused, rather than at endAfter, with no ruling handed over.
func (s *FloorSuite) TestAPauseNoCheckRulesOnEndsAtTheOtherTranscribersEnd() {
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.mu.Unlock()
	s.say(1000, true)
	s.writes(1, stt.ModeReplacement, "what time do you open")
	s.say(220, false)
	s.answered()
	s.say(300, false)
	s.noFinal()

	s.writes(1, stt.ModeFinal, "What time do you open?")
	s.say(40, false)
	end := s.final()
	s.Equal("What time do you open?", end.Text)
	s.True(end.Floor)
	s.Empty(s.agent.rulings(), "no check ruled")
}

// A server that will never rule has the floor step aside for the session: it says so, and
// takes no more words, which are the listener's to act on as they would be with no floor.
func (s *FloorSuite) TestAFloorWhoseServerCantRuleStepsAside() {
	s.server.mu.Lock()
	s.server.noPolicy = true
	s.server.mu.Unlock()
	s.say(1000, true)
	s.True(s.writes(1, stt.ModeReplacement, "hello"))
	s.say(220, false)
	gone := next[stt.Disconnected](&s.transcribing)
	s.False(gone.Clean)
	s.False(s.writes(1, stt.ModeFinal, "Hello."), "the words are no longer the floor's")
}

// One that answers none of failoverAfter checks in a row over failoverWindow is down for the
// session as well; one that answers between them is not.
func (s *FloorSuite) TestAFloorWhoseServerAnswersNoCheckForAWhileStepsAside() {
	clock := time.Now()
	var mu sync.Mutex
	s.transcriber.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return clock
	}
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.mu.Unlock()
	for i := range failoverAfter {
		s.say(600, true)
		s.say(220, false)
		s.answered()
		s.say(2600, false)
		if i < failoverAfter-1 {
			s.True(s.writes(int64(i+1), stt.ModeFinal, "hello"), "still the floor's after %d", i+1)
		}
		mu.Lock()
		clock = clock.Add(failoverWindow / 2)
		mu.Unlock()
	}
	gone := next[stt.Disconnected](&s.transcribing)
	s.False(gone.Clean)
	s.False(s.writes(9, stt.ModeReplacement, "hello"))
}
