package audiofloor

import (
	"bytes"
	"fmt"
	"log/slog"
	"slices"
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
	// gaveWay says the agent has given the floor up over what it is saying, and heard that the
	// caller had heard it when they began: unset, they began before it could be heard.
	gaveWay bool
	heard   bool
	asked   int
	ruled   []Ruled
	// overheard are the rulings over the agent handed over at once.
	overheard []Ruled
	// hearing counts the times the agent was told the caller is in an utterance.
	hearing int
}

func (a *asking) Hearing(stt.Participant) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.hearing++
}

func (a *asking) heardSoFar() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hearing
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

func (a *asking) GivenWay() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.gaveWay
}

func (a *asking) HeardBefore(time.Time) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.heard
}

// givesWay has the agent give the floor up over what it is saying, as on a shortening.
func (a *asking) givesWay() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.gaveWay = true
}

// heardIt has the caller begin over the agent, having heard it.
func (a *asking) heardIt() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.heard = true
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
	s.InDelta(end.AudioDurationMs, rulings[0].AudioMs, 1, "the ruling says what audio its words are of, as the final does")
	s.Equal(end.ProcessingTimeMs, rulings[0].WordsMs)
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
	s.InDelta(ruled.AudioMs-float64(preroll.Milliseconds()), float64(time.Since(ruled.Began).Milliseconds()), 100,
		"the caller began at the utterance's first voiced frame, as long before the check as the utterance so far less its preroll")
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

// The agent is told the caller is in an utterance every interval while it lasts, voiced or in
// a pause the model is waiting on, whether or not any words have been written of it: a caller
// whose words come only at the turn's end, with interims off or the words so far unstable, is
// not a silent line the agent should fill. Once the utterance has ended it is told nothing.
func (s *PausesSuite) TestTheAgentIsToldTheCallerIsTalkingUntilTheTurnEnds() {
	s.ending(0.1)
	s.say(1000, true)
	voiced := s.agent.heardSoFar()
	s.GreaterOrEqual(voiced, 3, "told while the caller talked")
	s.say(900, false)
	s.Greater(s.agent.heardSoFar(), voiced, "told while the model waited on the pause")
	s.ending(0.999)
	// The next check ends the turn, once the pause's last check has answered: a frame at a
	// time, as each check's answer may come after the frames it was asked in have been sent,
	// and its final a moment after the answer.
	if _, _, ended := s.holding(600, false); !ended {
		for !next[stt.Transcript](&s.transcribing).Final() {
		}
	}
	told := s.agent.heardSoFar()
	s.say(2000, false)
	s.Equal(told, s.agent.heardSoFar(), "told nothing once the turn had ended")
}

// An utterance a check over the agent stopped it in goes on, uncut, and ends by the free
// floor's end-of-turn policy like any other once the agent has stopped: a pause whose words are
// the stop's alone and which the model waits on ("No, [pause] Porto.") does not end it, and the
// words after the pause are the same utterance's. Ended there, the next utterance began where
// the speech gate had last heard voice, after the unvoiced P, and "Porto" was heard "Corto".
func (s *PausesSuite) TestAnUtteranceThatStoppedTheAgentEndsByThePolicy() {
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.words = "No."
	s.server.mu.Unlock()
	s.say(400, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 1 }, 5*time.Second, time.Millisecond, "the stop was never asked about")
	s.agent.speaks(false)
	s.ending(0.17)
	s.say(400, false)
	s.none()
	s.server.mu.Lock()
	s.server.words = "No, Porto."
	s.server.mu.Unlock()
	s.say(600, true)
	s.ending(0.999)
	s.say(300, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance, "one utterance, through the pause the model waited on")
	s.Equal("No, Porto.", end.Text)
}

// A check over the agent that answers once its utterance has ended, a pause check over the
// agent having ended it with a ruling of its own, is about an utterance that is over: it is not
// acted on, and the final's ruling is the one that stands. Acted on, a check that misheard
// "don't stop" as "let's stop" stopped the agent after the pause check had heard it right.
func (s *PausesSuite) TestACheckOverTheAgentAnsweredAfterItsUtteranceEndedIsNotActedOn() {
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.unfinishedDecision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.words = "Don't stop."
	// Late, and inside the 250 ms a check over the agent has.
	s.server.unfinishedAfter = 200 * time.Millisecond
	s.server.mu.Unlock()
	s.say(400, true)
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return s.transcriber.over.inFlight
	}, 5*time.Second, time.Millisecond, "the caller was never asked about over the agent")
	// The agent is quiet by the pause, which ends the utterance by its bars.
	s.agent.speaks(false)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return !s.transcriber.over.inFlight
	}, 5*time.Second, time.Millisecond, "the check over the agent never answered")
	s.Empty(s.agent.overheardSoFar(), "a check about an utterance that has ended is not acted on")
	s.Len(s.agent.rulings(), 1)
	s.none()
}

// A check still out when speech resumes is waited for, the model having been asked about the
// pause as it was: a ruling that ended the turn stands, the final goes out with its words,
// and the speech is the next utterance's; one that did not ends the pause as speech does,
// and the next pause is asked about afresh, its words with it. Either way the words written
// at a pause the caller spoke through are shown as the words so far.
func (s *PausesSuite) TestACheckOutWhenSpeechResumesIsWaitedFor() {
	s.transcriber.interims = true
	s.server.mu.Lock()
	s.server.decideAfter = 150 * time.Millisecond
	s.server.mu.Unlock()
	s.ending(0.1)
	s.say(1000, true)
	s.True(next[stt.Transcript](&s.transcribing).Interim)
	s.say(220, false)
	s.say(400, true)
	s.answered()
	shown := next[stt.Transcript](&s.transcribing)
	s.True(shown.Interim, "the pause's words, though the caller went on")
	s.InDelta(float64(300+1000+200), shown.AudioDurationMs, 40)
	s.say(220, false)
	s.answered()
	decisions, withWords, _ := s.counts()
	s.Equal([2]int{2, 2}, [2]int{decisions, withWords}, "the next pause is asked about afresh, with its words")
	s.True(next[stt.Transcript](&s.transcribing).Interim)

	s.ending(0.999)
	s.say(400, true)
	s.say(220, false)
	s.say(400, true)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance, "the ruling that ended the turn stands, though the caller went on before it came")
	s.say(400, true)
	s.say(220, false)
	again := next[stt.Transcript](&s.transcribing)
	s.Equal(int64(2), again.Utterance, "the speech that resumed is the next utterance")
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

// The answers of the server's frames over the agent: at a pause it says what to do about the
// floor, with no p_eot.
const (
	stopping   = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"stop"}}`
	continuing = `{"answer":"continue","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
)

// final is the next final, past the words shown meanwhile.
func (s *PausesSuite) final() stt.Transcript {
	s.T().Helper()
	for {
		if said := next[stt.Transcript](&s.transcribing); said.Mode == stt.ModeFinal {
			return said
		}
	}
}

// pausing has the caller quiet for ms, a check at a time: frames come faster than the server
// answers, and a check is asked only once the one before has.
func (s *PausesSuite) pausing(ms int) {
	for ; ms > 0; ms -= 100 {
		s.say(100, false)
		s.answered()
	}
}

// sayingWords has the caller talk until words of the utterance are written while they talk,
// an interim: a caller saying words, where a noise the speech gate opened on has none. The
// test turns interims on before any audio.
func (s *PausesSuite) sayingWords() {
	s.T().Helper()
	s.say(int((interimEvery+100*time.Millisecond)/time.Millisecond), true)
	said := next[stt.Transcript](&s.transcribing)
	s.Require().True(said.Interim, "the words so far: %+v", said)
}

// noFinal is none for a caller whose words so far are shown: interims may come, a final not.
func (s *PausesSuite) noFinal() {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			said, ok := event.(stt.Transcript)
			s.Require().False(ok && !said.Interim, "a final: %+v", event)
		case <-time.After(100 * time.Millisecond):
			return
		}
	}
}

// Over the agent the model's answer at a pause says what to do about the floor, and not whether
// the caller has finished: it has no p_eot. A pause the agent talks through, in words the caller
// began before they could hear it, ends at the policy's cap, on the ruling of its one check,
// which stands until then: it costs one request, not one every interval to the cap.
func (s *PausesSuite) TestOverTheAgentAPauseEndsAtTheCap() {
	s.transcriber.interims = true
	s.server.mu.Lock()
	s.server.speakingDecision = continuing
	s.server.words = "Uh-huh."
	s.server.mu.Unlock()
	s.sayingWords()
	s.agent.speaks(true)
	s.say(280, true)
	s.pausing(2300)
	s.noFinal()
	s.Empty(s.agent.rulings(), "a check over the agent ended the turn before the cap")
	s.pausing(500)
	s.Equal("Uh-huh.", s.final().Text)
	ruled := s.agent.rulings()[0]
	s.True(ruled.Speaking)
	s.Equal(harness.Decided{Disposition: harness.Ignore, Floor: harness.Continue}, ruled.Ruling)
	s.GreaterOrEqual(ruled.Paused, 2500*time.Millisecond, "the cap ended the turn")
	decisions, withWords, transcripts := s.counts()
	s.Equal([3]int{2, 2, 1}, [3]int{decisions, withWords, transcripts}, "one check while the caller talked over the agent, one at the pause whose ruling stood to the cap, and the words so far")
	s.True(s.server.turns[0].Speaking)
}

// shortening is the server's ruling over the agent that the caller added one more item for
// the request being answered, at a pause (respond) and while they talk on (wait).
const (
	shortening  = `{"answer":"shorten","answer_probability":0.99,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"shorten"}}`
	shortenedOn = `{"answer":"shorten","answer_probability":0.99,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"shorten"}}`
)

// Once the agent has given way, the floor is the caller's, and a ruling at their pause that
// they take it ends their turn. On the TPU, "And in Lisbon." over a forecast was ruled shorten
// as it was said, the agent shortened the forecast, and the pause's first check, over the
// tail, ruled it shorten again: with the floor free the checks that followed heard "And in
// Lisbon." with no conversation and waited (p_eot 0.04 to 0.23), and it was answered 2.12 s
// after the caller's voice, against 0.35 s before #1829.
func (s *PausesSuite) TestAShortenedFollowUpIsAnsweredAtItsPausesFirstCheck() {
	s.server.mu.Lock()
	s.server.unfinishedDecision, s.server.speakingDecision = shortenedOn, shortening
	s.server.words = "And in Lisbon."
	s.server.mu.Unlock()
	s.ending(0.05)
	s.agent.speaks(true)
	s.say(800, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) > 0 }, 5*time.Second, time.Millisecond,
		"the shortening was never handed over")
	s.agent.givesWay()
	s.pausing(200)
	end := s.final()
	s.Equal("And in Lisbon.", end.Text)
	s.Equal(int64(1), end.Utterance)
	ruled := s.agent.rulings()[0]
	s.True(ruled.Speaking, "the check over the agent's tail ended the turn")
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Shorten}, ruled.Ruling)
	s.Less(ruled.Paused, 300*time.Millisecond, "at the pause's first check")
}

// Once the agent has given way, a ruling at the caller's pause that it go on ends nothing: on
// a call, "Last four 9821," shortened the reply in flight, and the check at the comma, over its
// tail, ruled continue with its words written as nothing. The caller goes on, and it is one
// utterance.
func (s *PausesSuite) TestOnceTheAgentGaveWayARulingThatItGoOnEndsNothing() {
	nothing := ""
	s.server.mu.Lock()
	s.server.speakingDecision, s.server.speakingWords = continuing, &nothing
	s.server.unfinishedDecision = continuing
	s.server.words = "Last four 9821, 14 Cedar Lane."
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.agent.givesWay()
	s.say(800, true)
	s.pausing(200)
	s.say(800, true)
	s.agent.speaks(false)
	s.pausing(200)
	end := s.final()
	s.Equal(int64(1), end.Utterance, "one utterance, through the pause at the comma")
	s.Equal("Last four 9821, 14 Cedar Lane.", end.Text)
}

// A ruling over the agent stands while it talks on, and once it is quiet the pause is the
// caller's: it is asked about at once with the floor free, not at the next interval, nor at
// the cap.
func (s *PausesSuite) TestAPauseOverTheAgentIsAskedAboutOnceItIsQuiet() {
	s.transcriber.interims = true
	s.server.mu.Lock()
	s.server.speakingDecision = continuing
	s.server.mu.Unlock()
	s.sayingWords()
	s.agent.speaks(true)
	s.say(280, true)
	s.pausing(200)
	s.pausing(300)
	decisions, _, _ := s.counts()
	s.Equal(2, decisions, "one check while the caller talked over the agent, and one at the pause, whose ruling stands while it talks on")
	s.agent.speaks(false)
	s.pausing(200)
	end := s.final()
	s.Equal(int64(1), end.Utterance)
	ruled := s.agent.rulings()[0]
	s.False(ruled.Speaking, "ended with the floor free")
	s.Less(ruled.Paused, 800*time.Millisecond, "asked about as soon as the agent was quiet, %v into the pause", ruled.Paused)
}

// warnings is what the transcriber warns of, from goroutines of its own.
type warnings struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (w *warnings) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *warnings) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String()
}

// An utterance that ends with no words sends nothing, and is a warning only where words were
// lost: the model ruled them to be answered and the server wrote none, saying no why. A murmur
// over the agent the server wrote nothing of, as it ruled it ignored, is no warning: on a call,
// every "uh-huh" over the agent would be one.
func (s *PausesSuite) TestOnlyWordsLostAreAWarning() {
	warned := &warnings{}
	s.transcriber.logger = slog.New(slog.NewTextHandler(warned, &slog.HandlerOptions{Level: slog.LevelWarn}))
	nothing := ""
	s.server.mu.Lock()
	s.server.speakingDecision, s.server.speakingWords, s.server.blanked = continuing, &nothing, "ignored"
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(280, true)
	s.pausing(2800)
	s.none()
	s.Empty(warned.String(), "a murmur over the agent")

	s.server.mu.Lock()
	s.server.words, s.server.blanked = "", "language"
	s.server.mu.Unlock()
	s.agent.speaks(false)
	s.say(400, true)
	s.pausing(300)
	s.none()
	s.Empty(warned.String(), "a voice in another language")

	s.server.mu.Lock()
	s.server.blanked = ""
	s.server.mu.Unlock()
	s.say(400, true)
	s.pausing(300)
	s.none()
	s.Contains(warned.String(), "came back with no words", "words ruled to be answered, with no reason for none")
}

// An utterance begun over the agent, while the caller could hear it, that the model rules the
// agent may talk through ends at its pause's first check, on that one request: a murmur, a
// cough, the agent's own voice heard back. What the caller says next is an utterance of its
// own. Run on to the cap, a murmur cost a check every interval, and the agent's echo took the
// caller's question that came after it in with it: with the floor free the model heard both
// and waited on them, and the question was answered 1.5 s later.
func (s *PausesSuite) TestATalkOverTheAgentGoesOnThroughEndsAtItsPause() {
	s.server.mu.Lock()
	s.server.speakingDecision, s.server.unfinishedDecision = continuing, continuing
	s.server.words = "Here are the forecasts you asked for."
	s.server.mu.Unlock()
	s.agent.heardIt()
	s.agent.speaks(true)
	s.say(800, true)
	s.pausing(200)
	end := s.final()
	s.Equal(int64(1), end.Utterance)
	ruled := s.agent.rulings()[0]
	s.True(ruled.Speaking)
	s.Equal(harness.Decided{Disposition: harness.Ignore, Floor: harness.Continue}, ruled.Ruling)
	s.Less(ruled.Paused, 300*time.Millisecond, "at the pause's first check")
	s.server.mu.Lock()
	pauseChecks := 0
	for _, turn := range s.server.turns {
		if turn.Speaking && !turn.Unfinished {
			pauseChecks++
		}
	}
	s.server.words = "Sorry, what was the second one?"
	s.server.mu.Unlock()
	s.Equal(1, pauseChecks, "one request for the pause")

	s.say(800, true)
	s.pausing(200)
	again := s.final()
	s.Equal(int64(2), again.Utterance, "the caller's own utterance")
	s.Equal("Sorry, what was the second one?", again.Text)
}

// Over the agent a check at a pause ends nothing before the cap. On a call the caller
// dictated "Last four 9821, 14 Cedar Lane" while a reply was in flight, and the check 200 ms
// into the pause at the comma ended the utterance on its first half, its words written as
// nothing: they were never heard. The utterance goes on through the pause, and ends by the bars
// once the agent is quiet, all its words in one final.
func (s *PausesSuite) TestAPauseOverTheAgentDoesNotEndTheUtterance() {
	nothing := ""
	s.server.mu.Lock()
	s.server.speakingDecision, s.server.speakingWords = continuing, &nothing
	s.server.unfinishedDecision = continuing
	s.server.words = "Last four 9821, 14 Cedar Lane."
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(800, true)
	// The caller's words so far are written by the check over the agent, which answers on a
	// goroutine of its own: the pause comes once it has. Sent at once, the pause could come
	// first, and its check found nothing written and ended the utterance as a noise.
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return !s.transcriber.over.inFlight
	}, 5*time.Second, time.Millisecond, "the check over the agent never answered")
	s.pausing(200)
	s.say(800, true)
	s.agent.speaks(false)
	s.pausing(200)
	end := s.final()
	s.Equal(int64(1), end.Utterance, "one utterance, through the pause at the comma")
	s.Equal("Last four 9821, 14 Cedar Lane.", end.Text)
	s.Len(s.agent.rulings(), 1)
}

// An utterance none of which has been written by its pause over the agent is a noise the speech
// gate opened on before the reply, not a caller going on with their turn, and ends at that check
// as one begun over the agent does: what the caller says next is an utterance of its own. Left
// standing, a room's babble that never paused for the cap ran on through the reply and took
// the caller's next line in with it, answered a babble cycle late (6 of 15 restaurant
// confirmations on the TPU, against none before) or cut at the utterance limit.
func (s *PausesSuite) TestANoiseBegunBeforeTheReplyEndsAtItsPausesFirstCheck() {
	nothing := ""
	s.server.mu.Lock()
	s.server.speakingDecision, s.server.speakingWords = continuing, &nothing
	s.server.unfinishedDecision, s.server.unfinishedWords = continuing, &nothing
	s.server.mu.Unlock()
	s.say(280, true)
	s.agent.speaks(true)
	s.say(280, true)
	s.pausing(200)
	s.none()
	s.agent.speaks(false)
	s.say(800, true)
	s.pausing(200)
	end := s.final()
	s.Equal(int64(2), end.Utterance, "the caller's line is an utterance of its own")
	s.Equal("Two tickets to Bilbao, please.", end.Text)
}

// A stop ruled at a pause over the agent is acted on at once, with the pause's words, as one
// ruled while the caller talks on is (Overheard): a "stop" too short for a check over the agent
// without a pause, voiced for less than overAfter, is ruled on at its pause alone. The turn
// ends by the bars once the agent is quiet.
func (s *PausesSuite) TestAStopAtAPauseOverTheAgentIsActedOnAtOnce() {
	s.server.mu.Lock()
	s.server.speakingDecision = stopping
	s.server.words = "Stop."
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(280, true)
	s.pausing(200)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 1 }, 5*time.Second, time.Millisecond, "the stop was not acted on")
	ruled := s.agent.overheardSoFar()[0]
	s.Equal(harness.Stop, ruled.Ruling.Floor)
	s.Equal("Stop.", ruled.Words)
	s.True(ruled.Overheard)
	s.Equal(int64(1), ruled.Utterance)
	s.False(ruled.Began.IsZero(), "when the caller began the stop")
	s.Empty(s.agent.rulings(), "a stop is no end of the turn")
	s.none()

	s.agent.speaks(false)
	s.pausing(300)
	end := s.final()
	s.Equal("Stop.", end.Text)
	s.Equal(int64(1), end.Utterance)
	s.False(s.agent.rulings()[0].Speaking, "the turn was ended by a check with the floor free")
}

// overEcho is a bare "no, no" said over the agent's forecast on a line that hears the agent
// back, as the server writes a pause's words: the agent's own voice, then the caller's.
const overEcho = "Tomorrow in Lisbon, it will be sunny. No, no."

// stopAtAPause has the caller say words over the agent, voiced for less than a check over
// the agent waits for, and pause: the server rules at the pause's first check that the agent
// stop, with words written.
func (s *PausesSuite) stopAtAPause(words string) {
	s.T().Helper()
	s.server.mu.Lock()
	s.server.speakingDecision = stopping
	s.server.words = words
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(280, true)
	s.say(240, false)
	s.transcriber.mu.Lock()
	asked := s.transcriber.pause.checks
	s.transcriber.mu.Unlock()
	s.Require().Equal(1, asked, "the pause was not asked about")
}

// handedOver says the stop ruled at the pause reached the agent once, with the pause's words.
func (s *PausesSuite) handedOver() {
	s.T().Helper()
	s.Require().Eventually(func() bool { return len(s.agent.overheardSoFar()) > 0 }, 2*time.Second, time.Millisecond,
		"the stop was thrown away with the pause")
	s.Never(func() bool { return len(s.agent.overheardSoFar()) > 1 }, 100*time.Millisecond, time.Millisecond, "handed over twice")
	ruled := s.agent.overheardSoFar()[0]
	s.Equal(harness.Stop, ruled.Ruling.Floor)
	s.Equal(overEcho, ruled.Words, "the pause's words, which the agent tells its own echo by")
	s.True(ruled.Overheard)
	s.InDelta(200, float64(ruled.Paused.Milliseconds()), 40, "ruled at the pause's first check")
	s.Equal(int64(1), ruled.Utterance)
	s.False(ruled.Began.IsZero(), "when the caller began the stop")
	s.Empty(s.agent.rulings(), "a stop is no end of the turn")
}

// A stop ruled at a pause over the agent is handed over with the pause's words though the
// caller is voiced again while its check is out: on a line that hears the agent back, or in a
// noisy room, the speech gate opens again almost at once, which ends the pause and not what
// was ruled on it. Thrown away with the pause, a bare "no, no" over the agent's forecast, ruled
// a stop at its 200 ms pause, left the agent talking to the end of its reply.
func (s *PausesSuite) TestAStopAtAPauseOverTheAgentIsHandedOverThoughSpeechResumesWhileItsCheckIsOut() {
	s.server.mu.Lock()
	s.server.decideAfter = 150 * time.Millisecond
	s.server.mu.Unlock()
	s.stopAtAPause(overEcho)
	s.say(200, true)
	s.handedOver()
}

// So is one whose check answered before speech resumed, its words still to come.
func (s *PausesSuite) TestAStopAtAPauseOverTheAgentIsHandedOverThoughSpeechResumesBeforeItsWords() {
	s.server.mu.Lock()
	s.server.wordsAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.stopAtAPause(overEcho)
	s.answered()
	s.Empty(s.agent.overheardSoFar(), "not before its words")
	s.say(200, true)
	s.handedOver()
}

// A stop at a pause whose words come back empty is not handed over, the caller voiced again or
// not: there is nothing to tell the agent's own echo from the caller by, and nothing heard is
// no reason to stop.
func (s *PausesSuite) TestAStopAtAPauseWithNoWordsIsNotHandedOver() {
	s.server.mu.Lock()
	s.server.wordsAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.stopAtAPause("")
	s.answered()
	s.say(200, true)
	s.Never(func() bool { return len(s.agent.overheardSoFar()) > 0 }, 600*time.Millisecond, time.Millisecond,
		"a stop with no words was handed over")
}

// A stop at a pause whose words come only once its utterance has ended, a later pause's check
// having ended it with a ruling of its own, is about an utterance that is over: it is not
// handed over, as a check over the agent answered that late is not (overtaken), and the
// final's ruling is the one that stands.
func (s *PausesSuite) TestAStopAtAPauseIsNotHandedOverOnceItsUtteranceHasEnded() {
	gate := make(chan struct{})
	release := sync.OnceFunc(func() { close(gate) })
	s.T().Cleanup(release)
	s.server.mu.Lock()
	s.server.wordsGate = gate
	s.server.mu.Unlock()
	s.agent.heardIt()
	s.stopAtAPause(overEcho)
	s.answered()
	// The caller goes on, and at their next pause the model lets the agent talk on, which ends
	// the utterance, as they had heard the agent when they began it.
	s.server.mu.Lock()
	s.server.speakingDecision = continuing
	s.server.words = "Tomorrow in Lisbon, it will be sunny with a high of 24."
	s.server.mu.Unlock()
	s.say(200, true)
	s.say(240, false)
	end := s.final()
	s.Equal(int64(1), end.Utterance)
	s.Require().Len(s.agent.rulings(), 1)
	s.say(20, false)
	release()
	s.Never(func() bool { return len(s.agent.overheardSoFar()) > 0 }, 300*time.Millisecond, time.Millisecond,
		"a stop about an utterance that has ended was handed over")
}

// The checks in a row a point asks for are counted from a pause's first check with the floor
// free: one asked over the agent has no p_eot to count. Counted as the pause's first, it left
// the run of an earlier pause going into this one, and a point of two checks in a row ended
// the turn on one.
func (s *PausesSuite) TestChecksInARowAreCountedFromThePausesFirstWithTheFloorFree() {
	s.agent.hearer = NewHearer(s.client, EndOfTurn{MaxWaitMs: 1400}, slog.New(slog.DiscardHandler))
	s.transcriber.interims = true
	s.server.mu.Lock()
	s.server.speakingDecision = continuing
	s.server.mu.Unlock()
	s.sayingWords()
	s.pausing(200)
	s.say(400, true)
	s.noFinal()
	s.agent.speaks(true)
	s.pausing(200)
	s.agent.speaks(false)
	s.pausing(300)
	s.noFinal()
	s.Empty(s.agent.rulings(), "the turn ended on one check with the floor free")
	s.pausing(300)
	s.Equal(stt.ModeFinal, s.final().Mode)
	// The first with the floor free was asked as soon as the agent was quiet, 220 ms in.
	s.Equal(520*time.Millisecond, s.agent.rulings()[0].Paused, "the second check in a row with the floor free ended it")
}

// An utterance is never let go of in silence: the words of the pause that ended it may come
// back empty where a check over the agent wrote some of it, as the server wrote nothing of
// "Last four: ninety-eight twenty-one." at its end on a call, and the final is those words.
func (s *PausesSuite) TestAFinalWithNoWordsKeepsWhatACheckOverTheAgentWrote() {
	over := "Last four: ninety-eight twenty-one."
	s.server.mu.Lock()
	s.server.unfinishedDecision, s.server.unfinishedWords = continuing, &over
	s.server.words = ""
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(800, true)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return slices.ContainsFunc(s.server.turns, func(turn Turn) bool { return turn.Unfinished })
	}, 5*time.Second, time.Millisecond, "the caller was never asked about over the agent")
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return !s.transcriber.over.inFlight
	}, 5*time.Second, time.Millisecond, "the check over the agent never answered")
	s.agent.speaks(false)
	s.pausing(200)
	end := s.final()
	s.Equal(over, end.Text)
	s.Equal(int64(1), end.Utterance)
	s.Len(s.agent.rulings(), 1)
}

// A pause whose words the server wrote none of on purpose, saying why, keeps none of what a
// check over the agent wrote either. The pause heard the whole utterance, a voice in another
// language; the check over the agent heard less of it and wrote it as words in the caller's
// ("They got ill."), which were the final and answered.
func (s *PausesSuite) TestAFinalTheServerBlankedOnPurposeKeepsNothingACheckOverTheAgentWrote() {
	over := "They got ill."
	s.server.mu.Lock()
	s.server.unfinishedDecision, s.server.unfinishedWords = continuing, &over
	s.server.words = ""
	s.server.mu.Unlock()
	s.agent.speaks(true)
	s.say(800, true)
	s.Eventually(func() bool {
		s.transcriber.mu.Lock()
		defer s.transcriber.mu.Unlock()
		return s.transcriber.over.words == over
	}, 5*time.Second, time.Millisecond, "the check over the agent never wrote its words")
	s.server.mu.Lock()
	s.server.blanked = "language"
	s.server.mu.Unlock()
	s.agent.speaks(false)
	s.pausing(200)
	s.none()
	s.Empty(s.agent.rulings())
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
