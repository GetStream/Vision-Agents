package audiofloor

import (
	"log/slog"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// holding sends ms of frames, voiced or quiet, 20 ms at a time, each once the checks asked by the
// one before have answered, so that a check's answer comes at the frame it was asked at, as it
// nearly does on a call, where a frame takes 20 ms and a check about as long. It returns the
// final, if one came, and how far into the frames it came.
func (s *transcribing) holding(ms int, voiced bool) (stt.Transcript, time.Duration, bool) {
	s.T().Helper()
	for sent := 0; sent < ms; sent += 20 {
		s.say(20, voiced)
		s.Eventually(func() bool {
			s.transcriber.mu.Lock()
			defer s.transcriber.mu.Unlock()
			return !s.transcriber.still.inFlight && !s.transcriber.pause.inFlight && !s.transcriber.interimOut.Load()
		}, 5*time.Second, time.Millisecond)
		for {
			select {
			case event := <-s.transcriber.Events():
				if said, ok := event.(stt.Transcript); ok && said.Final() {
					return said, time.Duration(sent+20) * time.Millisecond, true
				}
				continue
			default:
			}
			break
		}
	}
	return stt.Transcript{}, 0, false
}

// saying has the server write words for what the caller has said so far.
func (s *PausesSuite) saying(words string) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.words = words
}

// freeChecks counts the decisions the server was asked for with the floor free, held checks and
// pauses' checks, and those that wrote words.
func (s *PausesSuite) freeChecks() (decisions, withWords int) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	for _, turn := range s.server.turns {
		if !turn.Speaking {
			decisions++
		}
	}
	return decisions, s.server.withWords
}

// A caller who finishes in a room that keeps the speech gate voiced, a TV or a street, never
// pauses: the policy's checks, counted from the voice's last frame, never begin. Their words hold
// still, and once they have for stillFor, the gate voiced all the while, a held check ends the turn
// on its own ruling, as the cap would, though the model is unsure in the room (p_eot under every
// bar). The turn ends within stillFor and the speech one writing of the words takes (heldEvery)
// after the caller's last word, and not before the words have held still for stillFor.
func (s *PausesSuite) TestANoisyWordlessTailEndsTheTurnWithinTheBound() {
	s.ending(0.3)
	ends := testutil.ToFloat64(stillEnds)
	s.saying("Two tickets")
	_, _, ended := s.holding(1200, true)
	s.Require().False(ended, "not while the caller talks")
	s.saying("Two tickets to Bilbao, please.")
	_, _, ended = s.holding(800, true)
	s.Require().False(ended, "not while the caller talks")

	// The caller has finished; the room holds the gate open, and the server writes nothing more.
	final, after, ended := s.holding(4000, true)
	s.Require().True(ended, "the turn ends though the gate never heard a pause")
	s.Equal("Two tickets to Bilbao, please.", final.Text)
	s.LessOrEqual(after, stillFor+heldEvery+100*time.Millisecond, "within the bound of the caller's last word")
	s.GreaterOrEqual(after, stillFor, "not before the words held still for stillFor")
	rulings := s.agent.rulings()
	s.Require().Len(rulings, 1)
	s.True(rulings[0].Ended)
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue}, rulings[0].Ruling, "the model's own ruling, as at the cap")
	s.Equal(final.Text, rulings[0].Words)
	s.Less(rulings[0].Paused, s.firstCheck(), "no pause the gate heard")
	s.Equal(ends+1, testutil.ToFloat64(stillEnds))

	// The utterance is over: the room after it is the next one, which has no words and ends nothing,
	// so that a caller who begins as it would have ended keeps their first word.
	s.saying("")
	_, _, ended = s.holding(3000, true)
	s.False(ended, "words that are none end nothing")
	s.Equal(ends+1, testutil.ToFloat64(stillEnds), "words that are none end nothing")
}

// A caller who talks on, their words changing, is not cut however long the gate stays voiced:
// held checks write their words as they go and end nothing. Their pause ends the turn as ever.
func (s *PausesSuite) TestACallerStillSpeakingIsNotCut() {
	said := []string{"I'd like"}
	for _, word := range strings.Fields("to book two tickets on the morning crossing to Bilbao on Friday for me and my wife") {
		s.saying(strings.Join(said, " "))
		_, _, ended := s.holding(400, true)
		s.Require().False(ended, "cut while still talking at %q", strings.Join(said, " "))
		said = append(said, word)
	}
	decisions, withWords := s.freeChecks()
	s.GreaterOrEqual(decisions, 4, "held checks wrote the words as they changed")
	s.Equal(decisions, withWords, "each with its words")

	final, _, ended := s.holding(400, false)
	s.Require().True(ended, "their pause ends the turn")
	s.Equal(strings.Join(said[:len(said)-1], " "), final.Text)
	s.GreaterOrEqual(s.agent.rulings()[0].Paused, s.firstCheck(), "ended by the pause's check")
}

// A slow dictation on a quiet line, a digit voiced for 300 ms and 1.4 s of quiet after it, the
// model waiting on each pause, is as it was: each pause's checks rule on it, nothing is asked
// while the gate is voiced, and the turn ends at the cap of the pause after the last digit.
func (s *PausesSuite) TestASlowDictationStaysAsItWas() {
	s.ending(0.1)
	digits := strings.Fields("four one two two")
	for i := range digits {
		s.saying(strings.Join(digits[:i+1], " "))
		_, _, ended := s.holding(300, true)
		s.Require().False(ended)
		if i < len(digits)-1 {
			_, _, ended = s.holding(1400, false)
			s.Require().False(ended, "the model waits on the pause after %q", digits[i])
		}
	}
	final, after, ended := s.holding(3000, false)
	s.Require().True(ended)
	s.Equal("four one two two", final.Text)
	_, _, cap := s.transcriber.checkTimes()
	s.GreaterOrEqual(after, cap, "at the cap")
	_, withWords := s.freeChecks()
	s.Equal(len(digits), withWords, "one writing of the words a pause, its first check's: no held check")
}

// Coughs in a pause on a quiet line, words unchanged, are no room holding the gate: voiced for less
// than half the time the words have held still, they end nothing and have nothing asked, and the
// pauses' checks rule as they did, the turn ending at the cap of the pause after the last cough.
func (s *PausesSuite) TestCoughsInAQuietPauseEndNothing() {
	s.ending(0.1)
	_, _, ended := s.holding(1000, true)
	s.Require().False(ended)
	for range 2 {
		_, _, ended = s.holding(600, false)
		s.Require().False(ended, "the model waits")
		_, _, ended = s.holding(300, true)
		s.Require().False(ended, "a cough ends nothing")
	}
	final, after, ended := s.holding(3000, false)
	s.Require().True(ended)
	_, _, cap := s.transcriber.checkTimes()
	s.GreaterOrEqual(after, cap, "at the cap of the last pause")
	s.Equal("Two tickets to Bilbao, please.", final.Text)
	_, withWords := s.freeChecks()
	s.Equal(3, withWords, "a writing of the words a pause, its first check's: no held check")
}

// Where interims write the words while the caller talks, they are the clock: no held check is
// asked to write them, and the one held check is the one that ends the turn, once an interim has
// read the words again and they have held still for stillFor.
func (s *PausesSuite) TestInterimsWriteTheWordsAndAHeldCheckOnlyEnds() {
	s.transcriber.interims = true
	s.ending(0.3)
	s.saying("Two tickets")
	_, _, ended := s.holding(1000, true)
	s.Require().False(ended)
	s.saying("Two tickets to Bilbao, please.")
	_, _, ended = s.holding(1000, true)
	s.Require().False(ended)
	final, after, ended := s.holding(4000, true)
	s.Require().True(ended)
	s.Equal("Two tickets to Bilbao, please.", final.Text)
	s.LessOrEqual(after, stillFor+heldEvery+100*time.Millisecond)
	decisions, _ := s.freeChecks()
	s.Equal(1, decisions, "the held check that ended the turn, and no other")
	s.GreaterOrEqual(s.transcribed(), 3, "interims wrote the words")
}

// A held check that writes nothing, the server failing, is asked again only after heldEvery more
// of speech, as one that wrote the words is: never at every frame.
func (s *PausesSuite) TestAHeldCheckThatFailsWaitsOutHeldEvery() {
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.mu.Unlock()
	_, _, ended := s.holding(4000, true)
	s.False(ended)
	s.agent.mu.Lock()
	asked := s.agent.asked
	s.agent.mu.Unlock()
	s.InDelta(3, asked, 1, "one held check every heldEvery of speech, at 1.1, 2.2 and 3.3 s")
}

// Over the agent nothing is held: a caller voiced over it is the checks over the agent's to ask
// about, however long their words hold still.
func (s *PausesSuite) TestNothingIsHeldOverTheAgent() {
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"continue","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
	s.server.mu.Unlock()
	_, _, ended := s.holding(4000, true)
	s.False(ended)
	decisions, _ := s.freeChecks()
	s.Zero(decisions, "nothing asked with the floor free")
}

// A session no agent of the app's asks for, transcription alone or an agent that acts on nothing
// the model says, is as it was: its utterances end by its own agent's checks, and nothing is held.
func (s *PausesSuite) TestNothingIsHeldForATranscriberOfItsOwn() {
	s.transcriber.AskOnPauses(neutralAgent{hearer: NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))})
	_, _, ended := s.holding(4000, true)
	s.False(ended)
	decisions, _ := s.freeChecks()
	s.Zero(decisions)
}

// An agent whose point lets a caller who has finished wait longer than stillFor has the words hold
// still that long: 1.4 s under the point fitted to 1400 ms.
func (s *PausesSuite) TestTheWordsHoldStillAsLongAsTheAgentsPointWaits() {
	s.agent.hearer = NewHearer(s.client, EndOfTurn{MaxWaitMs: 1400}, slog.New(slog.DiscardHandler))
	s.ending(0.3)
	s.saying("Two tickets to Bilbao, please.")
	_, _, ended := s.holding(1200, true)
	s.Require().False(ended)
	final, after, ended := s.holding(4000, true)
	s.Require().True(ended)
	s.Equal("Two tickets to Bilbao, please.", final.Text)
	s.GreaterOrEqual(after, 1400*time.Millisecond-100*time.Millisecond, "the words held still for the point's wait")
}

func (s *PausesSuite) firstCheck() time.Duration {
	first, _, _ := s.transcriber.checkTimes()
	return first
}
