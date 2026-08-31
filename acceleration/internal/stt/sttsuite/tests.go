//go:build integration

package sttsuite

import (
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// TestAlmostEveryWordComesBackTheWayItWasSaid is the one that says the provider works at
// all. Looking for a phrase would pass a transcript that dropped half the sentence, so
// the whole thing is scored against what the fixture actually says.
func (s *Suite) TestAlmostEveryWordComesBackTheWayItWasSaid() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}

// TestTheTurnSettlesQuicklyAfterTheCallerStops measures the wait that is actually felt:
// from the last word spoken to the settled transcript the rest of the turn is built on.
// Deciding whether to answer, the reply and the voice all queue behind this, so it is the
// first place a conversation that drags is worth looking.
func (s *Suite) TestTheTurnSettlesQuicklyAfterTheCallerStops() {
	timing := s.Measure()

	s.Less(timing.ToSettle, s.MaxSettle,
		"a caller should not wait this long after speaking for the turn to settle")
}

// TestWordsArriveWhileTheCallerIsStillSpeaking is what interim transcription buys. A
// provider that only reports the settled turn leaves the call looking unanswered until
// a second after the caller stopped talking.
func (s *Suite) TestWordsArriveWhileTheCallerIsStillSpeaking() {
	timing := s.Measure()

	s.NotZero(timing.WhileSpeaking,
		"nothing was written down until the caller stopped, which is what interim "+
			"transcription is for")
	s.Less(timing.ToFirstWords, timing.SpokeFor,
		"the first words should appear long before the last ones are spoken")
	s.Less(timing.ToFirstWords, s.MaxToFirstWords,
		"a caller should not talk this long before seeing anything")
}

// TestTheTranscriptSaysWhoSpokeAndWhichModelHeardIt covers what the router reads off a
// transcript once it has the text: who said it, who heard it, and whether it supersedes
// the last one or settles the turn.
func (s *Suite) TestTheTranscriptSaysWhoSpokeAndWhichModelHeardIt() {
	provider := s.Started()
	defer s.Hangup(provider)

	var heard, finals []stt.Transcript
	var sawConnected bool
	for _, event := range s.Collect(provider) {
		switch typed := event.(type) {
		case stt.Connected:
			sawConnected = true
		case stt.Transcript:
			if typed.Final() {
				finals = append(finals, typed)
			} else {
				heard = append(heard, typed)
			}
		case stt.Error:
			s.Failf("provider error", "%v", typed.Err)
		}
	}

	s.True(sawConnected, "should report the session becoming ready")
	s.Require().NotEmpty(heard, "should report the words as they are heard, not only at the end")
	s.Require().NotEmpty(finals, "should settle the turn")

	for _, hypothesis := range heard {
		s.Equal(stt.ModeReplacement, hypothesis.Mode,
			"each hypothesis restates the turn, so appending them would say it twice over")
	}
	s.Less(len(heard[0].Text), len(heard[len(heard)-1].Text),
		"the hypothesis should grow as the caller keeps talking")

	final := finals[0]
	s.Equal(speaker.UserID, final.Participant.UserID)
	s.Equal(provider.Provider(), final.Provider)
	s.Equal(provider.Model(), final.Model)
}

// TestClosingSettlesTheTailOfTheCall is about the caller who is cut off mid-sentence.
// There is no trailing silence to end the turn, so only ending the audio stream can
// settle what the server is still holding.
func (s *Suite) TestClosingSettlesTheTailOfTheCall() {
	if !s.SettlesOnClose {
		s.T().Skip("this provider does not settle the tail when the audio stream ends")
	}

	provider := s.Started()

	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
		}
		collected <- events
	}()

	s.Speak(provider, tailMs/s.ChunkMs)
	s.Hangup(provider)

	var events []stt.Event
	select {
	case events = <-collected:
	case <-time.After(patience):
		s.FailNow("the event channel was not closed")
	}

	var finals []stt.Transcript
	for _, event := range events {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			finals = append(finals, transcript)
		}
	}
	s.Require().NotEmpty(finals, "closing should settle the audio still being transcribed")
	s.Contains(strings.ToLower(finals[0].Text), s.opening())
}
