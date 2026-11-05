package audiofloor

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// InterimSuite is a transcriber writing the words so far while the caller talks, asking
// about each pause for its own agent.
type InterimSuite struct{ transcribing }

func TestInterimSuite(t *testing.T) {
	suite.Run(t, new(InterimSuite))
}

func (s *InterimSuite) SetupTest() {
	s.interims = true
	s.transcribing.SetupTest()
	s.ending(0.999)
}

// ending has the server answer p_eot, with the floor free, and a ruling to respond.
func (s *InterimSuite) ending(pEOT float64) {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.server.decision = fmt.Sprintf(`{"answer":"respond","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`, pEOT, pEOT)
}

// outcomes is what became of the interims asked so far, by outcome.
func outcomes() map[string]float64 {
	out := map[string]float64{}
	for _, outcome := range []string{"written", "late", "unstable", "empty", "foreign", "refused", "failed"} {
		out[outcome] = testutil.ToFloat64(interims.WithLabelValues(outcome))
	}
	return out
}

// askedIn are the languages each request for words alone named, in order.
func (s *InterimSuite) askedIn() []string {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	return append([]string(nil), s.server.languages...)
}

// became waits for one more interim to have had outcome.
func (s *InterimSuite) became(outcome string, before map[string]float64) {
	s.T().Helper()
	s.Eventually(func() bool { return outcomes()[outcome] > before[outcome] }, 5*time.Second, time.Millisecond,
		"no interim became %s: %v, then %v", outcome, before, outcomes())
}

// With no languages declared, an interim's language is the utterance's own: the one the
// server heard its first words in, at a pause, which the words so far are then asked for in.
// Before that nothing is asked for, as a guess on a second of speech would be, and words
// that come back in another language, the server's own refusal (noise: language) or words it
// names otherwise, are dropped. A pause's words in another language are kept for the final
// and not shown either.
func (s *InterimSuite) TestInterimsAreAskedForInTheUtterancesOwnLanguage() {
	s.languages = nil
	s.reopen(time.Now)
	s.ending(0.1)
	s.say(2000, true)
	s.none()
	s.Zero(s.transcribed(), "no words asked for alone before the utterance's language is known")

	s.say(220, false)
	s.answered()
	shown := next[stt.Transcript](&s.transcribing)
	s.True(shown.Interim, "the pause's words, in the language the server heard")
	s.say(1000, true)
	first := next[stt.Transcript](&s.transcribing)
	s.True(first.Interim)
	s.Equal([]string{"English"}, s.askedIn(), "asked for in the language of the pause's words")

	for _, foreign := range []struct{ language, noise string }{{"Hindi", ""}, {"English", "language"}} {
		before := outcomes()
		s.server.mu.Lock()
		s.server.language, s.server.noise = foreign.language, foreign.noise
		s.server.mu.Unlock()
		s.say(1000, true)
		s.became("foreign", before)
		s.none()
	}
	s.server.mu.Lock()
	s.server.language, s.server.noise, s.server.words = "Portuguese", "", "Duas passagens para Bilbau, por favor."
	s.server.mu.Unlock()
	s.say(220, false)
	s.answered()
	s.none()
	s.server.mu.Lock()
	s.server.language, s.server.words = "English", "Two tickets to Bilbao, please, for tonight."
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Equal("Two tickets to Bilbao, please, for tonight.", next[stt.Transcript](&s.transcribing).Text)
}

// With languages declared, the words so far are asked for in them until the utterance's own
// is known, and in that alone from then on.
func (s *InterimSuite) TestInterimsNarrowToTheLanguageOfTheFirstWords() {
	s.ending(0.1)
	s.say(1000, true)
	s.True(next[stt.Transcript](&s.transcribing).Interim)
	s.say(220, false)
	s.answered()
	s.True(next[stt.Transcript](&s.transcribing).Interim, "the pause's words")
	s.say(1000, true)
	s.True(next[stt.Transcript](&s.transcribing).Interim)
	s.Equal([]string{"en,pt-BR", "English"}, s.askedIn())
}

// An interim never takes a watcher's line back. Two writings of the same audio can differ
// wholesale over noise or speech in another language, so words that don't carry on the ones
// shown are dropped: fewer words, a script the shown words are not in, or an opening and
// words that are not theirs. The final is as it is.
func (s *InterimSuite) TestAnInterimThatGoesBackOnTheWordsShownIsDropped() {
	s.say(1000, true)
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](&s.transcribing).Text)
	for _, unstable := range []string{
		"Two tickets to Bilbao.",
		"Two tickets to Bilbao, please. 性格 木气 太 木",
		"Bye, bye, biga. Bye, there. Bye, biga. Bye, kill the jago.",
		"Two tickets, two tickets, two tickets, two tickets, two tickets.",
	} {
		before := outcomes()
		s.server.mu.Lock()
		s.server.words = unstable
		s.server.mu.Unlock()
		s.say(1000, true)
		s.became("unstable", before)
		s.none()
	}
	s.server.mu.Lock()
	s.server.words = "Two tickets to Bilbao, please, for tonight."
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Equal("Two tickets to Bilbao, please, for tonight.", next[stt.Transcript](&s.transcribing).Text)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please, for tonight.", end.Text)
}

// While the caller talks on, the words so far are written about once a second of speech, as
// an interim replacement of the utterance (Interim), the whole of it each time, asked for
// alone; the final at the turn's end is as it was, from the pause's check.
func (s *InterimSuite) TestTheWordsSoFarAreWrittenWhileTheCallerTalks() {
	s.say(980, true)
	s.Zero(s.transcribed(), "not before a second of speech")
	s.say(20, true)
	first := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeReplacement, first.Mode)
	s.True(first.Interim)
	s.Equal(int64(1), first.Utterance)
	s.Equal("Two tickets to Bilbao, please.", first.Text)
	s.InDelta(float64(300+1000), first.AudioDurationMs, 40, "the utterance so far, its preroll included")

	s.say(1000, true)
	second := next[stt.Transcript](&s.transcribing)
	s.True(second.Interim)
	s.InDelta(float64(300+2000), second.AudioDurationMs, 40, "the whole utterance again")

	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.False(end.Interim)
	s.Equal(int64(1), end.Utterance)
	s.Equal(2, s.transcribed(), "two interims, and the final's words with its decision")
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	s.Equal(1, s.server.withWords)
}

// One interim is out at a time: a slow server is asked for the next once it has answered,
// and the words that come are of all the speech so far.
func (s *InterimSuite) TestOneInterimIsOutAtATime() {
	gate := make(chan struct{})
	s.server.mu.Lock()
	s.server.transcribeGate = gate
	s.server.mu.Unlock()
	s.say(3000, true)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond)
	s.Equal(1, s.transcribed(), "one out, none behind it")
	s.server.mu.Lock()
	s.server.transcribeGate = nil
	s.server.mu.Unlock()
	close(gate)
	first := next[stt.Transcript](&s.transcribing)
	s.True(first.Interim)
	s.InDelta(float64(300+1000), first.AudioDurationMs, 40)
	s.Eventually(func() bool { return !s.transcriber.interimOut.Load() }, 5*time.Second, time.Millisecond)
	s.say(20, true)
	again := next[stt.Transcript](&s.transcribing)
	s.True(again.Interim)
	s.InDelta(float64(300+3020), again.AudioDurationMs, 40, "asked again as soon as the first had answered, with all the speech since")
}

// An interim the server refuses, busy, or fails to write is dropped, not asked again: the
// next second of speech asks for the next, and the final is unaffected.
func (s *InterimSuite) TestAnInterimTheServerRefusesIsDropped() {
	for status, outcome := range map[int]string{http.StatusServiceUnavailable: "refused", http.StatusTooManyRequests: "refused", http.StatusInternalServerError: "failed"} {
		s.Run(http.StatusText(status), func() {
			s.SetupTest()
			before := outcomes()
			s.server.mu.Lock()
			s.server.failing, s.server.failWith = 1, status
			s.server.mu.Unlock()
			s.say(1000, true)
			s.Eventually(func() bool { return outcomes()[outcome] > before[outcome] }, 5*time.Second, time.Millisecond)
			s.none()
			s.say(1000, true)
			s.True(next[stt.Transcript](&s.transcribing).Interim, "the next second asks again")
			s.say(220, false)
			s.Equal(stt.ModeFinal, next[stt.Transcript](&s.transcribing).Mode)
			s.Equal(1, s.transcribed(), "the refused interim was not asked again")
		})
	}
}

// A server that answers without words, every lane that writes them busy, is a refusal too.
func (s *InterimSuite) TestAnInterimAnsweredBusyIsDropped() {
	before := outcomes()
	s.server.mu.Lock()
	s.server.busy = 1
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Eventually(func() bool { return outcomes()["refused"] > before["refused"] }, 5*time.Second, time.Millisecond)
	s.none()
}

// The final takes precedence: an interim the server answers after the turn has ended is
// discarded rather than sent after the final, where it would read as the next utterance.
func (s *InterimSuite) TestAnInterimAnsweredAfterTheFinalIsDiscarded() {
	before := outcomes()
	gate := make(chan struct{})
	s.server.mu.Lock()
	s.server.transcribeGate = gate
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	close(gate)
	s.Eventually(func() bool { return outcomes()["late"] > before["late"] }, 5*time.Second, time.Millisecond)
	s.none()
}

// An interim holds up nothing: a pause's check is asked at its time, and the turn ends on it,
// while the interim is still out.
func (s *InterimSuite) TestAnInterimHoldsUpNoCheck() {
	gate := make(chan struct{})
	defer close(gate)
	s.server.mu.Lock()
	s.server.transcribeGate = gate
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Eventually(func() bool {
		s.server.mu.Lock()
		defer s.server.mu.Unlock()
		return s.server.held == 1
	}, 5*time.Second, time.Millisecond)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.InDelta(float64(300+1000+200), end.AudioDurationMs, 40, "ended 200 ms into the pause")
}

// The words of a pause the model waits on are shown as well, as an interim, rather than
// kept for the turn's end: they were written anyway, with the pause's first check.
func (s *InterimSuite) TestAPauseTheModelWaitsOnShowsItsWords() {
	s.ending(0.1)
	s.say(800, true)
	s.say(220, false)
	s.answered()
	shown := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeReplacement, shown.Mode)
	s.True(shown.Interim)
	s.Equal("Two tickets to Bilbao, please.", shown.Text)
	s.InDelta(float64(300+800+200), shown.AudioDurationMs, 40, "the pause's first check's words")
	s.Zero(s.transcribed(), "asked for no words of its own")
	for quiet := 220; quiet < int(endAfter/time.Millisecond); quiet += 300 {
		s.say(300, false)
		s.answered()
	}
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
}

// An interim sends the whole utterance so far, which a replacement must. The audio an
// interim sends a second is bounded by what a decision hears: past 16 s of utterance one is
// asked less often as it grows, so a caller who talks on costs a transcription lane at most
// the words of 16 s a second, however long they go on.
func (s *InterimSuite) TestInterimsComeLessOftenAsTheUtteranceGrows() {
	s.say(48000, true)
	s.Eventually(func() bool { return !s.transcriber.interimOut.Load() }, 5*time.Second, time.Millisecond)
	requests := s.requests()
	s.GreaterOrEqual(len(requests), 20, "one a second of speech to 16 s, then fewer")
	s.Less(len(requests), 40, "fewer than one a second past 16 s")
	var total int
	for i, samples := range requests {
		if i > 0 {
			s.Greater(samples, requests[i-1], "the whole utterance so far, each time")
		}
		total += samples
	}
	s.LessOrEqual(total, 48*int(Window/time.Second)*SampleRate, "at most 16 s of audio a second of speech")
	for i, samples := range requests {
		at := time.Duration(samples) * time.Second / SampleRate
		if i > 0 {
			since := at - time.Duration(requests[i-1])*time.Second/SampleRate
			s.GreaterOrEqual(since+2*frame, time.Duration(float64(interimEvery)*max(1, float64(at)/float64(Window))), "asked after %s of speech, %s since the last", at, since)
		}
	}
}

// With interims off, nothing is written while the caller talks: neither the words so far
// nor those of a pause the model waits on, which are kept for the turn's end, as they were.
func (s *InterimSuite) TestInterimsCanBeTurnedOff() {
	s.transcriber.interims = false
	s.ending(0.5)
	s.say(3000, true)
	s.say(220, false)
	s.answered()
	s.none()
	s.ending(0.999)
	s.say(300, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Zero(s.transcribed())
}
