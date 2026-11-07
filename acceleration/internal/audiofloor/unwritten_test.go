package audiofloor

import (
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// told is an agent that has its transcriber ask about each pause, and is told of each
// utterance that ended with no final (Unwritten).
type told struct {
	*asking
	lock      sync.Mutex
	unwritten map[int64]string
}

func (t *told) Unwritten(_ stt.Participant, utterance int64, why string) {
	t.lock.Lock()
	defer t.lock.Unlock()
	t.unwritten[utterance] = why
}

// unwrittenAs says whether utterance was told unwritten, and why.
func (t *told) unwrittenAs(utterance int64) (string, bool) {
	t.lock.Lock()
	defer t.lock.Unlock()
	why, ok := t.unwritten[utterance]
	return why, ok
}

// UnwrittenSuite is a transcriber whose agent is told of each utterance that ends with no
// final, with any words a check over the agent wrote of it that no final will carry.
type UnwrittenSuite struct {
	transcribing
	agent *told
}

func TestUnwrittenSuite(t *testing.T) {
	suite.Run(t, new(UnwrittenSuite))
}

func (s *UnwrittenSuite) SetupTest() {
	s.transcribing.SetupTest()
	s.agent = &told{asking: &asking{hearer: NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))}, unwritten: map[int64]string{}}
	s.transcriber.AskOnPauses(s.agent)
}

// finals are the finals sent before no more come.
func (s *UnwrittenSuite) finals() []stt.Transcript {
	var finals []stt.Transcript
	for {
		select {
		case event := <-s.transcriber.Events():
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				finals = append(finals, transcript)
			}
		case <-time.After(200 * time.Millisecond):
			return finals
		}
	}
}

// What a check over the agent wrote of an utterance goes out in its final, or the agent is
// told the utterance ended with none: it acted on those words, and no final carries them. On
// telecom.interrupt (t2 and t3) a check over the agent wrote "Last four: ninety-eight
// twenty-one." and shortened the reply, and the words of the pause after them came back empty
// under a ruling to ignore. The utterance ended with no final and no line in the log.
func (s *UnwrittenSuite) TestAnUtteranceACheckOverTheAgentWroteEndsInAFinalOrIsToldUnwritten() {
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"shorten","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"shorten"}}`
	s.server.words = "Last four: ninety-eight twenty-one."
	s.server.mu.Unlock()
	s.say(1000, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) > 0 }, 5*time.Second, time.Millisecond, "no check over the agent")
	s.Equal(int64(1), s.agent.overheardSoFar()[0].Utterance)

	s.server.mu.Lock()
	s.server.decision = `{"answer":"continue","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"ignore","floor":"continue"}}`
	s.server.words = ""
	s.server.mu.Unlock()
	// Past the policy's cap and checkGrace, where the pause has ended whatever the model says.
	s.say(3600, false)

	finals := s.finals()
	why, unwritten := s.agent.unwrittenAs(1)
	s.True(len(finals) == 1 && finals[0].Utterance == 1 || unwritten,
		"the digits went out in no final, and the agent was not told: finals %v", finals)
	if unwritten {
		s.Empty(finals, "told unwritten, with a final")
		s.Empty(why, "no reason the server gave")
	}
}

// Words the server hears in a language the callers don't speak come back with no text, for
// that reason, and an utterance that ends on them has no final: the agent is told why.
func (s *UnwrittenSuite) TestAnUtteranceInAnotherLanguageIsToldUnwrittenForIt() {
	s.server.mu.Lock()
	s.server.failDecide = true
	s.server.noise = "language"
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(3000, false)
	s.Empty(s.finals())
	s.Eventually(func() bool {
		why, ok := s.agent.unwrittenAs(1)
		return ok && why == "language"
	}, 5*time.Second, time.Millisecond, "the agent was not told the utterance was in another language")
}
