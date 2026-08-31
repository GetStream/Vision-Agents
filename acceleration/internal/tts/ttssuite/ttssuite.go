//go:build integration

// Package ttssuite is what every streaming text-to-speech provider is held to on a real
// call: the words come back as speech, the speech starts arriving before the sentence is
// finished, and the agent can stop talking when the caller cuts in.
//
// A provider suite embeds Suite, says how to build a provider and what it may be held to,
// and inherits those tests. Anything only one provider does, such as Fish's sample rate
// option or Breeze's voice described in words, stays in that provider's own file.
package ttssuite

import (
	"context"
	"os"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// Sentence takes a couple of seconds to say, which is long enough for the duration and
// chunking assertions to mean something.
const Sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

const (
	// defaultTimeout bounds a session and the wait for the events it produces. It is
	// generous because it is not a target: it is the point past which a provider has
	// stopped answering rather than merely being slow.
	defaultTimeout = 60 * time.Second
	// minAudioMs and maxAudioMs bracket what Sentence should cost to say. Roughly 15
	// words at a conversational pace.
	minAudioMs = 2_000.0
	maxAudioMs = 20_000.0
)

// Provider is what the suite is run against: the tts.TTS contract, plus the sample rate
// every provider reports but the interface does not carry.
type Provider interface {
	tts.TTS
	SampleRate() int
}

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds an unstarted provider configured for an ordinary turn.
	New func() Provider
	// Requires are the environment variables without which the provider cannot be
	// reached, and whose absence skips rather than fails.
	Requires []string

	// Timeout is the context a session is opened with and the longest a test waits for
	// the events it expects. A deployment that scales to zero needs a cold start's worth
	// on top.
	Timeout time.Duration
	// MaxTimeToFirstByte is how long a listener may wait to hear anything, in
	// milliseconds. Zero leaves it unasserted, which is right for a provider whose
	// latency is not what it is being kept for.
	MaxTimeToFirstByte float64
	// Interruptible marks a provider that can cut an utterance short, rather than
	// generating all of it whatever the listener does.
	Interruptible bool
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}
}

// Started builds and opens an ordinary provider.
func (s *Suite) Started() Provider {
	provider := s.New()
	s.Start(provider)
	return provider
}

// Start opens a provider the caller built, so a test can use options of its own and still
// get the suite's timeout.
func (s *Suite) Start(provider Provider) {
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
}

// Hangup ends the session the way the end of a turn would.
func (s *Suite) Hangup(provider Provider) {
	s.Require().NoError(provider.Close())
}

// Collect reads events until the predicate is satisfied. A provider error fails the test
// straight away, so a rejected request reports what went wrong instead of timing out.
func (s *Suite) Collect(provider Provider, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(s.Timeout)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if failure, failed := event.(tts.Error); failed {
				s.FailNowf("provider error", "%v", failure.Err)
			}
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for audio")
			return events
		}
	}
}

// Say synthesises one utterance and returns how it finished, with the audio it produced.
func (s *Suite) Say(provider Provider, request tts.Request) (tts.SynthesisComplete, []tts.AudioChunk) {
	s.Require().NoError(provider.Synthesize(request))
	return s.Settled(provider)
}

// Settled waits for the utterance in flight to finish, for a test that sent the text
// itself. It is what Say is built out of.
func (s *Suite) Settled(provider Provider) (tts.SynthesisComplete, []tts.AudioChunk) {
	events := s.Collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks []tts.AudioChunk
	var complete tts.SynthesisComplete
	for _, event := range events {
		switch typed := event.(type) {
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}
	return complete, chunks
}
