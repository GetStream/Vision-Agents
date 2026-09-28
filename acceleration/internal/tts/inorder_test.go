package tts

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// perSentenceVoice stands in for a provider that takes each sentence as its own request.
// A test sends its events by hand, in whatever order the case needs.
type perSentenceVoice struct {
	emitter    *Emitter
	refuse     map[string]bool
	interrupts int
}

func (v *perSentenceVoice) Start(context.Context) error { return nil }

func (v *perSentenceVoice) Synthesize(request Request) error {
	if v.refuse[request.ID] {
		return errors.New("refused")
	}
	return nil
}

func (v *perSentenceVoice) Interrupt() error     { v.interrupts++; return nil }
func (v *perSentenceVoice) Events() <-chan Event { return v.emitter.Events() }
func (v *perSentenceVoice) Close() error         { v.emitter.Close(); return nil }
func (v *perSentenceVoice) Provider() string     { return "stub" }
func (v *perSentenceVoice) Model() string        { return "stub-model" }
func (v *perSentenceVoice) Streaming() bool      { return false }
func (v *perSentenceVoice) Prompt() string       { return "" }
func (v *perSentenceVoice) Performs() bool       { return false }
func (v *perSentenceVoice) Voice() string        { return "Kore" }

type InOrderSuite struct {
	suite.Suite
	provider *perSentenceVoice
	voice    TTS
}

func TestInOrderSuite(t *testing.T) {
	suite.Run(t, new(InOrderSuite))
}

func (s *InOrderSuite) SetupTest() {
	s.provider = &perSentenceVoice{emitter: NewEmitter(64), refuse: map[string]bool{}}
	s.voice = InOrder(s.provider)
	s.T().Cleanup(func() { _ = s.voice.Close() })
}

func (s *InOrderSuite) say(ids ...string) {
	for _, id := range ids {
		s.Require().NoError(s.voice.Synthesize(Request{ID: id, Text: id, Final: true}))
	}
}

func (s *InOrderSuite) chunk(id string, index int) {
	s.provider.emitter.Send(AudioChunk{SynthesisID: id, Index: index, Audio: speech(20)})
}

func (s *InOrderSuite) complete(id string) {
	s.provider.emitter.Send(SynthesisComplete{SynthesisID: id})
}

// next reads the next event the wrapper passes on.
func (s *InOrderSuite) next() Event {
	select {
	case event := <-s.voice.Events():
		return event
	case <-time.After(2 * time.Second):
		s.FailNow("nothing was passed on")
		return nil
	}
}

// heard reads n events, naming each as the sentence and what it was.
func (s *InOrderSuite) heard(n int) []string {
	var names []string
	for range n {
		names = append(names, name(s.next()))
	}
	return names
}

// quiet reports whether nothing more is passed on for a moment.
func (s *InOrderSuite) quiet() bool {
	select {
	case event := <-s.voice.Events():
		s.Failf("expected nothing", "got %s", name(event))
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

func name(event Event) string {
	switch typed := event.(type) {
	case SynthesisStarted:
		return typed.SynthesisID + " started"
	case AudioChunk:
		return typed.SynthesisID + " audio"
	case SynthesisComplete:
		if typed.Interrupted {
			return typed.SynthesisID + " interrupted"
		}
		return typed.SynthesisID + " done"
	case Error:
		return typed.SynthesisID + " error"
	}
	return "other"
}

func (s *InOrderSuite) TestSentencesAreHeardInTheOrderTheyWereAskedFor() {
	s.say("a", "b", "c")

	s.chunk("b", 0)
	s.chunk("a", 0)
	s.chunk("c", 0)
	s.chunk("b", 1)
	s.complete("c")
	s.complete("b")
	s.chunk("a", 1)
	s.complete("a")

	s.Equal([]string{
		"a audio", "a audio", "a done",
		"b audio", "b audio", "b done",
		"c audio", "c done",
	}, s.heard(8))
}

func (s *InOrderSuite) TestTheSentenceBeingHeardIsPassedOnAsItArrives() {
	s.say("a", "b")

	s.chunk("a", 0)

	s.Equal([]string{"a audio"}, s.heard(1), "the head must not wait for anything")
}

func (s *InOrderSuite) TestAStartIsPassedOnStraightAway() {
	s.say("a", "b")

	s.provider.emitter.Send(SynthesisStarted{SynthesisID: "b"})

	s.Equal([]string{"b started"}, s.heard(1))
}

func (s *InOrderSuite) TestABargeInSilencesEverythingButStillSettlesEverySentence() {
	s.say("a", "b", "c")
	s.chunk("a", 0)
	s.chunk("b", 0)
	s.complete("b")
	s.chunk("c", 0)
	s.Equal([]string{"a audio"}, s.heard(1))

	s.Require().NoError(s.voice.Interrupt())
	s.chunk("a", 1)
	s.complete("a")
	s.chunk("c", 1)
	s.complete("c")

	s.ElementsMatch([]string{"a interrupted", "b interrupted", "c interrupted"}, s.heard(3),
		"a sentence nobody heard in full is settled as interrupted, and nothing more is heard")
	s.True(s.quiet())
	s.Equal(1, s.provider.interrupts, "the provider is stopped too")
}

func (s *InOrderSuite) TestTheNextTurnIsNotHeldBehindOneABargeInAbandoned() {
	s.say("a")
	s.chunk("a", 0)
	s.Equal([]string{"a audio"}, s.heard(1))
	s.Require().NoError(s.voice.Interrupt())

	s.say("d")
	s.chunk("d", 0)
	s.complete("d")

	s.Equal([]string{"d audio", "d done"}, s.heard(2))

	s.complete("a")
	s.Equal([]string{"a interrupted"}, s.heard(1))
}

func (s *InOrderSuite) TestAFailedSentenceLetsTheNextOneBeHeard() {
	s.say("a", "b")
	s.chunk("b", 0)
	s.complete("b")
	s.provider.emitter.Send(Error{SynthesisID: "a", Err: errors.New("boom")})

	s.Equal([]string{"a error"}, s.heard(1), "an error is passed on before the end it explains")
	s.complete("a")

	s.Equal([]string{"a done", "b audio", "b done"}, s.heard(3))
}

func (s *InOrderSuite) TestASentenceTheProviderRefusesDoesNotHoldUpTheNext() {
	s.provider.refuse["a"] = true
	s.Require().Error(s.voice.Synthesize(Request{ID: "a", Text: "a", Final: true}))
	s.say("b")

	s.chunk("b", 0)

	s.Equal([]string{"b audio"}, s.heard(1))
}

func (s *InOrderSuite) TestASentenceWithoutAnIdIsStillKeptInOrder() {
	s.say("a")
	s.Require().NoError(s.voice.Synthesize(Request{Text: "unnamed", Final: true}))
	s.say("c")

	s.chunk("c", 0)
	s.complete("c")
	s.complete("a")

	s.Equal([]string{"a done"}, s.heard(1))
	s.True(s.quiet(), "c waits behind the unnamed sentence asked for before it")
}

func (s *InOrderSuite) TestWhatIsStillHeldWhenTheProviderClosesIsPassedOn() {
	s.say("a", "b")
	s.chunk("b", 0)
	s.complete("b")

	s.Require().NoError(s.voice.Close())

	s.Equal([]string{"b audio", "b done"}, s.heard(2))
	_, open := <-s.voice.Events()
	s.False(open, "the events close with the provider")
}

func (s *InOrderSuite) TestItSpeaksInTheProvidersVoice() {
	voiced, ok := s.voice.(Voiced)
	s.Require().True(ok)
	s.Equal("Kore", voiced.Voice())
	s.False(s.voice.Streaming())
}
