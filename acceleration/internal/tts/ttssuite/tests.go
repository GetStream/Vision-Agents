//go:build integration

package ttssuite

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// deltas are Sentence's opening words, fed the way an LLM produces them.
var deltas = []string{"The quick ", "brown fox ", "jumps over ", "the lazy dog."}

// TestSynthesizesRealSpeech is the one that says the provider works at all: a sentence
// goes up, speech comes back in the format the caller was promised, and the summary the
// stats are billed from adds up.
func (s *Suite) TestSynthesizesRealSpeech() {
	provider := s.Started()
	defer s.Hangup(provider)

	complete, chunks := s.Say(provider, tts.Request{ID: "u1", Text: Sentence, Final: true})

	s.Require().NotEmpty(chunks, "the model should have said something")
	s.Greater(len(chunks), 1, "audio should stream rather than arrive in one lump")
	s.Equal(provider.SampleRate(), chunks[0].Audio.SampleRate)
	s.Equal(1, chunks[0].Audio.Channels)

	s.EqualValues(len(Sentence), complete.Characters)
	s.Greater(complete.AudioDurationMs, minAudioMs)
	s.Less(complete.AudioDurationMs, maxAudioMs)
	s.Positive(complete.TimeToFirstByteMs)
	s.Less(complete.TimeToFirstByteMs, complete.SynthesisTimeMs)
	s.False(complete.Interrupted)

	if s.MaxTimeToFirstByte > 0 {
		s.Less(complete.TimeToFirstByteMs, s.MaxTimeToFirstByte,
			"a buffering server would sit on the text instead of starting to say it")
	}
}

// TestDeltasBecomeOneUtterance feeds a sentence the way an LLM produces it, a piece at a
// time into one context closed at the end of the turn. A streaming provider sends each
// piece upstream and a buffering one waits for the last, but both owe the caller a single
// utterance billed for the whole sentence rather than one per delta.
func (s *Suite) TestDeltasBecomeOneUtterance() {
	provider := s.Started()
	defer s.Hangup(provider)

	for _, delta := range deltas {
		s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: delta}))
	}
	complete, _ := s.Say(provider, tts.Request{ID: "u1", Final: true})

	s.EqualValues(len(strings.Join(deltas, "")), complete.Characters)
	s.Greater(complete.AudioDurationMs, 1_000.0)
}

// TestInterruptStopsALongUtterance is barge-in: the caller talks over the agent, and the
// agent stops saying the rest rather than finishing a paragraph nobody is listening to.
func (s *Suite) TestInterruptStopsALongUtterance() {
	if !s.Interruptible {
		s.T().Skip("this provider generates the whole utterance whatever the listener does")
	}

	provider := s.Started()
	defer s.Hangup(provider)

	long := strings.Repeat(Sentence+" ", 4)
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "u1", Text: long, Final: true}))

	// Cut in as soon as the first sound arrives, the way a user talking over the agent
	// would.
	s.Collect(provider, func(event tts.Event) bool {
		_, ok := event.(tts.AudioChunk)
		return ok
	})
	s.Require().NoError(provider.Interrupt())

	complete, _ := s.Settled(provider)

	s.True(complete.Interrupted)
	s.Less(complete.AudioDurationMs, maxAudioMs, "barge-in should not bill the whole utterance")
}
