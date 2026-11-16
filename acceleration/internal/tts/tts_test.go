package tts

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

var errUnauthorized = errors.New("unauthorized")

type TTSSuite struct {
	suite.Suite
}

func TestTTSSuite(t *testing.T) {
	suite.Run(t, new(TTSSuite))
}

// speech returns the given milliseconds of 24 kHz mono audio.
func speech(durationMs int) audio.PcmData {
	return audio.PcmData{
		Samples:    make([]int16, 24_000*durationMs/1000),
		SampleRate: 24_000,
		Channels:   1,
	}
}

func (s *TTSSuite) TestSynthesisGeneratesAnIDWhenTheCallerHasNone() {
	first := NewSynthesis("")
	second := NewSynthesis("")

	s.NotEmpty(first.ID)
	s.NotEqual(first.ID, second.ID, "two utterances must not share an id")
	s.Equal("mine", NewSynthesis("mine").ID, "a caller's id is kept")
}

func (s *TTSSuite) TestSynthesisCountsCharactersAcrossDeltas() {
	synthesis := NewSynthesis("s1")
	synthesis.AddText("Hello ")
	synthesis.AddText("world")

	complete := synthesis.Complete("elevenlabs", "eleven_flash_v2_5", false)

	s.EqualValues(11, complete.Characters)
}

func (s *TTSSuite) TestSynthesisCountsCharactersNotBytes() {
	// Billing is per character, so a multi-byte character must count once.
	synthesis := NewSynthesis("s1")
	synthesis.AddText("héllo")

	s.EqualValues(5, synthesis.Complete("fish", "s2-pro", false).Characters)
}

func (s *TTSSuite) TestSynthesisNumbersChunksInOrder() {
	synthesis := NewSynthesis("s1")

	first := synthesis.Chunk(speech(100))
	second := synthesis.Chunk(speech(100))

	s.Equal(0, first.Index)
	s.Equal(1, second.Index)
	s.Equal("s1", second.SynthesisID)
}

func (s *TTSSuite) TestSynthesisSumsTheAudioItProduced() {
	synthesis := NewSynthesis("s1")
	synthesis.Chunk(speech(120))
	synthesis.Chunk(speech(80))

	complete := synthesis.Complete("elevenlabs", "eleven_flash_v2_5", false)

	s.InDelta(200.0, complete.AudioDurationMs, 0.001)
}

func (s *TTSSuite) TestSynthesisMeasuresTimeToFirstByte() {
	synthesis := NewSynthesis("s1")
	time.Sleep(20 * time.Millisecond)
	synthesis.Chunk(speech(100))
	time.Sleep(20 * time.Millisecond)
	synthesis.Chunk(speech(100))

	complete := synthesis.Complete("elevenlabs", "eleven_flash_v2_5", false)

	s.GreaterOrEqual(complete.TimeToFirstByteMs, 15.0, "the wait for the first chunk")
	s.Less(complete.TimeToFirstByteMs, complete.SynthesisTimeMs,
		"the first chunk arrives before the last one")
}

func (s *TTSSuite) TestSynthesisWithNoAudioReportsNoTimeToFirstByte() {
	complete := NewSynthesis("s1").Complete("elevenlabs", "eleven_flash_v2_5", false)

	s.Zero(complete.TimeToFirstByteMs, "nothing was heard, so there was no first byte")
	s.Zero(complete.AudioDurationMs)
}

func (s *TTSSuite) TestSynthesisReportsBeingInterrupted() {
	synthesis := NewSynthesis("s1")
	synthesis.AddText("this will be cut short")
	synthesis.Chunk(speech(50))

	complete := synthesis.Complete("elevenlabs", "eleven_flash_v2_5", true)

	s.True(complete.Interrupted)
	s.InDelta(50.0, complete.AudioDurationMs, 0.001, "the audio that did play still counts")
}

func (s *TTSSuite) TestEmitterDeliversEventsInOrder() {
	emitter := NewEmitter(4)
	defer emitter.Close()

	emitter.Send(SynthesisStarted{SynthesisID: "s1"})
	emitter.Send(AudioChunk{SynthesisID: "s1", Audio: speech(20)})

	started, ok := (<-emitter.Events()).(SynthesisStarted)
	s.Require().True(ok)
	s.Equal("s1", started.SynthesisID)

	chunk, ok := (<-emitter.Events()).(AudioChunk)
	s.Require().True(ok)
	s.InDelta(20.0, chunk.Audio.DurationMs(), 0.001)
}

func (s *TTSSuite) TestSendAfterCloseDoesNotPanic() {
	emitter := NewEmitter(1)
	emitter.Close()

	s.NotPanics(func() { emitter.Send(Connected{Provider: "elevenlabs"}) })
}

func (s *TTSSuite) TestErrorUnwrapsToTheProviderFailure() {
	cause := Error{Provider: "fish", Err: errUnauthorized, Context: "handshake"}

	s.ErrorIs(cause, errUnauthorized)
	s.Equal("unauthorized", cause.Error())
}
