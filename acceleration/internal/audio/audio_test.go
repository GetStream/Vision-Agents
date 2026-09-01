package audio

import (
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type AudioSuite struct {
	suite.Suite
}

func TestAudioSuite(t *testing.T) {
	suite.Run(t, new(AudioSuite))
}

func (s *AudioSuite) TestDurationMsCountsFramesNotSamples() {
	mono := PcmData{Samples: make([]int16, 16000), SampleRate: 16000, Channels: 1}
	s.InDelta(1000.0, mono.DurationMs(), 0.001)

	// The same sample count spread over two channels is half the wall-clock time.
	stereo := PcmData{Samples: make([]int16, 16000), SampleRate: 16000, Channels: 2}
	s.InDelta(500.0, stereo.DurationMs(), 0.001)
}

func (s *AudioSuite) TestDurationMsIsZeroWhenUnset() {
	s.Zero(PcmData{}.DurationMs())
}

func (s *AudioSuite) TestDurationMsFollowsTheSampleRate() {
	// A second of 24 kHz audio holds more samples but lasts just as long.
	pcm := PcmData{Samples: make([]int16, 24000), SampleRate: 24000, Channels: 1}
	s.InDelta(1000.0, pcm.DurationMs(), 0.001)
}

func (s *AudioSuite) TestBytesEncodesLittleEndianPCM16() {
	pcm := PcmData{Samples: []int16{0, 1, -1, 32767, -32768}, SampleRate: 16000, Channels: 1}

	raw := pcm.Bytes()
	s.Len(raw, len(pcm.Samples)*2)

	for i, want := range pcm.Samples {
		got := int16(binary.LittleEndian.Uint16(raw[i*2:]))
		s.Equal(want, got, "sample %d should survive the round trip", i)
	}
}

func (s *AudioSuite) TestFromBytesReversesBytes() {
	original := PcmData{Samples: []int16{0, 1, -1, 32767, -32768}, SampleRate: 24000, Channels: 1}

	decoded := FromBytes(original.Bytes(), 24000, 1)

	s.Equal(original, decoded)
}

func (s *AudioSuite) TestFromBytesDropsATrailingOddByte() {
	decoded := FromBytes([]byte{0x01, 0x02, 0x03}, 16000, 1)

	s.Len(decoded.Samples, 1, "half a sample cannot be decoded")
	s.EqualValues(0x0201, decoded.Samples[0])
}

func (s *AudioSuite) TestValidateChecksTheRateTheCallerWants() {
	s.NoError(PcmData{SampleRate: 16000, Channels: 1}.Validate(16000))
	s.NoError(PcmData{SampleRate: 44100, Channels: 1}.Validate(44100))
	s.ErrorContains(PcmData{SampleRate: 48000, Channels: 1}.Validate(16000), "sample rate must be 16000")
	s.ErrorContains(PcmData{SampleRate: 16000, Channels: 2}.Validate(16000), "channels must be 1")
}

func (s *AudioSuite) TestTwentyFourKilohertzSpeechBecomesTheSixteenEveryTranscriberWants() {
	// A second of a 400 Hz tone, which is inside the range a voice occupies and well under
	// what either rate can carry, so resampling it must not change what it is.
	source := tone(24_000, 400, time.Second)

	resampled := Resample(source, 16_000, 1)

	s.Equal(16_000, resampled.SampleRate)
	s.Equal(1, resampled.Channels)
	s.InDelta(16_000, len(resampled.Samples), 2)
	// The tone survives: it crosses zero as often as it did, so it is still 400 Hz rather
	// than a stretched version of it or one folded back down by aliasing.
	s.InDelta(crossings(source.Samples), crossings(resampled.Samples), 4)
}

func (s *AudioSuite) TestAudioAlreadyAtTheRateItIsWantedAtIsLeftAlone() {
	source := tone(16_000, 400, 100*time.Millisecond)

	resampled := Resample(source, 16_000, 1)

	s.Equal(source.Samples, resampled.Samples)
}

func (s *AudioSuite) TestTwoChannelsAreFoldedIntoTheOneATranscriberReads() {
	// Left is loud and right is silent, so the fold has to be an average rather than one
	// channel picked: picking the right would return silence.
	source := PcmData{
		Samples:    []int16{1000, 0, 2000, 0, 3000, 0},
		SampleRate: 16_000,
		Channels:   2,
	}

	resampled := Resample(source, 16_000, 1)

	s.Equal([]int16{500, 1000, 1500}, resampled.Samples)
}

// tone is a sine wave, which is the simplest thing whose identity survives resampling.
func tone(sampleRate, hertz int, length time.Duration) PcmData {
	count := sampleRate * int(length/time.Millisecond) / 1000
	samples := make([]int16, count)
	for i := range samples {
		samples[i] = int16(8000 * math.Sin(2*math.Pi*float64(hertz)*float64(i)/float64(sampleRate)))
	}
	return PcmData{Samples: samples, SampleRate: sampleRate, Channels: 1}
}

// crossings counts how often the wave passes through zero, which is its frequency read off
// without a transform.
func crossings(samples []int16) int {
	count := 0
	for i := 1; i < len(samples); i++ {
		if (samples[i-1] < 0) != (samples[i] < 0) {
			count++
		}
	}
	return count
}
