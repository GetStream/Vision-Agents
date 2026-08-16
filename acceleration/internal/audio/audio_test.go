package audio

import (
	"encoding/binary"
	"testing"

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
