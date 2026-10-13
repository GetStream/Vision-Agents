package main

import (
	"math/rand/v2"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type StampSuite struct {
	suite.Suite
	// sent is the moment the stamps under test carry.
	sent time.Time
}

func TestStampSuite(t *testing.T) {
	suite.Run(t, new(StampSuite))
}

func (s *StampSuite) SetupTest() {
	s.sent = time.UnixMilli(1_760_000_000_123)
}

// listen plays audio into a detector in 20 ms packets, the first arriving at start, and
// returns what it heard.
func (s *StampSuite) listen(audio []int16, start time.Time) ([]Heard, int) {
	var detector Detector
	var heard []Heard
	for i := 0; i < len(audio); i += packetSamples {
		end := min(i+packetSamples, len(audio))
		at := start.Add(time.Duration(i/packetSamples) * packetInterval)
		heard = append(heard, detector.Push(audio[i:end], at)...)
	}
	return heard, detector.Garbled()
}

// call is silence, then a stamp after lead samples, then silence, as a call carries it.
func (s *StampSuite) call(lead int, stamp []int16) []int16 {
	audio := make([]int16, lead)
	audio = append(audio, stamp...)
	return append(audio, make([]int16, sampleRate/2)...)
}

func (s *StampSuite) TestAStampReadsBackAsTheMillisecondItCarried() {
	heard, garbled := s.listen(s.call(sampleRate, Stamp(Millis(s.sent))), s.sent)

	s.Require().Len(heard, 1)
	s.Equal(Millis(s.sent), heard[0].Sent)
	s.Zero(garbled)
}

func (s *StampSuite) TestTheLatencyIsWhenThePacketHoldingTheOnsetArrived() {
	// The stamp starts 1.5 s into the audio, in the packet that arrives 1.5 s after the
	// first, and the network held everything for 180 ms on top.
	arrived := s.sent.Add(180 * time.Millisecond)
	heard, _ := s.listen(s.call(sampleRate*3/2, Stamp(Millis(s.sent))), arrived.Add(-1500*time.Millisecond))

	s.Require().Len(heard, 1)
	// The onset is found a few samples into the stamp's fade-in.
	s.InDelta(180*time.Millisecond, heard[0].Latency(), float64(time.Millisecond))
}

func (s *StampSuite) TestAnOnsetPartWayIntoAPacketIsDueByItsPlaceInIt() {
	// The carrier repacketised: the stamp starts 80 samples, 10 ms, into the packet.
	heard, _ := s.listen(s.call(sampleRate+80, Stamp(Millis(s.sent))), s.sent.Add(-time.Second))

	s.Require().Len(heard, 1)
	s.InDelta(10*time.Millisecond, heard[0].Latency(), float64(time.Millisecond))
}

func (s *StampSuite) TestAStampSurvivesMuLawAQuieterLineAndNoise() {
	random := rand.New(rand.NewPCG(1, 2))
	audio := s.call(sampleRate, Stamp(Millis(s.sent)))
	for i, sample := range audio {
		// 12 dB down and a noise floor around -50 dBFS.
		audio[i] = sample/4 + int16(random.IntN(201)-100)
	}
	audio = DecodeULaw(EncodeULaw(audio))

	heard, garbled := s.listen(audio, s.sent)

	s.Require().Len(heard, 1)
	s.Equal(Millis(s.sent), heard[0].Sent)
	s.Zero(garbled)
}

func (s *StampSuite) TestAStampWithAWrongToneIsCountedAsGarbledNotMisread() {
	stamp := Stamp(Millis(s.sent))
	// Replace the third data symbol with another stamp's, which the checksum must catch.
	other := Stamp(Millis(s.sent) ^ 0x00F00000)
	copy(stamp[4*symbolSamples:5*symbolSamples], other[4*symbolSamples:5*symbolSamples])

	heard, garbled := s.listen(s.call(sampleRate, stamp), s.sent)

	s.Empty(heard)
	s.Equal(1, garbled)
}

func (s *StampSuite) TestStampsOneAfterAnotherAreEachRead() {
	first := s.sent
	second := s.sent.Add(2 * time.Second)
	audio := s.call(sampleRate, Stamp(Millis(first)))
	audio = append(audio, s.call(sampleRate, Stamp(Millis(second)))...)

	heard, _ := s.listen(audio, s.sent)

	s.Require().Len(heard, 2)
	s.Equal(Millis(first), heard[0].Sent)
	s.Equal(Millis(second), heard[1].Sent)
}

func (s *StampSuite) TestNoiseWithoutAStampIsNotHeard() {
	random := rand.New(rand.NewPCG(3, 4))
	audio := make([]int16, 3*sampleRate)
	for i := range audio {
		audio[i] = int16(random.IntN(1001) - 500)
	}

	heard, garbled := s.listen(audio, s.sent)

	s.Empty(heard)
	s.Zero(garbled)
}

func (s *StampSuite) TestAMillisecondFromBeforeTheCounterWrappedResolvesToItsOwnTime() {
	// The low 32 bits of Unix milliseconds wrap every 49.7 days; a stamp sent just before
	// that and heard just after must still be 250 ms old, not 49 days younger.
	wrap := time.UnixMilli((s.sent.UnixMilli() >> 32 << 32) + 1<<32)
	sent := wrap.Add(-100 * time.Millisecond)
	heard := Heard{Sent: Millis(sent), At: wrap.Add(150 * time.Millisecond)}

	s.Equal(250*time.Millisecond, heard.Latency())
}

func (s *StampSuite) TestMuLawKeepsSpeechLevelsWithinItsQuantisingStep() {
	samples := []int16{0, 1, -1, 100, -100, 1000, -1000, 9000, -9000, 32767, -32768}

	decoded := DecodeULaw(EncodeULaw(samples))

	for i, sample := range samples {
		magnitude := int(sample)
		if magnitude < 0 {
			magnitude = -magnitude
		}
		step := max(magnitude/16, 8)
		s.InDelta(sample, decoded[i], float64(step), "sample %d", sample)
	}
}
