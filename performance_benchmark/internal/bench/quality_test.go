package bench

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/audio"
)

func TestToneSuite(t *testing.T) { suite.Run(t, new(ToneSuite)) }

type ToneSuite struct {
	suite.Suite
}

// reply is the mock TTS's tone for that long, between two half seconds of silence, with
// noise at that amplitude added throughout.
func (s *ToneSuite) reply(seconds float64, noise float64) []int16 {
	const amplitude = 0.2 * math.MaxInt16
	random := rand.New(rand.NewPCG(1, 2))
	silence := audio.Rate / 2
	samples := make([]int16, 2*silence+int(seconds*audio.Rate))
	for i := range samples {
		x := random.NormFloat64() * noise
		if i >= silence && i < len(samples)-silence {
			x += amplitude * math.Sin(2*math.Pi*agentToneHz*float64(i)/audio.Rate)
		}
		samples[i] = int16(x)
	}
	return samples
}

func TestRecordingSuite(t *testing.T) { suite.Run(t, new(RecordingSuite)) }

type RecordingSuite struct {
	suite.Suite
}

// play hands the recording the reply a frame a tick, with the frames in skip not arriving.
func (s *RecordingSuite) play(samples []int16, skip map[int]bool) []int16 {
	r := &recording{}
	for frame := 0; frame*audio.FrameSamples < len(samples); frame++ {
		if skip[frame] {
			r.missed++
			continue
		}
		pcm := samples[frame*audio.FrameSamples : (frame+1)*audio.FrameSamples]
		r.add(pcm, audio.FrameEnergy(pcm) > audio.DefaultSpeechThreshold)
	}
	return r.pcm
}

func (s *RecordingSuite) TestASilenceTheAgentDidNotSendIsAPause() {
	// The agent sends the first frame of a pause and then nothing until it speaks again.
	samples := (&ToneSuite{}).reply(1, 0)
	middle := len(samples) / 2 / audio.FrameSamples
	clear(samples[middle*audio.FrameSamples : (middle+10)*audio.FrameSamples])
	skip := map[int]bool{}
	for frame := middle + 1; frame < middle+10; frame++ {
		skip[frame] = true
	}

	quality, ok := measureTone(s.play(samples, skip))

	s.True(ok)
	s.Zero(quality.dropouts)
	s.Zero(quality.glitches)
}

func (s *RecordingSuite) TestASilenceIsAPauseWhenItsFirstFrameArrivesLate() {
	// The jitter buffer only lets the last frame before a gap go once the next one is in.
	samples := (&ToneSuite{}).reply(1, 0)
	frames := len(samples) / audio.FrameSamples
	middle := frames / 2
	clear(samples[middle*audio.FrameSamples : (middle+10)*audio.FrameSamples])
	r := &recording{}
	for frame := range frames {
		pcm := samples[frame*audio.FrameSamples : (frame+1)*audio.FrameSamples]
		switch {
		case frame == middle:
			continue
		case frame > middle && frame < middle+10:
			r.missed++
			continue
		case frame == middle+10:
			r.add(samples[middle*audio.FrameSamples:(middle+1)*audio.FrameSamples], false)
		}
		r.add(pcm, audio.FrameEnergy(pcm) > audio.DefaultSpeechThreshold)
	}

	quality, ok := measureTone(r.pcm)

	s.True(ok)
	s.Zero(quality.dropouts)
}

func (s *RecordingSuite) TestAFrameLostFromSpeechIsStillAGlitch() {
	samples := (&ToneSuite{}).reply(2, 0)

	quality, ok := measureTone(s.play(samples, map[int]bool{len(samples) / 2 / audio.FrameSamples: true}))

	s.True(ok)
	s.Equal(1, quality.glitches)
}

func (s *ToneSuite) TestACleanToneScoresHigh() {
	quality, ok := measureTone(s.reply(2, 0))

	s.True(ok)
	s.Greater(quality.snrDb, 50.0)
	s.Zero(quality.dropouts)
	s.Zero(quality.glitches)
	s.Positive(quality.windows)
}

func (s *ToneSuite) TestASkippedFrameIsAGlitch() {
	samples := s.reply(2, 0)
	middle := len(samples) / 2 / audio.FrameSamples * audio.FrameSamples
	samples = append(samples[:middle], samples[middle+audio.FrameSamples:]...)

	quality, ok := measureTone(samples)

	s.True(ok)
	s.Equal(1, quality.glitches)
	s.Zero(quality.dropouts)
}

func (s *ToneSuite) TestNoiseReadsAsItsLevel() {
	// White noise 30 dB under the tone's power, which is its amplitude over root two.
	noise := 0.2 * math.MaxInt16 / math.Sqrt2 * math.Pow(10, -30.0/20)

	quality, ok := measureTone(s.reply(2, noise))

	s.True(ok)
	s.InDelta(30, quality.snrDb, 1)
}

func (s *ToneSuite) TestMissingFramesAreDropouts() {
	samples := s.reply(2, 0)
	middle := len(samples) / 2 / audio.FrameSamples * audio.FrameSamples
	clear(samples[middle : middle+3*audio.FrameSamples])

	quality, ok := measureTone(samples)

	s.True(ok)
	s.Equal(3, quality.dropouts)
}

func (s *ToneSuite) TestAPauseBetweenSentencesIsNotADropout() {
	samples := s.reply(2, 0)
	middle := len(samples) / 2 / audio.FrameSamples * audio.FrameSamples
	clear(samples[middle : middle+8*audio.FrameSamples])

	quality, ok := measureTone(samples)

	s.True(ok)
	s.Zero(quality.dropouts)
	s.Greater(quality.snrDb, 50.0)
}

func (s *ToneSuite) TestAFadeInIsNotNoise() {
	samples := s.reply(2, 0)
	silence, fade := audio.Rate/2, audio.Rate*16/100
	for i := range fade {
		samples[silence+i] = int16(float64(samples[silence+i]) * float64(i) / float64(fade))
	}

	quality, ok := measureTone(samples)

	s.True(ok)
	s.Greater(quality.snrDb, 35.0)
}

func (s *ToneSuite) TestClicksLowerTheScore() {
	samples := s.reply(2, 0)
	for i := audio.Rate; i < len(samples)-audio.Rate; i += audio.Rate / 20 {
		samples[i] = math.MaxInt16
	}

	quality, ok := measureTone(samples)

	s.True(ok)
	s.Less(quality.snrDb, 40.0)
}

func (s *ToneSuite) TestAReplyTooShortIsNotScored() {
	_, ok := measureTone(s.reply(0.1, 0))

	s.False(ok)
}
