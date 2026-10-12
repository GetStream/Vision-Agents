package audio

import (
	"math"
	"testing"

	"github.com/stretchr/testify/suite"
)

type LevellerSuite struct {
	suite.Suite
}

func TestLevellerSuite(t *testing.T) {
	suite.Run(t, new(LevellerSuite))
}

const levelRate = 24000

// sine is ms of a 220 Hz sine at amplitude, as a fraction of full scale.
func sine(ms int, amplitude float64) []int16 {
	samples := make([]int16, levelRate*ms/1000)
	for i := range samples {
		samples[i] = int16(amplitude * 32767 * math.Sin(2*math.Pi*220*float64(i)/levelRate))
	}
	return samples
}

// stream levels samples in 20 ms chunks, the way a voice delivers them.
func stream(leveller *Leveller, samples []int16) []int16 {
	var out []int16
	chunk := levelRate / 50
	for start := 0; start < len(samples); start += chunk {
		end := min(start+chunk, len(samples))
		out = append(out, leveller.Level(PcmData{Samples: samples[start:end], SampleRate: levelRate, Channels: 1}).Samples...)
	}
	return out
}

// rmsDB is the RMS level of samples in dBFS.
func rmsDB(samples []int16) float64 {
	var sum float64
	for _, sample := range samples {
		value := float64(sample) / 32768
		sum += value * value
	}
	return 10 * math.Log10(sum/float64(len(samples))+1e-12)
}

func (s *LevellerSuite) TestAQuietPhraseIsRaisedTowardsTheRestOfTheSpeech() {
	// A sentence at about -23 dBFS, then a phrase 20 dB below it, as ElevenLabs read "I have
	// your reservation for" in Voicebench.
	loud := sine(1000, 0.1)
	quiet := sine(1500, 0.01)

	out := stream(NewLeveller(), append(append([]int16{}, loud...), quiet...))

	before := rmsDB(quiet[len(quiet)/2:])
	after := rmsDB(out[len(loud)+len(quiet)/2:])
	s.Greater(after-before, 15.0, "the second half of the quiet phrase is raised most of the way")
	s.InDelta(rmsDB(loud), rmsDB(out[:len(loud)]), 1.0, "speech already at its level is left alone")
}

func (s *LevellerSuite) TestLoudSpeechIsNeverTurnedDown() {
	loud := sine(1000, 0.5)

	out := stream(NewLeveller(), loud)

	s.Equal(loud, out)
}

func (s *LevellerSuite) TestSilenceStaysSilent() {
	// A pause between quiet phrases must not bring up whatever noise is in it.
	pause := make([]int16, levelRate)
	for i := range pause {
		pause[i] = int16(i%3 - 1)
	}

	out := stream(NewLeveller(), append(sine(500, 0.01), pause...))

	// The first 10 ms of the pause ramp the gain back down from the phrase before it.
	settled := levelRate/2 + levelRate/100
	s.Equal(pause[levelRate/100:], out[settled:])
}

func (s *LevellerSuite) TestARaisedPhraseNeverClips() {
	// A quiet phrase raises the gain; a loud one straight after must not clip on the way down.
	out := stream(NewLeveller(), append(sine(1500, 0.01), sine(500, 0.95)...))

	for _, sample := range out {
		s.Less(math.Abs(float64(sample))/32768, 0.95, "no sample is raised past the limit")
	}
}

func (s *LevellerSuite) TestTheGainNeverSteps() {
	// The gain is ramped within a block, so a quiet phrase being raised has no clicks: no
	// sample moves further from its neighbour than the louder of the two tones allows.
	out := stream(NewLeveller(), append(sine(500, 0.1), sine(1500, 0.005)...))

	slope := 2 * math.Pi * 220 / levelRate * 0.1 * 32767
	for i := 1; i < len(out); i++ {
		s.LessOrEqual(math.Abs(float64(out[i])-float64(out[i-1])), slope*1.1)
	}
}

func (s *LevellerSuite) TestTheChunkPassedInIsLeftUntouched() {
	quiet := sine(1000, 0.01)
	original := append([]int16{}, quiet...)

	NewLeveller().Level(PcmData{Samples: quiet, SampleRate: levelRate, Channels: 1})

	s.Equal(original, quiet)
}
