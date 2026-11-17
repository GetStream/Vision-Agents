package bench

import (
	"math"
	"slices"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/audio"
)

const (
	// agentToneHz is the pitch the router's mock TTS speaks in
	// (acceleration/internal/tts/mock), so whatever else is in a reply came from the
	// path between it and the caller.
	agentToneHz = 220
	// toneWindow is 100 ms at 16 kHz: a whole number of the tone's cycles, which keeps
	// the fit in measureTone exact.
	toneWindow      = audio.Rate / 10
	framesPerWindow = toneWindow / audio.FrameSamples
	// edgeFrames are left unscored at each end of a stretch of speech, where Opus fades
	// the tone in and out.
	edgeFrames = 2
	// pauseFrames is the shortest quiet the mock TTS leaves between sentences. A quiet
	// stretch shorter than this inside a reply is audio that went missing.
	pauseFrames = 5
	// dropoutLevel is how far under the reply's median level a frame has to fall to
	// count as quiet.
	dropoutLevel = 0.25
	// glitchDb is the score under which a window was not the tone. Clean Opus scores
	// about 34 dB, and one 20 ms frame skipped inside a window about 4.
	glitchDb = 20
)

// toneQuality is how close one reply came to the tone the mock TTS sent.
type toneQuality struct {
	// snrDb is the tone's power over everything else while the agent spoke, in
	// decibels. Clicks, distortion and concealed packets lower it.
	snrDb float64
	// windows is how many 100 ms windows of speech were scored, and glitches how many
	// of them were not the tone.
	windows  int
	glitches int
	// dropouts counts 20 ms frames that went quiet in the middle of speech.
	dropouts int
}

// measureTone scores one reply recorded at 16 kHz, and reports false for one with too
// little speech to score.
func measureTone(samples []int16) (toneQuality, bool) {
	frames := len(samples) / audio.FrameSamples
	levels := make([]float64, frames)
	var loud []float64
	for i := range frames {
		levels[i] = audio.FrameEnergy(samples[i*audio.FrameSamples : (i+1)*audio.FrameSamples])
		if levels[i] > audio.DefaultSpeechThreshold {
			loud = append(loud, levels[i])
		}
	}
	if len(loud) == 0 {
		return toneQuality{}, false
	}
	slices.Sort(loud)
	quiet := dropoutLevel * loud[len(loud)/2]

	quality := toneQuality{}
	var tone, rest float64
	for i := 0; i < frames; {
		speaking := levels[i] >= quiet
		end := i
		for end < frames && (levels[end] >= quiet) == speaking {
			end++
		}
		switch {
		case speaking:
			for start := (i + edgeFrames) * audio.FrameSamples; start+toneWindow <= (end-edgeFrames)*audio.FrameSamples; start += toneWindow {
				fitted, power := fitTone(samples[start : start+toneWindow])
				tone += fitted
				rest += max(power-fitted, 1)
				quality.windows++
				if 10*math.Log10(fitted/max(power-fitted, 1)) < glitchDb {
					quality.glitches++
				}
			}
		case i > 0 && end < frames && end-i < pauseFrames:
			quality.dropouts += end - i
		}
		i = end
	}
	if quality.windows == 0 {
		return toneQuality{}, false
	}
	quality.snrDb = 10 * math.Log10(tone/rest)
	return quality, true
}

// fitTone fits the tone to one window, letting its level change linearly across it so a
// fade in or out is not counted as noise, and returns the fitted energy and the window's.
// Over a whole number of cycles the four shapes are orthogonal, so each is a projection.
func fitTone(window []int16) (fitted, power float64) {
	step := 2 * math.Pi * agentToneHz / audio.Rate
	middle := float64(len(window)-1) / 2
	var sin, cos, rampSin, rampCos, ramp float64
	for n, sample := range window {
		x, t := float64(sample), float64(n)-middle
		s, c := math.Sin(float64(n)*step), math.Cos(float64(n)*step)
		sin += x * s
		cos += x * c
		rampSin += x * t * s
		rampCos += x * t * c
		ramp += t * t
		power += x * x
	}
	return 2*(sin*sin+cos*cos)/float64(len(window)) + 2*(rampSin*rampSin+rampCos*rampCos)/ramp, power
}
