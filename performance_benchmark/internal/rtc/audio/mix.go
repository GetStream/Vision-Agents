package audio

import "math"

func meanSquare(samples []int16) float64 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		v := float64(s)
		sum += v * v
	}
	return sum / float64(len(samples))
}

func clip(v float64) int16 {
	if v > 32767 {
		return 32767
	}
	if v < -32768 {
		return -32768
	}
	return int16(v)
}

const speechRefAmp = 8000.0

// ScaleNoiseForSNR scales noise against a typical speech amplitude. New benchmark
// recordings should use ScaleNoiseForSignalSNR with their actual caller audio.
func ScaleNoiseForSNR(noise []int16, snrDB float64) []int16 {
	return scaleNoise(noise, speechRefAmp*speechRefAmp, snrDB)
}

// ScaleNoiseForSignalSNR scales noise against the measured power of signal.
func ScaleNoiseForSignalSNR(noise, signal []int16, snrDB float64) []int16 {
	return scaleNoise(noise, meanSquare(signal), snrDB)
}

func scaleNoise(noise []int16, signalPower, snrDB float64) []int16 {
	if len(noise) == 0 {
		return nil
	}
	noisePower := meanSquare(noise)
	if noisePower == 0 || signalPower == 0 {
		return append([]int16(nil), noise...)
	}
	target := signalPower / math.Pow(10, snrDB/10)
	scale := math.Sqrt(target / noisePower)
	out := make([]int16, len(noise))
	for i, sample := range noise {
		out[i] = clip(float64(sample) * scale)
	}
	return out
}

// MeasuredSNRDB returns the power ratio of signal to noise in decibels.
func MeasuredSNRDB(signal, noise []int16) float64 {
	signalPower := meanSquare(signal)
	noisePower := meanSquare(noise)
	if signalPower == 0 || noisePower == 0 {
		return 0
	}
	return 10 * math.Log10(signalPower/noisePower)
}

// Add mixes a looping bed into frame starting at offset samples.
func Add(frame, bed []int16, offset int) []int16 {
	if len(bed) == 0 {
		return frame
	}
	for i, s := range frame {
		frame[i] = clip(float64(s) + float64(bed[(offset+i)%len(bed)]))
	}
	return frame
}

// KitchenNoise is bursty clatter.
func KitchenNoise(samples int, seed int64) []int16 {
	rng := newLCG(seed)
	out := make([]int16, samples)
	burst := 0
	for i := range out {
		if burst <= 0 && rng.float64() < 0.02 {
			burst = Rate/100 + int(rng.float64()*Rate/40)
		}
		if burst > 0 {
			out[i] = int16((rng.float64()*2 - 1) * 12000)
			burst--
			continue
		}
		out[i] = int16((rng.float64()*2 - 1) * 800)
	}
	return out
}

// StreetNoise is a low rumble with occasional spikes.
func StreetNoise(samples int, seed int64) []int16 {
	rng := newLCG(seed)
	out := make([]int16, samples)
	var low float64
	for i := range out {
		low = low*0.97 + (rng.float64()*2-1)*0.03
		v := low * 8000
		if rng.float64() < 0.004 {
			v += (rng.float64()*2 - 1) * 10000
		}
		out[i] = clip(v)
	}
	return out
}

// ConversationNoise is overlapping syllabic babble, not a pair of sines.
func ConversationNoise(samples int, seed int64) []int16 {
	rng := newLCG(seed)
	a := voicedTalker(samples, 110+rng.float64()*20, 4.2, rng, 0)
	b := voicedTalker(samples, 175+rng.float64()*25, 5.1, rng, math.Pi/3)
	out := make([]int16, samples)
	for i := range out {
		out[i] = clip(float64(a[i])*0.55 + float64(b[i])*0.55)
	}
	return out
}

// MusicNoise is background music: a four-chord loop with a bass line and a soft beat, the way
// a bar or a radio sounds behind a caller.
func MusicNoise(samples int, seed int64) []int16 {
	rng := newLCG(seed)
	// I-V-vi-IV in C, two seconds a chord.
	chords := [][]float64{{261.6, 329.6, 392.0}, {196.0, 246.9, 293.7}, {220.0, 261.6, 329.6}, {174.6, 220.0, 261.6}}
	offset := rng.float64() * 8
	out := make([]int16, samples)
	for i := range out {
		t := float64(i)/Rate + offset
		chord := chords[int(t/2)%len(chords)]
		v := 0.0
		for _, f := range chord {
			v += math.Sin(2*math.Pi*f*t) + 0.25*math.Sin(2*math.Pi*2*f*t)
		}
		v *= 0.25
		v += 0.6 * math.Sin(2*math.Pi*chord[0]/2*t)
		beat := math.Mod(t, 0.5)
		v += math.Exp(-beat*30) * math.Sin(2*math.Pi*60*beat) * 1.4
		out[i] = clip(v * 4200)
	}
	return out
}

// TVNoise is a television in the room: music under someone talking.
func TVNoise(samples int, seed int64) []int16 {
	rng := newLCG(seed)
	music := MusicNoise(samples, seed+1)
	voice := voicedTalker(samples, 120+rng.float64()*20, 4.4, rng, 0.7)
	out := make([]int16, samples)
	for i := range out {
		out[i] = clip(float64(music[i])*0.45 + float64(voice[i])*0.7)
	}
	return out
}

// Talker is a ~1.2s other-conversation clip for mid-utterance overlap.
func Talker(seed int64) []int16 {
	n := Rate * 6 / 5
	rng := newLCG(seed)
	return voicedTalker(n, 145+rng.float64()*30, 4.6, rng, 0.4)
}

func voicedTalker(samples int, f0, syllHz float64, rng *lcg, phase float64) []int16 {
	out := make([]int16, samples)
	gapUntil := 0
	for i := range out {
		if i >= gapUntil && rng.float64() < 0.0008 {
			gapUntil = i + int(rng.float64()*0.18*Rate)
		}
		if i < gapUntil {
			continue
		}
		t := float64(i) / Rate
		syll := 0.5 + 0.5*math.Sin(2*math.Pi*syllHz*t+phase)
		if syll < 0.22 {
			continue
		}
		buzz := math.Sin(2*math.Pi*f0*t) + 0.35*math.Sin(2*math.Pi*2*f0*t+0.3)
		formant := 0.6 + 0.4*math.Sin(2*math.Pi*(f0*4.5)*t)
		out[i] = clip(buzz * formant * syll * 14000)
	}
	return out
}

// Cough is a short noisy burst (~200 ms).
func Cough(seed int64) []int16 {
	n := Rate / 5
	rng := newLCG(seed)
	out := make([]int16, n)
	for i := range out {
		env := math.Sin(math.Pi * float64(i) / float64(n))
		out[i] = clip((rng.float64()*2 - 1) * 18000 * env)
	}
	return out
}

// Backchannel is a short "mm-hm" like hum (~250 ms).
func Backchannel() []int16 {
	n := Rate / 4
	out := make([]int16, n)
	for i := range out {
		t := float64(i) / Rate
		env := math.Sin(math.Pi * float64(i) / float64(n))
		out[i] = clip(math.Sin(2*math.Pi*140*t) * 9000 * env)
	}
	return out
}

// Tone generates a voiced-like tone used when TTS is unavailable.
func Tone(samples int, hz float64, amp float64) []int16 {
	out := make([]int16, samples)
	for i := range out {
		t := float64(i) / Rate
		out[i] = clip(math.Sin(2*math.Pi*hz*t) * amp)
	}
	return out
}

type lcg struct{ s uint64 }

func newLCG(seed int64) *lcg {
	if seed == 0 {
		seed = 1
	}
	return &lcg{s: uint64(seed)}
}

func (r *lcg) float64() float64 {
	r.s = r.s*6364136223846793005 + 1
	return float64(r.s>>11) / float64(1<<53)
}

// NoiseNamed returns a generated noise buffer.
func NoiseNamed(name string, samples int, seed int64) []int16 {
	switch name {
	case "kitchen":
		return KitchenNoise(samples, seed)
	case "street":
		return StreetNoise(samples, seed)
	case "conversation":
		return ConversationNoise(samples, seed)
	case "cough":
		return Cough(seed)
	case "backchannel":
		return Backchannel()
	case "talker":
		return Talker(seed)
	case "music":
		return MusicNoise(samples, seed)
	case "tv":
		return TVNoise(samples, seed)
	default:
		return Silence(samples)
	}
}
