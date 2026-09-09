package audio

import "math"

// Health is cheap clip-level diagnostics for synthesized or recorded speech.
type Health struct {
	DurationMS    int     `json:"duration_ms"`
	ClipFraction  float64 `json:"clip_fraction"`
	RMS           float64 `json:"rms"`
	LeadSilenceMS int     `json:"lead_silence_ms"`
	TailSilenceMS int     `json:"tail_silence_ms"`
	SilenceRatio  float64 `json:"silence_ratio"`
	Peak          int     `json:"peak"`
}

// MeasureHealth reports clipping, loudness, and silence around speech.
func MeasureHealth(samples []int16, rate int) Health {
	out := Health{DurationMS: durationMS(len(samples), rate)}
	if len(samples) == 0 || rate <= 0 {
		return out
	}
	var sumSquares float64
	clipped := 0
	peak := 0
	for _, s := range samples {
		v := int(s)
		if v < 0 {
			v = -v
		}
		if v > peak {
			peak = v
		}
		if v >= 32767 {
			clipped++
		}
		f := float64(s) / 32768.0
		sumSquares += f * f
	}
	out.Peak = peak
	out.ClipFraction = float64(clipped) / float64(len(samples))
	out.RMS = math.Sqrt(sumSquares / float64(len(samples)))
	spans := DetectSpeech(samples, rate, DefaultSpeechThreshold, DefaultHangoverMs)
	if len(spans) == 0 {
		out.LeadSilenceMS = out.DurationMS
		out.TailSilenceMS = out.DurationMS
		out.SilenceRatio = 1
		return out
	}
	out.LeadSilenceMS = spans[0].StartMs
	last := spans[len(spans)-1]
	if last.EndMs < out.DurationMS {
		out.TailSilenceMS = out.DurationMS - last.EndMs
	}
	speechMS := 0
	for _, span := range spans {
		if span.EndMs > span.StartMs {
			speechMS += span.EndMs - span.StartMs
		}
	}
	if out.DurationMS > 0 {
		silent := out.DurationMS - speechMS
		if silent < 0 {
			silent = 0
		}
		out.SilenceRatio = float64(silent) / float64(out.DurationMS)
	}
	return out
}

func durationMS(n, rate int) int {
	if rate <= 0 {
		return 0
	}
	return n * 1000 / rate
}
