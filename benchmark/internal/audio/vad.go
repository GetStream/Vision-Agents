package audio

// FrameEnergy is mean-square energy of a 20 ms frame, scaled to 0..1.
func FrameEnergy(frame []int16) float64 {
	if len(frame) == 0 {
		return 0
	}
	var sum float64
	for _, s := range frame {
		v := float64(s) / 32768.0
		sum += v * v
	}
	return sum / float64(len(frame))
}

// Span is a speech region in milliseconds from the start of a recording.
type Span struct {
	StartMs int
	EndMs   int
}

// DetectSpeech finds speech spans using energy with hangover.
// threshold is mean-square energy; hangoverMs keeps a span open through brief dips.
func DetectSpeech(samples []int16, rate int, threshold float64, hangoverMs int) []Span {
	if rate <= 0 || len(samples) == 0 {
		return nil
	}
	frame := rate / 50
	if frame < 1 {
		frame = 1
	}
	hangoverFrames := (hangoverMs * rate / 1000) / frame
	if hangoverFrames < 1 {
		hangoverFrames = 1
	}
	var spans []Span
	inSpeech := false
	start := 0
	silent := 0
	for i := 0; i+frame <= len(samples); i += frame {
		e := FrameEnergy(samples[i : i+frame])
		if e >= threshold {
			if !inSpeech {
				inSpeech = true
				start = i
			}
			silent = 0
			continue
		}
		if !inSpeech {
			continue
		}
		silent++
		if silent >= hangoverFrames {
			end := i - (silent-1)*frame
			if end <= start {
				end = i
			}
			spans = append(spans, Span{
				StartMs: start * 1000 / rate,
				EndMs:   end * 1000 / rate,
			})
			inSpeech = false
			silent = 0
		}
	}
	if inSpeech {
		spans = append(spans, Span{
			StartMs: start * 1000 / rate,
			EndMs:   len(samples) * 1000 / rate,
		})
	}
	return spans
}

// DefaultSpeechThreshold is a conservative energy gate for 8 kHz mu-law recordings.
const DefaultSpeechThreshold = 0.0015

// DefaultHangoverMs covers a short pause without splitting a turn.
const DefaultHangoverMs = 250
