package sts

import (
	"math"
	"sync"
)

// charsPerToken is the rule of thumb for English text, as llm's estimate uses.
const charsPerToken = 4

// Estimator stands in for the usage of a reply the vendor never reported, which is a reply
// cut off before the event that would have said: the caller hanging up, the connection
// dropping, a newer reply taking its place. The vendor generated it all the same.
//
// It learns from the replies the vendor did report in the session: how many output audio
// tokens a millisecond of streamed audio came to, and what the last reply read, since every
// reply reads the whole conversation again. One instance serves one session.
type Estimator struct {
	mu sync.Mutex
	// fallback is output audio tokens per millisecond of audio until a reply has reported.
	fallback float64
	// audioMs and audioTokens are the streamed audio and the output audio tokens of the
	// replies that reported both.
	audioMs     float64
	audioTokens int64
	// split is whether the vendor reports output by modality, which the estimate follows: an
	// output split is priced in place of the total.
	split bool
	// input is the last reported reply's input.
	input Usage
}

// NewEstimator returns an estimator that, until the session's replies say otherwise, counts
// audioTokensPerSecond output tokens for each second of audio a reply streamed, and splits
// the output by modality where split says the vendor does.
func NewEstimator(audioTokensPerSecond float64, split bool) *Estimator {
	return &Estimator{fallback: audioTokensPerSecond / 1000, split: split}
}

// Reported learns from a reply the vendor counted.
func (e *Estimator) Reported(turn *Turn, usage Usage) {
	if !usage.Counted() || usage.Estimated {
		return
	}
	audioMs, _ := turn.streamed()
	e.mu.Lock()
	defer e.mu.Unlock()
	if usage.OutputAudioTokens > 0 || usage.OutputTextTokens > 0 {
		e.split = true
	}
	if usage.OutputAudioTokens > 0 && audioMs > 0 {
		e.audioMs += audioMs
		e.audioTokens += usage.OutputAudioTokens
	}
	if usage.InputTokens > 0 {
		e.input = Usage{
			InputTokens:            usage.InputTokens,
			CachedInputTokens:      usage.CachedInputTokens,
			InputTextTokens:        usage.InputTextTokens,
			InputAudioTokens:       usage.InputAudioTokens,
			InputImageTokens:       usage.InputImageTokens,
			CachedInputTextTokens:  usage.CachedInputTextTokens,
			CachedInputAudioTokens: usage.CachedInputAudioTokens,
			CachedInputImageTokens: usage.CachedInputImageTokens,
		}
	}
}

// Estimate is what a reply the vendor never counted used: the last reported input, and the
// output its audio and transcript came to. A reply that streamed nothing and follows no
// reported one estimates nothing.
func (e *Estimator) Estimate(turn *Turn) Usage {
	audioMs, said := turn.streamed()
	e.mu.Lock()
	defer e.mu.Unlock()
	rate := e.fallback
	if e.audioMs > 0 {
		rate = float64(e.audioTokens) / e.audioMs
	}
	audioTokens := int64(math.Ceil(audioMs * rate))
	textTokens := int64((said + charsPerToken - 1) / charsPerToken)
	usage := e.input
	usage.OutputTokens = audioTokens + textTokens
	if e.split {
		usage.OutputAudioTokens, usage.OutputTextTokens = audioTokens, textTokens
	}
	usage.Estimated = usage.Counted()
	return usage
}

// streamed is the audio and the bytes of transcript the reply has streamed.
func (t *Turn) streamed() (float64, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.audioMs, t.said
}
