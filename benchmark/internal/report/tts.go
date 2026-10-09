package report

import (
	"fmt"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// Health grades for one clip, and for a target's share of healthy clips.
const (
	HealthGood = "good"
	HealthWarn = "warn"
	HealthFail = "fail"
)

// The clip-health and grade thresholds. They are provisional: borrowed in shape from
// Inworld's open-tts-eval and meant to be recalibrated on a known-good baseline run, so a
// change to any of them is a methodology change.
const (
	// healthyGood and healthyWarn are the shares of healthy clips a target needs for a good
	// or a warn grade.
	healthyGood = 0.99
	healthyWarn = 0.95
	// leadWarnMs and leadFailMs bound the silence before the first word, which a caller
	// hears as latency on top of TTFB.
	leadWarnMs = 300
	leadFailMs = 1000
	// tailWarnMs is trailing silence long enough to delay the turn after it.
	tailWarnMs = 1000
	// clipFailFraction is the share of full-scale samples past which clipping is audible.
	clipFailFraction = 0.001
	// quietPeak is a peak below which the clip is too quiet to be heard over a call.
	quietPeak = 1000
)

// TTSClip is one corpus line spoken by one target. Latencies are from the moment the text
// was sent, on the benchmark's clock.
type TTSClip struct {
	ID     string `json:"id"`
	Target string `json:"target"`
	Text   string `json:"text"`
	// Returned is whether any audio came back.
	Returned bool `json:"returned"`
	// TTFBMs is from sending the text to the first audio; SynthesisMs to the end of the
	// utterance. RTF is synthesis time over audio duration: below 1 is faster than spoken.
	TTFBMs      int     `json:"ttfb_ms"`
	SynthesisMs int     `json:"synthesis_ms"`
	AudioMs     int     `json:"audio_ms"`
	RTF         float64 `json:"rtf"`
	// Heard is what the scoring ASR transcribed from the audio. Round-trip WER compares it
	// with the text: ASR errors inflate it, so it is a check on intelligibility, not a MOS.
	Heard      string          `json:"heard"`
	Raw        score.Alignment `json:"raw"`
	Normalized score.Alignment `json:"normalized"`
	Health     audio.Health    `json:"health"`
	Grade      string          `json:"grade"`
	Issues     []string        `json:"issues,omitempty"`
	Error      string          `json:"error,omitempty"`
}

// GradeHealth grades one clip's audio and says what was wrong with it.
func GradeHealth(h audio.Health) (string, []string) {
	var fails, warns []string
	switch {
	case h.DurationMS == 0 || h.SilenceRatio >= 1:
		fails = append(fails, "no speech")
	case h.LeadSilenceMS > leadFailMs:
		fails = append(fails, fmt.Sprintf("%d ms of silence before speech", h.LeadSilenceMS))
	case h.LeadSilenceMS > leadWarnMs:
		warns = append(warns, fmt.Sprintf("%d ms of silence before speech", h.LeadSilenceMS))
	}
	if h.ClipFraction > clipFailFraction {
		fails = append(fails, fmt.Sprintf("%.2f%% of samples clipped", 100*h.ClipFraction))
	} else if h.ClipFraction > 0 {
		warns = append(warns, "some samples clipped")
	}
	if h.TailSilenceMS > tailWarnMs {
		warns = append(warns, fmt.Sprintf("%d ms of trailing silence", h.TailSilenceMS))
	}
	if h.DurationMS > 0 && h.Peak < quietPeak {
		warns = append(warns, fmt.Sprintf("peak %d is too quiet", h.Peak))
	}
	switch {
	case len(fails) > 0:
		return HealthFail, append(fails, warns...)
	case len(warns) > 0:
		return HealthWarn, warns
	default:
		return HealthGood, nil
	}
}

// TTSSummary is one target's aggregate over the corpus.
type TTSSummary struct {
	Target string `json:"target"`
	Clips  int    `json:"clips"`
	// Failed counts the clips that ended in an error, from the router, the provider or the
	// scoring ASR, so a run with any is not one to compare or store.
	Failed       int     `json:"failed"`
	ReturnedRate float64 `json:"returned_rate"`
	// Latency percentiles are over the clips that returned audio.
	TimedClips     int     `json:"timed_clips"`
	TTFBP50Ms      int     `json:"ttfb_p50_ms"`
	TTFBP95Ms      int     `json:"ttfb_p95_ms"`
	TTFBP99Ms      int     `json:"ttfb_p99_ms"`
	SynthesisP50Ms int     `json:"synthesis_p50_ms"`
	RTFP50         float64 `json:"rtf_p50"`
	RTFP95         float64 `json:"rtf_p95"`
	// Round-trip WER is over the clips the scoring ASR transcribed.
	PooledWERRaw float64 `json:"pooled_wer_raw"`
	PooledWER    float64 `json:"pooled_wer"`
	MeanWER      float64 `json:"mean_wer"`
	// The health grid: clips per grade, the share that were good, and the target's grade.
	Good              int     `json:"health_good"`
	Warn              int     `json:"health_warn"`
	Fail              int     `json:"health_fail"`
	HealthyRate       float64 `json:"healthy_rate"`
	Grade             string  `json:"grade"`
	ScoringASR        string  `json:"scoring_asr"`
	NormalizerVersion string  `json:"normalizer_version"`
}

// SummarizeTTS aggregates one target's clips.
func SummarizeTTS(target string, clips []TTSClip) TTSSummary {
	out := TTSSummary{Target: target, Clips: len(clips), ScoringASR: score.ScoringASR, NormalizerVersion: score.NormalizerVersion}
	if len(clips) == 0 {
		return out
	}
	var ttfb, synthesis, rtf []int
	var refWords, errRaw, errNorm, scored, returned int
	var meanWER float64
	for _, clip := range clips {
		if clip.Error != "" {
			out.Failed++
		}
		switch clip.Grade {
		case HealthGood:
			out.Good++
		case HealthWarn:
			out.Warn++
		default:
			out.Fail++
		}
		if !clip.Returned {
			continue
		}
		returned++
		ttfb = append(ttfb, clip.TTFBMs)
		// An utterance that never finished has no synthesis time, rather than one of zero.
		if clip.SynthesisMs > 0 {
			synthesis = append(synthesis, clip.SynthesisMs)
			rtf = append(rtf, int(clip.RTF*1000))
		}
		if clip.Raw.Reference == 0 {
			continue
		}
		scored++
		refWords += clip.Raw.Reference
		errRaw += clip.Raw.Errors()
		errNorm += clip.Normalized.Errors()
		meanWER += clip.Normalized.WER
	}
	out.ReturnedRate = float64(returned) / float64(len(clips))
	out.HealthyRate = float64(out.Good) / float64(len(clips))
	switch {
	case out.HealthyRate >= healthyGood:
		out.Grade = HealthGood
	case out.HealthyRate >= healthyWarn:
		out.Grade = HealthWarn
	default:
		out.Grade = HealthFail
	}
	if refWords > 0 {
		out.PooledWERRaw = float64(errRaw) / float64(refWords)
		out.PooledWER = float64(errNorm) / float64(refWords)
		out.MeanWER = meanWER / float64(scored)
	}
	out.TimedClips = len(ttfb)
	if len(ttfb) > 0 {
		sort.Ints(ttfb)
		out.TTFBP50Ms = score.Percentile(ttfb, 50)
		out.TTFBP95Ms = score.Percentile(ttfb, 95)
		out.TTFBP99Ms = score.Percentile(ttfb, 99)
	}
	if len(synthesis) > 0 {
		sort.Ints(synthesis)
		sort.Ints(rtf)
		out.SynthesisP50Ms = score.Percentile(synthesis, 50)
		out.RTFP50 = float64(score.Percentile(rtf, 50)) / 1000
		out.RTFP95 = float64(score.Percentile(rtf, 95)) / 1000
	}
	return out
}

// TTSMarkdown renders a TTS run: one row per target.
func TTSMarkdown(sum Summary) string {
	var b strings.Builder
	b.WriteString("# Voicebench TTS\n\n")
	fmt.Fprintf(&b, "TTFB is from sending the text to the first audio; RTF is synthesis time over audio duration. Round-trip WER transcribes the audio with `%s` and normalizer `%s`: ASR errors inflate it, so it checks intelligibility rather than naturalness. Health grades each clip on silence, clipping and level; the grade is the share of good clips (good at %.0f%%, warn at %.0f%%), with provisional thresholds.\n\n",
		score.ScoringASR, score.NormalizerVersion, 100*healthyGood, 100*healthyWarn)
	b.WriteString("| Target | Clips | Failed | TTFB P50 | P95 | P99 | Synthesis P50 | RTF P50 | P95 | Round-trip WER | Raw | Good / Warn / Fail | Grade |\n")
	b.WriteString("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
	for _, s := range sum.TTS {
		latency := "— | — | — | — | — | —"
		if s.TimedClips > 0 {
			latency = fmt.Sprintf("%d ms | %d ms | %d ms | %d ms | %.2f | %.2f",
				s.TTFBP50Ms, s.TTFBP95Ms, s.TTFBP99Ms, s.SynthesisP50Ms, s.RTFP50, s.RTFP95)
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %s | %.3f | %.3f | %d / %d / %d | %s |\n",
			s.Target, s.Clips, s.Failed, latency, s.PooledWER, s.PooledWERRaw, s.Good, s.Warn, s.Fail, s.Grade)
	}
	return b.String()
}

// TTSSlackText is a TTS run as a chat message: one line per target.
func TTSSlackText(title string, sum Summary) string {
	var b strings.Builder
	lines := 0
	if len(sum.TTS) > 0 {
		lines = sum.TTS[0].Clips
	}
	fmt.Fprintf(&b, "*%s*\n%d lines · round trip through %s\n", title, lines, score.ScoringASR)
	for _, s := range sum.TTS {
		fmt.Fprintf(&b, "• %s: TTFB P50 %s, P95 %s · RTF %.2f · round-trip WER %.1f%% · health %s (%d/%d/%d)",
			s.Target, seconds(float64(s.TTFBP50Ms)), seconds(float64(s.TTFBP95Ms)), s.RTFP50, 100*s.PooledWER, s.Grade, s.Good, s.Warn, s.Fail)
		if s.Failed > 0 {
			fmt.Fprintf(&b, " · %d failed", s.Failed)
		}
		b.WriteString("\n")
	}
	return b.String()
}
