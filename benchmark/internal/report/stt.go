package report

import (
	"fmt"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// STTClip is one clip transcribed by one target. The timings are measured against the
// clock the audio was sent on, from the speech in the clip rather than from the file's
// edges, so leading and trailing room tone is not credited to the provider.
type STTClip struct {
	ID         string `json:"id"`
	Target     string `json:"target"`
	Reference  string `json:"reference"`
	Hypothesis string `json:"hypothesis"`
	// Provider and Model are what the router answered with, which a shortcut target leaves
	// open.
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
	// Returned is whether a settled transcript with any words arrived.
	Returned bool `json:"returned"`
	// Timed is whether the clip was streamed, so the timings below were measured.
	Timed bool `json:"timed"`
	// ToFirstWordsMs is from the start of the speech to the first transcript of any kind.
	ToFirstWordsMs int `json:"to_first_words_ms"`
	// ToSettleMs (TTFS) is from the last word spoken to the last settled transcript.
	ToSettleMs int `json:"to_settle_ms"`
	// WhileSpeaking counts the transcripts that arrived before the speech ended.
	WhileSpeaking int             `json:"while_speaking"`
	Raw           score.Alignment `json:"raw"`
	Normalized    score.Alignment `json:"normalized"`
	Error         string          `json:"error,omitempty"`
}

// STTSummary is one target's aggregate over a clip set.
type STTSummary struct {
	Target string `json:"target"`
	Clips  int    `json:"clips"`
	// PooledWER is total word errors over total reference words, so a short clip cannot
	// dominate it; MeanWER is the unweighted mean of per-clip WER. Both normalized, with the
	// raw pooled figure beside them so normalization cannot hide errors.
	PooledWERRaw  float64 `json:"pooled_wer_raw"`
	PooledWER     float64 `json:"pooled_wer"`
	MeanWER       float64 `json:"mean_wer"`
	Substitutions int     `json:"substitutions"`
	Insertions    int     `json:"insertions"`
	Deletions     int     `json:"deletions"`
	PerfectRate   float64 `json:"perfect_rate"`
	ReturnedRate  float64 `json:"returned_rate"`
	// Failed counts the clips that ended in an error, from the router or the provider. Their
	// missing words are in the WER, so a run with any is not one to compare or store.
	Failed int `json:"failed"`
	// Timing percentiles are over the clips that were streamed and returned a transcript.
	TimedClips         int      `json:"timed_clips"`
	ToSettleP50Ms      int      `json:"to_settle_p50_ms"`
	ToSettleP95Ms      int      `json:"to_settle_p95_ms"`
	ToSettleP99Ms      int      `json:"to_settle_p99_ms"`
	ToFirstWordsP50Ms  int      `json:"to_first_words_p50_ms"`
	ToFirstWordsP95Ms  int      `json:"to_first_words_p95_ms"`
	WhileSpeakingMean  float64  `json:"while_speaking_mean"`
	NormalizerVersion  string   `json:"normalizer_version"`
	ProvidersAnswering []string `json:"providers_answering,omitempty"`
}

// SummarizeSTT aggregates one target's clips.
func SummarizeSTT(target string, clips []STTClip) STTSummary {
	out := STTSummary{Target: target, Clips: len(clips), NormalizerVersion: score.NormalizerVersion}
	if len(clips) == 0 {
		return out
	}
	var refWords, errRaw, errNorm, perfect, returned, whileSpeaking int
	var meanWER float64
	var settle, firstWords []int
	answering := map[string]bool{}
	for _, clip := range clips {
		refWords += clip.Raw.Reference
		errRaw += clip.Raw.Errors()
		errNorm += clip.Normalized.Errors()
		meanWER += clip.Normalized.WER
		out.Substitutions += clip.Normalized.Substitutions
		out.Insertions += clip.Normalized.Insertions
		out.Deletions += clip.Normalized.Deletions
		if clip.Error != "" {
			out.Failed++
		}
		if clip.Normalized.WER == 0 && clip.Returned {
			perfect++
		}
		if !clip.Returned {
			continue
		}
		returned++
		if clip.Provider != "" {
			answering[strings.TrimSuffix(clip.Provider+"/"+clip.Model, "/")] = true
		}
		if !clip.Timed {
			continue
		}
		settle = append(settle, clip.ToSettleMs)
		firstWords = append(firstWords, clip.ToFirstWordsMs)
		whileSpeaking += clip.WhileSpeaking
	}
	if refWords > 0 {
		out.PooledWERRaw = float64(errRaw) / float64(refWords)
		out.PooledWER = float64(errNorm) / float64(refWords)
	}
	out.MeanWER = meanWER / float64(len(clips))
	out.PerfectRate = float64(perfect) / float64(len(clips))
	out.ReturnedRate = float64(returned) / float64(len(clips))
	out.TimedClips = len(settle)
	if len(settle) > 0 {
		sort.Ints(settle)
		sort.Ints(firstWords)
		out.ToSettleP50Ms = score.Percentile(settle, 50)
		out.ToSettleP95Ms = score.Percentile(settle, 95)
		out.ToSettleP99Ms = score.Percentile(settle, 99)
		out.ToFirstWordsP50Ms = score.Percentile(firstWords, 50)
		out.ToFirstWordsP95Ms = score.Percentile(firstWords, 95)
		out.WhileSpeakingMean = float64(whileSpeaking) / float64(len(settle))
	}
	for name := range answering {
		out.ProvidersAnswering = append(out.ProvidersAnswering, name)
	}
	sort.Strings(out.ProvidersAnswering)
	return out
}

// STTMarkdown renders an STT run: one row per target.
func STTMarkdown(sum Summary) string {
	var b strings.Builder
	b.WriteString("# Voicebench STT\n\n")
	fmt.Fprintf(&b, "Normalizer `%s`. WER is normalized unless marked raw. TTFS is from the last word spoken to the settled transcript; first words is from the first word spoken to the first transcript of any kind. Timings are over the clips streamed that returned a transcript.\n\n", score.NormalizerVersion)
	b.WriteString("| Target | Clips | Pooled WER | Raw | Mean WER | S / I / D | Perfect | Returned | Failed | TTFS P50 | P95 | P99 | First words P50 | Interims while speaking |\n")
	b.WriteString("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, s := range sum.STT {
		timing := "— | — | — | —"
		interims := "—"
		if s.TimedClips > 0 {
			timing = fmt.Sprintf("%d ms | %d ms | %d ms | %d ms", s.ToSettleP50Ms, s.ToSettleP95Ms, s.ToSettleP99Ms, s.ToFirstWordsP50Ms)
			interims = fmt.Sprintf("%.1f", s.WhileSpeakingMean)
		}
		fmt.Fprintf(&b, "| %s | %d | %.3f | %.3f | %.3f | %d / %d / %d | %.0f%% | %.0f%% | %d | %s | %s |\n",
			s.Target, s.Clips, s.PooledWER, s.PooledWERRaw, s.MeanWER, s.Substitutions, s.Insertions, s.Deletions,
			100*s.PerfectRate, 100*s.ReturnedRate, s.Failed, timing, interims)
	}
	return b.String()
}
