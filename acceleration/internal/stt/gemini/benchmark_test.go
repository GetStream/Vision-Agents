//go:build integration

package gemini

import (
	"os"
	"slices"
	"time"
)

// benchmarkEnvVar gates the benchmark, which is a measurement rather than a test: it has
// nothing to fail on, and it spends a couple of minutes and ten sessions on the API.
const benchmarkEnvVar = "STT_BENCHMARK"

// benchmarkRuns is how many turns each mode is timed over. The spread between runs is
// wider than the difference being looked for, so the median of a handful is the smallest
// honest answer.
const benchmarkRuns = 5

// TestModeLatencyBenchmark times VERBATIM against SMART and prints what each costs, which
// is what BENCHMARK.md in this directory records.
func (s *GeminiIntegrationSuite) TestModeLatencyBenchmark() {
	if os.Getenv(benchmarkEnvVar) == "" {
		s.T().Skipf("set %s=1 to time the transcription modes", benchmarkEnvVar)
	}

	for _, mode := range []TranscriptionMode{ModeVerbatim, ModeSmart} {
		var toFirstWords, toSettle []time.Duration
		var scores []float64

		for range benchmarkRuns {
			provider := s.started(Options{Mode: mode})
			timing := s.MeasureOn(provider)
			s.Hangup(provider)

			toFirstWords = append(toFirstWords, timing.ToFirstWords)
			toSettle = append(toSettle, timing.ToSettle)
			scores = append(scores, s.Accuracy(timing.Text))
		}

		s.T().Logf("%s over %d runs: first words after %.0fms, settled %.0fms after the "+
			"last word, %.0f%% of the words",
			mode, benchmarkRuns,
			median(toFirstWords).Seconds()*1000,
			median(toSettle).Seconds()*1000,
			median(scores)*100)
	}
}

func median[T int64 | float64 | time.Duration](values []T) T {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}
