package agent

import (
	"math"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	// echoWords is the fewest of the agent's own words a transcript has to end on to be
	// taken for its voice coming back down the caller's line.
	echoWords = 3
	// echoMiss is the share of an echo's words the transcriber may have heard wrong.
	echoMiss = 0.25
	// echoTail is how long after the agent's speech was last queued its echo can still be
	// in the caller's audio: the edge's queue, the trip out and back, and the line.
	echoTail = 1500 * time.Millisecond
	// echoRun is the gap in the agent's speech that begins another run of it.
	echoRun = time.Second
)

var echoes = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "agent_chorus_asr_echoes_total",
	Help:      "Transcripts of chorus-asr that ended on the agent's own words heard back on the caller's line, by whether anything of the caller's was left.",
}, []string{"left"})

// withoutEcho is a chorus-asr transcript without the agent's own words heard back at its
// end, and how many words of them it ended on. The model hears the caller's line whole, and
// a phone with no echo cancellation sends the agent's voice back down it, which the model
// writes as the caller's words, or adds to theirs when it comes within their utterance. A
// transcript heard while the agent spoke, or within echoTail of it, that ends on a run of
// at least echoWords of what the agent said, near enough, is taken for that echo: what is
// left before it is the caller's, and an echo alone is nothing heard. Other transcribers,
// and the words of a caller who repeats the agent once it has stopped, are left as they are.
func (a *Agent) withoutEcho(transcript stt.Transcript) (string, int) {
	if transcript.Provider != audiofloor.ProviderName {
		return transcript.Text, 0
	}
	a.mu.Lock()
	from, to := a.spokeFrom, a.lastSpokeAt
	said := a.saying + " " + lastAssistantSaid(a.history)
	a.mu.Unlock()
	if to.IsZero() {
		return transcript.Text, 0
	}
	// The audio the words are of ended as the transcriber began writing them.
	ended := time.Now().Add(-time.Duration(transcript.ProcessingTimeMs * float64(time.Millisecond)))
	began := ended.Add(-time.Duration(transcript.AudioDurationMs * float64(time.Millisecond)))
	if ended.Before(from) || began.After(to.Add(echoTail)) {
		return transcript.Text, 0
	}
	return echoed(transcript.Text, said)
}

// echoed is text without the run of reference's words it ends on, of at least echoWords
// near enough, the longest there is, and how many words that run was. A run opens on two
// words the agent said, so that the caller's last words before an echo stay theirs.
func echoed(text, reference string) (string, int) {
	fields := strings.Fields(text)
	var heard []string
	var field []int
	for i, f := range fields {
		for _, w := range strings.Fields(strings.ToLower(words(f))) {
			heard, field = append(heard, w), append(field, i)
		}
	}
	spoken := strings.Fields(strings.ToLower(words(reference)))
	if len(spoken) < echoWords {
		return text, 0
	}
	said := map[string]bool{}
	for _, w := range spoken {
		said[w] = true
	}
	for at := 0; at+echoWords <= len(heard); at++ {
		if said[heard[at]] && said[heard[at+1]] && within(heard[at:], spoken) {
			return strings.Join(fields[:field[at]], " "), len(heard) - at
		}
	}
	return text, 0
}

// within says run is somewhere in spoken, word for word but for echoMiss of its words:
// the fewest words changed, dropped or added to make it a stretch of spoken.
func within(run, spoken []string) bool {
	previous, current := make([]int, len(spoken)+1), make([]int, len(spoken)+1)
	for k := 1; k <= len(run); k++ {
		current[0] = k
		for j := 1; j <= len(spoken); j++ {
			changed := previous[j-1]
			if run[k-1] != spoken[j-1] {
				changed++
			}
			current[j] = min(changed, previous[j]+1, current[j-1]+1)
		}
		previous, current = current, previous
	}
	fewest := len(run)
	for _, d := range previous {
		fewest = min(fewest, d)
	}
	return float64(fewest) <= math.Round(float64(len(run))*echoMiss)
}
