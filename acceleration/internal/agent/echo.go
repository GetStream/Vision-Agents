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
	// echoWords is the fewest of the agent's own words in a row a transcript has to carry
	// to be taken for its voice coming back down the caller's line.
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
	Help:      "Transcripts of chorus-asr with the agent's own words heard back on the caller's line in them, by whether anything of the caller's was left.",
}, []string{"left"})

// withoutEcho is a chorus-asr transcript without the agent's own words heard back in it, and
// how many words of them there were. The model hears the caller's line whole, and a phone
// with no echo cancellation sends the agent's voice back down it, which the model writes as
// the caller's words, or around theirs when they speak over it. In a transcript heard while
// the agent spoke, or within echoTail of it, every run of at least echoWords of what the
// agent said, near enough, is taken for that echo (echoed): what is left is the caller's,
// and an echo alone is nothing heard. Other transcribers, and the words of a caller who
// repeats the agent once it has stopped, are left as they are.
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

// echoed is text without the agent's words heard back in it, and how many of them there
// were. The model writes the agent's voice coming back down the line before, after and
// around what the caller says over it, so the echo is every run of at least echoWords of
// reference's words, in order and near enough, wherever it is in the text: what is before,
// between and after the runs is the caller's. Near enough is a run with at most echoMiss of
// its words misheard, one at a time, and no more than one of the agent's words dropped
// between two heard: two words in a row that are not the agent's are the caller's, however
// short, and end the run, which opens and closes on the agent's words.
func echoed(text, reference string) (string, int) {
	fields := strings.Fields(text)
	var heard []string
	var field []int
	for i, f := range fields {
		for _, w := range echoTokens(f) {
			heard, field = append(heard, w), append(field, i)
		}
	}
	spoken := echoTokens(reference)
	if len(spoken) < echoWords || len(heard) < echoWords {
		return text, 0
	}
	at := align(heard, spoken)
	drop := make([]bool, len(fields))
	echo := 0
	for start := 0; start < len(heard); {
		end := agentsRun(at, start)
		if end == start {
			start++
			continue
		}
		for i := start; i < end; i++ {
			drop[field[i]] = true
		}
		echo, start = echo+end-start, end
	}
	if echo == 0 {
		return text, 0
	}
	var left []string
	for i, f := range fields {
		if !drop[i] {
			left = append(left, f)
		}
	}
	return strings.Join(left, " "), echo
}

// echoTokens is text as the words an echo is compared by: lowercase, without punctuation,
// and a hyphenated number as its words, which the voice says and the model writes either way.
func echoTokens(text string) []string {
	return strings.Fields(strings.ToLower(words(strings.ReplaceAll(text, "-", " "))))
}

// agentsRun is the end of the longest run of the agent's words from start in at, aligned's
// word-by-word account of a transcript, or start when none begins there: the run opens and
// closes on the agent's words, has at least echoWords in it and at most echoMiss of them
// misheard, no two misheard in a row, and no more than one of the agent's words skipped
// between two heard.
func agentsRun(at []aligned, start int) int {
	if !at[start].same {
		return start
	}
	end, misses, last := start, 0, at[start].to
	for i := start; i < len(at); i++ {
		w := at[i]
		if !w.same {
			if !at[i-1].same {
				break
			}
			misses++
			if w.to >= 0 {
				// Misheard, the word still stands where the agent's did.
				last = w.to
			}
			continue
		}
		if w.to-last > 2 {
			break
		}
		last = w.to
		if float64(misses) <= math.Round(float64(i+1-start)*echoMiss) {
			end = i + 1
		}
	}
	if end-start < echoWords {
		return start
	}
	return end
}

// aligned is one heard word's place in the agent's line: the index of the word of the line
// it stands for, -1 for one said between the agent's, and whether it is that word.
type aligned struct {
	to   int
	same bool
}

// align is each word of heard in the agent's line, under the alignment of all of heard to a
// stretch of spoken that changes, drops or adds the fewest words.
func align(heard, spoken []string) []aligned {
	h, s := len(heard), len(spoken)
	cost := make([]int, (h+1)*(s+1))
	cell := func(i, j int) int { return i*(s+1) + j }
	// The stretch may begin anywhere in spoken: the first row costs nothing.
	for i := 1; i <= h; i++ {
		cost[cell(i, 0)] = i
		for j := 1; j <= s; j++ {
			changed := cost[cell(i-1, j-1)]
			if heard[i-1] != spoken[j-1] {
				changed++
			}
			cost[cell(i, j)] = min(changed, cost[cell(i-1, j)]+1, cost[cell(i, j-1)]+1)
		}
	}
	// And end anywhere: the cheapest cell of the last row, the furthest on among equals so
	// that the agent's last words are read as its words rather than as the caller's, read
	// back to the first.
	j := 0
	for k := 1; k <= s; k++ {
		if cost[cell(h, k)] <= cost[cell(h, j)] {
			j = k
		}
	}
	at := make([]aligned, h)
	for i := h; i > 0; {
		c := cost[cell(i, j)]
		switch {
		case j > 0 && heard[i-1] == spoken[j-1] && c == cost[cell(i-1, j-1)]:
			at[i-1], i, j = aligned{to: j - 1, same: true}, i-1, j-1
		case j > 0 && c == cost[cell(i-1, j-1)]+1:
			at[i-1], i, j = aligned{to: j - 1}, i-1, j-1
		case c == cost[cell(i-1, j)]+1:
			at[i-1], i = aligned{to: -1}, i-1
		default:
			j--
		}
	}
	return at
}
