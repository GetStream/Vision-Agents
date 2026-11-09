package agent

import (
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	// echoWords is the fewest of the agent's own words in a row a transcript has to carry
	// to be taken for its voice coming back down the caller's line.
	echoWords = 3
	// echoMiss is the share of an echo's words the transcriber may have heard wrong.
	echoMiss = 0.25
	// echoDelay is the longest the agent's voice takes, once the edge's track has sent it, to
	// come back as echo in the caller's audio as the router hears it: out to the caller, from
	// their speaker into their microphone, and back. Each way is at most a call's one-way
	// delay, for which ITU-T G.114 sets 400 ms as the upper limit for network planning. The
	// edge's own queue is not in it: when the voice goes out is tracked (voiceOut).
	echoDelay = 800 * time.Millisecond
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
// the caller's words, or around theirs when they speak over it. Echo is the agent's voice in
// the caller's microphone, so it can only be in audio that came while that voice was going
// out or within echoDelay of it: in a transcript whose audio did, every run of at least
// echoWords of what the agent said, near enough, is taken for that echo (echoed), what is
// left is the caller's, and an echo alone is nothing heard. The model writes no word's time,
// so what is weighed is the audio the words are of, from the start of its utterance: one
// that began more than echoDelay after the agent's last audible sample went out has none of
// its voice in it, and a caller who repeats the agent then said the words. The same goes for
// another transcriber's words of an utterance chorus-asr's floor ended (stt.Transcript.Floor),
// timed by the floor's clock. Other transcribers are left as they are.
func (a *Agent) withoutEcho(transcript stt.Transcript) (string, int) {
	if transcript.Provider != audiofloor.ProviderName && !transcript.Floor {
		return transcript.Text, 0
	}
	a.mu.Lock()
	from, to := a.voiceOut.from, a.voiceOut.to
	said := a.saying + " " + lastAssistantSaid(a.history)
	a.mu.Unlock()
	if to.IsZero() {
		return transcript.Text, 0
	}
	// The audio the words are of ended as the transcriber began writing them, and began, the
	// transcriber's lead ahead of the voice included, as long before as it lasts.
	ended := time.Now().Add(-time.Duration(transcript.ProcessingTimeMs * float64(time.Millisecond)))
	began := ended.Add(-time.Duration(transcript.AudioDurationMs * float64(time.Millisecond)))
	if ended.Before(from) || began.After(to.Add(echoDelay)) {
		return transcript.Text, 0
	}
	return echoed(transcript.Text, said)
}

// droppingEcho is transcript's words without the agent's own voice heard back at their end
// (withoutEcho), logged and counted as who heard them, and whether any of the caller's are
// left: an echo alone is nothing heard.
func (a *Agent) droppingEcho(transcript stt.Transcript, who string) (string, bool) {
	left, echo := a.withoutEcho(transcript)
	if echo == 0 {
		return transcript.Text, true
	}
	a.logger.Debug(who+" heard the agent's own voice on the caller's line",
		"provider", transcript.Provider, "participant", transcript.Participant.ID, "mode", transcript.Mode,
		"text", transcript.Text, "echo_words", echo, "left", left)
	heard := strings.TrimSpace(left) != ""
	echoes.WithLabelValues(strconv.FormatBool(heard)).Inc()
	return left, heard
}

// echoed is text without the agent's words heard back in it, and how many of them there
// were. The model writes the agent's voice coming back down the line before, after and
// around what the caller says over it, so the echo is every run of at least echoWords of
// reference's words, in order and near enough, wherever it is in the text: what is before,
// between and after the runs is the caller's. Near enough is a run with at most echoMiss of
// its words misheard, one at a time, and no more than one of the agent's words dropped
// between two heard: two words that are not the agent's, in a row or with one of its words
// between them, are the caller's, however short, and end the run, which opens and closes on
// the agent's words.
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
// and a number as its words, which the voice says and the model writes either way: a
// hyphenated one, and one of one or two digits standing alone ("24 degrees" is the voice's
// "twenty four degrees"). A longer number is left as written: the voice reads a year, a price
// or a code each its own way.
func echoTokens(text string) []string {
	var tokens []string
	for _, field := range strings.Fields(strings.ReplaceAll(text, "-", " ")) {
		if said, ok := numberSaid(strings.TrimFunc(field, notLetterOrDigit)); ok {
			tokens = append(tokens, said...)
			continue
		}
		tokens = append(tokens, strings.Fields(strings.ToLower(words(field)))...)
	}
	return tokens
}

func notLetterOrDigit(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }

var (
	numberOnes = [...]string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
		"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	numberTens = [...]string{2: "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
)

// numberSaid is digits, a number of one or two of them, as its words, false for anything else.
func numberSaid(digits string) ([]string, bool) {
	if len(digits) == 0 || len(digits) > 2 {
		return nil, false
	}
	n := 0
	for _, d := range digits {
		if d < '0' || d > '9' {
			return nil, false
		}
		n = n*10 + int(d-'0')
	}
	switch {
	case n < len(numberOnes):
		return []string{numberOnes[n]}, true
	case n%10 == 0:
		return []string{numberTens[n/10]}, true
	}
	return []string{numberTens[n/10], numberOnes[n%10]}, true
}

// agentsRun is the end of the longest run of the agent's words from start in at, aligned's
// word-by-word account of a transcript, or start when none begins there: the run opens and
// closes on the agent's words, has at least echoWords in it and at most echoMiss of them
// misheard, each one on its own among the agent's words (callersNear), and no more than one of
// the agent's words skipped between two heard.
func agentsRun(at []aligned, start int) int {
	if !at[start].same {
		return start
	}
	end, misses, last := start, 0, at[start].to
	for i := start; i < len(at); i++ {
		w := at[i]
		if !w.same {
			if callersNear(at, i) {
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

// callersNear says at[i], a word that is not the agent's, has another that is not within two
// words of it, either side: the caller's, not the agent's words misheard. The model mishears
// the agent's voice a word at a time, among words it hears right. A caller over the agent
// says words of their own together, and a short word of theirs that is also in the agent's
// line lines up between them by chance: "Wait a minute" over "…and warm, with a high of…"
// is the line's "with a high" with two words misheard, and taking it for the echo left
// "minute", a murmur, of the caller's stop. One of the agent's words between two that are
// not its own is no more its echo than none.
func callersNear(at []aligned, i int) bool {
	for _, j := range [...]int{i - 2, i - 1, i + 1, i + 2} {
		if j >= 0 && j < len(at) && !at[j].same {
			return true
		}
	}
	return false
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

// spoke notes a chunk of the agent's speech published to the edge: handed over at queued, the
// publish returning at returned.
func (a *Agent) spoke(queued, returned time.Time, speech audio.PcmData) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.voiceOut.published(queued, returned, speech)
	a.lastSpokeAt = returned
}

// voiceOut is when the agent's voice goes out on the edge's track, which is what can come
// back as echo. Publishing doesn't say: a voice sends speech faster than it is heard, and the
// edge queues it and sends it at the pace it plays, its publish returning while the end is
// still queued. So a chunk goes out from when it was queued or when what was queued ahead of
// it has gone, whichever is later, for as long as it lasts, and is not over before its publish
// returned; an edge with no queue sends it at once, which this takes for later than it was.
// Only the audible samples count: a voice's silences, the padding after an utterance's last
// word included, carry no words. Agent.mu guards it.
type voiceOut struct {
	// until is when the last of the speech published so far has gone out.
	until time.Time
	// from and to are when the first and the last audible sample of the run of speech going
	// out went or go out, and before is the run before it.
	from, to time.Time
	before   [2]time.Time
}

// published notes a chunk handed to the edge at queued, its publish returning at returned.
func (v *voiceOut) published(queued, returned time.Time, speech audio.PcmData) {
	length := pcmLength(speech)
	start := later(later(v.until, queued), returned.Add(-length))
	v.until = start.Add(length)
	first, last, ok := audibleSpan(speech)
	if !ok {
		return
	}
	if began := start.Add(first); v.to.IsZero() || began.Sub(v.to) > echoRun {
		v.before = [2]time.Time{v.from, v.to}
		v.from = began
	}
	v.to = start.Add(last)
}

// dropped notes the speech still queued at the edge thrown away at at, which never goes out.
// A run none of which had gone out was never heard, and the one before it is the last.
func (v *voiceOut) dropped(at time.Time) {
	if v.until.After(at) {
		v.until = at
	}
	if v.from.After(at) {
		v.from, v.to, v.before = v.before[0], v.before[1], [2]time.Time{}
	}
	if v.to.After(at) {
		v.to = at
	}
}

// pcmLength is how long pcm plays for.
func pcmLength(pcm audio.PcmData) time.Duration {
	if pcm.SampleRate <= 0 || pcm.Channels <= 0 {
		return 0
	}
	return time.Duration(len(pcm.Samples)/pcm.Channels) * time.Second / time.Duration(pcm.SampleRate)
}

// audibleSpan is where in pcm its first audible sample begins and its last ends, as times
// from its start, false for a chunk with nothing to hear (audio.AudibleSample).
func audibleSpan(pcm audio.PcmData) (first, last time.Duration, ok bool) {
	if pcm.SampleRate <= 0 || pcm.Channels <= 0 {
		return 0, 0, false
	}
	loud := func(s int16) bool { return s >= audio.AudibleSample || s <= -audio.AudibleSample }
	i := slices.IndexFunc(pcm.Samples, loud)
	if i < 0 {
		return 0, 0, false
	}
	j := len(pcm.Samples) - 1
	for !loud(pcm.Samples[j]) {
		j--
	}
	at := func(frame int) time.Duration {
		return time.Duration(frame) * time.Second / time.Duration(pcm.SampleRate)
	}
	return at(i / pcm.Channels), at(j/pcm.Channels + 1), true
}

// later is whichever of a and b is later.
func later(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}
