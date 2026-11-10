package audiofloor

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// interimEvery is the speech after which the words so far are written while the caller
// talks on, as an interim (stt.Transcript.Interim), when no pause has had them written
// meanwhile. An interim sends the whole utterance so far, which a replacement must, and the
// words have no 16 s limit; so that a caller who talks on costs a transcription lane no more
// as they go, the audio an interim sends a second of speech is bounded by Window, what a
// decision hears: past Window of utterance one is asked less often as it grows, every 2 s at
// 32 s and every 3 s at 48 s. On the TPU the words of 16 s take 50 to 250 ms, so a caller
// talking on costs at most a quarter of one lane, whatever the length of their turn.
const interimEvery = time.Second

var interims = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_interims_total",
	Help: "Words asked for while the caller talked on, by what became of them: written, late (newer words or the utterance's final went out first), " +
		"unstable (they did not carry on the words shown), empty, foreign (in a language other than the utterance's), refused (every lane busy: 503, 429 or busy) or failed. None is asked for again.",
}, []string{"outcome"})

// sentWords is the last words of an utterance to have gone out, which later words of it are
// held to: none after its final, which has the last word, and none of less audio than words
// already out, as an interim answered late would be, so that what a watcher sees never goes
// back. Every transcript the session sends goes through it (send), so a final and an interim
// answered as it goes out can't cross. It also keeps the language an utterance is spoken in,
// the one the server heard its first words in at a pause, which its interims are held to.
type sentWords struct {
	mu sync.Mutex
	// utterance is the newest utterance words have gone out of, upTo how much of its audio
	// the last of them were of, text what they said, and stable whether they carried on the
	// words before them.
	utterance int64
	upTo      time.Duration
	text      string
	stable    bool
	// final is the newest utterance whose final has gone out, or that ended with nothing
	// heard or its final lost: words of it, or of one before it, still to come are of an
	// utterance that is over. It is the utterance's own final that ends it, never newer
	// words of the next, which an interim of the next utterance is.
	final int64
	// language is the language of utterance spokenOf's first words at a pause, as the
	// server named it (Words.Language).
	language string
	spokenOf int64
}

// send sends transcript unless words of its utterance that supersede it have gone out, and
// says what became of it: written, or why not. A final goes whatever went out before it,
// and ends its utterance. An interim is held to the words shown as well: never in a script
// they are not in, and once they are stable, carried on by the words after them, only words
// that carry them on in turn (carriesOn). Words shown once and never carried on may be
// anything, the agent's echo or the room, and the next words replace them.
func (t *Transcriber) send(transcript stt.Transcript) string {
	s := &t.sent
	s.mu.Lock()
	defer s.mu.Unlock()
	upTo := time.Duration(transcript.AudioDurationMs * float64(time.Millisecond))
	same := transcript.Utterance == s.utterance
	switch {
	case transcript.Utterance <= s.final:
		// After its final, or of an utterance before one that has ended.
		return "late"
	case transcript.Final():
	case transcript.Utterance < s.utterance || (same && upTo <= s.upTo):
		return "late"
	case transcript.Interim && same && scripts(transcript.Text)&^scripts(s.text) != 0:
		return "unstable"
	case transcript.Interim && same && s.stable && !carriesOn(s.text, transcript.Text):
		return "unstable"
	}
	if transcript.Final() {
		s.final = transcript.Utterance
	}
	// Whoever wrote the words, the session timed them: it ended the utterance.
	transcript.Floor = true
	if transcript.Utterance >= s.utterance {
		s.stable = same && !transcript.Final() && carriesOn(s.text, transcript.Text)
		s.utterance, s.upTo, s.text = transcript.Utterance, upTo, transcript.Text
	}
	t.emitter.Send(transcript)
	return "written"
}

// ended records that utterance is over with nothing more to send: it ended on words that
// came back empty, so nothing was heard, or its final was lost, and words of it still to
// come are dropped.
func (s *sentWords) ended(utterance int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.final = max(s.final, utterance)
}

// over says utterance's final has gone out, or it ended with nothing heard: words of it
// still to be asked for or answered are of an utterance that is over.
func (s *sentWords) over(utterance int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return utterance <= s.final
}

// wrote says words of utterance have gone out: an interim shown while the caller talked.
func (s *sentWords) wrote(utterance int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.utterance == utterance && s.text != ""
}

// establish keeps language as the one utterance is spoken in, unless its first words have
// said already, and says which it is. An empty language, from a server that does not say,
// establishes nothing.
func (s *sentWords) establish(utterance int64, language string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if utterance > s.spokenOf && language != "" {
		s.language, s.spokenOf = language, utterance
	}
	return s.spokenLocked(utterance)
}

// spoken is the language utterance is spoken in, or empty until its first words at a pause
// have said.
func (s *sentWords) spoken(utterance int64) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.spokenLocked(utterance)
}

func (s *sentWords) spokenLocked(utterance int64) string {
	if s.spokenOf != utterance {
		return ""
	}
	return s.language
}

// carriesOn says next, an interim, goes on from shown, the words of the utterance a watcher
// sees: with at least as many words, opening on the same word and keeping at least half of
// shown's. The model writes the whole utterance each time, and over noise or speech in
// another language two writings of the same audio can differ wholesale: a watcher keeps the
// words that were carried on once, and words that don't carry them on wait for the final.
func carriesOn(shown, next string) bool {
	if shown == "" {
		return false
	}
	before, after := plainWords(shown), plainWords(next)
	if len(before) == 0 || len(after) < len(before) || before[0] != after[0] {
		return false
	}
	has := map[string]bool{}
	for _, w := range after {
		has[w] = true
	}
	var kept int
	for _, w := range before {
		if has[w] {
			kept++
		}
	}
	return 2*kept >= len(before)
}

// plainWords is text as words: lowercase, letters, digits and apostrophes alone.
func plainWords(text string) []string {
	return strings.Fields(strings.Map(func(r rune) rune {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '\'':
			return unicode.ToLower(r)
		}
		return ' '
	}, text))
}

// scriptTables are the scripts letters are told apart by; a letter in none is one more.
var scriptTables = []*unicode.RangeTable{unicode.Latin, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul,
	unicode.Devanagari, unicode.Cyrillic, unicode.Arabic, unicode.Hebrew, unicode.Greek, unicode.Thai}

// scripts is the set of scripts text's letters are in, a bit each.
func scripts(text string) (set uint) {
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		bit := uint(1) << len(scriptTables)
		for i, table := range scriptTables {
			if unicode.Is(table, r) {
				bit = 1 << i
				break
			}
		}
		set |= bit
	}
	return set
}

// interim asks for the words so far while the caller talks on, once interimEvery of speech
// has come since they were last asked for, by a pause's first check or an interim, less
// often as the utterance grows past Window: one at a time, on a goroutine of its own, so
// that it holds up no pause check and no final. Not before the utterance before has had its
// final, which a slow server may still be writing: an interim out first would show the next
// utterance's words before the last one's. The words are asked for in the utterance's own
// language once its first words at a pause have said which (sentWords.establish), and until
// then in the languages the callers were declared to speak; with none declared they are not
// asked for yet, as the server would guess the language of a second of speech, and a watcher
// would see the words switch language and script from one interim to the next. It runs on
// the session's goroutine, on a frame of speech.
func (t *Transcriber) interim(participant stt.Participant, heard int64) {
	if !t.interims || t.interimOut.Load() {
		return
	}
	utterance := t.utterance + 1
	if !t.sent.over(utterance - 1) {
		return
	}
	languages, spoken := t.languages, t.sent.spoken(utterance)
	switch {
	case spoken != "":
		languages = []string{spoken}
	case len(languages) == 0:
		return
	}
	length := time.Duration(heard-t.start) * frame
	every := time.Duration(float64(interimEvery) * max(1, float64(length)/float64(Window)))
	if time.Duration(t.spoken)*frame < every {
		return
	}
	t.spoken = 0
	if t.client.backingOff(kindInterim) {
		// The server asked for time: the words of the pauses and the utterance's end, which
		// end turns, are asked for meanwhile, and the words so far are not.
		backedOff.WithLabelValues(kindInterim).Inc()
		return
	}
	t.interimOut.Store(true)
	t.running.Add(1)
	go t.writeInterim(participant, utterance, heard, t.talked, t.heard.Last(length), languages, spoken)
}

// writeInterim asks for the words of audio, the utterance so far, in languages, and sends
// them as an interim of the utterance unless newer words of it, or its final, have gone out
// meanwhile, or they are in a language other than spoken, the utterance's own, when that is
// known. A server that is busy or fails costs nothing but this request: it is not asked
// again. Words it wrote, none included, move the clock of a pause the speech gate can't hear,
// as of frame at with talked voiced frames (stillWords); words blanked as another language
// say nothing of whether the caller went on.
func (t *Transcriber) writeInterim(participant stt.Participant, utterance, at, talked int64, audio harness.Audio, languages []string, spoken string) {
	defer t.running.Done()
	defer t.interimOut.Store(false)
	start := time.Now()
	ctx, cancel := context.WithTimeout(t.ctx, transcribeTimeout+audio.Duration)
	said, err := t.client.Transcribe(ctx, audio, languages)
	cancel()
	if t.ctx.Err() != nil {
		return
	}
	if err != nil {
		interims.WithLabelValues(interimFailure(err)).Inc()
		t.logger.Debug("the chorus-asr server did not write an interim, which is not asked for again",
			"participant", participant.ID, "utterance", utterance, "error", err)
		return
	}
	words := strings.TrimSpace(said.Text)
	if said.Noise == "" && (spoken == "" || said.Language == "" || said.Language == spoken) {
		t.wrote(utterance, at, talked, words)
	}
	switch {
	case said.Noise == "language", spoken != "" && said.Language != "" && said.Language != spoken:
		interims.WithLabelValues("foreign").Inc()
		t.logger.Debug("an interim in a language other than the utterance's is not shown",
			"participant", participant.ID, "utterance", utterance, "language", said.Language, "spoken", spoken, "noise", said.Noise)
		return
	case words == "":
		interims.WithLabelValues("empty").Inc()
		return
	}
	interims.WithLabelValues(t.send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeReplacement,
		Interim:          true,
		Utterance:        utterance,
		Text:             words,
		Provider:         ProviderName,
		Model:            t.model,
		ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000,
		AudioDurationMs:  float64(audio.Duration.Microseconds()) / 1000,
	})).Inc()
}

// interimFailure is why an interim was not written, as interims counts it: refused by a
// server shedding load, or failed.
func interimFailure(err error) string {
	var transcript *TranscriptError
	var status *StatusError
	switch {
	case errors.As(err, &transcript):
		return "refused"
	case errors.As(err, &status) && (status.Code == http.StatusServiceUnavailable || status.Code == http.StatusTooManyRequests):
		return "refused"
	}
	return "failed"
}
