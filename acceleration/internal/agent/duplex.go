package agent

import (
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Prefixes that say what a turn is for. Audio is gated on the current turn, so every
// noise the agent makes needs one, and telling them apart is what stops a murmur being
// treated as a reply worth interrupting.
const (
	replyPrefix       = "turn-"
	speculationPrefix = "guess-"
	backchannelPrefix = "back-"
)

// Defaults for listening while someone else is talking.
const (
	// defaultBackchannelWords is how much someone must have said before letting them
	// know you are still there is worth doing. Acknowledging three words is interrupting.
	defaultBackchannelWords = 14
	// defaultBackchannelGap is the least time between two murmurs. A listener who says
	// "mhm" every other second is not listening, they are heckling.
	defaultBackchannelGap = 6 * time.Second
)

// defaultPhrases are what the agent murmurs to show it is still listening. They are
// short on purpose: anything longer is a turn, and taking a turn is interrupting.
var defaultPhrases = []string{"Mhm.", "Okay.", "Right.", "I see."}

// uncertainNote is what the model is told about a turn the transcriber was doubtful
// about. Checking is cheaper than confidently answering the wrong question.
const uncertainNote = "You did not catch all of that. Check what they meant before " +
	"answering, rather than answering as though you were sure."

// DuplexOptions configure listening and talking at the same time.
//
// Both halves are off by default, because both trade a guess for latency and the guess
// is only worth making when the transcriber is good enough to revoke it.
type DuplexOptions struct {
	// Backchannel makes the agent murmur while someone is still talking, the way a person
	// on the phone does. It never reaches the model: a listening noise is not a turn.
	Backchannel bool
	// Speculate starts answering as soon as the transcriber provisionally ends a turn,
	// rather than waiting for it to be sure. The reply is held back until the turn really
	// does settle on the same words, and thrown away unheard when it does not. What it
	// buys is the model's time to first token, which is most of the wait.
	Speculate bool
	// Phrases are what the agent murmurs. Empty means the built-in ones.
	Phrases []string
	// BackchannelWords is how much someone must have said before it is worth
	// acknowledging. Zero means the default.
	BackchannelWords int
	// BackchannelGap is the least time between two murmurs. Zero means the default.
	BackchannelGap time.Duration
	// MinConfidence is how sure the transcriber has to be for the agent to answer a turn
	// as though it heard it properly. Below it the agent checks what they meant instead.
	// Zero turns this off.
	MinConfidence float64
}

// enabled reports whether either half of duplex is on.
func (o DuplexOptions) enabled() bool { return o.Backchannel || o.Speculate }

// duplex tracks what each participant is in the middle of saying.
//
// Turn boundaries come from the transcriber, which reports the start of speech, a
// provisional end that it may yet revoke, and a settled end. Taking the provisional one
// seriously is what lets the agent be answering already when the turn really ends.
type duplex struct {
	options DuplexOptions

	mu sync.Mutex
	// speakers is one state per participant, because two people talking at once are two
	// separate turns.
	speakers map[string]*speaker
	// phrase rotates the murmurs, so the agent does not say "mhm" four times running.
	phrase int
}

// speaker is what one participant is in the middle of.
type speaker struct {
	// interim is the latest revision of what they are saying.
	interim string
	// murmured is when the agent last let them know it was still there.
	murmured time.Time
	// guess is the reply started on a provisional end of turn, empty when there is none.
	guess string
	// guessed is the transcript that reply was started on, which the settled one has to
	// agree with for the reply to be worth keeping.
	guessed string
}

func newDuplex(options DuplexOptions) *duplex {
	if options.BackchannelWords <= 0 {
		options.BackchannelWords = defaultBackchannelWords
	}
	if options.BackchannelGap <= 0 {
		options.BackchannelGap = defaultBackchannelGap
	}
	if len(options.Phrases) == 0 {
		options.Phrases = defaultPhrases
	}
	return &duplex{options: options, speakers: map[string]*speaker{}}
}

// Began records that a participant has started, or gone back to, talking. It returns the
// speculative reply to abandon: they carried on, so whatever was being answered was only
// half of what they had to say.
func (d *duplex) Began(participant stt.Participant) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	current := d.speakerFor(participant)
	abandoned := current.guess
	current.guess, current.guessed = "", ""
	return abandoned
}

// Heard records a revision of what someone is saying and returns a murmur worth making,
// or empty when there is none. Quiet says whether the agent has the floor: talking over
// someone to tell them you are listening is not listening.
func (d *duplex) Heard(participant stt.Participant, text string, quiet bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	current := d.speakerFor(participant)
	// The revision is kept whatever happens next: a provisional end of turn carries no
	// words of its own, so this is what a reply guessed at it would be answering.
	current.interim = text

	if !d.options.Backchannel || !quiet {
		return ""
	}
	if len(strings.Fields(text)) < d.options.BackchannelWords {
		return ""
	}
	if time.Since(current.murmured) < d.options.BackchannelGap {
		return ""
	}
	// A murmur made while a reply is being guessed at would land on top of the reply
	// itself a moment later.
	if current.guess != "" {
		return ""
	}

	current.murmured = time.Now()
	phrase := d.options.Phrases[d.phrase%len(d.options.Phrases)]
	d.phrase++
	return phrase
}

// Eager records that the transcriber has provisionally ended a turn, and returns the
// reply worth starting on it. The text is what the reply should answer, which is the
// provisional transcript rather than the empty string a bare end-of-turn carries.
func (d *duplex) Eager(participant stt.Participant, text string) (string, string, bool) {
	if !d.options.Speculate || strings.TrimSpace(text) == "" {
		return "", "", false
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	current := d.speakerFor(participant)
	// Guessing twice at the same words would be paying for the same reply twice.
	if current.guess != "" && sameWords(current.guessed, text) {
		return "", "", false
	}

	abandoned := current.guess
	current.guess = speculationPrefix + turnStamp()
	current.guessed = text
	return current.guess, abandoned, true
}

// Settled records that a turn is over, and reports what to do with the reply that was
// guessed at: promote it if the words held, abandon it if they did not.
func (d *duplex) Settled(participant stt.Participant, text string) (string, string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	current := d.speakerFor(participant)
	guess, guessed := current.guess, current.guessed
	current.guess, current.guessed, current.interim = "", "", ""

	if guess == "" {
		return "", ""
	}
	if sameWords(guessed, text) {
		return guess, ""
	}
	return "", guess
}

// Note is what the model should know about a turn beyond its words, which for now is
// only whether it was heard clearly.
//
// A transcriber that reports no confidence at all reports zero, and never having been
// told is not the same as having been told the caller was inaudible.
func (d *duplex) Note(confidence float64) string {
	if d.options.MinConfidence <= 0 || confidence <= 0 {
		return ""
	}
	if confidence >= d.options.MinConfidence {
		return ""
	}
	return uncertainNote
}

// Interim is the latest revision of what a participant is saying.
func (d *duplex) Interim(participant stt.Participant) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.speakerFor(participant).interim
}

// Forget drops a participant's state, so a reconnection does not inherit half a turn.
func (d *duplex) Forget(participant stt.Participant) {
	d.mu.Lock()
	defer d.mu.Unlock()
	delete(d.speakers, participant.ID)
}

// speakerFor returns a participant's state, starting one on first hearing them. It must
// be called with the lock held.
func (d *duplex) speakerFor(participant stt.Participant) *speaker {
	current, ok := d.speakers[participant.ID]
	if !ok {
		current = &speaker{}
		d.speakers[participant.ID] = current
	}
	return current
}

// sameWords reports whether two transcripts say the same thing.
//
// A provisional transcript and the settled one that follows it differ in punctuation and
// capitalisation far more often than in words, and a reply to "book a table for four" is
// still the right reply to "Book a table for four."
func sameWords(first, second string) bool {
	return strings.EqualFold(words(first), words(second))
}

func words(text string) string {
	var kept strings.Builder
	for _, symbol := range text {
		if unicode.IsLetter(symbol) || unicode.IsDigit(symbol) || unicode.IsSpace(symbol) {
			kept.WriteRune(symbol)
		}
	}
	return strings.Join(strings.Fields(kept.String()), " ")
}
