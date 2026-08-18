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
	backchannelPrefix = "back-"
	handoffPrefix     = "hand-"
	toolPrefix        = "tool-"
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

// duplex tracks acknowledgements and confidence for each participant.
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
	// murmured is when the agent last let them know it was still there.
	murmured time.Time
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

// Heard records a revision of what someone is saying and returns a murmur worth making,
// or empty when there is none. Quiet says whether the agent has the floor: talking over
// someone to tell them you are listening is not listening.
func (d *duplex) Heard(participant stt.Participant, text string, quiet bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	current := d.speakerFor(participant)

	if !d.options.Backchannel || !quiet {
		return ""
	}
	if len(strings.Fields(text)) < d.options.BackchannelWords {
		return ""
	}
	if time.Since(current.murmured) < d.options.BackchannelGap {
		return ""
	}
	current.murmured = time.Now()
	return d.nextPhraseLocked()
}

// Presence returns a short acknowledgement after a long active listening or thinking
// gap. An otherwise idle call stays quiet.
func (d *duplex) Presence(participant stt.Participant, lastSpokeAt time.Time, quiet bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.options.Backchannel || !quiet || time.Since(lastSpokeAt) < d.options.BackchannelGap {
		return ""
	}
	current := d.speakerFor(participant)
	if time.Since(current.murmured) < d.options.BackchannelGap {
		return ""
	}
	current.murmured = time.Now()
	return d.nextPhraseLocked()
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

func (d *duplex) nextPhraseLocked() string {
	phrase := d.options.Phrases[d.phrase%len(d.options.Phrases)]
	d.phrase++
	return phrase
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
