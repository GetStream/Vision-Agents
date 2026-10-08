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
	// writtenPrefix is an answer nobody hears, so it names a request rather than a turn:
	// no audio is gated on it because none is produced.
	writtenPrefix = "text-"
)

// Defaults for listening while someone else is talking.
const (
	// defaultBackchannelWords is how much someone must have said before letting them
	// know you are still there is worth doing. Acknowledging three words is interrupting.
	defaultBackchannelWords = 14
	// defaultBackchannelGap is the least time between two murmurs. A listener who says
	// "mhm" every other second is not listening, they are heckling.
	defaultBackchannelGap = 6 * time.Second
	// defaultIdleGap is how long a call has to go silent before the agent asks whether
	// there is anything else. Long enough to let somebody think, short enough that they
	// are not left wondering whether the agent is still on the line.
	defaultIdleGap = 30 * time.Second
	// idleAsks is how often one silence is asked about before the agent lets it stand.
	// Somebody who has not answered twice is not going to answer a third time, and
	// asking again is nagging a caller who has walked away.
	idleAsks = 2
)

// defaultPhrases are what the agent murmurs to show it is still listening. They are
// short on purpose: anything longer is a turn, and taking a turn is interrupting.
var defaultPhrases = []string{"Mhm.", "Okay.", "Right.", "I see."}

// lostReply is what the agent says when the model fails before it has said anything, so
// a caller waiting on an answer hears that it is not coming rather than nothing at all.
const lostReply = "Sorry, something went wrong on my side. Could you ask me that again?"

// updateGaps are how long a caller waiting on work hears nothing before being told it is
// still going, one per update. After the last one the agent waits quietly: a status read
// out every few seconds is nagging, and the answer is what they are waiting for.
var updateGaps = []time.Duration{10 * time.Second, 15 * time.Second}

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
	// updateGaps is how long a caller waiting on work hears nothing before each update.
	// Empty means the built-in ones.
	updateGaps []time.Duration
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
	// asked counts how often the silence in hand has been asked about, and is cleared
	// when somebody speaks. It caps the nagging without deciding the words.
	asked int
	// updating is the work the caller was last told is still running, and updates how
	// often they were told about it since they last spoke.
	updating string
	updates  int
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
	if len(options.updateGaps) == 0 {
		options.updateGaps = updateGaps
	}
	return &duplex{options: options, speakers: map[string]*speaker{}}
}

// Heard records a revision of what someone is saying and returns a murmur worth making,
// or empty when there is none. Quiet says whether the agent has the floor: talking over
// someone to tell them you are listening is not listening.
func (d *duplex) Heard(participant stt.Participant, text string, quiet bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Somebody is talking, so a silence that had been given up on is over and a later
	// one is worth asking about again.
	d.asked = 0
	d.updates = 0
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

// Idle reports whether a call nobody has said anything on for a while is due a question,
// so a silence ends in an invitation rather than in the caller wondering whether anyone
// is still there. A call where nothing has happened at all is not idle yet: the agent has
// not so much as greeted anyone.
//
// Like Update, and unlike a murmur, it is not tied to the backchannel option: leaving
// somebody in silence until they hang up is never what was wanted.
func (d *duplex) Idle(lastActivity time.Time, quiet bool) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !quiet || lastActivity.IsZero() || time.Since(lastActivity) < defaultIdleGap {
		return false
	}
	if d.asked >= idleAsks {
		return false
	}
	d.asked++
	return true
}

// Update reports whether a caller who has heard nothing for a while is due word that the
// work they are waiting on is still going. Work names what is running, so new work starts
// its updates over.
func (d *duplex) Update(work string, lastSpokeAt time.Time) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	if work != d.updating {
		d.updating = work
		d.updates = 0
	}
	gaps := d.options.updateGaps
	if work == "" || lastSpokeAt.IsZero() || d.updates >= len(gaps) ||
		time.Since(lastSpokeAt) < gaps[d.updates] {
		return false
	}
	d.updates++
	return true
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
