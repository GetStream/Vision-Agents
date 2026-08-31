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

// workingPhrases are what the agent says when it has gone off to do something and would
// otherwise leave the caller listening to nothing. They are rotated so a caller who asks
// twice is not answered with the same words.
var workingPhrases = []string{
	"One moment.",
	"Let me check that.",
	"One second, looking that up.",
	"Bear with me a moment.",
}

// idlePhrases are what the agent says to a call nobody is talking on, so a silence ends
// in an invitation rather than in the caller wondering whether anyone is still there.
//
// They rotate across the whole call rather than restarting with each silence. A caller
// who pauses several times over a long call would otherwise be asked the same opening
// question every time, which is the point at which a stock phrase starts to grate.
var idlePhrases = []string{
	"Is there anything else I can help with?",
	"Anything else on your mind?",
	"Was there anything else?",
	"What else can I do for you?",
	"Happy to keep going if there is more.",
	"Anything else you wanted to look at?",
}

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
	// working rotates what is said while a tool runs, separately from the murmurs so
	// that using one does not skip the other along.
	working int
	// asked counts how often the silence in hand has been asked about, and is cleared
	// when somebody speaks. It caps the nagging without deciding the words.
	asked int
	// idle rotates what is said to a call that has gone quiet. It runs on across the
	// whole call, so a later silence does not open with the same question as the first.
	idle int
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

	// Somebody is talking, so a silence that had been given up on is over and a later
	// one is worth asking about again. Only the count is cleared: the rotation carries
	// on, so the next silence is not opened with the same question as the last.
	d.asked = 0
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

// Idle returns a question to put to a call nobody has said anything on for a while, or
// empty when it has not been quiet for long enough. A call where nothing has happened at
// all is not idle yet: the agent has not so much as greeted anyone.
//
// Like Working, and unlike a murmur, it is not tied to the backchannel option: leaving
// somebody in silence until they hang up is never what was wanted.
func (d *duplex) Idle(lastActivity time.Time, quiet bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !quiet || lastActivity.IsZero() || time.Since(lastActivity) < defaultIdleGap {
		return ""
	}
	if d.asked >= idleAsks {
		return ""
	}
	phrase := idlePhrases[d.idle%len(idlePhrases)]
	d.asked++
	d.idle++
	return phrase
}

// Working returns something to say while a tool runs, for a turn where the model reached
// for one without a word to the caller.
//
// It is not tied to the backchannel option: murmuring over someone who is still talking
// is a judgement call, but going quiet on somebody who asked a question is never what
// was wanted.
func (d *duplex) Working() string {
	d.mu.Lock()
	defer d.mu.Unlock()

	phrase := workingPhrases[d.working%len(workingPhrases)]
	d.working++
	return phrase
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
