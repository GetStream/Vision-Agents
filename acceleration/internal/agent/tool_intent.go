package agent

import (
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// What an agent acting on chorus-asr does with the tool a ruling heard the caller ask for
// (audiofloor.Answer.Wanted), in a turn it ended with the floor free and the reply is about
// to answer. The tool is the one the caller asks for, which is often not the one the reply
// calls first: an agent told to check before it books, or to verify a caller before it
// reschedules, is asked to book or to reschedule. What is said at once therefore names no
// tool and no argument, and the reply is told which tool only where the model was right
// about the call often enough to be worth it.
const (
	// acknowledgeAt is how sure the model must be that a tool is wanted for the agent to say
	// an acknowledgment at once. On the voice bench's calls every turn it was this sure of was
	// followed by a call, and below it some were answered with a question instead.
	acknowledgeAt = 0.999
	// steerAt is how sure it must be for the reply to be told which tool, where a deployment
	// turns steering on. At no bar on those calls was the tool the model heard asked for the
	// reply's first call in more than one turn in eight, which is why steering is off unless a
	// deployment turns it on.
	steerAt = 0.999
)

// acknowledgments are what the agent says at once, by the language the caller was heard in,
// a primary BCP 47 subtag, in turn so that a caller does not hear the same words before
// every request. None names a tool or what it is asked: the reply has not worked that out,
// and the tool the caller asks for may not be the first it calls.
var acknowledgments = map[string][]string{
	"en": {"One moment.", "Just a moment.", "Give me a second."},
	"es": {"Un momento.", "Un segundo."},
	"fr": {"Un instant.", "Un moment."},
	"de": {"Einen Moment.", "Einen Augenblick."},
	"pt": {"Um momento.", "Um instante."},
	"it": {"Un attimo.", "Un momento."},
	"nl": {"Een moment.", "Een ogenblik."},
}

// languageTags are the server's names for the languages acknowledgments has
// (audiofloor.Words.Language).
var languageTags = map[string]string{
	"english": "en", "spanish": "es", "french": "fr", "german": "de", "portuguese": "pt", "italian": "it", "dutch": "nl",
}

// acknowledgedNote tells the reply that the acknowledgment was said, as the hold phrase the
// use policy asks it to open the sentence before a call with, so that the wait has one.
const acknowledgedNote = "You have already said %q to the caller, at the start of this reply, while you work out " +
	"what to do: that is your hold phrase, so do not say another or repeat it. Go straight on with the " +
	"rest: whatever your instructions ask you to say before acting, then the call."

// steeredNote tells the reply which tool the caller was heard asking for. It is a note, not
// a tool choice: every provider's named tool choice forces the call, and only the caller's
// words and the agent's instructions may say whether it is time for it.
const steeredNote = "From the caller's voice, what they are asking for is the %s tool. Call it in this reply " +
	"if their words and your instructions support doing so now; if your instructions ask for " +
	"another step first, do that step."

var (
	toolIntents = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_tool_intents_total",
		Help: "Turns answered on a chorus-asr ruling that heard the caller ask for one of the agent's tools, by what " +
			"the agent did about the acknowledgment and the steering: said or steered, or why not.",
	}, []string{"acknowledgment", "steering"})
	acknowledgmentOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_tool_acknowledgments_total",
		Help:      "Acknowledgments of a turn asking for a tool, by whether they were said once the reply was asked for, or why not.",
	}, []string{"outcome"})
)

// wantedTool is the tool chorus-asr heard a caller ask for in a turn it ended with the floor
// free: the model's probability of it, of the turn having ended, and the language the server
// heard the turn's words in, empty when it did not say.
type wantedTool struct {
	tool     string
	p        float64
	pEOT     float64
	language string
}

// toolIntent holds the acknowledgment the reply to a turn is to open with, and when it was
// decided on, until its request is accepted (acknowledge); the last one said and the reply it
// opened; and how many have been decided on, which picks the next.
type toolIntent struct {
	mu               sync.Mutex
	turn, line       string
	decidedAt        time.Time
	saidTurn, saidAs string
	said             int
}

// heardIntent decides what the agent does with the tool chorus-asr heard the caller ask for
// in the turn ready is, which is about to be answered, and returns what the reply is to be
// told about it. adopted says a reply started before the ruling answers the turn, which was
// asked for before any of this could be said or told.
func (a *Agent) heardIntent(ready candidate, adopted bool) string {
	v := a.audioFloor
	if v == nil {
		return ""
	}
	// An acknowledgment is for the reply asked for after it, and none other: a turn answered
	// again, as when the caller went on before hearing any of the reply, has its own say.
	v.intent.mu.Lock()
	v.intent.turn, v.intent.line = "", ""
	v.intent.mu.Unlock()
	wanted, ok := v.wantedFor(ready.ID)
	if !ok {
		return ""
	}
	line, acknowledgment := a.acknowledgment(wanted, adopted)
	steering := a.steering(wanted, adopted)
	toolIntents.WithLabelValues(acknowledgment, steering).Inc()
	a.logger.Info("chorus-asr heard the caller ask for a tool", "turn", ready.ID, "participant", ready.Participant.ID,
		"tool", wanted.tool, "p", wanted.p, "p_eot", wanted.pEOT, "language", wanted.language,
		"acknowledgment", acknowledgment, "line", line, "steering", steering)
	var notes []string
	if line != "" {
		v.intent.mu.Lock()
		v.intent.turn, v.intent.line, v.intent.decidedAt = ready.ID, line, time.Now()
		v.intent.said++
		v.intent.mu.Unlock()
		notes = append(notes, fmt.Sprintf(acknowledgedNote, line))
	}
	if steering == "steered" {
		notes = append(notes, fmt.Sprintf(steeredNote, wanted.tool))
	}
	return strings.Join(notes, "\n\n")
}

// acknowledgment is what to say at once about wanted, and "said", or empty and why not.
func (a *Agent) acknowledgment(wanted wantedTool, adopted bool) (string, string) {
	switch {
	case !a.audioFloor.client.Acknowledges():
		return "", "off"
	case adopted:
		return "", "speculated"
	case wanted.p < acknowledgeAt:
		return "", "below_bar"
	case a.options.Text || a.native() || a.voice() == nil:
		return "", "no_voice"
	case !a.answeredAloud(wanted.tool):
		return "", "no_answer_follows"
	}
	phrases := acknowledgments[a.languageOf(wanted)]
	if len(phrases) == 0 {
		return "", "no_language"
	}
	a.audioFloor.intent.mu.Lock()
	defer a.audioFloor.intent.mu.Unlock()
	return phrases[a.audioFloor.intent.said%len(phrases)], "said"
}

// steering is "steered" when the reply is to be told which tool the caller asked for, or why
// it is not.
func (a *Agent) steering(wanted wantedTool, adopted bool) string {
	switch {
	case !a.audioFloor.client.Steers():
		return "off"
	case adopted:
		return "speculated"
	case wanted.p < steerAt:
		return "below_bar"
	case !a.answeredAloud(wanted.tool):
		return "no_answer_follows"
	}
	return "steered"
}

// answeredAloud says the agent offers tool and that its result is followed by a reply the
// caller hears, which is when a pause before it is worth filling (fillsPause).
func (a *Agent) answeredAloud(tool string) bool {
	if _, offered := a.availableTools().Lookup(tool); !offered {
		return false
	}
	return a.followsTool(harness.ToolRequested{Call: llm.ToolCall{Name: tool}})
}

// languageOf is the tag of the language the caller was heard in: the server's word for the
// turn, or else the agent's own when its language hints all name one. Empty is not known.
func (a *Agent) languageOf(wanted wantedTool) string {
	if tag, ok := languageTags[strings.ToLower(wanted.language)]; ok {
		return tag
	}
	if wanted.language != "" {
		return ""
	}
	tag := ""
	for _, hint := range a.options.LanguageHints {
		primary, _, _ := strings.Cut(strings.ToLower(hint), "-")
		if tag != "" && primary != tag {
			return ""
		}
		tag = primary
	}
	return tag
}

// acknowledge says the acknowledgment the reply to turnID was to open with, now that its
// request has been accepted, as that reply's first words: they are the reply's, so a caller
// who talks on before hearing them drops them with it, and one who talks over them is
// talking over the reply. It is not said to a caller talking already, or for a reply that
// has been given up on.
//
// It runs on the goroutine that speaks, like the reply's own words.
func (a *Agent) acknowledge(turnID string) {
	v := a.audioFloor
	if v == nil {
		return
	}
	v.intent.mu.Lock()
	line, decidedAt := "", v.intent.decidedAt
	if v.intent.turn == turnID {
		line = v.intent.line
		v.intent.turn, v.intent.line = "", ""
	}
	v.intent.mu.Unlock()
	if line == "" {
		return
	}
	a.mu.Lock()
	heard := a.lastHeardAt.After(decidedAt)
	a.mu.Unlock()
	outcome := "said"
	switch {
	case !a.speaking(turnID):
		outcome = "abandoned"
	case heard || a.converse.Listening():
		// The caller has been heard since the turn was answered, in words or in an utterance
		// the model has not ended: what they say now is theirs to finish.
		outcome = "caller_talking"
	}
	acknowledgmentOutcomes.WithLabelValues(outcome).Inc()
	if outcome != "said" {
		a.logger.Info("did not say the acknowledgment of a turn asking for a tool", "turn", turnID, "why", outcome)
		return
	}
	a.claimReply(turnID)
	// The model's words follow it in the same message and the same utterance, so they are
	// counted from after it, wherever a span of them is reckoned from.
	span := a.appendSpoken(line)
	a.spoken.WriteString(" ")
	lead := utf8.RuneCountInString(line) + 1
	a.chunk.Said(lead)
	if a.performs {
		a.performed = append(a.performed, []rune(line+" ")...)
	}
	a.mu.Lock()
	a.saying = a.spoken.String()
	a.mu.Unlock()
	v.intent.mu.Lock()
	v.intent.saidTurn, v.intent.saidAs = turnID, line
	v.intent.mu.Unlock()
	a.emitter.Send(ResponseDelta{TurnID: turnID, Text: line + " "})
	if err := a.speakSentence(turnID, line, span); err != nil {
		a.fail(err, "tts")
		return
	}
	a.logger.Info("acknowledged a turn asking for a tool while the reply works it out", "turn", turnID, "line", line)
}

// saidNothingOfItsOwn says the reply to turnID has said nothing but the acknowledgment it
// opened with, if it opened with one: a reply that fails then has left the caller waiting
// on an answer that is not coming, as one that said nothing at all has. It runs on the
// goroutine that speaks.
func (a *Agent) saidNothingOfItsOwn(turnID string) bool {
	said := strings.TrimSpace(a.spoken.String())
	if v := a.audioFloor; v != nil {
		v.intent.mu.Lock()
		if v.intent.saidTurn == turnID {
			said = strings.TrimSpace(strings.TrimPrefix(said, v.intent.saidAs))
		}
		v.intent.mu.Unlock()
	}
	return said == ""
}
