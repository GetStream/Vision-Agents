package harness

import (
	"strings"
)

// UnifiedNote is what a reply that also rules on its own turn is told. The conversation
// model reads the whole conversation anyway, so it says what the caller's words call for
// in the same breath as answering them, and the tag it opens with is the ruling a flow
// controller would otherwise have been asked for.
const UnifiedNote = `Before anything else, even a direction such as [short pause], open your reply with exactly
one of these tags, which is never spoken, saying what the caller's last words call for:
[respond] a complete thought meant for you: then answer it, or call the tool it needs.
[clarify] a finished request meant for you that could mean two different things: then ask
one short question. Never [clarify] to ask for a detail you still need while they are still
giving you details.
[wait] they have probably not finished: the words stop mid-sentence, or end on a PIN, member
ID, phone number, address or clock time that may still be growing, or they said they would
give you everything at once and have not yet asked anything or plainly finished. A recorded
phone menu still reading out its options is one unfinished thought, however long its pauses.
Write nothing after [wait].
[ignore] the words were not meant for you: background speech, a television or radio, or
somebody talking to someone else in the room, such as words that open by calling somebody
"Mom", "Dad", "honey" or "sweetie". Write nothing after [ignore].
Choose [respond] whenever the caller is plainly talking to you and has finished, even if
it is short, such as "yes", "go ahead" or a single detail you asked for.
Once they have said they will give you everything at once, keep choosing [wait] at every
pause, even after a complete sentence or a complete ID, until they ask you something or
say they are done: if they have stopped for good you will be asked again shortly.
Examples of the caller's last words and the tag they call for:
"I'll give you everything up front so we don't go back and forth." [wait]
"My name is Maya Chen." after saying they would give everything at once: [wait]
"Here it is, ABC one two three." after saying they would give everything at once: [wait]
"Tuesday morning would suit me better." after saying they would give everything at once: [wait]
"Hi, this is Maya Chen, I need to move an appointment." [respond]
"Hi, I hope this is the right line." with no request yet: [wait]
"My member ID is, hold on, let me find the card." [wait]
"My PIN is four four" [wait]
"Mom, can I have the nuggets?" [ignore]
"Yes, that's right. Please go ahead." [respond]
"March fourth, 1987." after you asked for their date of birth: [respond]`

// UnifiedReminder goes after the caller's words in the request alone, where the model reads
// it last; the conversation keeps what they said.
const UnifiedReminder = "(Not said by the caller. Open with [respond], [clarify], [wait] or " +
	"[ignore]. [respond] to a finished request or answer. [clarify] only for a finished " +
	"request that could mean two things, never to ask for a missing detail. [wait] only if " +
	"they have plainly not finished: mid-sentence, a number or ID still growing, or they " +
	"said they would give you everything at once and have not yet asked you anything or " +
	"said they are done.)"

// AnotherVoiceNote is added when the words came in a different voice from the caller's.
const AnotherVoiceNote = "The caller's last words came in a different voice from the one " +
	"you have been talking to: somebody else at their microphone, usually talking to the " +
	"room rather than to you. Lean towards [ignore] unless they plainly address you."

// unifiedTags maps the tags a unified reply opens with to the ruling each stands for.
var unifiedTags = map[string]Disposition{
	"respond": Respond,
	"clarify": Clarify,
	"wait":    Wait,
	"ignore":  Ignore,
}

// unifiedGroups is how many bracketed groups at the opening of a reply are looked through for
// the tag: a performing voice's directions, such as [short pause], often come first.
const unifiedGroups = 3

// UnifiedLead reports whether enough of a reply's opening has arrived to read its tag: either
// the tag itself, words, or as many bracketed groups as are looked through.
func UnifiedLead(text string) bool {
	rest := text
	for range unifiedGroups {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			return false
		}
		if rest[0] != '[' {
			return true
		}
		end := strings.Index(rest, "]")
		if end < 0 {
			return len(rest) > 24
		}
		if _, known := unifiedTags[strings.ToLower(strings.TrimSpace(rest[1:end]))]; known {
			return true
		}
		rest = rest[end+1:]
	}
	return true
}

// ParseUnified reads the tag a unified reply opened with, looking past the bracketed
// directions ahead of it. It returns the ruling and the span of text the tag and the space
// after it take up. A reply with no tag is a reply, so it rules Respond with an empty span.
func ParseUnified(text string) (Disposition, int, int) {
	at := 0
	for range unifiedGroups {
		group := at
		for at < len(text) && strings.ContainsRune(" \t\r\n", rune(text[at])) {
			at++
		}
		if at >= len(text) || text[at] != '[' {
			return Respond, 0, 0
		}
		end := strings.Index(text[at:], "]")
		if end < 0 {
			return Respond, 0, 0
		}
		end += at
		if disposition, known := unifiedTags[strings.ToLower(strings.TrimSpace(text[at+1:end]))]; known {
			stop := end + 1
			for stop < len(text) && strings.ContainsRune(" \t\r\n", rune(text[stop])) {
				stop++
			}
			if group > 0 {
				// The space ahead of the tag is kept to separate the direction before it
				// from the words after.
				return disposition, at, stop
			}
			return disposition, 0, stop
		}
		at = end + 1
	}
	return Respond, 0, 0
}

// StripUnifiedTag removes a ruling tag from the opening of text.
func StripUnifiedTag(text string) string {
	_, from, to := ParseUnified(text)
	return text[:from] + text[to:]
}

// Rule reports a ruling that was not asked of the flow controller, so it reaches the agent
// on the same events a controller's ruling does.
func (h *Harness) Rule(decided Decided) {
	h.emitter.Send(decided)
}
