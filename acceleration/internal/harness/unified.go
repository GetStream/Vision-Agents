package harness

import (
	"strings"
)

// UnifiedNote is what a reply that also rules on its own turn is told. The conversation
// model reads the whole conversation anyway, so it says what the caller's words call for
// in the same breath as answering them, and the tag it opens with is the ruling a flow
// controller would otherwise have been asked for.
const UnifiedNote = `Before anything else, open your reply with exactly one of these tags, which is
never spoken, saying what the caller's last words call for:
[respond] a complete thought meant for you: then answer it, or call the tool it needs.
[clarify] meant for you, but you cannot tell what they want: then ask one short question.
[wait] they have probably not finished: the words stop mid-sentence, or end on a PIN, member
ID, phone number, address or clock time that may still be growing, or they said they would
give you everything at once and have not yet asked anything or plainly finished. A recorded
phone menu still reading out its options is one unfinished thought, however long its pauses.
Write nothing after [wait].
[ignore] the words were not meant for you: background speech, a television or radio, or
somebody talking to someone else in the room, such as words that open by calling somebody
"Mom", "Dad", "honey" or "sweetie". Write nothing after [ignore].
Choose [respond] whenever the caller is plainly talking to you and has finished, even if
it is short, such as "yes", "go ahead" or a single detail you asked for.`

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

// UnifiedLead reports whether enough of a reply's opening has arrived to read its tag.
func UnifiedLead(text string) bool {
	trimmed := strings.TrimLeft(text, " \t\r\n")
	if trimmed == "" {
		return false
	}
	if trimmed[0] != '[' {
		return true
	}
	return strings.Contains(trimmed, "]") || len(trimmed) > 16
}

// ParseUnified reads the tag a unified reply opened with. It returns the ruling and how many
// bytes of the text the tag and the space after it take up. A reply that opened without a
// tag is a reply, so it rules Respond and trims nothing.
func ParseUnified(text string) (Disposition, int) {
	trimmed := strings.TrimLeft(text, " \t\r\n")
	lead := len(text) - len(trimmed)
	if !strings.HasPrefix(trimmed, "[") {
		return Respond, 0
	}
	end := strings.Index(trimmed, "]")
	if end < 0 {
		return Respond, 0
	}
	name := strings.ToLower(strings.TrimSpace(trimmed[1:end]))
	disposition, known := unifiedTags[name]
	if !known {
		return Respond, 0
	}
	rest := trimmed[end+1:]
	after := len(rest) - len(strings.TrimLeft(rest, " \t\r\n"))
	return disposition, lead + end + 1 + after
}

// StripUnifiedTag removes a ruling tag from the front of text, for a reply asked without the
// protocol that wrote one anyway.
func StripUnifiedTag(text string) string {
	if _, n := ParseUnified(text); n > 0 {
		return text[n:]
	}
	return text
}

// Rule reports a ruling that was not asked of the flow controller, so it reaches the agent
// on the same events a controller's ruling does.
func (h *Harness) Rule(decided Decided) {
	h.emitter.Send(decided)
}
