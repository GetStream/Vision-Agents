package simulation

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// said is the conversation so far. It is the stored shape rather than a private one,
// because everything that reads a conversation -- the caller deciding what to say next, the
// judge ruling on it, the dashboard showing it -- reads the same lines.
type said []store.SimulationLine

// render is the conversation as a model reads it. It is quoted rather than replayed as
// turns for the same reason a review is: a model handed a conversation to continue
// continues it, and both the caller and the judge are being asked for something else.
func (s said) render() string {
	if len(s) == 0 {
		return "Nothing has been said yet. You are speaking first."
	}

	var written strings.Builder
	written.WriteString("The conversation so far, in order:\n\n")
	for _, line := range s {
		speaker := "Agent"
		if line.Caller {
			speaker = "You"
		}
		written.WriteString(speaker)
		written.WriteString(": ")
		written.WriteString(line.Text)
		written.WriteString("\n")
	}
	return written.String()
}

// turns is how many times the caller has spoken, which is what the turn limit counts.
func (s said) turns() int {
	spoke := 0
	for _, line := range s {
		if line.Caller {
			spoke++
		}
	}
	return spoke
}
