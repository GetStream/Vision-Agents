package agent

import (
	"fmt"
	"strings"
	"time"
)

// statusPurpose takes what whoever runs a call said about it, for a line in the agent's own
// words.
const statusPurpose = "What they asked for is still being worked on, and whoever is doing it " +
	"reports: %q. Tell them where it has got to, in terms of what they asked for. It is still " +
	"running, so do not guess its result."

// toolSaidPurpose marks words whoever runs a call asked to be said as written.
const toolSaidPurpose = "tool-said"

// toolUpdate is one update waiting for the line to be free.
type toolUpdate struct {
	text     string
	verbatim bool
}

// ToolUpdate takes something whoever runs a call says about it while it runs: a status for
// the agent to say in its own words, or verbatim words to say as written. It is reported
// either way. On a call it is said once nobody is talking and nothing is being written, a
// newer update for the same call replacing it and the call's result dropping it, since the
// result says more. In writing a status is only reported, and words are posted as Say posts
// them. An update for a call that is not running is dropped.
func (a *Agent) ToolUpdate(id, text string, verbatim bool) {
	text = strings.TrimSpace(text)
	a.mu.Lock()
	name, running := a.toolsRunning[id]
	turnID := a.toolTurns[id]
	if !running || text == "" {
		a.mu.Unlock()
		return
	}
	spoken := !a.options.Text && !a.native()
	if spoken {
		if a.toolUpdates == nil {
			a.toolUpdates = map[string]toolUpdate{}
		}
		a.toolUpdates[id] = toolUpdate{text: text, verbatim: verbatim}
	}
	a.mu.Unlock()

	a.emitter.Send(ToolUpdated{ID: id, TurnID: turnID, Tool: name, Text: text, Verbatim: verbatim})
	if a.options.Text && verbatim {
		if err := a.sayFor(text, PurposeUpdate); err != nil {
			a.logger.Warn("could not post what a tool asked to say", "tool", name, "error", err)
		}
	}
}

// sayToolUpdate says one update waiting for the line, when it is free: the agent is not
// speaking or writing, owes no reply, and the caller is not talking.
func (a *Agent) sayToolUpdate() {
	state := a.floor()
	if state.Talking || state.Owed || state.Composing || a.converse.Listening() {
		return
	}
	a.mu.Lock()
	var id string
	var update toolUpdate
	for queued, waiting := range a.toolUpdates {
		id, update = queued, waiting
		break
	}
	if id == "" || a.composing || a.tts == nil {
		a.mu.Unlock()
		return
	}
	delete(a.toolUpdates, id)
	participant, ctx := a.lastParticipant, a.ctx
	asked := time.Now()
	if update.verbatim {
		a.mu.Unlock()
		a.sayComposed(participant, update.text, toolSaidPurpose, asked)
		return
	}
	a.composing = true
	a.mu.Unlock()

	go func() {
		defer func() {
			a.mu.Lock()
			a.composing = false
			a.mu.Unlock()
		}()
		line, err := a.compose(ctx, fmt.Sprintf(statusPurpose, update.text))
		if err != nil {
			a.logger.Warn("could not write a tool's status to say", "error", err)
			return
		}
		a.mu.Lock()
		_, running := a.toolsRunning[id]
		a.mu.Unlock()
		if !running {
			return
		}
		a.sayComposed(participant, line, statusPurpose, asked)
	}()
}
