package api

import (
	"log/slog"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
)

// callUpdates is what whoever runs a speech-to-speech conversation's calls says about them
// while they run, kept until the line is free: no reply is being spoken and the caller is
// not talking. The rules are a session's: a newer update for a call replaces one not yet
// said, and the call's result drops it, since the result says more.
type callUpdates struct {
	session *stsrouter.Session
	logger  *slog.Logger

	mu           sync.Mutex
	instructions string
	// open is the calls the model is waiting on, and waiting what is to be said about them.
	open    map[string]struct{}
	waiting map[string]sts.CallUpdate
	// replying is set from a reply starting, or an update being asked for, until it ends.
	replying, listening bool
}

func newCallUpdates(session *stsrouter.Session, instructions string, logger *slog.Logger) *callUpdates {
	return &callUpdates{
		session:      session,
		logger:       logger,
		instructions: instructions,
		open:         map[string]struct{}{},
		waiting:      map[string]sts.CallUpdate{},
	}
}

// heard follows what the model is doing, from each event before it goes down the socket.
func (u *callUpdates) heard(event sts.Event) {
	u.mu.Lock()
	switch typed := event.(type) {
	case sts.ToolCall:
		u.open[typed.CallID] = struct{}{}
	case sts.ToolCancel:
		for _, id := range typed.CallIDs {
			delete(u.open, id)
			delete(u.waiting, id)
		}
	case sts.ResponseStarted:
		u.replying = true
	case sts.ResponseComplete:
		u.replying = false
	case sts.SpeechStarted:
		u.listening = true
	case sts.SpeechStopped:
		u.listening = false
	}
	u.mu.Unlock()
	u.say()
}

// update takes a status for an open call, or with verbatim, words to say as written. One
// for a call the model is not waiting on, or to a model that cannot be told, is dropped.
func (u *callUpdates) update(id, text string, verbatim bool) {
	if text == "" || !u.session.UpdatesCalls() {
		return
	}
	u.mu.Lock()
	if _, open := u.open[id]; open {
		u.waiting[id] = sts.CallUpdate{CallID: id, Status: text, Verbatim: verbatim}
	}
	u.mu.Unlock()
	u.say()
}

// answered closes a call, which drops what was still to be said about it.
func (u *callUpdates) answered(id string) {
	u.mu.Lock()
	delete(u.open, id)
	delete(u.waiting, id)
	u.mu.Unlock()
}

// holds reports whether a call is still open, and so whether it can be held. Nothing on
// this socket cancels a call but the model, which stops waiting on it all the same.
func (u *callUpdates) holds(id string) bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	_, open := u.open[id]
	return open
}

func (u *callUpdates) setInstructions(instructions string) {
	u.mu.Lock()
	u.instructions = instructions
	u.mu.Unlock()
}

// say asks the model for one waiting update, when the line is free.
func (u *callUpdates) say() {
	u.mu.Lock()
	if u.replying || u.listening || len(u.waiting) == 0 {
		u.mu.Unlock()
		return
	}
	var update sts.CallUpdate
	for id, waiting := range u.waiting {
		update = waiting
		delete(u.waiting, id)
		break
	}
	update.Guidance = sts.Guidance(update.Status, update.Verbatim)
	update.Instructions = u.instructions
	u.replying = true
	u.mu.Unlock()

	if err := u.session.UpdateCall(update); err != nil {
		u.mu.Lock()
		u.replying = false
		u.mu.Unlock()
		u.logger.Warn("could not tell the caller how a tool is going", "call", update.CallID, "error", err)
	}
}
