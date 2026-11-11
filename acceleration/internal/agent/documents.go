package agent

import (
	"context"
	"errors"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Document is a file the caller attached to the conversation, read as text.
type Document struct {
	// ID names it where it is kept, so a conversation carried on pins it once.
	ID   string
	Name string
	// Pages is 0 for a file with none to count.
	Pages int
	Text  string
}

// MaxDocumentRunes is the most text, in runes, the files pinned in one conversation hold
// together. Every reply is sent all of them, so this is the room they take in every prompt.
const MaxDocumentRunes = 120_000

// Runes is the room a document takes in a prompt.
func (d Document) Runes() int { return utf8.RuneCountInString(d.Text) }

// documentAlone is what the turn says of a file sent with no words: the model is asked about
// it rather than handed an empty message.
const documentAlone = "The caller sent this document without a message. Say briefly what it is about."

type documentKey struct{}

// WithDocument says that the turn RespondTo answers on ctx brought a file, which is pinned
// ahead of the conversation for that reply and every one after it.
func WithDocument(ctx context.Context, document Document) context.Context {
	return context.WithValue(ctx, documentKey{}, document)
}

// PinDocuments pins files attached earlier in the conversation, as a conversation carried on
// or forked has them, ahead of every reply. One already pinned is left as it is.
func (a *Agent) PinDocuments(documents ...Document) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, document := range documents {
		a.pinLocked(document)
	}
}

// ReplaceDocuments pins documents in place of every file pinned before, as a conversation
// rewound to before some of them was sent has them.
func (a *Agent) ReplaceDocuments(documents ...Document) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.documents, a.pinned = nil, nil
	for _, document := range documents {
		a.pinLocked(document)
	}
}

// pinLocked pins one document, and reports whether it was not pinned already.
func (a *Agent) pinLocked(document Document) bool {
	if slices.ContainsFunc(a.documents, func(pinned Document) bool { return pinned.ID == document.ID }) {
		return false
	}
	a.documents = append(a.documents, document)
	a.pinned = append(a.pinned, harness.DocumentMessage(document.Name, document.Pages, document.Text))
	return true
}

// unpinLocked takes back the document pinned for a turn that did not start.
func (a *Agent) unpinLocked(id string) {
	// Copied rather than deleted in place: a reply already started holds pinned unlocked.
	if at := slices.IndexFunc(a.documents, func(pinned Document) bool { return pinned.ID == id }); at >= 0 {
		a.documents = slices.Concat(a.documents[:at], a.documents[at+1:])
		a.pinned = slices.Concat(a.pinned[:at], a.pinned[at+1:])
	}
}

// runesPerToken is how many runes of a document a token is counted as when it is weighed
// against the model's window. Prose runs nearer four; a low count errs to refusing a file.
const runesPerToken = 3

// DocumentRoom is how much text, in runes, one more file may hold: what is left of
// MaxDocumentRunes, and of a quarter of the conversation model's window when it declares one,
// after the files already pinned.
func (a *Agent) DocumentRoom() int {
	budget := a.documentBudget()
	a.mu.Lock()
	defer a.mu.Unlock()
	return max(budget-a.pinnedRunesLocked(), 0)
}

// documentBudget is the room every file of the conversation shares: MaxDocumentRunes, or a
// quarter of the conversation model's window when it declares less.
func (a *Agent) documentBudget() int {
	a.mu.Lock()
	current := a.harness
	a.mu.Unlock()
	budget := MaxDocumentRunes
	if current != nil {
		if window := current.ContextWindow(); window > 0 {
			budget = min(budget, int(window/4)*runesPerToken)
		}
	}
	return budget
}

func (a *Agent) pinnedRunesLocked() int {
	used := 0
	for _, document := range a.documents {
		used += document.Runes()
	}
	return used
}

// Documents are the files pinned in the conversation, oldest first.
func (a *Agent) Documents() []Document {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.documents)
}

// attachDocument pins the file ctx brings, if any, for the turn about to start, and returns
// the words the turn answers: text, or a request to read the file when it came with none.
// undo takes the pin back when the turn does not start.
func (a *Agent) attachDocument(ctx context.Context, text string) (string, func(), error) {
	document, ok := ctx.Value(documentKey{}).(Document)
	if !ok {
		return text, func() {}, nil
	}
	budget := a.documentBudget()
	a.mu.Lock()
	// Weighed again as it is pinned: two questions sent at once were each weighed against
	// the room before either was pinned.
	if a.pinnedRunesLocked()+document.Runes() > budget && !slices.ContainsFunc(a.documents, func(pinned Document) bool { return pinned.ID == document.ID }) {
		a.mu.Unlock()
		return "", nil, stack.Wrap(ErrNoRoomForDocument)
	}
	added := a.pinLocked(document)
	a.mu.Unlock()
	if strings.TrimSpace(text) == "" {
		text = documentAlone
	}
	if !added {
		return text, func() {}, nil
	}
	return text, func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		a.unpinLocked(document.ID)
	}, nil
}

// ErrNoRoomForDocument is a file that does not fit beside the files the conversation holds.
var ErrNoRoomForDocument = errors.New("agent: the conversation has no room for the file")

// pinnedLocked is what every reply is sent ahead of the conversation. The slice is never
// written in place, only appended to or replaced, so a turn may hold it unlocked.
func (a *Agent) pinnedLocked() []llm.Message {
	return a.pinned[:len(a.pinned):len(a.pinned)]
}
