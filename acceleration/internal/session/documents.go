package session

import (
	"context"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// documents are the files pinned in a text session as it opens: those a fork carries in
// (Spec.Documents), kept under the new session as well so a fork of it carries them on, and
// those its conversation was sent before, read back only for the person it is with. A
// document that cannot be kept or read leaves the session without it rather than unopened,
// as the rest of a session's records do.
func (m *Manager) documents(ctx context.Context, spec Spec) []agent.Document {
	carried := slices.Clone(spec.Documents)
	written := m.options.Store
	if written != nil && !spec.Incognito {
		for i, document := range carried {
			kept := document
			kept.ID, kept.SessionID, kept.ConversationID, kept.ResponseID = store.NewID(), spec.ID, spec.ConversationID, ""
			kept.CustomerID, kept.OwnerUserID, kept.CreatedAt = spec.CustomerID, spec.Caller.UserID, document.CreatedAt
			if err := written.AddConversationDocument(ctx, &kept); err != nil {
				m.logger.Error("could not keep a document a fork carried", "session", spec.ID, "error", err)
				continue
			}
			carried[i] = kept
		}
	}
	documents := pinnable(carried)
	// Only a conversation that was there before can have been sent one.
	if written == nil || spec.ConversationID == "" && spec.Reopened.IsZero() {
		return documents
	}
	stored, err := written.ConversationDocuments(ctx, spec.CustomerID, store.DocumentScope{
		ConversationID: spec.ConversationID, SessionID: spec.ID, Owner: spec.Caller.UserID,
	})
	if err != nil {
		m.logger.Error("could not read the documents of a conversation", "session", spec.ID, "error", err)
		return documents
	}
	for _, document := range pinnable(stored) {
		if !slices.ContainsFunc(documents, func(pinned agent.Document) bool { return pinned.ID == document.ID }) {
			documents = append(documents, document)
		}
	}
	return newestWithin(documents, agent.MaxDocumentRunes)
}

// newestWithin is the newest documents whose text fits in budget runes together, oldest
// first. A conversation is refused a file past the budget as it arrives, so only a budget
// lowered since leaves an older one out.
func newestWithin(documents []agent.Document, budget int) []agent.Document {
	at := len(documents)
	for at > 0 && documents[at-1].Runes() <= budget {
		budget -= documents[at-1].Runes()
		at--
	}
	return documents[at:]
}

// pinnable is kept documents as the agent pins them.
func pinnable(kept []store.ConversationDocument) []agent.Document {
	documents := make([]agent.Document, 0, len(kept))
	for _, document := range kept {
		documents = append(documents, agent.Document{ID: document.ID, Name: document.Name, Pages: document.Pages, Text: document.Markdown})
	}
	return documents
}
