//go:build integration

package store

import (
	"time"
)

// attached keeps a document sent in session with turn, as sessionID's owner owner.
func (s *StoreSuite) attached(id, sessionID, conversationID, owner, turn string, at time.Time) {
	s.Require().NoError(s.store.AddConversationDocument(s.ctx, &ConversationDocument{
		ID: id, CustomerID: "app", OwnerUserID: owner, SessionID: sessionID, ConversationID: conversationID,
		ResponseID: turn, Name: id + ".pdf", Pages: 2, SHA256: "digest", Markdown: "text of " + id, CreatedAt: at,
	}))
}

// documentIDs are the ids of the documents in scope, in the order they are read.
func (s *StoreSuite) documentIDs(scope DocumentScope) []string {
	documents, err := s.store.ConversationDocuments(s.ctx, "app", scope)
	s.Require().NoError(err)
	ids := make([]string, 0, len(documents))
	for _, document := range documents {
		ids = append(ids, document.ID)
	}
	return ids
}

func (s *StoreSuite) TestADocumentIsOnlyReadForThePersonWhoSentIt() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "read this", s.base)
	s.attached("lease", "one", "", "jlahey", "first", s.base)

	s.Equal([]string{"lease"}, s.documentIDs(DocumentScope{SessionID: "one", Owner: "jlahey"}))
	s.Empty(s.documentIDs(DocumentScope{SessionID: "one", Owner: "rlafleur"}), "another person reads none")
	s.Empty(s.documentIDs(DocumentScope{SessionID: "one", Owner: ""}), "nor does a caller who names nobody")
}

func (s *StoreSuite) TestADocumentIsOnlyItsOwnCustomers() {
	s.opened("one", "app", s.base, nil)
	s.attached("lease", "one", "", "jlahey", "", s.base)

	documents, err := s.store.ConversationDocuments(s.ctx, "other-app", DocumentScope{SessionID: "one", Owner: "jlahey"})
	s.Require().NoError(err)
	s.Empty(documents)
}

func (s *StoreSuite) TestAConversationReadsTheDocumentsOfEverySessionThatHeldIt() {
	s.attached("first", "one", "agent:chat-1", "jlahey", "", s.base)
	s.attached("second", "two", "agent:chat-1", "jlahey", "", s.base.Add(time.Hour))
	s.attached("elsewhere", "three", "agent:chat-2", "jlahey", "", s.base)

	s.Equal([]string{"first", "second"}, s.documentIDs(DocumentScope{ConversationID: "agent:chat-1", SessionID: "two", Owner: "jlahey"}),
		"oldest first, from both sessions, and none of another conversation's")
}

func (s *StoreSuite) TestADocumentSentWithARewoundTurnIsNotRead() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "read this", s.base)
	s.responded("one", "second", "and this", s.base.Add(time.Hour))
	s.attached("kept", "one", "", "jlahey", "first", s.base)
	s.attached("taken-back", "one", "", "jlahey", "second", s.base.Add(time.Hour))

	s.Require().NoError(s.store.RewindResponses(s.ctx, "app", "one", "first", s.base.Add(2*time.Hour)))

	s.Equal([]string{"kept"}, s.documentIDs(DocumentScope{SessionID: "one", Owner: "jlahey"}))
}

func (s *StoreSuite) TestAForkStoppingAtATurnReadsOnlyTheDocumentsSentUpToIt() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "read this", s.base)
	s.responded("one", "second", "and this", s.base.Add(time.Hour))
	s.attached("carried", "one", "", "jlahey", "", s.base.Add(-time.Hour))
	// Written just after its turn's row, as a document is.
	s.attached("with-first", "one", "", "jlahey", "first", s.base.Add(time.Second))
	s.attached("with-second", "one", "", "jlahey", "second", s.base.Add(time.Hour+time.Second))

	s.Equal([]string{"carried", "with-first"}, s.documentIDs(DocumentScope{SessionID: "one", Owner: "jlahey", UntilResponse: "first"}))
}

func (s *StoreSuite) TestAForkStoppingAtATurnThatIsNotThereIsRefused() {
	_, err := s.store.ConversationDocuments(s.ctx, "app", DocumentScope{SessionID: "one", Owner: "jlahey", UntilResponse: "missing"})

	s.ErrorIs(err, ErrUnknownResponse)
}

func (s *StoreSuite) TestDeletingASessionDeletesItsDocuments() {
	s.opened("one", "app", s.base, nil)
	s.attached("lease", "one", "agent:chat-1", "jlahey", "", s.base)
	s.attached("other", "two", "agent:chat-1", "jlahey", "", s.base)

	s.Require().NoError(s.store.DeleteSession(s.ctx, "app", "one"))

	s.Equal([]string{"other"}, s.documentIDs(DocumentScope{ConversationID: "agent:chat-1", Owner: "jlahey"}),
		"the person's file goes with the session it was sent in")
}

func (s *StoreSuite) TestARegeneratedTurnKeepsACopyOfItsQuestionsDocuments() {
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "asked", "", "", s.base)
	s.attached("lease", "one", "", "jlahey", "asked", s.base.Add(time.Second))
	// A regenerate leaves the conversation at the replaced turn's parent, then answers again.
	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "", s.base.Add(time.Minute)))
	s.branchAnswered(s.store, "asked-again", "", "asked", s.base.Add(time.Minute))

	s.Require().NoError(s.store.CopyResponseDocuments(s.ctx, "app", "one", "asked", "asked-again"))

	documents, err := s.store.ConversationDocuments(s.ctx, "app", DocumentScope{SessionID: "one", Owner: "jlahey"})
	s.Require().NoError(err)
	s.Require().Len(documents, 1, "the replaced turn's own is off the conversation")
	copied := documents[0]
	s.NotEqual("lease", copied.ID, "a copy has an id of its own")
	s.Equal("asked-again", copied.ResponseID)
	s.Equal([]string{"jlahey", "lease.pdf", "digest", "text of lease"}, []string{copied.OwnerUserID, copied.Name, copied.SHA256, copied.Markdown})
	s.Equal(2, copied.Pages)
	s.True(copied.CreatedAt.Equal(s.base.Add(time.Second)), "attached when the original was")

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "asked", s.base.Add(2*time.Minute)))
	s.Equal([]string{"lease"}, s.documentIDs(DocumentScope{SessionID: "one", Owner: "jlahey"}),
		"activating the replaced turn finds its own, and the copy goes with the regenerated one")
}

func (s *StoreSuite) TestCopyingATurnsDocumentsLeavesAnotherCustomersAlone() {
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "asked", "", "", s.base)
	s.attached("lease", "one", "", "jlahey", "asked", s.base.Add(time.Second))
	s.Require().NoError(s.store.AddConversationDocument(s.ctx, &ConversationDocument{
		ID: "theirs", CustomerID: "other-app", OwnerUserID: "jlahey", SessionID: "one", ResponseID: "asked",
		Name: "theirs.pdf", SHA256: "digest", Markdown: "text of theirs", CreatedAt: s.base,
	}))

	s.Require().NoError(s.store.CopyResponseDocuments(s.ctx, "app", "one", "asked", "asked-again"))

	documents, err := s.store.ConversationDocuments(s.ctx, "app", DocumentScope{SessionID: "one", Owner: "jlahey"})
	s.Require().NoError(err)
	s.Len(documents, 2, "the original and its one copy")
	for _, document := range documents {
		s.Equal("lease.pdf", document.Name)
	}
	theirs, err := s.store.ConversationDocuments(s.ctx, "other-app", DocumentScope{SessionID: "one", Owner: "jlahey"})
	s.Require().NoError(err)
	s.Require().Len(theirs, 1, "no copy of another customer's file, and none of ours made into theirs")
	s.Equal("theirs", theirs[0].ID)
}

// The regenerated turn was answered by a router of the other shard: the copy is written there,
// beside the turn, so it drops out with it.
func (s *ShardedStoreSuite) TestACopyIsWrittenBesideTheTurnItWasCopiedTo() {
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "asked", "", "", s.base)
	s.attached("lease", "one", "", "jlahey", "asked", s.base.Add(time.Second))
	s.Require().NoError(s.other.SaveSession(s.ctx, &AgentSession{ID: "one", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", CreatedAt: s.base}))
	s.Require().NoError(s.other.MoveActivePath(s.ctx, "app", "one", "", s.base.Add(time.Minute)))
	s.branchAnswered(s.other, "asked-again", "", "asked", s.base.Add(time.Minute))

	s.Require().NoError(s.other.CopyResponseDocuments(s.ctx, "app", "one", "asked", "asked-again"))

	documents, err := s.store.ConversationDocuments(s.ctx, "app", DocumentScope{SessionID: "one", Owner: "jlahey"})
	s.Require().NoError(err)
	s.Require().Len(documents, 1)
	s.Equal("asked-again", documents[0].ResponseID)
	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "asked", s.base.Add(2*time.Minute)))
	s.Equal([]string{"lease"}, s.documentIDs(DocumentScope{SessionID: "one", Owner: "jlahey"}),
		"the copy is rewound with the turn beside it")
}
