package store

import (
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ConversationDocument is a file a person attached to a conversation of theirs, read as text.
type ConversationDocument struct {
	bun.BaseModel `bun:"table:conversation_documents,alias:cd"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// OwnerUserID is the person whose conversation it is.
	OwnerUserID string `bun:"owner_user_id,notnull"`
	SessionID   string `bun:"session_id,notnull"`
	// ConversationID is the Chat channel the conversation is kept in, empty for none.
	ConversationID string `bun:"conversation_id,notnull"`
	// ResponseID is the turn it was sent with, empty for one carried into a fork.
	ResponseID string    `bun:"response_id,notnull"`
	Name       string    `bun:"name,notnull"`
	Pages      int       `bun:"pages,notnull"`
	SHA256     string    `bun:"sha256,notnull"`
	Markdown   string    `bun:"markdown,notnull"`
	CreatedAt  time.Time `bun:"created_at,notnull"`
}

// AddConversationDocument keeps a file attached to a conversation. A retried write of the same
// id is not an error.
func (s *Store) AddConversationDocument(ctx context.Context, document *ConversationDocument) error {
	if document.ID == "" || document.CustomerID == "" || document.SessionID == "" {
		return stack.Wrap(errors.New("store: a document, a customer and a session id are required"))
	}
	if document.CreatedAt.IsZero() {
		document.CreatedAt = time.Now().UTC()
	}
	_, err := s.shard.NewInsert().Model(document).On("CONFLICT (id) DO NOTHING").Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: add conversation document: %w", err))
	}
	return nil
}

// DocumentScope is the conversation whose documents are read, and whose they must be.
type DocumentScope struct {
	// ConversationID reads a conversation kept in Chat, across every session that held it.
	// Empty reads SessionID's alone.
	ConversationID string
	SessionID      string
	// Owner is the person reading. A document of anybody else's is never read.
	Owner string
	// UntilResponse leaves out documents sent with turns after this one, as a fork that stops
	// there carries none of them. Empty is all of them.
	UntilResponse string
}

// ConversationDocuments are the documents in scope, oldest first, but for those sent with a
// turn since rewound.
func (s *Store) ConversationDocuments(ctx context.Context, customerID string, scope DocumentScope) ([]ConversationDocument, error) {
	if customerID == "" || scope.ConversationID == "" && scope.SessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a conversation or session id are required"))
	}
	// The turn a fork stops at may be on another shard than the documents, so when it was is
	// read first rather than joined in each shard.
	var until time.Time
	if scope.UntilResponse != "" {
		stopped, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (AgentResponse, error) {
			var response AgentResponse
			err := db.NewSelect().Model(&response).Column("created_at").
				Where("id = ?", scope.UntilResponse).Where("customer_id = ?", customerID).Scan(ctx)
			return response, err
		})
		if errors.Is(err, sql.ErrNoRows) {
			return nil, stack.Wrap(ErrUnknownResponse)
		}
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("store: conversation documents: %w", err))
		}
		until = stopped.CreatedAt
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]ConversationDocument, error) {
		query := db.NewSelect().Model((*ConversationDocument)(nil)).
			Where("cd.customer_id = ?", customerID).
			Where("cd.owner_user_id = ?", scope.Owner).
			Where("NOT EXISTS (SELECT 1 FROM agent_responses AS r WHERE r.id = cd.response_id AND r.rewound_at IS NOT NULL)").
			Order("cd.created_at ASC", "cd.id ASC")
		if scope.ConversationID != "" {
			query = query.Where("cd.conversation_id = ?", scope.ConversationID)
		} else {
			query = query.Where("cd.session_id = ?", scope.SessionID)
		}
		// A document is written by the router that took its turn, beside the turn's row.
		if !until.IsZero() {
			query = query.Where("cd.response_id = '' OR EXISTS (SELECT 1 FROM agent_responses AS sent "+
				"WHERE sent.id = cd.response_id AND sent.created_at <= ?)", until)
		}
		var documents []ConversationDocument
		err := query.Scan(ctx, &documents)
		return documents, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: conversation documents: %w", err))
	}
	var documents []ConversationDocument
	for _, list := range lists {
		documents = append(documents, list...)
	}
	sortDocuments(documents)
	return documents, nil
}

// sortDocuments puts documents read from several shards back in the order they were attached.
func sortDocuments(documents []ConversationDocument) {
	slices.SortFunc(documents, func(a, b ConversationDocument) int {
		return cmp.Or(a.CreatedAt.Compare(b.CreatedAt), cmp.Compare(a.ID, b.ID))
	})
}
