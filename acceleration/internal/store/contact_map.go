package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// How a person is known to the contact map (20261007042000_contact_map.sql).
const (
	// ContactPhone is an E.164 number: a call, an SMS, a WhatsApp message or an iMessage.
	ContactPhone = "phone"
	// ContactSlack is a Slack user in one workspace, «<team id>:<user id>».
	ContactSlack = "slack"
)

// ContactMapEntry is one address a person is reached at, for one agent of a customer, and the
// omni-channel their episode cards go to (20261007042000_contact_map.sql).
type ContactMapEntry struct {
	bun.BaseModel `bun:"table:contact_map,alias:cm"`

	ID            string `bun:"id,pk"`
	CustomerID    string `bun:"customer_id,notnull"`
	AgentConfigID string `bun:"agent_config_id,notnull"`
	// Kind is ContactPhone or ContactSlack, and Address the number or the Slack user.
	Kind    string `bun:"kind,notnull"`
	Address string `bun:"address,notnull"`
	// ConversationID is the omni-channel's cid, agent:omni-<uuid>.
	ConversationID string `bun:"conversation_id,notnull"`
	// UserID is the end user the address is linked to; empty until linked.
	UserID string `bun:"user_id,nullzero"`
	// StreamAppPK is the Stream app the omni-channel is in; zero, stored as NULL, is the
	// deployment's own.
	StreamAppPK int64     `bun:"stream_app_pk,nullzero"`
	CreatedAt   time.Time `bun:"created_at,notnull"`
	UpdatedAt   time.Time `bun:"updated_at,notnull"`
}

// MapContact returns the contact map row of the address entry names, making it with
// entry.ConversationID and entry.StreamAppPK when the address has none yet. created is
// whether it did. An existing row keeps its omni-channel and its pin. Two first sightings of
// one address at once make one row: the unique index on the address decides which.
func (s *Store) MapContact(ctx context.Context, entry *ContactMapEntry) (created bool, err error) {
	if entry.CustomerID == "" || entry.AgentConfigID == "" || entry.Kind == "" || entry.Address == "" || entry.ConversationID == "" {
		return false, stack.Wrap(errors.New("store: a customer, an agent config, a kind, an address and a conversation are required"))
	}
	proposed := entry.ConversationID
	now := time.Now().UTC().Truncate(time.Microsecond)
	entry.CreatedAt, entry.UpdatedAt = now, now
	if entry.ID == "" {
		entry.ID = newID()
	}
	// The update is there only so RETURNING hands back the row that won: DO NOTHING returns
	// none.
	_, err = s.db.NewInsert().Model(entry).
		On("CONFLICT (customer_id, agent_config_id, kind, address) DO UPDATE").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("*").
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: map contact: %w", err))
	}
	return entry.ConversationID == proposed, nil
}

// ErrNoContact says the contact map holds no row for a person: the agent has had no episode
// with them, so they have no omni-channel and no cards.
var ErrNoContact = errors.New("store: no contact map row for this person")

// Contact is the contact map row of an address, for one agent of a customer. Unlike
// MapContact it never makes one: reading a person's cards writes nothing.
func (s *Store) Contact(ctx context.Context, customerID, agentConfigID, kind, address string) (ContactMapEntry, error) {
	var entry ContactMapEntry
	err := s.db.NewSelect().Model(&entry).
		Where("customer_id = ?", customerID).Where("agent_config_id = ?", agentConfigID).
		Where("kind = ?", kind).Where("address = ?", address).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ContactMapEntry{}, stack.Wrap(ErrNoContact)
	}
	if err != nil {
		return ContactMapEntry{}, stack.Wrap(fmt.Errorf("store: contact: %w", err))
	}
	return entry, nil
}

// ThreadContact is the contact map row of the person a thread channel's episodes are with,
// for one agent of a customer: the one who started the thread, since OpenEpisode keeps the
// first message's episode for the later ones. The newest episode of the thread decides. A
// thread with no episode, or only one of another customer or agent, has none.
func (s *Store) ThreadContact(ctx context.Context, customerID, agentConfigID, threadChannel string) (ContactMapEntry, error) {
	var entry ContactMapEntry
	err := s.db.NewSelect().Model(&entry).
		Join("JOIN episodes AS ep ON ep.contact_id = cm.id").
		Where("ep.customer_id = ?", customerID).Where("ep.thread_channel = ?", threadChannel).
		Where("ep.session_id IS NULL").
		Where("cm.customer_id = ?", customerID).Where("cm.agent_config_id = ?", agentConfigID).
		OrderExpr("ep.started_at DESC").Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ContactMapEntry{}, stack.Wrap(ErrNoContact)
	}
	if err != nil {
		return ContactMapEntry{}, stack.Wrap(fmt.Errorf("store: thread contact: %w", err))
	}
	return entry, nil
}
