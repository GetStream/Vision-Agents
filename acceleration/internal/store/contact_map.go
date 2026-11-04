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
	// ContactStreamUser is a Stream user of the app chat in one Stream app, by the user id
	// Stream signed, «<app pk>:<user id>» (StreamUserAddress, 20261212090000_chat_episodes.sql).
	ContactStreamUser = "stream_user"
)

// ContactMapEntry is one address a person is reached at, for one agent of a customer, and the
// omni-channel their episode cards go to (20261007042000_contact_map.sql).
type ContactMapEntry struct {
	bun.BaseModel `bun:"table:contact_map,alias:cm"`

	ID            string `bun:"id,pk"`
	CustomerID    string `bun:"customer_id,notnull"`
	AgentConfigID string `bun:"agent_config_id,notnull"`
	// Kind is ContactPhone, ContactSlack or ContactStreamUser, and Address the number, the Slack
	// user or the Stream user in its app; retired:<id> for a number's row a link set aside
	// (retireHistory).
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
// one address at once make one row: the unique index on the address decides which. A row made
// for a linked Stream user or number takes the omni-channel and pin of its partner's row under
// the same agent, the older (joinPartner).
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
	created = entry.ConversationID == proposed
	if !created {
		return false, nil
	}
	return true, s.joinPartner(ctx, entry)
}

// joinPartner points a row just made at its linked partner's omni-channel of the same agent,
// when the customer linked its Stream user or its number (LinkUserPhone), and reads the row
// back. Read after the row is committed, so a link made at the same time is either seen here
// or sees the row. Only a new row: a row that was there when the link was made was joined
// then, so a message on a known row costs no extra query.
func (s *Store) joinPartner(ctx context.Context, entry *ContactMapEntry) error {
	link, linked, err := s.linkOf(ctx, entry)
	if err != nil || !linked {
		return err
	}
	if err := s.joinLinked(ctx, link); err != nil {
		return err
	}
	if err := s.db.NewSelect().Model(entry).WherePK().Scan(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: read a joined contact: %w", err))
	}
	return nil
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
