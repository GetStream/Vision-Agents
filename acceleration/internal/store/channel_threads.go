package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoChannelThread says no external thread is linked to a thread channel: the channel is a
// plain Stream Chat conversation, which the channel bridge sends nothing from.
var ErrNoChannelThread = errors.New("store: no external thread is linked to this channel")

// channelMessageKeep is how long an inbound message's id is kept to drop a retried delivery of
// it. Slack's last retry comes 5 minutes after the first delivery
// (https://docs.slack.dev/apis/events-api/, «Retries»); a day is that with a wide margin, a
// choice rather than a measurement. Other providers' retry spans are unverified.
const channelMessageKeep = 24 * time.Hour

// channelWaitingKeep is how long a reply waits for the message that links its thread
// (WaitChannelThreadMessage). Slack's last retry of an event comes about 6 minutes after its
// first delivery: one «nearly immediately», then after 1 minute and after 5 minutes
// (https://docs.slack.dev/apis/events-api/, «Retries»). 10 minutes is that with a margin, a
// choice rather than a measurement. Other providers' retry spans are unverified.
const channelWaitingKeep = 10 * time.Minute

// ChannelThread links one external thread, such as a Slack thread, to the thread channel the
// channel bridge writes it into (20261007003100_channel_threads.sql).
type ChannelThread struct {
	bun.BaseModel `bun:"table:channel_threads,alias:ct"`

	// ChannelID is the thread channel's id in the agent channel type.
	ChannelID      string `bun:"channel_id,pk"`
	CustomerID     string `bun:"customer_id,notnull"`
	ConnectorID    string `bun:"connector_id,notnull"`
	ProviderUnitID string `bun:"provider_unit_id,notnull"`
	ThreadKey      string `bun:"thread_key,notnull"`
	// ConnectionID is the app-owned connection replies are sent with.
	ConnectionID string `bun:"connection_id,notnull"`
	// ThreadParts are the thread key's named parts, which a reply names as {thread.<name>}.
	ThreadParts map[string]string `bun:"thread_parts,type:jsonb,notnull"`
	// StreamAppPK is the Stream app the thread channel is in; zero, stored as NULL, is the
	// deployment's own.
	StreamAppPK   int64     `bun:"stream_app_pk,nullzero"`
	LastInboundAt time.Time `bun:"last_inbound_at,notnull"`
	CreatedAt     time.Time `bun:"created_at,notnull"`
	// TurnHolder and TurnUntil are the lease on the thread's running turn
	// (TakeChannelThreadTurn); empty and nil while none runs.
	TurnHolder string     `bun:"turn_holder,nullzero"`
	TurnUntil  *time.Time `bun:"turn_until"`
}

// TakeChannelThreadTurn leases a thread channel's one running turn to holder until until,
// when no turn holds it or the one that did ran out. False is a turn another holder has: two
// routers answering one thread at once would talk over each other, and each one's session
// would miss what the other said. One statement, so two routers asking at once get one
// lease between them.
func (s *Store) TakeChannelThreadTurn(ctx context.Context, channelID, holder string, until time.Time) (bool, error) {
	if channelID == "" || holder == "" {
		return false, stack.Wrap(errors.New("store: a channel and a holder are required"))
	}
	result, err := s.db.NewUpdate().Model((*ChannelThread)(nil)).
		Set("turn_holder = ?", holder).
		Set("turn_until = ?", until.UTC()).
		Where("channel_id = ?", channelID).
		Where("(turn_until IS NULL OR turn_until < now() OR turn_holder = ?)", holder).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: take channel thread turn: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: take channel thread turn: %w", err))
	}
	return affected == 1, nil
}

// ReleaseChannelThreadTurn ends holder's lease on a thread channel's turn, so the next turn
// need not wait for it to run out. A lease another holder took since is left alone.
func (s *Store) ReleaseChannelThreadTurn(ctx context.Context, channelID, holder string) error {
	_, err := s.db.NewUpdate().Model((*ChannelThread)(nil)).
		Set("turn_holder = NULL").
		Set("turn_until = NULL").
		Where("channel_id = ?", channelID).
		Where("turn_holder = ?", holder).
		Exec(ctx)
	return stack.Wrap(err)
}

// ReleaseChannelThreadMessage forgets a claim, so the step that took it can take it again:
// a reply whose send failed is sent on a later attempt.
func (s *Store) ReleaseChannelThreadMessage(ctx context.Context, channelID, kind, messageID string) error {
	_, err := s.db.NewRaw("DELETE FROM channel_thread_messages WHERE channel_id = ? AND kind = ? AND message_id = ?",
		channelID, kind, messageID).Exec(ctx)
	return stack.Wrap(err)
}

// LinkChannelThread returns the thread channel of the external thread thread names, linking it
// to thread.ChannelID when the thread has none yet. created is whether it did. An existing
// link keeps its channel and pin, takes thread's connection, which the latest message names,
// and records the message as the last inbound one. Two first messages of one thread at once
// link one channel: the unique index on the thread decides which.
func (s *Store) LinkChannelThread(ctx context.Context, thread *ChannelThread) (created bool, err error) {
	if thread.ChannelID == "" || thread.CustomerID == "" || thread.ConnectorID == "" || thread.ThreadKey == "" || thread.ConnectionID == "" {
		return false, stack.Wrap(errors.New("store: a channel, a customer, a connector, a thread key and a connection are required"))
	}
	proposed := thread.ChannelID
	now := time.Now().UTC().Truncate(time.Microsecond)
	thread.LastInboundAt, thread.CreatedAt = now, now
	if thread.ThreadParts == nil {
		thread.ThreadParts = map[string]string{}
	}
	_, err = s.db.NewInsert().Model(thread).
		On("CONFLICT (customer_id, connector_id, provider_unit_id, thread_key) DO UPDATE").
		Set("connection_id = EXCLUDED.connection_id").
		Set("last_inbound_at = EXCLUDED.last_inbound_at").
		Returning("*").
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: link channel thread: %w", err))
	}
	return thread.ChannelID == proposed, nil
}

// ChannelThreadLinked is whether an external thread of a customer's provider unit is linked to
// a thread channel already.
func (s *Store) ChannelThreadLinked(ctx context.Context, customerID, connectorID, providerUnitID, threadKey string) (bool, error) {
	linked, err := s.db.NewSelect().Model((*ChannelThread)(nil)).
		Where("customer_id = ? AND connector_id = ? AND provider_unit_id = ? AND thread_key = ?",
			customerID, connectorID, providerUnitID, threadKey).
		Exists(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: channel thread linked: %w", err))
	}
	return linked, nil
}

// ChannelThread returns the external thread linked to a thread channel, or ErrNoChannelThread.
func (s *Store) ChannelThread(ctx context.Context, channelID string) (ChannelThread, error) {
	if channelID == "" {
		return ChannelThread{}, stack.Wrap(errors.New("store: a channel id is required"))
	}
	var thread ChannelThread
	err := s.db.NewSelect().Model(&thread).Where("channel_id = ?", channelID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelThread{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoChannelThread, channelID))
	}
	if err != nil {
		return ChannelThread{}, stack.Wrap(fmt.Errorf("store: channel thread: %w", err))
	}
	return thread, nil
}

// waitingMessage is one row of channel_thread_waiting (20261011220000_channel_thread_waiting.sql).
type waitingMessage struct {
	ConnectorID       string    `bun:"connector_id"`
	ProviderUnitID    string    `bun:"provider_unit_id"`
	ThreadKey         string    `bun:"thread_key"`
	ProviderMessageID string    `bun:"provider_message_id"`
	AuthorID          string    `bun:"author_id"`
	Text              string    `bun:"text"`
	Raw               []byte    `bun:"raw"`
	CreatedAt         time.Time `bun:"created_at"`
}

// WaitChannelThreadMessage keeps a customer's message on an external thread no thread channel
// is linked to, for the message that links the thread to take
// (TakeWaitingChannelThreadMessages). False is one already waiting, delivered again. Rows past
// channelWaitingKeep are dropped first, so the table holds minutes of them.
func (s *Store) WaitChannelThreadMessage(ctx context.Context, customerID string, message core.InboundMessage) (bool, error) {
	if customerID == "" || message.ConnectorID == "" || message.ThreadKey == "" || message.ProviderMessageID == "" {
		return false, stack.Wrap(errors.New("store: a customer, a connector, a thread key and a message id are required"))
	}
	now := time.Now().UTC()
	if _, err := s.db.NewRaw("DELETE FROM channel_thread_waiting WHERE created_at < ?", now.Add(-channelWaitingKeep)).Exec(ctx); err != nil {
		return false, stack.Wrap(fmt.Errorf("store: wait channel thread message: %w", err))
	}
	result, err := s.db.NewRaw(
		"INSERT INTO channel_thread_waiting (customer_id, connector_id, provider_unit_id, thread_key, provider_message_id, author_id, text, raw, created_at) "+
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
		customerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey, message.ProviderMessageID,
		message.AuthorID, message.Text, message.Raw, now).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: wait channel thread message: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: wait channel thread message: %w", err))
	}
	return affected == 1, nil
}

// UnwaitChannelThreadMessage takes one waiting message back. False is one that is not waiting:
// the message that linked its thread took it first, or it was never kept.
func (s *Store) UnwaitChannelThreadMessage(ctx context.Context, customerID string, message core.InboundMessage) (bool, error) {
	result, err := s.db.NewRaw(
		"DELETE FROM channel_thread_waiting WHERE customer_id = ? AND connector_id = ? AND provider_unit_id = ? AND thread_key = ? AND provider_message_id = ?",
		customerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey, message.ProviderMessageID).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: unwait channel thread message: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: unwait channel thread message: %w", err))
	}
	return affected == 1, nil
}

// TakeWaitingChannelThreadMessages takes every message waiting on a customer's external thread
// that is not past channelWaitingKeep, oldest first. Each is taken once: a message that takes
// itself back at the same time (UnwaitChannelThreadMessage) gets it, or this does.
func (s *Store) TakeWaitingChannelThreadMessages(ctx context.Context, customerID, connectorID, providerUnitID, threadKey string) ([]core.InboundMessage, error) {
	var rows []waitingMessage
	err := s.db.NewRaw(
		"DELETE FROM channel_thread_waiting WHERE customer_id = ? AND connector_id = ? AND provider_unit_id = ? AND thread_key = ? AND created_at >= ? "+
			"RETURNING connector_id, provider_unit_id, thread_key, provider_message_id, author_id, text, raw, created_at",
		customerID, connectorID, providerUnitID, threadKey, time.Now().UTC().Add(-channelWaitingKeep)).Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: take waiting channel thread messages: %w", err))
	}
	slices.SortStableFunc(rows, func(a, b waitingMessage) int { return a.CreatedAt.Compare(b.CreatedAt) })
	messages := make([]core.InboundMessage, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, core.InboundMessage{
			ConnectorID: row.ConnectorID, ProviderUnitID: row.ProviderUnitID, ThreadKey: row.ThreadKey,
			AuthorID: row.AuthorID, Text: row.Text, ProviderMessageID: row.ProviderMessageID, Raw: row.Raw,
		})
	}
	return messages, nil
}

// What a claimed message of a thread channel is, each with its own ids
// (20261007003100_channel_threads.sql).
const (
	// ClaimInbound is a provider's message the channel bridge took, by the provider's id.
	ClaimInbound = "inbound"
	// ClaimTurn is a person's message the message hook handed to the session, by its Stream
	// Chat id.
	ClaimTurn = "turn"
	// ClaimReply is an agent's reply the bridge sent to the external thread, by its Stream
	// Chat id.
	ClaimReply = "reply"
)

// ClaimChannelThreadMessage records a message of a thread channel as acted on by one step,
// kind, by that step's id for it. False is one already taken: delivered or told again. Ids
// kept longer than channelMessageKeep are dropped first, so the table holds a day of them.
func (s *Store) ClaimChannelThreadMessage(ctx context.Context, channelID, kind, messageID string) (bool, error) {
	if channelID == "" || kind == "" || messageID == "" {
		return false, stack.Wrap(errors.New("store: a channel, a kind and a message id are required"))
	}
	now := time.Now().UTC()
	if _, err := s.db.NewRaw(
		"DELETE FROM channel_thread_messages WHERE channel_id = ? AND created_at < ?",
		channelID, now.Add(-channelMessageKeep)).Exec(ctx); err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim channel thread message: %w", err))
	}
	result, err := s.db.NewRaw(
		"INSERT INTO channel_thread_messages (channel_id, kind, message_id, created_at) "+
			"VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
		channelID, kind, messageID, now).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim channel thread message: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim channel thread message: %w", err))
	}
	return affected == 1, nil
}

// AppConnectionByAccount returns the customer's live app-owned connection of a connector whose
// account id is accountID: the connection of the provider unit a provider app's event names,
// such as the bot token of the Slack workspace the app is installed in, whose identity is the
// team alone.
func (s *Store) AppConnectionByAccount(ctx context.Context, customerID, connectorID, accountID string) (ConnectorConnection, error) {
	if customerID == "" || connectorID == "" || accountID == "" {
		return ConnectorConnection{}, stack.Wrap(errors.New("store: a customer, a connector and an account id are required"))
	}
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID).
		Where("owner_type = ?", OwnerApp).
		Where("account_id = ?", accountID).
		Where("deleted_at IS NULL").
		Order("created_at DESC", "id DESC").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("%w: %s account %s", ErrNoConnectorConnection, connectorID, accountID))
	}
	if err != nil {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("store: app connection by account: %w", err))
	}
	return connection, nil
}

// bindsFixedConnection is the WHERE clause that finds the agent configs that bind a connection
// as their fixed connection, matched as boundByConfig matches them.
const bindsFixedConnection = "connectors @> jsonb_build_array(jsonb_build_object('connection', jsonb_build_object('type', 'fixed', 'connection_id', ?::text)))"

// AgentConfigsBindingConnection returns the customer's live agent configs that bind the
// connection as their fixed connection, oldest first: the agents that answer on the provider
// unit the connection is, matched as boundByConfig matches them. A test copy (DraftOfTag) is
// left out: it answers no channel message and subscribes to no event (AI-1049, AI-1048). The
// first is the connection's owner, which refuseSecondChannelAgent keeps the only one on a
// channel connection.
func (s *Store) AgentConfigsBindingConnection(ctx context.Context, customerID, connectionID string) ([]AgentConfig, error) {
	if customerID == "" || connectionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	var configs []AgentConfig
	err := s.db.NewSelect().Model(&configs).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Where(notTestCopy).
		Where(bindsFixedConnection, connectionID).
		Order("created_at", "id").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: agent configs binding connection: %w", err))
	}
	return configs, nil
}
