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

// States of a queued Stream Chat message (ChatInbound.State).
const (
	ChatInboundQueued = "queued"
	ChatInboundDone   = "done"
	ChatInboundFailed = "failed"
)

// chatInboundDoneKeep is how long an answered message's row is kept, so a delivery Stream
// makes again queues nothing. It is channelMessageKeep, the span thread channels already drop
// a message's second delivery in.
const chatInboundDoneKeep = channelMessageKeep

// chatInboundFailedKeep is how long the row of a message whose retries ran out is kept, for
// someone to find why it was never answered. 30 days is a choice, not a measurement, the same
// as channelInboundFailedKeep.
const chatInboundFailedKeep = 30 * 24 * time.Hour

// ChatInbound is one Stream Chat message queued to be answered
// (20261204120000_chat_inbound.sql), with where its delivery came from.
type ChatInbound struct {
	bun.BaseModel `bun:"table:chat_inbound,alias:cd"`

	// Seq is the order the message was queued in.
	Seq int64 `bun:"seq,pk"`
	// StreamApp and MessageID are the Stream app and its id for the message. The app is always
	// known, as the internal endpoint reads it from the signed body.
	StreamApp int64  `bun:"stream_app,notnull"`
	MessageID string `bun:"message_id,notnull"`
	// CustomerID, Deployment and APIKey are, with StreamApp, where the delivery came from as
	// the router verified it.
	CustomerID string `bun:"customer_id,notnull"`
	Deployment bool   `bun:"deployment,notnull"`
	APIKey     string `bun:"api_key,notnull"`
	// Body is the verified event as Stream sent it.
	Body  []byte `bun:"body,notnull"`
	State string `bun:"state,notnull"`
	// Attempts is how many answers of the message failed.
	Attempts int `bun:"attempts,notnull"`
	// NextAttemptAt is when the message is due; on a claimed row, the end of the claim's lease.
	NextAttemptAt time.Time  `bun:"next_attempt_at,notnull"`
	ReceivedAt    time.Time  `bun:"received_at,notnull"`
	FinishedAt    *time.Time `bun:"finished_at"`
}

// chatInboundColumns are the columns a ChatInbound is read from, named rather than taken
// with *, so a router older than a migration that adds one still reads the rows.
const chatInboundColumns = "cd.seq, cd.stream_app, cd.message_id, cd.customer_id, cd.deployment, cd.api_key, cd.body, " +
	"cd.state, cd.attempts, cd.next_attempt_at, cd.received_at, cd.finished_at"

// QueueChatInbound queues delivery to be answered, due at delivery.NextAttemptAt. A message
// already queued under the same app and id, answered or given up within its keep, is not
// queued again: queued is false.
func (s *Store) QueueChatInbound(ctx context.Context, delivery ChatInbound) (queued bool, err error) {
	return queueChatInbound(ctx, s.db, delivery)
}

// queueChatInbound is QueueChatInbound on db, the store's or a transaction's.
func queueChatInbound(ctx context.Context, db bun.IDB, delivery ChatInbound) (bool, error) {
	if delivery.MessageID == "" || len(delivery.Body) == 0 {
		return false, stack.Wrap(errors.New("store: a message id and a body are required"))
	}
	if delivery.StreamApp == 0 {
		return false, stack.Wrap(errors.New("store: a Stream app is required: a message queued without one would be queued again once the app is known"))
	}
	result, err := db.NewRaw(`
INSERT INTO chat_inbound (stream_app, message_id, customer_id, deployment, api_key, body, next_attempt_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (stream_app, message_id) DO NOTHING`,
		delivery.StreamApp, delivery.MessageID, delivery.CustomerID, delivery.Deployment, delivery.APIKey, delivery.Body,
		delivery.NextAttemptAt.UTC()).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue chat inbound: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue chat inbound: %w", err))
	}
	return affected == 1, nil
}

// ClaimChatInbound takes at most limit messages due at now, oldest due first, and moves
// each one's next attempt to leaseUntil, so no other worker takes it while this one answers
// it. Rows another worker is claiming at the same moment are skipped, not waited for.
func (s *Store) ClaimChatInbound(ctx context.Context, now time.Time, limit int, leaseUntil time.Time) ([]ChatInbound, error) {
	return s.claimChatInbound(ctx, claimChatInboundQuery, now, limit, leaseUntil)
}

// claimChatInboundQuery is ClaimChatInbound' statement.
//
// due checks state and next_attempt_at on the row it locks. When another router's claim, or
// its finish, commits after the statement started, Postgres locks the row's newer version and
// checks due's WHERE on it again («the second updater ... re-evaluates its WHERE condition»:
// Read Committed Isolation Level, https://www.postgresql.org/docs/16/transaction-iso.html), so
// a row another router just leased or answered is skipped.
const claimChatInboundQuery = `
WITH due AS (
    SELECT cd.seq FROM chat_inbound AS cd
    WHERE cd.state = 'queued' AND cd.next_attempt_at <= ?
    ORDER BY cd.next_attempt_at, cd.seq
    LIMIT ?
    FOR UPDATE OF cd SKIP LOCKED
)
UPDATE chat_inbound AS cd
SET next_attempt_at = ?
FROM due
WHERE cd.seq = due.seq
RETURNING ` + chatInboundColumns

// claimChatInbound runs query, ClaimChatInbound' statement or a test's variant of it
// with a pause in it, with ClaimChatInbound' arguments.
func (s *Store) claimChatInbound(ctx context.Context, query string, now time.Time, limit int, leaseUntil time.Time) ([]ChatInbound, error) {
	var claimed []ChatInbound
	if limit < 1 {
		return claimed, nil
	}
	if err := s.db.NewRaw(query, now.UTC(), limit, leaseUntil.UTC()).Scan(ctx, &claimed); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim chat inbound: %w", err))
	}
	return claimed, nil
}

// NextChatInboundAt is when the queued message due first is due, leased ones at the end of
// their lease. found is false when no message is queued at all.
func (s *Store) NextChatInboundAt(ctx context.Context) (next time.Time, found bool, err error) {
	var at sql.NullTime
	if err := s.db.NewRaw("SELECT min(next_attempt_at) FROM chat_inbound WHERE state = 'queued'").Scan(ctx, &at); err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next chat inbound: %w", err))
	}
	return at.Time, at.Valid, nil
}

// FinishChatInbound records that a claimed message was answered, at at.
//
// It, RetryChatInbound and FailChatInbound change the row only while the claim's lease is
// the row's: a worker whose lease ran out while it answered, and whose message another worker
// took since, changes nothing, so it cannot finish or reschedule the other's attempt. changed
// is false then. A row the answer settled as it started (StartChatInboundAnswer) keeps its
// lease and when it left the queue, so it is still this claim's to finish.
func (s *Store) FinishChatInbound(ctx context.Context, claimed ChatInbound, at time.Time) (changed bool, err error) {
	return updateClaimedChatInbound(ctx, claimed, []string{ChatInboundQueued, ChatInboundDone},
		s.db.NewUpdate().Model((*ChatInbound)(nil)).
			Set("state = ?", ChatInboundDone).
			Set("attempts = ?", claimed.Attempts).
			Set("finished_at = COALESCE(finished_at, ?)", at.UTC()))
}

// FailChatInbound records that a claimed message's last attempt failed and it is given up,
// at at.
func (s *Store) FailChatInbound(ctx context.Context, claimed ChatInbound, at time.Time) (changed bool, err error) {
	return updateClaimedChatInbound(ctx, claimed, []string{ChatInboundQueued}, s.db.NewUpdate().Model((*ChatInbound)(nil)).
		Set("state = ?", ChatInboundFailed).
		Set("attempts = ?", claimed.Attempts+1).
		Set("finished_at = ?", at.UTC()))
}

// RetryChatInbound records a failed attempt of a claimed message and when the next one is
// due.
func (s *Store) RetryChatInbound(ctx context.Context, claimed ChatInbound, next time.Time) (changed bool, err error) {
	return updateClaimedChatInbound(ctx, claimed, []string{ChatInboundQueued}, s.db.NewUpdate().Model((*ChatInbound)(nil)).
		Set("attempts = ?", claimed.Attempts+1).
		Set("next_attempt_at = ?", next.UTC()))
}

func updateClaimedChatInbound(ctx context.Context, claimed ChatInbound, states []string, update *bun.UpdateQuery) (bool, error) {
	result, err := update.
		Where("seq = ?", claimed.Seq).
		Where("state IN (?)", bun.In(states)).
		Where("next_attempt_at = ?", claimed.NextAttemptAt.UTC()).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: update chat inbound: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: update chat inbound: %w", err))
	}
	return affected == 1, nil
}

// StartChatInboundAnswer settles a queued message as its answer starts, at at, so no later try
// of it, on any router, answers it again: a peer that answered and whose reply was lost, an
// attempt Close cut short after the peer took it, or a router that stopped before it recorded
// the answer. started is false when the message's answer already started, it was given up, or
// nothing is queued under streamApp and messageID: there is nothing to answer then. The lease
// is left as it is, for the worker holding it to finish the row (FinishChatInbound).
func (s *Store) StartChatInboundAnswer(ctx context.Context, streamApp int64, messageID string, at time.Time) (started bool, err error) {
	result, err := s.db.NewRaw(`
UPDATE chat_inbound SET state = 'done', finished_at = ?
WHERE stream_app = ? AND message_id = ? AND state = 'queued'`,
		at.UTC(), streamApp, messageID).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: start chat inbound answer: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: start chat inbound answer: %w", err))
	}
	return affected == 1, nil
}

// ReleaseChatInboundAnswer queues again a message whose answer started (StartChatInboundAnswer)
// and that nothing then took, such as a dispatch worker that was not connected, so a later try
// answers it. The lease is left as it is, for the worker holding it to retry the row.
func (s *Store) ReleaseChatInboundAnswer(ctx context.Context, streamApp int64, messageID string) error {
	if _, err := s.db.NewRaw(`
UPDATE chat_inbound SET state = 'queued', finished_at = NULL
WHERE stream_app = ? AND message_id = ? AND state = 'done'`,
		streamApp, messageID).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: release chat inbound answer: %w", err))
	}
	return nil
}

// PruneChatInbound deletes the rows of answered messages finished a day before now, and of
// given-up ones 30 days before, and returns how many it deleted.
func (s *Store) PruneChatInbound(ctx context.Context, now time.Time) (int, error) {
	result, err := s.db.NewRaw(`
DELETE FROM chat_inbound
WHERE (state = 'done' AND finished_at < ?) OR (state = 'failed' AND finished_at < ?)`,
		now.Add(-chatInboundDoneKeep).UTC(), now.Add(-chatInboundFailedKeep).UTC()).Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: prune chat inbound: %w", err))
	}
	pruned, _ := result.RowsAffected()
	return int(pruned), nil
}
