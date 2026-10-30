package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// States of a queued inbound message (ChannelInbound.State).
const (
	ChannelInboundQueued = "queued"
	ChannelInboundDone   = "done"
	ChannelInboundFailed = "failed"
)

// channelInboundDoneKeep is how long a written message's row is kept, so a provider's retry
// of it queues nothing. It is channelMessageKeep, the span the bridge already drops retries
// in.
const channelInboundDoneKeep = channelMessageKeep

// channelInboundFailedKeep is how long the row of a message whose retries ran out is kept,
// for someone to find why it was never answered. 30 days is a choice, not a measurement.
const channelInboundFailedKeep = 30 * 24 * time.Hour

// ChannelInbound is one inbound message queued to be written into its thread channel
// (20261205120000_channel_thread_inbound.sql), with what writing it and opening its episode
// card take.
type ChannelInbound struct {
	bun.BaseModel `bun:"table:channel_thread_inbound,alias:cti"`

	// Seq is the order the message was queued in.
	Seq               int64  `bun:"seq,pk"`
	ChannelID         string `bun:"channel_id,notnull"`
	ProviderMessageID string `bun:"provider_message_id,notnull"`
	CustomerID        string `bun:"customer_id,notnull"`
	ConnectorID       string `bun:"connector_id,notnull"`
	ProviderUnitID    string `bun:"provider_unit_id,notnull"`
	ThreadKey         string `bun:"thread_key,notnull"`
	// ConfigID and ConfigName are the agent config that answers the thread.
	ConfigID   string              `bun:"config_id,notnull"`
	ConfigName string              `bun:"config_name,notnull"`
	AuthorID   string              `bun:"author_id,notnull"`
	Text       string              `bun:"text,notnull"`
	Media      ChannelInboundMedia `bun:"media,type:jsonb,notnull"`
	// Raw is the verified request body.
	Raw []byte `bun:"raw,notnull"`
	// Forward is the delivery's forward to the event destinations that take what no agent
	// answered; nil when none would take it.
	Forward *ChannelInboundForward `bun:"forward,type:jsonb"`
	State   string                 `bun:"state,notnull"`
	// KeywordAnswered is whether a worker answered the message's carrier keyword
	// (MarkChannelInboundKeywordAnswered).
	KeywordAnswered bool `bun:"keyword_answered,notnull"`
	// Attempts is how many writes of the message failed.
	Attempts int `bun:"attempts,notnull"`
	// NextAttemptAt is when the message is due; on a claimed row, the end of the claim's lease.
	NextAttemptAt time.Time  `bun:"next_attempt_at,notnull"`
	CreatedAt     time.Time  `bun:"created_at,notnull"`
	FinishedAt    *time.Time `bun:"finished_at"`
}

// ChannelInboundForward is what forwarding a delivery to the customer's event destinations
// takes besides its body, which is the message's Raw (eventforward.Event).
type ChannelInboundForward struct {
	// ID is the provider's own id for the delivery; empty keys the forward by its body.
	ID string `json:"id"`
	// Headers are the provider's headers a receiver verifies the body with.
	Headers map[string]string `json:"headers"`
	// HeadersUntil is when the signature in Headers stops verifying; nil when it does not age.
	HeadersUntil *time.Time `json:"headers_until,omitempty"`
}

// ChannelInboundMedia is a queued message's media. Its JSON, the row's media column, is an
// array of {"url", "mime_type"}, never null: core.Media has no json tags, and keying the rows
// by its field names would empty the media of rows queued before a rename of one.
type ChannelInboundMedia []core.Media

// channelInboundFile is one file of ChannelInboundMedia as the row holds it.
type channelInboundFile struct {
	URL      string `json:"url"`
	MIMEType string `json:"mime_type"`
}

// MarshalJSON writes the media as the row holds it.
func (m ChannelInboundMedia) MarshalJSON() ([]byte, error) {
	files := make([]channelInboundFile, 0, len(m))
	for _, file := range m {
		files = append(files, channelInboundFile(file))
	}
	return json.Marshal(files)
}

// UnmarshalJSON reads the media as the row holds it.
func (m *ChannelInboundMedia) UnmarshalJSON(data []byte) error {
	var files []channelInboundFile
	if err := json.Unmarshal(data, &files); err != nil {
		return err
	}
	*m = nil
	for _, file := range files {
		*m = append(*m, core.Media(file))
	}
	return nil
}

// Message is the queued message as the verifier handed it over.
func (m ChannelInbound) Message() core.InboundMessage {
	return core.InboundMessage{
		ConnectorID:       m.ConnectorID,
		ProviderUnitID:    m.ProviderUnitID,
		ThreadKey:         m.ThreadKey,
		AuthorID:          m.AuthorID,
		Text:              m.Text,
		Media:             m.Media,
		ProviderMessageID: m.ProviderMessageID,
		Raw:               m.Raw,
	}
}

// AcceptChannelInbound claims message in its thread channel by the provider's id for it
// (ClaimInbound) and queues it, due at message.NextAttemptAt, in one transaction, so a message
// is claimed only with its row: a store that fails leaves it unclaimed, and the provider's
// retry takes it again. fresh is false for a message claimed before, which is not queued again.
//
// With waited not nil, the message starts its external thread, and the transaction also takes
// the messages that waited on that thread (takeWaitingChannelThreadMessages), whether or not the
// message is fresh, and claims and queues the row waited makes of each, after the message's: on
// a failure they wait again, and each waiting message is taken once.
func (s *Store) AcceptChannelInbound(ctx context.Context, message ChannelInbound, waited func(core.InboundMessage) (ChannelInbound, error)) (fresh bool, err error) {
	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		fresh, err = acceptChannelInbound(ctx, tx, message)
		if err != nil || waited == nil {
			return err
		}
		taken, err := takeWaitingChannelThreadMessages(ctx, tx, message.CustomerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey)
		if err != nil {
			return err
		}
		for _, reply := range taken {
			row, err := waited(reply)
			if err != nil {
				return err
			}
			if _, err := acceptChannelInbound(ctx, tx, row); err != nil {
				return err
			}
		}
		return nil
	})
	return fresh, stack.Wrap(err)
}

// acceptChannelInbound claims and queues one message in tx. A message claimed before is not
// queued.
func acceptChannelInbound(ctx context.Context, tx bun.Tx, message ChannelInbound) (bool, error) {
	if err := validChannelInbound(message); err != nil {
		return false, err
	}
	fresh, err := claimChannelThreadMessage(ctx, tx, message.ChannelID, ClaimInbound, message.ProviderMessageID)
	if err != nil || !fresh {
		return false, err
	}
	_, err = queueChannelInbound(ctx, tx, message)
	return err == nil, err
}

// validChannelInbound refuses a message missing what writing it takes.
func validChannelInbound(message ChannelInbound) error {
	if message.ChannelID == "" || message.ProviderMessageID == "" || message.CustomerID == "" || message.ConnectorID == "" || message.ConfigID == "" {
		return stack.Wrap(errors.New("store: a channel, a provider message id, a customer, a connector and an agent config are required"))
	}
	return nil
}

// queueChannelInbound queues message in tx to be written into its thread channel, due at
// message.NextAttemptAt. A message already queued for the channel under the same provider id,
// written or given up within its keep, is not queued again: queued is false.
//
// It first locks the message's thread channel row until tx ends, so the messages of one
// thread channel are queued one transaction at a time. seq is taken at the insert, and a
// claim sees only committed rows: without the lock, a message queued in a transaction that
// commits after a later message's would get the smaller seq and become its thread's oldest
// queued row while the later one is already being written, and the two would be written at
// once, out of order. The lock is FOR NO KEY UPDATE, which the foreign key's FOR KEY SHARE on
// the same row does not wait for.
func queueChannelInbound(ctx context.Context, tx bun.Tx, message ChannelInbound) (bool, error) {
	if _, err := tx.NewRaw("SELECT 1 FROM channel_threads WHERE channel_id = ? FOR NO KEY UPDATE", message.ChannelID).Exec(ctx); err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue channel inbound: %w", err))
	}
	media, err := json.Marshal(message.Media)
	if err != nil {
		return false, stack.Wrap(err)
	}
	var forward *string
	if message.Forward != nil {
		encoded, err := json.Marshal(message.Forward)
		if err != nil {
			return false, stack.Wrap(err)
		}
		forward = new(string(encoded))
	}
	result, err := tx.NewRaw(`
INSERT INTO channel_thread_inbound (channel_id, provider_message_id, customer_id, connector_id, provider_unit_id, thread_key,
    config_id, config_name, author_id, text, media, raw, forward, next_attempt_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?::jsonb, ?)
ON CONFLICT (channel_id, provider_message_id) DO NOTHING`,
		message.ChannelID, message.ProviderMessageID, message.CustomerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey,
		message.ConfigID, message.ConfigName, message.AuthorID, message.Text, string(media), message.Raw, forward,
		message.NextAttemptAt.UTC()).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue channel inbound: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue channel inbound: %w", err))
	}
	return affected == 1, nil
}

// ClaimChannelInbound takes at most limit messages due at now, oldest due first, and moves
// each one's next attempt to leaseUntil, so no other worker takes it while this one writes it.
// Rows another worker is claiming at the same moment are skipped, not waited for.
//
// It takes only the oldest queued message of a thread channel, so a thread's messages are
// written in the order they were queued, one at a time: the ones behind a message being
// written or waiting for its retry wait with it.
func (s *Store) ClaimChannelInbound(ctx context.Context, now time.Time, limit int, leaseUntil time.Time) ([]ChannelInbound, error) {
	return s.claimChannelInbound(ctx, claimChannelInboundQuery, now, limit, leaseUntil)
}

// claimChannelInboundQuery is ClaimChannelInbound's statement. head is the oldest queued row
// of each thread channel; due locks the heads that are due.
//
// due checks state and next_attempt_at on the row it locks, not only in head, which reads the
// rows as they were when the statement started. When another router's claim, or its finish,
// commits in between, Postgres locks the row's newer version and checks only due's own WHERE
// on it again («the second updater ... re-evaluates its WHERE condition»: Read Committed
// Isolation Level, https://www.postgresql.org/docs/16/transaction-iso.html). Without the
// check in due, a row another router just leased is leased again.
//
// RETURNING names ChannelInbound's columns rather than cti.*: bun refuses a column the model
// lacks, so after a later migration adds one, a build rolled back to this one would lease rows
// and fail to read them.
const claimChannelInboundQuery = `
WITH head AS (
    SELECT DISTINCT ON (cti.channel_id) cti.seq
    FROM channel_thread_inbound AS cti
    WHERE cti.state = 'queued'
    ORDER BY cti.channel_id, cti.seq
), due AS (
    SELECT cti.seq FROM channel_thread_inbound AS cti
    JOIN head ON head.seq = cti.seq
    WHERE cti.state = 'queued' AND cti.next_attempt_at <= ?
    ORDER BY cti.next_attempt_at, cti.seq
    LIMIT ?
    FOR UPDATE OF cti SKIP LOCKED
)
UPDATE channel_thread_inbound AS cti
SET next_attempt_at = ?
FROM due
WHERE cti.seq = due.seq
RETURNING cti.seq, cti.channel_id, cti.provider_message_id, cti.customer_id, cti.connector_id, cti.provider_unit_id,
    cti.thread_key, cti.config_id, cti.config_name, cti.author_id, cti.text, cti.media, cti.raw, cti.forward, cti.state,
    cti.keyword_answered, cti.attempts, cti.next_attempt_at, cti.created_at, cti.finished_at`

// claimChannelInbound runs query, ClaimChannelInbound's statement or a test's variant of it
// with a pause in it, with ClaimChannelInbound's arguments.
func (s *Store) claimChannelInbound(ctx context.Context, query string, now time.Time, limit int, leaseUntil time.Time) ([]ChannelInbound, error) {
	var claimed []ChannelInbound
	if limit < 1 {
		return claimed, nil
	}
	if err := s.db.NewRaw(query, now.UTC(), limit, leaseUntil.UTC()).Scan(ctx, &claimed); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim channel inbound: %w", err))
	}
	return claimed, nil
}

// NextChannelInboundAt is when the message due first is due, leased ones at the end of their
// lease, of the ones a claim may take: the oldest queued message of each thread channel.
// found is false when no message is queued at all.
func (s *Store) NextChannelInboundAt(ctx context.Context) (next time.Time, found bool, err error) {
	var at sql.NullTime
	if err := s.db.NewRaw(`
SELECT min(head.next_attempt_at) FROM (
    SELECT DISTINCT ON (channel_id) next_attempt_at
    FROM channel_thread_inbound
    WHERE state = 'queued'
    ORDER BY channel_id, seq
) AS head`).Scan(ctx, &at); err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next channel inbound: %w", err))
	}
	return at.Time, at.Valid, nil
}

// FinishChannelInbound records that a claimed message was written, at at.
//
// It, RetryChannelInbound and FailChannelInbound change the row only while the claim's lease
// is the row's: a worker whose lease ran out while it wrote, and whose message another worker
// took since, changes nothing, so it cannot finish or reschedule the other's attempt. changed
// is false then.
func (s *Store) FinishChannelInbound(ctx context.Context, claimed ChannelInbound, at time.Time) (changed bool, err error) {
	return leaveChannelInbound(ctx, s.db, claimed, ChannelInboundDone, at)
}

// FailChannelInbound records that a claimed message's last attempt failed and it is given up,
// at at, and calls forward, when not nil, to forward the message's delivery: when it has one,
// and no other message of the delivery forwarded it. The event destinations key a forward by
// the delivery only while it is queued (QueueEventDeliveries; FinishEventDelivery deletes it
// once sent), and one thread's messages are given up minutes apart, so a delivery whose two
// messages were never written would otherwise reach a destination twice. A message counts as
// forwarded once forward returned nil, so a forward that fails, or a router that stops before
// it, leaves the delivery to the next of its messages given up; forward logs its own error.
func (s *Store) FailChannelInbound(ctx context.Context, claimed ChannelInbound, at time.Time, forward func(context.Context) error) (changed bool, err error) {
	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if claimed.Forward != nil {
			if err := lockChannelInboundDelivery(ctx, tx, claimed); err != nil {
				return err
			}
		}
		changed, err = leaveChannelInbound(ctx, tx, claimed, ChannelInboundFailed, at)
		if err != nil || !changed || claimed.Forward == nil || forward == nil {
			return err
		}
		forwarded, err := channelInboundDeliveryForwarded(ctx, tx, claimed)
		if err != nil || forwarded {
			return err
		}
		if forward(ctx) != nil {
			return nil
		}
		if _, err := tx.NewRaw("UPDATE channel_thread_inbound SET forward = jsonb_set(forward, '{forwarded}', 'true') WHERE seq = ?",
			claimed.Seq).Exec(ctx); err != nil {
			return fmt.Errorf("store: fail channel inbound: %w", err)
		}
		return nil
	})
	if err != nil {
		return false, stack.Wrap(err)
	}
	return changed, nil
}

// lockChannelInboundDelivery locks claimed's delivery until tx ends, so of two messages of it
// given up at once on two routers, the second sees what the first did.
func lockChannelInboundDelivery(ctx context.Context, tx bun.Tx, claimed ChannelInbound) error {
	if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
		"channel_thread_inbound/"+claimed.CustomerID+"/"+claimed.ConnectorID+"/"+claimed.Forward.ID); err != nil {
		return fmt.Errorf("store: fail channel inbound: %w", err)
	}
	return nil
}

// channelInboundDeliveryForwarded is whether another message of claimed's delivery forwarded
// it: one with the same provider id for the delivery, or with none and the same body, as the
// event destinations key a forward (eventforward.deliveryID).
func channelInboundDeliveryForwarded(ctx context.Context, tx bun.Tx, claimed ChannelInbound) (bool, error) {
	var forwarded bool
	if err := tx.NewRaw(`
SELECT EXISTS (
    SELECT 1 FROM channel_thread_inbound
    WHERE state = 'failed' AND forward->>'forwarded' = 'true' AND customer_id = ? AND connector_id = ?
      AND forward->>'id' = ? AND (? <> '' OR raw = ?)
)`, claimed.CustomerID, claimed.ConnectorID, claimed.Forward.ID, claimed.Forward.ID, claimed.Raw).Scan(ctx, &forwarded); err != nil {
		return false, fmt.Errorf("store: fail channel inbound: %w", err)
	}
	return forwarded, nil
}

func leaveChannelInbound(ctx context.Context, db bun.IDB, claimed ChannelInbound, state string, at time.Time) (bool, error) {
	attempts := claimed.Attempts
	if state == ChannelInboundFailed {
		attempts++
	}
	return updateClaimedChannelInbound(ctx, claimed, db.NewUpdate().Model((*ChannelInbound)(nil)).
		Set("state = ?", state).
		Set("attempts = ?", attempts).
		Set("finished_at = ?", at.UTC()))
}

// MarkChannelInboundKeywordAnswered records that a claimed message's carrier keyword is being
// answered, before the answer is sent, so no later attempt of the row sends it again: an
// answer is sent at most once, as a reply is (ClaimReply). changed is false when an earlier
// attempt marked it, or when the claim's lease is no longer the row's and the worker that took
// it since answers it.
func (s *Store) MarkChannelInboundKeywordAnswered(ctx context.Context, claimed ChannelInbound) (changed bool, err error) {
	return updateClaimedChannelInbound(ctx, claimed, s.db.NewUpdate().Model((*ChannelInbound)(nil)).
		Set("keyword_answered = true").
		Where("NOT keyword_answered"))
}

// RetryChannelInbound records a failed attempt of a claimed message and when the next one is
// due.
func (s *Store) RetryChannelInbound(ctx context.Context, claimed ChannelInbound, next time.Time) (changed bool, err error) {
	return updateClaimedChannelInbound(ctx, claimed, s.db.NewUpdate().Model((*ChannelInbound)(nil)).
		Set("attempts = ?", claimed.Attempts+1).
		Set("next_attempt_at = ?", next.UTC()))
}

func updateClaimedChannelInbound(ctx context.Context, claimed ChannelInbound, update *bun.UpdateQuery) (bool, error) {
	result, err := update.
		Where("seq = ?", claimed.Seq).
		Where("state = ?", ChannelInboundQueued).
		Where("next_attempt_at = ?", claimed.NextAttemptAt.UTC()).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: update channel inbound: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: update channel inbound: %w", err))
	}
	return affected == 1, nil
}

// PruneChannelInbound deletes the rows of written messages finished a day before now, and of
// given-up ones 30 days before, and returns how many it deleted.
func (s *Store) PruneChannelInbound(ctx context.Context, now time.Time) (int, error) {
	result, err := s.db.NewRaw(`
DELETE FROM channel_thread_inbound
WHERE (state = 'done' AND finished_at < ?) OR (state = 'failed' AND finished_at < ?)`,
		now.Add(-channelInboundDoneKeep).UTC(), now.Add(-channelInboundFailedKeep).UTC()).Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: prune channel inbound: %w", err))
	}
	pruned, _ := result.RowsAffected()
	return int(pruned), nil
}
