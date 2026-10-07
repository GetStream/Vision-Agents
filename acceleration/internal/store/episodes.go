package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// What an episode came in on, the source field of its card (20261007040100_episodes.sql).
// The table also takes sms, whatsapp and imessage, which T53, T51 and T36 name here when they
// put those channels on the bridge.
const (
	EpisodeCall  = "call"
	EpisodeSlack = "slack"
)

// episodeInProgress is an episode not closed yet, the status its card is written with. T55
// adds the statuses that close it.
const episodeInProgress = "in_progress"

// Episode is one call, or one run of messages on one external thread, and its card in the
// person's omni-channel (20261007040100_episodes.sql).
type Episode struct {
	bun.BaseModel `bun:"table:episodes,alias:ep"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// ContactID is the contact map row of the person; the card is in its omni-channel.
	ContactID string `bun:"contact_id,notnull"`
	Source    string `bun:"source,notnull"`
	// ThreadChannel is the cid of the channel with the raw text: a thread channel, or a
	// call channel.
	ThreadChannel string `bun:"thread_channel,notnull"`
	// CallID and SessionID are a call's Stream call and router session; empty for a thread.
	CallID    string `bun:"call_id,nullzero"`
	SessionID string `bun:"session_id,nullzero"`
	// CardMessageID is the card's Stream Chat message id.
	CardMessageID string     `bun:"card_message_id,notnull"`
	Status        string     `bun:"status,notnull"`
	StartedAt     time.Time  `bun:"started_at,notnull"`
	EndedAt       *time.Time `bun:"ended_at"`
	// StreamAppPK is the Stream app the omni-channel is in; zero, stored as NULL, is the
	// deployment's own.
	StreamAppPK int64     `bun:"stream_app_pk,nullzero"`
	CreatedAt   time.Time `bun:"created_at,notnull,default:current_timestamp"`
}

// OpenEpisode opens the episode episode describes, unless its thread already has one open,
// or its call session already has one; episode is then that one. opened is whether this call
// opened it, and so whether its card is still to be written. Two first messages of one
// thread at once open one episode: the unique indexes decide which.
func (s *Store) OpenEpisode(ctx context.Context, episode *Episode) (opened bool, err error) {
	if episode.CustomerID == "" || episode.ContactID == "" || episode.Source == "" || episode.ThreadChannel == "" {
		return false, stack.Wrap(errors.New("store: a customer, a contact, a source and a thread channel are required"))
	}
	if (episode.Source == EpisodeCall) != (episode.SessionID != "") {
		return false, stack.Wrap(errors.New("store: a call episode, and only a call episode, names its session"))
	}
	if episode.ID == "" {
		episode.ID = newID()
	}
	if episode.CardMessageID == "" {
		episode.CardMessageID = "episode-" + episode.ID
	}
	episode.Status = episodeInProgress
	if episode.StartedAt.IsZero() {
		episode.StartedAt = time.Now()
	}
	episode.StartedAt = episode.StartedAt.UTC().Truncate(time.Microsecond)
	result, err := s.db.NewInsert().Model(episode).On("CONFLICT DO NOTHING").Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: open episode: %w", err))
	}
	inserted, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: open episode: %w", err))
	}
	if inserted == 1 {
		return true, nil
	}
	query := s.db.NewSelect().Model(episode).Where("customer_id = ?", episode.CustomerID)
	if episode.SessionID != "" {
		query = query.Where("session_id = ?", episode.SessionID)
	} else {
		query = query.Where("thread_channel = ?", episode.ThreadChannel).
			Where("status = ?", episodeInProgress).Where("session_id IS NULL")
	}
	if err := query.Scan(ctx); err != nil {
		return false, stack.Wrap(fmt.Errorf("store: open episode: %w", err))
	}
	return false, nil
}
