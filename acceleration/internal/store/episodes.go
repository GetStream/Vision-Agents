package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// What an episode came in on, the source field of its card (20261007042100_episodes.sql).
// The table also takes sms, whatsapp and imessage, which T53, T51 and T36 name here when they
// put those channels on the bridge.
const (
	EpisodeCall  = "call"
	EpisodeSlack = "slack"
)

// episodeInProgress is an episode not closed yet, the status its card is written with. T55
// adds the statuses that close it.
const episodeInProgress = "in_progress"

// EpisodeSummarized is an episode whose card holds its summary (20261007042100_episodes.sql;
// T55 sets it). A card in any other status is read by the last lines of its thread channel.
const EpisodeSummarized = "summarized"

// Episode is one call, or one run of messages on one external thread, and its card in the
// person's omni-channel (20261007042100_episodes.sql).
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

// EpisodeCard is an episode as a session reads its card: the episode, and until when its
// thread channel's lines are its own.
type EpisodeCard struct {
	Episode `bun:",extend"`
	// Until is the earliest of when the episode ended, when its call's session closed and
	// when the next episode in its thread channel started. A call channel can hold several
	// callers' calls (episodes_call_session), so a call's lines end there. Nil while none
	// of them is known.
	Until *time.Time `bun:"until,scanonly"`
}

// CardsQuery is whose episode cards a session reads, and which of them it leaves out
// because it reads them word for word already.
type CardsQuery struct {
	CustomerID    string
	AgentConfigID string
	// ConversationID is the person's omni-channel: every contact map row of the customer
	// and agent that points at it is the person, whatever they came in on.
	ConversationID string
	// ExceptThread and ExceptSession leave out the session's own thread channel and its
	// own call's episode. Empty leaves out nothing.
	ExceptThread  string
	ExceptSession string
	Limit         int
}

// EpisodeCards are the newest episodes of one person, for one agent of a customer: those of
// the contact map rows of that customer and agent that point at the person's omni-channel,
// never another customer's or another agent's.
func (s *Store) EpisodeCards(ctx context.Context, query CardsQuery) ([]EpisodeCard, error) {
	if query.CustomerID == "" || query.AgentConfigID == "" || query.ConversationID == "" || query.Limit <= 0 {
		return nil, stack.Wrap(errors.New("store: a customer, an agent config, an omni-channel and a limit are required"))
	}
	var cards []EpisodeCard
	err := s.db.NewSelect().Model(&cards).
		ColumnExpr("ep.*").
		// LEAST ignores NULLs, so Until is nil only when all three are.
		ColumnExpr(`LEAST(ep.ended_at, asn.closed_at, (SELECT min(nx.started_at) FROM episodes AS nx
			WHERE nx.customer_id = ep.customer_id AND nx.thread_channel = ep.thread_channel
			AND nx.started_at > ep.started_at)) AS until`).
		Join("JOIN contact_map AS cm ON cm.id = ep.contact_id").
		Join("LEFT JOIN agent_sessions AS asn ON asn.id = ep.session_id AND asn.customer_id = ep.customer_id").
		Where("ep.customer_id = ?", query.CustomerID).
		Where("cm.customer_id = ?", query.CustomerID).
		Where("cm.agent_config_id = ?", query.AgentConfigID).
		Where("cm.conversation_id = ?", query.ConversationID).
		Where("ep.thread_channel <> ?", query.ExceptThread).
		Where("ep.session_id IS DISTINCT FROM ?", query.ExceptSession).
		OrderExpr("ep.started_at DESC").
		Limit(query.Limit).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: episode cards: %w", err))
	}
	return cards, nil
}
