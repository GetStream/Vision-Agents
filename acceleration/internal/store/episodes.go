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

// What an episode came in on, the source field of its card (20261007042100_episodes.sql).
// The table also takes sms and whatsapp, which T53 and T51 name here when they put those
// channels on the bridge.
const (
	EpisodeCall     = "call"
	EpisodeSlack    = "slack"
	EpisodeIMessage = "imessage"
)

// episodeInProgress is an episode not closed yet, the status its card is written with.
const episodeInProgress = "in_progress"

// The statuses that close an episode (20261007042100_episodes.sql; T55, omnichannel.Closer):
// ended once it is closed, then summarized or summary_failed once its summary is written or
// could not be.
const (
	EpisodeEnded         = "ended"
	EpisodeSummaryFailed = "summary_failed"
)

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
//
// Each message of a thread is the thread's last one for now, so a message that finds the
// episode open sets its last_message_at (episode_activity, 20261008140000_episodes_close.sql)
// to StartedAt's time, which keeps the episode from closing as idle (CloseIdleEpisodes); the
// first message is its started_at. The touch finds the episode open, or misses it, in one
// statement, and holds its row against a close until it commits: an episode the idle sweeper
// closed after the insert found it open is missed, and the message opens the next one.
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
	at := episode.StartedAt
	// Twice at most: a second miss is the thread's episode closed and another opened between
	// the insert and the touch, twice over.
	for range 2 {
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
		if episode.SessionID != "" {
			err := s.db.NewSelect().Model(episode).Where("customer_id = ?", episode.CustomerID).
				Where("session_id = ?", episode.SessionID).Scan(ctx)
			if err != nil {
				return false, stack.Wrap(fmt.Errorf("store: open episode: %w", err))
			}
			return false, nil
		}
		err = s.db.NewRaw(touchThreadEpisodeQuery, episode.CustomerID, episode.ThreadChannel, at).Scan(ctx, episode)
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return false, stack.Wrap(fmt.Errorf("store: open episode: %w", err))
		}
		return false, nil
	}
	return false, stack.Wrap(errors.New("store: open episode: the thread's episode closed twice while a message opened it"))
}

// touchThreadEpisodeQuery sets the last message of the thread's episode in progress, and
// reads the episode. open holds the episode's row until the message commits, so a sweep
// skips it meanwhile (closeIdleEpisodesQuery), and a touch that waited for a close finds the
// episode no longer open.
const touchThreadEpisodeQuery = `
WITH open AS (
    SELECT id FROM episodes
    WHERE customer_id = ? AND thread_channel = ? AND status = 'in_progress' AND session_id IS NULL
    FOR SHARE
), touched AS (
    INSERT INTO episode_activity (episode_id, last_message_at)
    SELECT id, ? FROM open
    ON CONFLICT (episode_id) DO UPDATE SET last_message_at = excluded.last_message_at
    RETURNING episode_id
)
SELECT ep.* FROM episodes AS ep JOIN touched ON touched.episode_id = ep.id`

// ClosedEpisode is an episode a closer holds to summarize: the episode, until when it holds
// it (episode_activity, 20261008140000_episodes_close.sql) and, from its contact map row, the
// omni-channel its card is in, the agent config whose LLM writes the summary and how the
// person is known, which a call's lines are checked against.
type ClosedEpisode struct {
	Episode `bun:",extend"`
	// SummaryLeaseUntil is until when the router that closed the episode, or took it again,
	// has it to summarize.
	SummaryLeaseUntil *time.Time `bun:"summary_lease_until,scanonly"`
	ConversationID    string     `bun:"conversation_id,scanonly"`
	AgentConfigID     string     `bun:"agent_config_id,scanonly"`
	ContactKind       string     `bun:"contact_kind,scanonly"`
	ContactAddress    string     `bun:"contact_address,scanonly"`
}

// leasedWithContact ends each statement that closes episodes: the closed CTE's rows are
// leased to this router until the one argument it takes, and read with their lease and their
// contact map row.
const leasedWithContact = `, leased AS (
    INSERT INTO episode_activity (episode_id, summary_lease_until)
    SELECT id, ? FROM closed
    ON CONFLICT (episode_id) DO UPDATE SET summary_lease_until = excluded.summary_lease_until
    RETURNING episode_id, summary_lease_until
)
SELECT closed.*, leased.summary_lease_until,
    cm.conversation_id, cm.agent_config_id, cm.kind AS contact_kind, cm.address AS contact_address
FROM closed
JOIN leased ON leased.episode_id = closed.id
JOIN contact_map AS cm ON cm.id = closed.contact_id`

// idleEpisodesQuery locks at most limit thread episodes whose last message, or whose start
// while they have none, is at or before the idle cutoff, and skips the ones another router,
// or a message (touchThreadEpisodeQuery), holds. It reads the last message of each open
// episode by its key: episode_activity keeps a row for every thread episode there was, and a
// join may read all of them.
const idleEpisodesQuery = `
SELECT ep.id FROM episodes AS ep
WHERE ep.status = 'in_progress' AND ep.session_id IS NULL
  AND COALESCE((SELECT act.last_message_at FROM episode_activity AS act WHERE act.episode_id = ep.id), ep.started_at) <= ?
ORDER BY ep.started_at
LIMIT ?
FOR UPDATE OF ep SKIP LOCKED`

// closeIdleEpisodesQuery closes the episodes idleEpisodesQuery locked, which no other router
// can close meanwhile, so each episode is closed once, by one router. It is a statement of
// its own, so it reads the status and the last messages as they are once the rows are locked:
// it checks them again, so an episode already ended is not closed twice, and an episode a
// message touched after idleEpisodesQuery started is left alone.
const closeIdleEpisodesQuery = `
WITH closed AS (
    UPDATE episodes AS ep
    SET status = 'ended', ended_at = ?
    WHERE ep.id IN (?) AND ep.status = 'in_progress' AND ep.session_id IS NULL
      AND COALESCE((SELECT act.last_message_at FROM episode_activity AS act WHERE act.episode_id = ep.id), ep.started_at) <= ?
    RETURNING ep.*
)` + leasedWithContact

// CloseIdleEpisodes ends, at now, at most limit thread episodes of any customer with no
// message since idleSince, each leased to this router for its summary until leaseUntil. A
// call's episode is never idle: its call ends it (EndCallEpisodes).
func (s *Store) CloseIdleEpisodes(ctx context.Context, idleSince, now time.Time, limit int, leaseUntil time.Time) ([]ClosedEpisode, error) {
	closed := []ClosedEpisode{}
	if limit < 1 {
		return closed, nil
	}
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var due []string
		if err := tx.NewRaw(idleEpisodesQuery, idleSince.UTC(), limit).Scan(ctx, &due); err != nil {
			return err
		}
		if len(due) == 0 {
			return nil
		}
		return tx.NewRaw(closeIdleEpisodesQuery, now.UTC(), bun.In(due), idleSince.UTC(), leaseUntil.UTC()).Scan(ctx, &closed)
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: close idle episodes: %w", err))
	}
	return closed, nil
}

// endCallEpisodesQuery ends the call episodes still in progress of one call in one app. The
// status is checked on the row the update locks, so the event delivered twice, to two
// routers, ends each episode once. session_id IS NOT NULL is every call episode
// (20261007042100_episodes.sql), named so episodes_open_call finds them.
const endCallEpisodesQuery = `
WITH closed AS (
    UPDATE episodes
    SET status = 'ended', ended_at = ?
    WHERE source = 'call' AND session_id IS NOT NULL AND call_id = ? AND status = 'in_progress' AND (?)
    RETURNING *
)` + leasedWithContact

// EndCallEpisodes ends, at now, the episodes in progress of the call callID in the app scope
// names, each leased to this router for its summary until leaseUntil. A call id is unique
// only within its app, so the scope is the app the call event came from, as
// ReleaseCallResourcesInApp takes it. None is no error: a call under an agent config without
// episode cards has none.
func (s *Store) EndCallEpisodes(ctx context.Context, scope AppScope, callID string, now, leaseUntil time.Time) ([]ClosedEpisode, error) {
	closed := []ClosedEpisode{}
	if callID == "" {
		return closed, nil
	}
	err := s.db.NewRaw(endCallEpisodesQuery, now.UTC(), callID, scope.clause(), leaseUntil.UTC()).Scan(ctx, &closed)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: end call episodes: %w", err))
	}
	return closed, nil
}

// claimEpisodeSummariesQuery takes at most limit ended episodes whose summary lease ran out:
// a router closed them and stopped before it wrote their summary. Only an ended episode has a
// lease: closing sets both, and finishing drops both (FinishEpisodeSummary). due locks the
// leases it takes and skips the ones another router holds; one that another router took or
// finished after the statement started is checked again as it locks it, and left alone.
const claimEpisodeSummariesQuery = `
WITH due AS (
    SELECT episode_id FROM episode_activity
    WHERE summary_lease_until <= ?
    ORDER BY summary_lease_until
    LIMIT ?
    FOR UPDATE SKIP LOCKED
), leased AS (
    UPDATE episode_activity AS act
    SET summary_lease_until = ?
    FROM due
    WHERE act.episode_id = due.episode_id
    RETURNING act.episode_id, act.summary_lease_until
)
SELECT ep.*, leased.summary_lease_until,
    cm.conversation_id, cm.agent_config_id, cm.kind AS contact_kind, cm.address AS contact_address
FROM leased
JOIN episodes AS ep ON ep.id = leased.episode_id
JOIN contact_map AS cm ON cm.id = ep.contact_id`

// ClaimEpisodeSummaries takes at most limit ended episodes whose summary lease ran out at now,
// each leased to this router until leaseUntil.
func (s *Store) ClaimEpisodeSummaries(ctx context.Context, now time.Time, limit int, leaseUntil time.Time) ([]ClosedEpisode, error) {
	claimed := []ClosedEpisode{}
	if limit < 1 {
		return claimed, nil
	}
	err := s.db.NewRaw(claimEpisodeSummariesQuery, now.UTC(), limit, leaseUntil.UTC()).Scan(ctx, &claimed)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim episode summaries: %w", err))
	}
	return claimed, nil
}

// finishEpisodeSummaryQuery sets an ended episode's status and drops its lease, in one
// statement.
const finishEpisodeSummaryQuery = `
WITH finished AS (
    UPDATE episodes SET status = ?
    WHERE customer_id = ? AND id = ? AND status = 'ended'
    RETURNING id
)
UPDATE episode_activity AS act SET summary_lease_until = NULL
FROM finished WHERE act.episode_id = finished.id`

// FinishEpisodeSummary sets an ended episode summarized or summary_failed and drops its
// lease. An episode no longer ended, finished already, is left as it is.
func (s *Store) FinishEpisodeSummary(ctx context.Context, customerID, id, status string) error {
	if status != EpisodeSummarized && status != EpisodeSummaryFailed {
		return stack.Wrap(fmt.Errorf("store: %q is not a status a summary finishes with", status))
	}
	_, err := s.db.NewRaw(finishEpisodeSummaryQuery, status, customerID, id).Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: finish episode summary: %w", err))
	}
	return nil
}

// AnyEpisodeCards reports whether an agent config of any customer has episode_cards on, so
// its calls may have episodes to close.
func (s *Store) AnyEpisodeCards(ctx context.Context) (bool, error) {
	var on bool
	err := s.db.NewRaw("SELECT EXISTS (SELECT 1 FROM agent_configs WHERE episode_cards AND deleted_at IS NULL)").Scan(ctx, &on)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: any episode cards: %w", err))
	}
	return on, nil
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
	// ContactKind and ContactAddress are how the episode's person is known, from its contact
	// map row: for a call, the caller's number, which is whose lines the card's are.
	ContactKind    string `bun:"contact_kind,scanonly"`
	ContactAddress string `bun:"contact_address,scanonly"`
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
		ColumnExpr("cm.kind AS contact_kind, cm.address AS contact_address").
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
