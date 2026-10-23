package store

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// How many sessions are handed back at once. A person scrolling a sidebar reads a page at a
// time, and a backend exporting everything pages rather than asking for it all.
const (
	defaultSessionLimit = 25
	maxSessionLimit     = 200
)

// How much of a conversation is handed back at once. A long session is thousands of items,
// and the only caller who wants every one is a caller walking the whole conversation, which
// is what paging is for.
const (
	defaultItemLimit = 200
	maxItemLimit     = 1000
)

// SessionLimit is the page size a session or turn list uses for the limit asked for.
//
// The queries return one row more than this, so a caller can tell the page is not the last
// without counting.
func SessionLimit(asked int) int { return clampLimit(asked, defaultSessionLimit, maxSessionLimit) }

// ItemLimit is the page size an item list uses for the limit asked for. SessionItems
// returns one row more than this, the same as the session queries.
func ItemLimit(asked int) int { return clampLimit(asked, defaultItemLimit, maxItemLimit) }

func clampLimit(asked, fallback, most int) int {
	if asked <= 0 {
		return fallback
	}
	return min(asked, most)
}

// SaveSession records that a session was opened.
//
// Written when the session starts rather than when it ends, so a session that is still
// running is findable and one the process died during is not simply lost. Saving the same
// session twice updates the row: a retried write must not become a second conversation. It
// also opens a text session that ended again, which is how one is carried on under its id.
//
// Incognito sessions never get here. The manager does not call this for one, which is what
// the flag means -- not a row that is hidden, but no row at all.
func (s *Store) SaveSession(ctx context.Context, session *AgentSession) error {
	if session.ID == "" {
		return errors.New("store: a session id is required")
	}
	if session.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	now := time.Now().UTC()
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	session.UpdatedAt = now
	if session.State == "" {
		session.State = SessionRunning
	}
	if session.Modality == "" {
		session.Modality = ModalityVoice
		if session.CallID == "" {
			session.Modality = ModalityText
		}
	}
	if session.Custom == nil {
		session.Custom = map[string]any{}
	}
	// The column's own default, which a nil slice would write as JSON null.
	if session.ConnectorSelections == nil {
		session.ConnectorSelections = []SessionConnectorSelection{}
	}

	_, err := s.shard.NewInsert().Model(session).
		On("CONFLICT (id) DO UPDATE").
		Set("title = EXCLUDED.title").
		Set("description = EXCLUDED.description").
		Set("custom = EXCLUDED.custom").
		Set("conversation_id = EXCLUDED.conversation_id").
		Set("state = EXCLUDED.state").
		Set("closed_at = NULL").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save session: %w", err)
	}
	return nil
}

// ErrNoSessionTools is a session that recorded no tools.
var ErrNoSessionTools = errors.New("store: the session recorded no tools")

// ToolDefinition is one tool as a session's model was offered it.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// agentSessionTools is the tools a session was offered, kept apart from its row.
type agentSessionTools struct {
	bun.BaseModel `bun:"table:agent_session_tools"`

	SessionID string           `bun:"session_id,pk"`
	Tools     []ToolDefinition `bun:"tools,type:jsonb,notnull"`
	UpdatedAt time.Time        `bun:"updated_at,notnull"`
}

// SaveSessionTools records what a session's model is offered, replacing what it was offered
// before: a reopened session is offered what it has now.
func (s *Store) SaveSessionTools(ctx context.Context, sessionID string, tools []ToolDefinition) error {
	if sessionID == "" {
		return errors.New("store: a session id is required")
	}
	if tools == nil {
		tools = []ToolDefinition{}
	}
	_, err := s.shard.NewInsert().
		Model(&agentSessionTools{SessionID: sessionID, Tools: tools, UpdatedAt: time.Now().UTC()}).
		On("CONFLICT (session_id) DO UPDATE").
		Set("tools = EXCLUDED.tools").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save session tools: %w", err)
	}
	return nil
}

// SessionTools is what a session's model was last offered. A session that recorded none is
// ErrNoSessionTools.
func (s *Store) SessionTools(ctx context.Context, sessionID string) ([]ToolDefinition, error) {
	stored, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (agentSessionTools, error) {
		var stored agentSessionTools
		err := db.NewSelect().Model(&stored).Where("session_id = ?", sessionID).Scan(ctx)
		return stored, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNoSessionTools
	}
	if err != nil {
		return nil, fmt.Errorf("store: session tools: %w", err)
	}
	return stored.Tools, nil
}

// CloseSession records that a session ended. One that already ended keeps the first time: a
// session is left once, and a second close is the same leaving reported again.
func (s *Store) CloseSession(ctx context.Context, id string, at time.Time) error {
	if id == "" {
		return errors.New("store: a session id is required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	_, err := s.shard.NewUpdate().Model((*AgentSession)(nil)).
		Set("closed_at = ?", at).
		Set("updated_at = ?", at).
		Set("state = ?", SessionClosed).
		Where("id = ?", id).
		Where("closed_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: close session: %w", err)
	}
	return nil
}

// SawVideo records that the agent saw the user's video on a session. The session's clock is
// left alone: the turn that looked at the video moves it.
func (s *Store) SawVideo(ctx context.Context, id string) error {
	if id == "" {
		return errors.New("store: a session id is required")
	}

	_, err := s.shard.NewUpdate().Model((*AgentSession)(nil)).
		Set("modality = ?", ModalityVideo).
		Where("id = ?", id).
		Where("modality <> ?", ModalityVideo).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: saw video: %w", err)
	}
	return nil
}

// SessionVoiced records that voice was started on a session, on the call it joined. A
// session that was text becomes a voice one; one that saw video stays a video one.
func (s *Store) SessionVoiced(ctx context.Context, id, callID, callType string) error {
	if id == "" {
		return errors.New("store: a session id is required")
	}

	_, err := s.shard.NewUpdate().Model((*AgentSession)(nil)).
		Set("call_id = ?", callID).
		Set("call_type = ?", callType).
		Set("modality = CASE WHEN modality = ? THEN ? ELSE modality END", ModalityText, ModalityVoice).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: session voiced: %w", err)
	}
	return nil
}

// DescribeSession changes what a session is called. It is separate from SaveSession because
// renaming a conversation is something a person does long after it ended, when there is no
// spec left to save.
func (s *Store) DescribeSession(ctx context.Context, customerID, id, title, description string, custom map[string]any) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	affected, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		query := db.NewUpdate().Model((*AgentSession)(nil)).
			Set("updated_at = ?", time.Now().UTC()).
			Set("title = ?", title).
			Set("description = ?", description).
			Where("id = ?", id).
			Where("customer_id = ?", customerID)
		if custom != nil {
			query = query.Set("custom = ?", jsonbOf(custom))
		}
		return query.Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: describe session: %w", err))
	}
	if affected == 0 {
		return unknownStoredSession(id)
	}
	return nil
}

// ChooseSessionConnection records connectionID as the session's selection for the connector
// binding called name, replacing one it had, in one statement so two writes for different
// names never drop each other's. A fork or a reopened chat re-resolves it as it does the
// selections the session was opened with. A session with no row, such as an incognito one,
// is left as it is.
func (s *Store) ChooseSessionConnection(ctx context.Context, customerID, id, name, connectionID string) error {
	if customerID == "" || id == "" || name == "" || connectionID == "" {
		return stack.Wrap(errors.New("store: a customer, a session, a binding name and a connection id are required"))
	}
	_, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentSession)(nil)).
			Set("updated_at = ?", time.Now().UTC()).
			Set(`connector_selections = (
			SELECT coalesce(jsonb_agg(chosen), '[]'::jsonb) FROM jsonb_array_elements(connector_selections) AS chosen
			WHERE chosen->>'name' <> ?) || jsonb_build_array(jsonb_build_object('name', ?::text, 'connection_id', ?::text))`,
				name, name, connectionID).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: choose session connection: %w", err))
	}
	return nil
}

// SessionExists reports whether any customer has a session with this id.
func (s *Store) SessionExists(ctx context.Context, id string) (bool, error) {
	_, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (bool, error) {
		exists, err := db.NewSelect().Model((*AgentSession)(nil)).Where("id = ?", id).Exists(ctx)
		if err == nil && !exists {
			err = sql.ErrNoRows
		}
		return exists, err
	})
	exists := err == nil
	if errors.Is(err, sql.ErrNoRows) {
		err = nil
	}
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: session exists: %w", err))
	}
	return exists, nil
}

// StoredSession returns one session a customer ran.
func (s *Store) StoredSession(ctx context.Context, customerID, id string) (AgentSession, error) {
	if customerID == "" || id == "" {
		return AgentSession{}, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	session, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (AgentSession, error) {
		var session AgentSession
		err := db.NewSelect().Model(&session).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Limit(1).
			Scan(ctx)
		return session, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return AgentSession{}, unknownStoredSession(id)
	}
	if err != nil {
		return AgentSession{}, stack.Wrap(fmt.Errorf("store: stored session: %w", err))
	}
	return session, nil
}

// ConversationPin is the Stream app a customer's conversation was kept in, read off the
// newest session that held it. Found is false for a conversation no session of theirs did.
func (s *Store) ConversationPin(ctx context.Context, customerID, conversationID string) (int64, bool, error) {
	type pinned struct {
		Pin       sql.NullInt64 `bun:"stream_app_pk"`
		CreatedAt time.Time     `bun:"created_at"`
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]pinned, error) {
		var newest []pinned
		err := db.NewSelect().Model((*AgentSession)(nil)).Column("stream_app_pk", "created_at").
			Where("customer_id = ?", customerID).
			Where("conversation_id = ?", conversationID).
			OrderExpr("created_at DESC").
			Limit(1).
			Scan(ctx, &newest)
		return newest, err
	})
	if err != nil {
		return 0, false, fmt.Errorf("store: conversation pin: %w", err)
	}
	newest := merged(lists, func(a, b pinned) int { return b.CreatedAt.Compare(a.CreatedAt) }, 1)
	if len(newest) == 0 {
		return 0, false, nil
	}
	return newest[0].Pin.Int64, true, nil
}

// DeleteSession deletes one of a customer's sessions, and its turns and their items with it.
// A session that is not there is not an error: a retried delete has nothing left to do.
func (s *Store) DeleteSession(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	_, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewDelete().Model((*AgentSession)(nil)).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete session: %w", err))
	}
	return nil
}

// QuerySessions returns a customer's sessions, most recently updated first, narrowed by the
// filter.
//
// This is the structured half of finding an old conversation: everything here is an exact
// match, so the answer is the same every time and a page can hold a cursor into it.
// SearchSessions is the other half, for when what the caller has is words rather than a
// filter.
func (s *Store) QuerySessions(ctx context.Context, customerID string, filter SessionFilter) ([]AgentSession, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentSession, error) {
		query := db.NewSelect().Model((*AgentSession)(nil)).
			Where("customer_id = ?", customerID).
			Order("updated_at DESC", "id DESC")
		if after := filter.Cursor; after != nil {
			query = query.Where("(updated_at, id) < (?, ?)", after.UpdatedAt, after.ID)
		}
		query = narrowSessions(query, filter)

		var sessions []AgentSession
		err := query.Scan(ctx, &sessions)
		return sessions, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: query sessions: %w", err))
	}
	return pageOfSessions(lists, filter, func(a, b AgentSession) int {
		return cmp.Or(b.UpdatedAt.Compare(a.UpdatedAt), strings.Compare(b.ID, a.ID))
	}), nil
}

// SearchSessions returns the sessions whose title, description, project or agent match the
// words given, best match first.
//
// Only what the caller wrote is searched, not the transcript. Searching what was said means
// either reading every conversation out of Stream Chat on every query or keeping a second
// copy of every message here, and the first is too slow to offer while the second is a
// transcript that can drift from the real one. Titles and descriptions are what a person
// names a conversation with, and naming them is the thing to encourage.
//
// An empty query is the same as no query: it falls through to QuerySessions rather than
// matching nothing, because a search box a person has not typed in yet should show them
// their conversations.
func (s *Store) SearchSessions(ctx context.Context, customerID, text string, filter SessionFilter) ([]AgentSession, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	if text == "" {
		return s.QuerySessions(ctx, customerID, filter)
	}

	// websearch_to_tsquery rather than to_tsquery, because the text comes from a search
	// box: it takes quoted phrases and bare words and never fails on punctuation, where
	// to_tsquery would answer a syntax error to somebody who typed an apostrophe.
	const rank = "ts_rank(searchable, websearch_to_tsquery('english', ?))"
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentSession, error) {
		query := db.NewSelect().Model((*AgentSession)(nil)).
			ColumnExpr("?TableColumns").
			ColumnExpr(rank+" AS rank", text).
			Where("customer_id = ?", customerID).
			Where("searchable @@ websearch_to_tsquery('english', ?)", text).
			Order("rank DESC", "updated_at DESC", "id DESC")
		if after := filter.Cursor; after != nil {
			query = query.Where("("+rank+", updated_at, id) < (?::real, ?, ?)",
				text, after.Rank, after.UpdatedAt, after.ID)
		}
		query = narrowSessions(query, filter)

		var sessions []AgentSession
		err := query.Scan(ctx, &sessions)
		return sessions, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: search sessions: %w", err))
	}
	return pageOfSessions(lists, filter, func(a, b AgentSession) int {
		return cmp.Or(cmp.Compare(b.Rank, a.Rank), b.UpdatedAt.Compare(a.UpdatedAt), strings.Compare(b.ID, a.ID))
	}), nil
}

// pageOfSessions merges every shard's page into one, keeping a session carried on in another
// shard once, by its newest row. Each shard's page is one longer than asked, so the merged one
// still tells the caller whether there is more.
func pageOfSessions(lists [][]AgentSession, filter SessionFilter, before func(a, b AgentSession) int) []AgentSession {
	if len(lists) == 1 {
		return lists[0]
	}
	sessions := distinct(merged(lists, before, 0), func(session AgentSession) string { return session.ID })
	if limit := SessionLimit(filter.Limit) + 1; len(sessions) > limit {
		sessions = sessions[:limit]
	}
	return sessions
}

// narrowSessions applies a filter to either kind of session query, so the two cannot drift
// apart in what they admit. That matters more than the duplication it saves: a filter
// honoured by query and forgotten by search is one a caller uses to read somebody else's
// conversations.
func narrowSessions(query *bun.SelectQuery, filter SessionFilter) *bun.SelectQuery {
	query = query.Limit(SessionLimit(filter.Limit) + 1)

	if filter.UserID != "" {
		query = query.Where("user_id = ?", filter.UserID)
	}
	if filter.ConfigID != "" {
		query = query.Where("config_id = ?", filter.ConfigID)
	}
	if filter.AgentName != "" {
		query = query.Where("agent_name = ?", filter.AgentName)
	}
	if filter.Project != "" {
		query = query.Where("project = ?", filter.Project)
	}
	if filter.Modality != "" {
		query = query.Where("modality = ?", filter.Modality)
	}
	if filter.AgentID != "" {
		query = query.Where("agent_id = ?", filter.AgentID)
	}
	if filter.State != "" {
		query = query.Where("state = ?", filter.State)
	}
	if len(filter.Custom) > 0 {
		// Containment rather than a key at a time, so the GIN index on custom is usable
		// and a caller asking for two labels gets the sessions carrying both. The index is
		// jsonb_path_ops, which serves @> and nothing else: no key-exists (?, ?|, ?&) filter.
		query = query.Where("custom @> ?::jsonb", jsonbOf(filter.Custom))
	}
	if !filter.After.IsZero() {
		query = query.Where("created_at >= ?", filter.After)
	}
	if !filter.Before.IsZero() {
		query = query.Where("created_at < ?", filter.Before)
	}
	return query
}

// StartResponse records that the agent began a turn.
func (s *Store) StartResponse(ctx context.Context, response *AgentResponse) error {
	if response.ID == "" || response.SessionID == "" {
		return errors.New("store: a response and a session id are required")
	}
	if response.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if response.CreatedAt.IsZero() {
		response.CreatedAt = time.Now().UTC()
	}
	if response.Status == "" {
		response.Status = ResponseRunning
	}
	response.Said = SafeLogText(response.Said)

	_, err := s.shard.NewInsert().Model(response).
		On("CONFLICT (id) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: start response: %w", err)
	}
	return nil
}

// FinishResponse records how a turn ended and stamps the session's last activity, which is
// what a most-recently-used ordering of conversations reads.
func (s *Store) FinishResponse(ctx context.Context, id, status, failure string, at time.Time) error {
	if id == "" {
		return errors.New("store: a response id is required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}
	if status == "" {
		status = ResponseCompleted
	}

	_, err := s.shard.NewUpdate().Model((*AgentResponse)(nil)).
		Set("status = ?", status).
		Set("error = ?", SafeLogText(failure)).
		Set("finished_at = ?", at).
		Where("id = ?", id).
		Where("finished_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish response: %w", err)
	}

	// The session's clock moves with its turns rather than with its row being touched, so a
	// conversation somebody came back to sorts above one that was merely renamed.
	_, err = s.shard.NewUpdate().Model((*AgentSession)(nil)).
		Set("last_response_at = ?", at).
		Set("updated_at = ?", at).
		TableExpr("agent_responses AS r").
		Where("r.id = ?", id).
		Where("asn.id = r.session_id").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish response: stamp session: %w", err)
	}
	return nil
}

// SessionResponses returns a session's turns, oldest first, which read in order are the
// conversation. A nil after is the first page.
func (s *Store) SessionResponses(ctx context.Context, customerID, sessionID string, limit int, after *ResponsePosition) ([]AgentResponse, error) {
	if customerID == "" || sessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponse, error) {
		query := db.NewSelect().Model((*AgentResponse)(nil)).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("rewound_at IS NULL").
			Order("created_at ASC", "id ASC").
			Limit(SessionLimit(limit) + 1)
		if after != nil {
			query = query.Where("(created_at, id) > (?, ?)", after.CreatedAt, after.ID)
		}
		var responses []AgentResponse
		err := query.Scan(ctx, &responses)
		return responses, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: session responses: %w", err))
	}
	return merged(lists, oldestResponseFirst, SessionLimit(limit)+1), nil
}

func oldestResponseFirst(a, b AgentResponse) int {
	return cmp.Or(a.CreatedAt.Compare(b.CreatedAt), strings.Compare(a.ID, b.ID))
}

// AppendResponseItems writes a batch of items. They are written together because they
// arrive together: a turn that uses three tools produces its items in bursts, and one round
// trip each would have the writer behind the conversation it is recording.
//
// Writing the same item twice is not an error. The writer batches on a timer as well as on
// a full buffer, so a flush overlapping the end of a turn can offer an item the previous
// flush already took, and the ordinal is what settles it.
func (s *Store) AppendResponseItems(ctx context.Context, items []AgentResponseItem) error {
	if len(items) == 0 {
		return nil
	}
	for i := range items {
		if items[i].ResponseID == "" || items[i].SessionID == "" {
			return errors.New("store: a response and a session id are required")
		}
		if items[i].Kind == "" {
			return errors.New("store: an item kind is required")
		}
		items[i].Text = SafeLogText(items[i].Text)
		if items[i].At.IsZero() {
			items[i].At = time.Now().UTC()
		}
	}

	_, err := s.shard.NewInsert().Model(&items).
		On("CONFLICT (response_id, ordinal) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: append response items: %w", err)
	}
	return nil
}

// redactToolOutputSQL is AgentResponseItem.Redact for up to ?1 rows of this shard whose
// redact_at is no later than ?0, oldest first. SKIP LOCKED because every router of the
// shard runs it: two of them on one row would wait on each other for nothing.
const redactToolOutputSQL = `
WITH due AS (
    SELECT response_id, ordinal FROM agent_response_items
    WHERE redact_at IS NOT NULL AND redact_at <= ?0
    ORDER BY redact_at
    LIMIT ?1
    FOR UPDATE SKIP LOCKED
)
UPDATE agent_response_items AS t SET
    text = '',
    payload = jsonb_strip_nulls(jsonb_build_object(
        'call_id', t.payload->'call_id',
        'product', t.payload->'product',
        'sdk', t.payload->'sdk',
        'error', CASE WHEN t.payload->'error' IS NOT NULL THEN to_jsonb('redacted'::text) END,
        'redacted', true)),
    redact_at = NULL
FROM due
WHERE t.response_id = due.response_id AND t.ordinal = due.ordinal`

// RedactToolOutput takes the arguments and output out of up to limit tool items in this
// router's shard whose time to be redacted has come by now, and returns how many it
// redacted. Each row's time was set from its own customer's retention when it was written,
// so a run reads no customer's settings and touches no row that is not due. The other
// shards' routers redact theirs.
func (s *Store) RedactToolOutput(ctx context.Context, now time.Time, limit int) (int64, error) {
	if limit <= 0 {
		return 0, nil
	}
	result, err := s.shard.NewRaw(redactToolOutputSQL, now, limit).Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: redact tool output: %w", err))
	}
	return result.RowsAffected()
}

// SessionItems returns a session's items in the order they happened, across every turn.
//
// One flat stream rather than a list per turn, because that is how a conversation reads and
// how it is rendered: the question, what the agent did about it, what it said, then the next
// question. A caller that wants one turn's items passes that response id. A nil after is
// the first page.
func (s *Store) SessionItems(ctx context.Context, customerID, sessionID, responseID string, limit int, after *ItemPosition) ([]AgentResponseItem, error) {
	if customerID == "" || sessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	// Joined to responses rather than trusting the session id on the item, because the
	// customer is on the response and an item is only this caller's if the turn that made
	// it was.
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponseItem, error) {
		query := db.NewSelect().Model((*AgentResponseItem)(nil)).
			Join("JOIN agent_responses AS r ON r.id = ari.response_id").
			Where("r.customer_id = ?", customerID).
			Where("ari.session_id = ?", sessionID).
			Where("r.rewound_at IS NULL").
			Order("ari.at ASC", "ari.response_id ASC", "ari.ordinal ASC").
			Limit(ItemLimit(limit) + 1)
		if responseID != "" {
			query = query.Where("ari.response_id = ?", responseID)
		}
		if after != nil {
			query = query.Where("(ari.at, ari.response_id, ari.ordinal) > (?, ?, ?)",
				after.At, after.ResponseID, after.Ordinal)
		}
		var items []AgentResponseItem
		err := query.Scan(ctx, &items)
		return items, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: session items: %w", err))
	}
	return merged(lists, func(a, b AgentResponseItem) int {
		return cmp.Or(a.At.Compare(b.At), strings.Compare(a.ResponseID, b.ResponseID), cmp.Compare(a.Ordinal, b.Ordinal))
	}, ItemLimit(limit)+1), nil
}

// ErrUnknownResponse is a response that is not part of the session's conversation: it does
// not exist, belongs to another session, or was rewound.
var ErrUnknownResponse = errors.New("store: that response is not part of this conversation")

// Exchanges returns a session's conversation as a model would be given it again, oldest
// first: each turn's question and everything it answered. A non-empty upTo stops after that
// response, which is where a rewind or a fork carries on from.
//
// Only questions and answers are kept. A tool's call and its result were the agent working
// the answer out, and the answer is what the conversation carries forward, which is also all
// a transcript read out of Chat gives a model.
func (s *Store) Exchanges(ctx context.Context, customerID, sessionID, upTo string) ([]Exchange, error) {
	if customerID == "" || sessionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a session id are required"))
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponse, error) {
		var responses []AgentResponse
		err := db.NewSelect().Model(&responses).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("rewound_at IS NULL").
			Order("created_at ASC", "id ASC").
			Scan(ctx)
		return responses, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: exchanges: %w", err))
	}
	responses := merged(lists, oldestResponseFirst, 0)
	if upTo != "" {
		end := -1
		for i, response := range responses {
			if response.ID == upTo {
				end = i
				break
			}
		}
		if end < 0 {
			return nil, stack.Wrap(ErrUnknownResponse)
		}
		responses = responses[:end+1]
	}
	if len(responses) == 0 {
		return nil, nil
	}

	ids := make([]string, 0, len(responses))
	for _, response := range responses {
		ids = append(ids, response.ID)
	}
	answerLists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]AgentResponseItem, error) {
		var answers []AgentResponseItem
		err := db.NewSelect().Model(&answers).
			Where("response_id IN (?)", bun.In(ids)).
			Where("kind = ?", ItemAnswer).
			Order("response_id ASC", "ordinal ASC").
			Scan(ctx)
		return answers, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: exchanges: answers: %w", err))
	}
	// An agent that speaks again once delegated work comes back answers one question twice,
	// and both halves are the answer. A response's items are all in the shard that ran it.
	answered := map[string][]string{}
	for _, answers := range answerLists {
		for _, answer := range answers {
			answered[answer.ResponseID] = append(answered[answer.ResponseID], answer.Text)
		}
	}

	exchanges := make([]Exchange, 0, len(responses))
	for _, response := range responses {
		exchanges = append(exchanges, Exchange{
			ResponseID: response.ID, Said: response.Said,
			Answer: strings.Join(answered[response.ID], "\n\n"),
		})
	}
	return exchanges, nil
}

// RewindResponses takes every response after kept out of the session's conversation.
func (s *Store) RewindResponses(ctx context.Context, customerID, sessionID, kept string, at time.Time) error {
	if customerID == "" || sessionID == "" || kept == "" {
		return stack.Wrap(errors.New("store: a customer, a session and a response id are required"))
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	// The response kept may be in another shard than the ones rewound, so its position is
	// read first rather than in each shard's update.
	keptAt, err := whicheverShard(ctx, s, func(ctx context.Context, db bun.IDB) (AgentResponse, error) {
		var response AgentResponse
		err := db.NewSelect().Model(&response).Column("created_at", "id").Where("id = ?", kept).Scan(ctx)
		return response, err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return nil
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: rewind responses: %w", err))
	}
	_, err = onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentResponse)(nil)).
			Set("rewound_at = ?", at).
			Where("customer_id = ?", customerID).
			Where("session_id = ?", sessionID).
			Where("rewound_at IS NULL").
			Where("(created_at, id) > (?, ?)", keptAt.CreatedAt, keptAt.ID).
			Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: rewind responses: %w", err))
	}
	return nil
}

// RecordUser stores an end user this app was seen acting for.
func (s *Store) RecordUser(ctx context.Context, user *User) error {
	if user.ID == "" {
		return stack.Wrap(errors.New("store: a user id is required"))
	}
	if user.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if user.Kind == "" {
		return stack.Wrap(errors.New("store: a user kind is required"))
	}
	if user.CreatedAt.IsZero() {
		user.CreatedAt = time.Now().UTC()
	}
	if user.Custom == nil {
		user.Custom = map[string]any{}
	}

	// Recording a user is get-or-create, so the same id arriving again is the same person
	// coming back rather than a conflict. A row already here is left alone: a guest who
	// signed up is claimed rather than overwritten, and overwriting would lose the kind
	// the claim path reads.
	_, err := s.db.NewInsert().Model(user).
		On("CONFLICT (customer_id, id) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: record user: %w", err))
	}
	return nil
}

// SetUserEmail records the address a server-side caller said a user signs in with. The row
// is created when there is none; on an existing row only the email moves, never the kind
// the claim path reads, and an email it already holds is not written again: a backend may
// say it on every request.
func (s *Store) SetUserEmail(ctx context.Context, customerID, userID, email string) error {
	if customerID == "" || userID == "" || email == "" {
		return nil
	}
	user := &User{CustomerID: customerID, ID: userID, Kind: UserKindAuthenticated, Email: email, CreatedAt: time.Now().UTC()}
	_, err := s.db.NewInsert().Model(user).
		On("CONFLICT (customer_id, id) DO UPDATE").
		Set("email = EXCLUDED.email").
		Where("u.email IS DISTINCT FROM EXCLUDED.email").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set user email: %w", err))
	}
	return nil
}

// UserEmail returns the address last recorded for a user, or empty.
func (s *Store) UserEmail(ctx context.Context, customerID, userID string) (string, error) {
	var user User
	err := s.db.NewSelect().Model(&user).
		Column("email").
		Where("customer_id = ?", customerID).
		Where("id = ?", userID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("store: user email: %w", err))
	}
	return user.Email, nil
}

// RecordGuest stores a guest this app handed out, so it can later be claimed.
func (s *Store) RecordGuest(ctx context.Context, guest *User) error {
	guest.Kind = UserKindGuest
	return s.RecordUser(ctx, guest)
}

// Guest returns one guest of a customer's. A user of some other kind is not one: an
// account that signed up is nobody's to claim.
func (s *Store) Guest(ctx context.Context, customerID, id string) (User, error) {
	if customerID == "" || id == "" {
		return User{}, stack.Wrap(errors.New("store: a customer and a guest id are required"))
	}

	var guest User
	err := s.db.NewSelect().Model(&guest).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("kind = ?", UserKindGuest).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return User{}, notAGuest(id, customerID)
	}
	if err != nil {
		return User{}, stack.Wrap(fmt.Errorf("store: guest: %w", err))
	}
	return guest, nil
}

// ClaimGuest moves a guest's sessions onto a real user and returns how many moved.
//
// One transaction, because the two halves are one fact. A guest marked claimed whose
// sessions still say the guest owns them is a person who signed up and lost their history;
// sessions moved without the guest being marked is a guest that can be claimed again, by
// somebody else.
//
// With shards, the guest is in the shared database and the sessions in every shard, and no
// transaction spans them. The guest is claimed first, and a claim by the same user again moves
// whatever sessions a failed one left behind.
func (s *Store) ClaimGuest(ctx context.Context, customerID, guestID, userID string) (int64, error) {
	if customerID == "" || guestID == "" || userID == "" {
		return 0, stack.Wrap(errors.New("store: a customer, a guest and a user id are required"))
	}
	if guestID == userID {
		return 0, stack.Wrap(errors.New("store: a guest cannot be claimed by itself"))
	}

	now := time.Now().UTC()
	moveSessions := func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentSession)(nil)).
			Set("user_id = ?", userID).
			Set("updated_at = ?", now).
			Where("customer_id = ?", customerID).
			Where("user_id = ?", guestID).
			Exec(ctx)
	}

	if !s.Sharded() && len(s.peers) == 0 {
		var moved int64
		err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			if err := claimGuest(ctx, tx, customerID, guestID, userID, now, false); err != nil {
				return err
			}
			result, err := moveSessions(ctx, tx)
			if err != nil {
				return fmt.Errorf("claim guest sessions: %w", err)
			}
			moved, err = result.RowsAffected()
			if err != nil {
				return fmt.Errorf("claim guest sessions: %w", err)
			}
			return nil
		})
		if err != nil {
			return 0, stack.Wrap(fmt.Errorf("store: %w", err))
		}
		return moved, nil
	}

	if err := claimGuest(ctx, s.db, customerID, guestID, userID, now, true); err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: %w", err))
	}
	moved, err := onEveryShard(ctx, s, moveSessions)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: claim guest sessions: %w", err))
	}
	return moved, nil
}

// claimGuest marks a guest claimed by userID. sameUserAgain lets a claim userID already made
// stand rather than fail, for a caller that has more to finish after it.
func claimGuest(ctx context.Context, db bun.IDB, customerID, guestID, userID string, now time.Time, sameUserAgain bool) error {
	claimed, err := db.NewUpdate().Model((*User)(nil)).
		Set("claimed_by = ?", userID).
		Set("claimed_at = ?", now).
		Where("id = ?", guestID).
		Where("customer_id = ?", customerID).
		Where("kind = ?", UserKindGuest).
		Where("claimed_by IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("claim guest: %w", err)
	}
	affected, err := claimed.RowsAffected()
	if err != nil {
		return fmt.Errorf("claim guest: %w", err)
	}
	if affected > 0 {
		return nil
	}
	// Either there is no such guest or somebody already claimed them, and the caller is told
	// which: a guest already claimed is a retry, and a guest that never existed is a
	// different mistake.
	var existing User
	err = db.NewSelect().Model(&existing).
		Where("id = ?", guestID).
		Where("customer_id = ?", customerID).
		Where("kind = ?", UserKindGuest).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return notAGuest(guestID, customerID)
	}
	if err != nil {
		return fmt.Errorf("claim guest: %w", err)
	}
	if sameUserAgain && existing.ClaimedBy == userID {
		return nil
	}
	return fmt.Errorf("guest %s was already claimed by %s", guestID, existing.ClaimedBy)
}

// jsonbOf renders a map for a jsonb comparison. A map that will not encode is sent as an
// empty object rather than failing the query: the values come from a query string, and the
// worst an unencodable one can mean is that nothing matches it.
func jsonbOf[V any](values map[string]V) string {
	encoded, err := json.Marshal(values)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

func unknownStoredSession(id string) error {
	return stack.Wrap(fmt.Errorf("store: there is no session %s", id))
}

func notAGuest(id, customerID string) error {
	return stack.Wrap(fmt.Errorf("store: %s is not a guest of %s", id, customerID))
}
