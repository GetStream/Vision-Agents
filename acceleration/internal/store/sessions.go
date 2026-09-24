package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/uptrace/bun"
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

// SaveSession records that a session was opened.
//
// Written when the session starts rather than when it ends, so a session that is still
// running is findable and one the process died during is not simply lost. Saving the same
// session twice updates the row: a retried write must not become a second conversation.
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
	if session.Custom == nil {
		session.Custom = map[string]any{}
	}

	_, err := s.db.NewInsert().Model(session).
		On("CONFLICT (id) DO UPDATE").
		Set("title = EXCLUDED.title").
		Set("description = EXCLUDED.description").
		Set("custom = EXCLUDED.custom").
		Set("conversation_id = EXCLUDED.conversation_id").
		Set("state = EXCLUDED.state").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save session: %w", err)
	}
	return nil
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

	_, err := s.db.NewUpdate().Model((*AgentSession)(nil)).
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

// DescribeSession changes what a session is called. It is separate from SaveSession because
// renaming a conversation is something a person does long after it ended, when there is no
// spec left to save.
func (s *Store) DescribeSession(ctx context.Context, customerID, id, title, description string, custom map[string]any) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a session id are required")
	}

	query := s.db.NewUpdate().Model((*AgentSession)(nil)).
		Set("updated_at = ?", time.Now().UTC()).
		Set("title = ?", title).
		Set("description = ?", description).
		Where("id = ?", id).
		Where("customer_id = ?", customerID)
	if custom != nil {
		query = query.Set("custom = ?", jsonbOf(custom))
	}

	result, err := query.Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: describe session: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: describe session: %w", err)
	}
	if affected == 0 {
		return unknownStoredSession(id)
	}
	return nil
}

// StoredSession returns one session a customer ran.
func (s *Store) StoredSession(ctx context.Context, customerID, id string) (AgentSession, error) {
	if customerID == "" || id == "" {
		return AgentSession{}, errors.New("store: a customer and a session id are required")
	}

	var session AgentSession
	err := s.db.NewSelect().Model(&session).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AgentSession{}, unknownStoredSession(id)
	}
	if err != nil {
		return AgentSession{}, fmt.Errorf("store: stored session: %w", err)
	}
	return session, nil
}

// QuerySessions returns a customer's sessions, newest first, narrowed by the filter.
//
// This is the structured half of finding an old conversation: everything here is an exact
// match or a range, so the answer is the same every time and a page can hold a cursor into
// it. SearchSessions is the other half, for when what the caller has is words rather than a
// filter.
func (s *Store) QuerySessions(ctx context.Context, customerID string, filter SessionFilter) ([]AgentSession, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	query := s.db.NewSelect().Model((*AgentSession)(nil)).
		Where("customer_id = ?", customerID).
		Order("created_at DESC", "id DESC")
	query = narrowSessions(query, filter)

	var sessions []AgentSession
	if err := query.Scan(ctx, &sessions); err != nil {
		return nil, fmt.Errorf("store: query sessions: %w", err)
	}
	return sessions, nil
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
		return nil, errors.New("store: customer id is required")
	}
	if text == "" {
		return s.QuerySessions(ctx, customerID, filter)
	}

	// websearch_to_tsquery rather than to_tsquery, because the text comes from a search
	// box: it takes quoted phrases and bare words and never fails on punctuation, where
	// to_tsquery would answer a syntax error to somebody who typed an apostrophe.
	query := s.db.NewSelect().Model((*AgentSession)(nil)).
		Where("customer_id = ?", customerID).
		Where("searchable @@ websearch_to_tsquery('english', ?)", text).
		OrderExpr("ts_rank(searchable, websearch_to_tsquery('english', ?)) DESC", text).
		Order("created_at DESC", "id DESC")
	query = narrowSessions(query, filter)

	var sessions []AgentSession
	if err := query.Scan(ctx, &sessions); err != nil {
		return nil, fmt.Errorf("store: search sessions: %w", err)
	}
	return sessions, nil
}

// narrowSessions applies a filter to either kind of session query, so the two cannot drift
// apart in what they admit. That matters more than the duplication it saves: a filter
// honoured by query and forgotten by search is one a caller uses to read somebody else's
// conversations.
func narrowSessions(query *bun.SelectQuery, filter SessionFilter) *bun.SelectQuery {
	limit := filter.Limit
	if limit <= 0 {
		limit = defaultSessionLimit
	}
	if limit > maxSessionLimit {
		limit = maxSessionLimit
	}
	query = query.Limit(limit)
	if filter.Offset > 0 {
		query = query.Offset(filter.Offset)
	}

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
	switch filter.State {
	case SessionRunning:
		query = query.Where("closed_at IS NULL")
	case SessionClosed:
		query = query.Where("closed_at IS NOT NULL")
	}
	if len(filter.Custom) > 0 {
		// Containment rather than a key at a time, so the GIN index on custom is usable
		// and a caller asking for two labels gets the sessions carrying both.
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

	_, err := s.db.NewInsert().Model(response).
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

	_, err := s.db.NewUpdate().Model((*AgentResponse)(nil)).
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
	_, err = s.db.NewUpdate().Model((*AgentSession)(nil)).
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
// conversation.
func (s *Store) SessionResponses(ctx context.Context, customerID, sessionID string, limit, offset int) ([]AgentResponse, error) {
	if customerID == "" || sessionID == "" {
		return nil, errors.New("store: a customer and a session id are required")
	}
	if limit <= 0 {
		limit = defaultSessionLimit
	}
	if limit > maxSessionLimit {
		limit = maxSessionLimit
	}

	query := s.db.NewSelect().Model((*AgentResponse)(nil)).
		Where("customer_id = ?", customerID).
		Where("session_id = ?", sessionID).
		Where("rewound_at IS NULL").
		Order("created_at ASC", "id ASC").
		Limit(limit)
	if offset > 0 {
		query = query.Offset(offset)
	}

	var responses []AgentResponse
	if err := query.Scan(ctx, &responses); err != nil {
		return nil, fmt.Errorf("store: session responses: %w", err)
	}
	return responses, nil
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

	_, err := s.db.NewInsert().Model(&items).
		On("CONFLICT (response_id, ordinal) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: append response items: %w", err)
	}
	return nil
}

// SessionItems returns a session's items in the order they happened, across every turn.
//
// One flat stream rather than a list per turn, because that is how a conversation reads and
// how it is rendered: the question, what the agent did about it, what it said, then the next
// question. A caller that wants one turn's items passes that response id.
func (s *Store) SessionItems(ctx context.Context, customerID, sessionID, responseID string, limit, offset int) ([]AgentResponseItem, error) {
	if customerID == "" || sessionID == "" {
		return nil, errors.New("store: a customer and a session id are required")
	}
	if limit <= 0 {
		limit = defaultItemLimit
	}
	if limit > maxItemLimit {
		limit = maxItemLimit
	}

	// Joined to responses rather than trusting the session id on the item, because the
	// customer is on the response and an item is only this caller's if the turn that made
	// it was.
	query := s.db.NewSelect().Model((*AgentResponseItem)(nil)).
		Join("JOIN agent_responses AS r ON r.id = ari.response_id").
		Where("r.customer_id = ?", customerID).
		Where("ari.session_id = ?", sessionID).
		Where("r.rewound_at IS NULL").
		Order("ari.at ASC", "ari.response_id ASC", "ari.ordinal ASC").
		Limit(limit)
	if responseID != "" {
		query = query.Where("ari.response_id = ?", responseID)
	}
	if offset > 0 {
		query = query.Offset(offset)
	}

	var items []AgentResponseItem
	if err := query.Scan(ctx, &items); err != nil {
		return nil, fmt.Errorf("store: session items: %w", err)
	}
	return items, nil
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
		return nil, errors.New("store: a customer and a session id are required")
	}

	var responses []AgentResponse
	err := s.db.NewSelect().Model(&responses).
		Where("customer_id = ?", customerID).
		Where("session_id = ?", sessionID).
		Where("rewound_at IS NULL").
		Order("created_at ASC", "id ASC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: exchanges: %w", err)
	}
	if upTo != "" {
		end := -1
		for i, response := range responses {
			if response.ID == upTo {
				end = i
				break
			}
		}
		if end < 0 {
			return nil, ErrUnknownResponse
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
	var answers []AgentResponseItem
	err = s.db.NewSelect().Model(&answers).
		Where("response_id IN (?)", bun.In(ids)).
		Where("kind = ?", ItemAnswer).
		Order("response_id ASC", "ordinal ASC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: exchanges: answers: %w", err)
	}
	// An agent that speaks again once delegated work comes back answers one question twice,
	// and both halves are the answer.
	answered := map[string][]string{}
	for _, answer := range answers {
		answered[answer.ResponseID] = append(answered[answer.ResponseID], answer.Text)
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
		return errors.New("store: a customer, a session and a response id are required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	_, err := s.db.NewUpdate().Model((*AgentResponse)(nil)).
		Set("rewound_at = ?", at).
		Where("customer_id = ?", customerID).
		Where("session_id = ?", sessionID).
		Where("rewound_at IS NULL").
		Where("(created_at, id) > (SELECT created_at, id FROM agent_responses WHERE id = ?)", kept).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: rewind responses: %w", err)
	}
	return nil
}

// RecordGuest stores a guest this app handed out, so it can later be claimed.
func (s *Store) RecordGuest(ctx context.Context, guest *GuestUser) error {
	if guest.ID == "" {
		return errors.New("store: a guest id is required")
	}
	if guest.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if guest.CreatedAt.IsZero() {
		guest.CreatedAt = time.Now().UTC()
	}
	if guest.Custom == nil {
		guest.Custom = map[string]any{}
	}

	// Getting a guest is get-or-create, so the same id arriving again is the same person
	// coming back rather than a conflict.
	_, err := s.db.NewInsert().Model(guest).
		On("CONFLICT (id) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: record guest: %w", err)
	}
	return nil
}

// Guest returns one guest of a customer's.
func (s *Store) Guest(ctx context.Context, customerID, id string) (GuestUser, error) {
	if customerID == "" || id == "" {
		return GuestUser{}, errors.New("store: a customer and a guest id are required")
	}

	var guest GuestUser
	err := s.db.NewSelect().Model(&guest).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return GuestUser{}, notAGuest(id, customerID)
	}
	if err != nil {
		return GuestUser{}, fmt.Errorf("store: guest: %w", err)
	}
	return guest, nil
}

// ClaimGuest moves a guest's sessions onto a real user and returns how many moved.
//
// One transaction, because the two halves are one fact. A guest marked claimed whose
// sessions still say the guest owns them is a person who signed up and lost their history;
// sessions moved without the guest being marked is a guest that can be claimed again, by
// somebody else.
func (s *Store) ClaimGuest(ctx context.Context, customerID, guestID, userID string) (int64, error) {
	if customerID == "" || guestID == "" || userID == "" {
		return 0, errors.New("store: a customer, a guest and a user id are required")
	}
	if guestID == userID {
		return 0, errors.New("store: a guest cannot be claimed by itself")
	}

	var moved int64
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		now := time.Now().UTC()
		claimed, err := tx.NewUpdate().Model((*GuestUser)(nil)).
			Set("claimed_by = ?", userID).
			Set("claimed_at = ?", now).
			Where("id = ?", guestID).
			Where("customer_id = ?", customerID).
			Where("claimed_by IS NULL").
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("claim guest: %w", err)
		}
		affected, err := claimed.RowsAffected()
		if err != nil {
			return fmt.Errorf("claim guest: %w", err)
		}
		if affected == 0 {
			// Either there is no such guest or somebody already claimed them, and the
			// caller is told which: a guest already claimed is a retry, and a guest that
			// never existed is a different mistake.
			var existing GuestUser
			err := tx.NewSelect().Model(&existing).
				Where("id = ?", guestID).
				Where("customer_id = ?", customerID).
				Limit(1).
				Scan(ctx)
			if errors.Is(err, sql.ErrNoRows) {
				return notAGuest(guestID, customerID)
			}
			if err != nil {
				return fmt.Errorf("claim guest: %w", err)
			}
			return fmt.Errorf("guest %s was already claimed by %s", guestID, existing.ClaimedBy)
		}

		result, err := tx.NewUpdate().Model((*AgentSession)(nil)).
			Set("user_id = ?", userID).
			Set("updated_at = ?", now).
			Where("customer_id = ?", customerID).
			Where("user_id = ?", guestID).
			Exec(ctx)
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
		return 0, fmt.Errorf("store: %w", err)
	}
	return moved, nil
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
	return fmt.Errorf("store: there is no session %s", id)
}

func notAGuest(id, customerID string) error {
	return fmt.Errorf("store: %s is not a guest of %s", id, customerID)
}
