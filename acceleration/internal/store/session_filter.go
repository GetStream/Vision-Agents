package store

import (
	"cmp"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"
)

// SessionField is a field of a session a caller's filter may name.
type SessionField string

const (
	// SessionFieldProject is what the conversation belongs to.
	SessionFieldProject SessionField = "project_id"
	// SessionFieldReviewScore is what the review made of it, from 1 to 5. A session nobody
	// reviewed has none, and matches no condition on it.
	SessionFieldReviewScore SessionField = "review_score"
	// SessionFieldAgent is the agent name the session was opened against.
	SessionFieldAgent SessionField = "agent"
	// SessionFieldUser is whose session it is.
	SessionFieldUser SessionField = "user_id"
	// SessionFieldConfig is the agent config the session ran under, which a rename leaves alone.
	SessionFieldConfig SessionField = "config_id"
	// SessionFieldState is SessionRunning or SessionClosed.
	SessionFieldState SessionField = "state"
	// SessionFieldCreatedAt is when the session began.
	SessionFieldCreatedAt SessionField = "created_at"
	// SessionFieldCustom is the caller's own labels, matched by containment.
	SessionFieldCustom SessionField = "custom"
)

// column is the agent_sessions column a field is kept in. Only these names are ever written
// into a statement: everything a caller sent is bound.
func (f SessionField) column() string {
	switch f {
	case SessionFieldProject:
		return "project"
	case SessionFieldReviewScore:
		return "review_score"
	case SessionFieldAgent:
		return "agent_name"
	case SessionFieldUser:
		return "user_id"
	case SessionFieldConfig, SessionFieldState, SessionFieldCreatedAt, SessionFieldCustom:
		return string(f)
	}
	return ""
}

// SessionOperator is how a condition compares its field.
type SessionOperator string

const (
	SessionEq  SessionOperator = "$eq"
	SessionIn  SessionOperator = "$in"
	SessionGt  SessionOperator = "$gt"
	SessionGte SessionOperator = "$gte"
	SessionLt  SessionOperator = "$lt"
	SessionLte SessionOperator = "$lte"
)

// comparison is the operator in SQL, empty for one a number is not compared by.
func (o SessionOperator) comparison() string {
	switch o {
	case SessionEq:
		return "="
	case SessionGt:
		return ">"
	case SessionGte:
		return ">="
	case SessionLt:
		return "<"
	case SessionLte:
		return "<="
	}
	return ""
}

// SessionExpr is a caller's filter: a condition on one field, or a group of expressions
// that all have to hold (AllOf) or of which one has to (AnyOf).
//
// It is what the request asked for and nothing else. Who the caller is allowed to see is
// not in here: SessionFilter's own fields carry that, and they are applied around the whole
// expression, so no $or inside it can widen them.
type SessionExpr struct {
	// Any makes a group one of whose Children has to hold rather than all of them.
	Any      bool
	Children []SessionExpr

	// Field is set on a condition, and a condition has no Children.
	Field SessionField
	Op    SessionOperator
	// Strings is what a text field is compared with: one value for $eq, several for $in.
	Strings []string
	// Number is what review_score is compared with.
	Number int
	// Time is what created_at is compared with.
	Time time.Time
	// Labels are the pairs custom has to hold, every one of them.
	Labels map[string]string
}

// AllOf is the expressions ANDed. None at all holds for every session.
func AllOf(children ...SessionExpr) SessionExpr { return SessionExpr{Children: children} }

// AnyOf is the expressions ORed. None at all holds for no session.
func AnyOf(children ...SessionExpr) SessionExpr { return SessionExpr{Any: true, Children: children} }

// SessionIs is a text field equal to one value.
func SessionIs(field SessionField, value string) SessionExpr {
	return SessionExpr{Field: field, Op: SessionEq, Strings: []string{value}}
}

// SessionIsOneOf is a text field equal to any of the values.
func SessionIsOneOf(field SessionField, values ...string) SessionExpr {
	return SessionExpr{Field: field, Op: SessionIn, Strings: values}
}

// ReviewScoreIs is the review score compared with a number by $eq, $gt, $gte, $lt or $lte.
func ReviewScoreIs(op SessionOperator, score int) SessionExpr {
	return SessionExpr{Field: SessionFieldReviewScore, Op: op, Number: score}
}

// CreatedAtIs is when the session began compared with a time by $eq, $gt, $gte, $lt or $lte.
func CreatedAtIs(op SessionOperator, at time.Time) SessionExpr {
	return SessionExpr{Field: SessionFieldCreatedAt, Op: op, Time: at}
}

// CustomHolds is the session's custom object holding every one of the pairs.
func CustomHolds(labels map[string]string) SessionExpr {
	return SessionExpr{Field: SessionFieldCustom, Op: SessionEq, Labels: labels}
}

// SessionValues is what a filter reads of one session, for a session that has no row to ask.
type SessionValues struct {
	Project  string
	Agent    string
	UserID   string
	ConfigID string
	// State is SessionRunning or SessionClosed.
	State string
	// CreatedAt is cut to the microsecond, as a row keeps it.
	CreatedAt time.Time
	Custom    map[string]any
	// ReviewScore is nil for a session nobody reviewed.
	ReviewScore *int
}

// Matches reports whether the expression holds for a session, the way the statement where
// renders decides it for a row. A nil expression holds for every session.
func (e *SessionExpr) Matches(values SessionValues) bool {
	if e == nil {
		return true
	}
	if e.Field == "" {
		if e.Any {
			return slices.ContainsFunc(e.Children, func(child SessionExpr) bool { return child.Matches(values) })
		}
		return !slices.ContainsFunc(e.Children, func(child SessionExpr) bool { return !child.Matches(values) })
	}

	if e.Field == SessionFieldReviewScore {
		// A missing score is NULL in the statement, which no comparison is true of.
		if values.ReviewScore == nil {
			return false
		}
		score := *values.ReviewScore
		switch e.Op {
		case SessionEq:
			return score == e.Number
		case SessionGt:
			return score > e.Number
		case SessionGte:
			return score >= e.Number
		case SessionLt:
			return score < e.Number
		case SessionLte:
			return score <= e.Number
		}
		return false
	}

	switch e.Field {
	case SessionFieldCreatedAt:
		compared := values.CreatedAt.Compare(e.Time)
		switch e.Op {
		case SessionEq:
			return compared == 0
		case SessionGt:
			return compared > 0
		case SessionGte:
			return compared >= 0
		case SessionLt:
			return compared < 0
		case SessionLte:
			return compared <= 0
		}
		return false
	case SessionFieldCustom:
		// Containment compares JSON values, so a number held is not the string it prints as.
		for key, wanted := range e.Labels {
			if held, ok := values.Custom[key].(string); !ok || held != wanted {
				return false
			}
		}
		return len(e.Labels) > 0
	}

	var held string
	switch e.Field {
	case SessionFieldProject:
		held = values.Project
	case SessionFieldAgent:
		held = values.Agent
	case SessionFieldUser:
		held = values.UserID
	case SessionFieldConfig:
		held = values.ConfigID
	case SessionFieldState:
		held = values.State
	default:
		return false
	}
	switch e.Op {
	case SessionEq:
		return len(e.Strings) == 1 && e.Strings[0] == held
	case SessionIn:
		return slices.Contains(e.Strings, held)
	}
	return false
}

// where is the expression as a parenthesised condition and its arguments. Every value is
// bound: the only text written into the statement is a column name and an operator this
// package chose. A condition this package does not know holds for nothing rather than for
// everything.
func (e *SessionExpr) where() (string, []any) {
	if e.Field == "" {
		if len(e.Children) == 0 {
			if e.Any {
				return "FALSE", nil
			}
			return "TRUE", nil
		}
		joined := " AND "
		if e.Any {
			joined = " OR "
		}
		parts := make([]string, 0, len(e.Children))
		var args []any
		for i := range e.Children {
			part, bound := e.Children[i].where()
			parts = append(parts, part)
			args = append(args, bound...)
		}
		return "(" + strings.Join(parts, joined) + ")", args
	}

	column := e.Field.column()
	switch {
	case column == "":
		return "FALSE", nil
	case e.Field == SessionFieldReviewScore:
		if comparison := e.Op.comparison(); comparison != "" {
			return "(" + column + " " + comparison + " ?)", []any{e.Number}
		}
	case e.Field == SessionFieldCreatedAt:
		if comparison := e.Op.comparison(); comparison != "" {
			return "(" + column + " " + comparison + " ?)", []any{e.Time}
		}
	case e.Field == SessionFieldCustom:
		// Containment, so the GIN index on custom serves it. The index is jsonb_path_ops,
		// which serves @> and nothing else.
		if len(e.Labels) > 0 {
			return "(" + column + " @> ?::jsonb)", []any{jsonbOf(e.Labels)}
		}
	case e.Op == SessionEq && len(e.Strings) == 1:
		return "(" + column + " = ?)", []any{e.Strings[0]}
	case e.Op == SessionIn && len(e.Strings) > 0:
		return "(" + column + " IN (?))", []any{bun.In(e.Strings)}
	}
	return "FALSE", nil
}

// SessionSortKey is what a session list is ordered by.
type SessionSortKey string

const (
	// SessionSortLastMessage is when the agent last answered, or when the session began for
	// one it has not answered in. The zero value, so a filter that names no order has it.
	SessionSortLastMessage SessionSortKey = ""
	// SessionSortReviewScore is the review's score, the sessions without one last.
	SessionSortReviewScore SessionSortKey = "review_score"
	// SessionSortRelevance is how well a search matched, which only a search has.
	SessionSortRelevance SessionSortKey = "relevance"
)

// SessionOrder is the order of a session list. The zero value is the most recent message
// first. Every order ends on the id, descending, so no two sessions ever tie and a cursor
// names exactly one place in the list.
type SessionOrder struct {
	By SessionSortKey
	// Ascending turns the key around: the oldest message, or the lowest score, first.
	// Relevance is only ever best first.
	Ascending bool
	// PinnedFirst lists every pinned session before any that is not.
	PinnedFirst bool
}

// lastMessageAt is when a session last had a message, as SQL: the expression the indexes
// are built on.
const lastMessageAt = "COALESCE(last_response_at, created_at)"

// LastMessageAt is when the agent last answered, or when the session began if it never did.
func (s *AgentSession) LastMessageAt() time.Time {
	if s.LastResponseAt != nil {
		return *s.LastResponseAt
	}
	return s.CreatedAt
}

// Position is where the session sits in a list, by every key any order sorts on.
func (s *AgentSession) Position() SessionPosition {
	return SessionPosition{
		LastMessageAt: s.LastMessageAt(),
		ID:            s.ID,
		Rank:          s.Rank,
		Pinned:        s.PinnedAt != nil,
		ReviewScore:   s.ReviewScore,
	}
}

// Compare orders two sessions: negative when a is listed before b. It is the statement's
// ORDER BY, for merging the pages of several shards and the sessions nothing records.
func (o SessionOrder) Compare(a, b SessionPosition) int {
	turned := func(compared int, ascending bool) int {
		if ascending {
			return compared
		}
		return -compared
	}
	var keys []int
	if o.PinnedFirst {
		keys = append(keys, -cmp.Compare(flag(a.Pinned), flag(b.Pinned)))
	}
	messages := a.LastMessageAt.Compare(b.LastMessageAt)
	switch o.By {
	case SessionSortRelevance:
		keys = append(keys, -cmp.Compare(a.Rank, b.Rank), -messages)
	case SessionSortReviewScore:
		keys = append(keys,
			cmp.Compare(flag(a.ReviewScore == nil), flag(b.ReviewScore == nil)),
			turned(cmp.Compare(scoreOf(a.ReviewScore), scoreOf(b.ReviewScore)), o.Ascending),
			-messages)
	default:
		keys = append(keys, turned(messages, o.Ascending))
	}
	keys = append(keys, -strings.Compare(a.ID, b.ID))
	return cmp.Or(keys...)
}

func flag(set bool) int {
	if set {
		return 1
	}
	return 0
}

func scoreOf(score *int) int {
	if score == nil {
		return 0
	}
	return *score
}

// sessionKey is one key of an order: its expression, which way it sorts, and where a
// cursor sits on it.
type sessionKey struct {
	expr string
	// args are what the expression itself binds, which only a search's rank does.
	args []any
	// placeholder is how the cursor's value is bound against the expression.
	placeholder string
	ascending   bool
	value       any
}

// keys is the order as the keys a statement sorts on, with the cursor's value for each.
// rank is a search's rank expression and rankArgs what it binds.
func (o SessionOrder) keys(rank string, rankArgs []any, at SessionPosition) []sessionKey {
	var keys []sessionKey
	if o.PinnedFirst {
		keys = append(keys, sessionKey{expr: "(pinned_at IS NOT NULL)", placeholder: "?", value: at.Pinned})
	}
	messages := sessionKey{expr: lastMessageAt, placeholder: "?", value: at.LastMessageAt}
	switch o.By {
	case SessionSortRelevance:
		keys = append(keys, sessionKey{expr: rank, args: rankArgs, placeholder: "?::real", value: at.Rank}, messages)
	case SessionSortReviewScore:
		// Two keys rather than NULLS LAST, so a cursor on a session without a score has a
		// value to compare: unreviewed sorts after reviewed whichever way the score runs.
		keys = append(keys,
			sessionKey{expr: "(review_score IS NULL)", placeholder: "?", ascending: true, value: at.ReviewScore == nil},
			sessionKey{expr: "COALESCE(review_score, 0)", placeholder: "?", ascending: o.Ascending, value: scoreOf(at.ReviewScore)},
			messages)
	default:
		messages.ascending = o.Ascending
		keys = append(keys, messages)
	}
	return append(keys, sessionKey{expr: "id", placeholder: "?", value: at.ID})
}

// orderSessions sorts a session statement and, with a cursor, starts it after the cursor's
// session.
func orderSessions(query *bun.SelectQuery, order SessionOrder, rank string, rankArgs []any, cursor *SessionPosition) *bun.SelectQuery {
	var at SessionPosition
	if cursor != nil {
		at = *cursor
	}
	keys := order.keys(rank, rankArgs, at)
	for _, key := range keys {
		direction := " DESC"
		if key.ascending {
			direction = " ASC"
		}
		query = query.OrderExpr(key.expr+direction, key.args...)
	}
	if cursor == nil {
		return query
	}
	condition, args := afterSession(keys)
	return query.Where(condition, args...)
}

// afterSession is the condition that a row sorts after the cursor.
//
// Keys that all run the same way are one row comparison, which is what an index on those
// keys is walked from. Keys that run both ways cannot be compared as a row, so they are the
// same thing spelled out: after on the first key, or level on it and after on the second,
// and so on down to the id.
func afterSession(keys []sessionKey) (string, []any) {
	mixed := slices.ContainsFunc(keys, func(key sessionKey) bool { return key.ascending != keys[0].ascending })
	if !mixed {
		exprs := make([]string, 0, len(keys))
		placeholders := make([]string, 0, len(keys))
		var args, values []any
		for _, key := range keys {
			exprs = append(exprs, key.expr)
			placeholders = append(placeholders, key.placeholder)
			args = append(args, key.args...)
			values = append(values, key.value)
		}
		comparison := " < "
		if keys[0].ascending {
			comparison = " > "
		}
		return "(" + strings.Join(exprs, ", ") + ")" + comparison + "(" + strings.Join(placeholders, ", ") + ")",
			append(args, values...)
	}

	steps := make([]string, 0, len(keys))
	var args []any
	for i, key := range keys {
		terms := make([]string, 0, i+1)
		for _, level := range keys[:i] {
			terms = append(terms, level.expr+" = "+level.placeholder)
			args = append(append(args, level.args...), level.value)
		}
		comparison := " < "
		if key.ascending {
			comparison = " > "
		}
		terms = append(terms, key.expr+comparison+key.placeholder)
		args = append(append(args, key.args...), key.value)
		steps = append(steps, "("+strings.Join(terms, " AND ")+")")
	}
	return "(" + strings.Join(steps, " OR ") + ")", args
}
