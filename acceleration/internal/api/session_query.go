package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SessionQuery is which of the caller's sessions to list, and in what order. It is written
// the way Stream's query endpoints are: a filter of fields joined by $and and $or, and a
// sort of {field, direction}.
type SessionQuery struct {
	Filter *SessionFilter `json:"filter,omitempty"`
	Sort   []SessionSort  `json:"sort,omitempty" maxItems:"2" doc:"Omitted is last_message_at descending, or relevance for a text search. One of: last_message_at; pinned, then last_message_at, which lists the pinned sessions first; review_score, for a server-side caller; relevance, for a text search and nothing else."`
	Limit  int            `json:"limit,omitempty" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string         `json:"cursor,omitempty" doc:"The next_cursor of the previous page, sent with the same filter and sort. Omitted is the first page."`

	// refused is why the body is not a session query, kept for the operation to answer
	// with: the body is checked here rather than against the schema, so that what is
	// refused is said in this endpoint's own words.
	refused error
}

func (*SessionQuery) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// UnmarshalJSON reads the query and notes a key it does not take, where the decoder alone
// would drop it and answer a different question than the caller asked.
func (q *SessionQuery) UnmarshalJSON(raw []byte) error {
	type plain SessionQuery
	var read plain
	if err := json.Unmarshal(raw, &read); err != nil {
		return err
	}
	*q = SessionQuery(read)

	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		return err
	}
	for _, key := range slices.Sorted(maps.Keys(keys)) {
		switch key {
		case "filter", "sort", "limit", "cursor":
		default:
			q.refused = stack.Wrap(fmt.Errorf("%q is not part of a session query, which takes filter, sort, limit and cursor", key))
			return nil
		}
	}
	return nil
}

// What a session filter may hold, checked before any query runs.
const (
	// maxFilterDepth is how deep $and and $or may nest.
	maxFilterDepth = 3
	// maxFilterItems is how many filters one $and or $or takes.
	maxFilterItems = 10
	// maxFilterValues is how many values one $in takes.
	maxFilterValues = 50
	// maxFilterConditions is how many field conditions a whole filter holds.
	maxFilterConditions = 30
)

// sessionFilterFields is said in every refusal of a key, so a caller reads what it could
// have sent instead.
const sessionFilterFields = "the fields that can be filtered are project_id, review_score, text, agent, " +
	"user_id, config_id, state, created_at and custom, combined with $and and $or"

// SessionFilter narrows a session query. It is kept as it was sent: a filter nests, so it is
// read by sessionFilterOf, which says what it refuses in its own words.
type SessionFilter struct {
	raw json.RawMessage
}

// UnmarshalJSON keeps the filter as sent. sessionFilterOf reads it.
func (f *SessionFilter) UnmarshalJSON(raw []byte) error {
	f.raw = slices.Clone(raw)
	return nil
}

// MarshalJSON writes the filter back as it was sent.
func (f SessionFilter) MarshalJSON() ([]byte, error) {
	if len(f.raw) == 0 {
		return []byte("{}"), nil
	}
	return f.raw, nil
}

// Schema describes the filter by hand: it nests, and its fields take a value or an operator
// object, neither of which a struct's tags can say.
func (SessionFilter) Schema(registry huma.Registry) *huma.Schema {
	// Each condition is a schema of its own, so a client generator names it once rather
	// than once per field that takes it.
	named := func(name string, schema *huma.Schema) *huma.Schema {
		schema.PrecomputeMessages()
		registry.Map()[name] = schema
		return &huma.Schema{Ref: "#/components/schemas/" + name}
	}
	described := func(ref *huma.Schema, description string) *huma.Schema {
		return &huma.Schema{Ref: ref.Ref, Description: description}
	}
	text := func() *huma.Schema { return &huma.Schema{Type: huma.TypeString, MinLength: new(1)} }
	score := func() *huma.Schema {
		return &huma.Schema{Type: huma.TypeInteger, Minimum: new(1.0), Maximum: new(5.0)}
	}
	nested := func(description string) *huma.Schema {
		return &huma.Schema{
			Type: huma.TypeArray, Description: description,
			Items:    &huma.Schema{Ref: "#/components/schemas/SessionFilter"},
			MinItems: new(1), MaxItems: new(maxFilterItems),
		}
	}

	equals := named("Equals", &huma.Schema{
		Description: `Matches one value exactly: "value" is short for {"$eq": "value"}.`,
		OneOf: []*huma.Schema{
			text(),
			{
				Type:                 huma.TypeObject,
				Properties:           map[string]*huma.Schema{"$eq": text()},
				Required:             []string{"$eq"},
				AdditionalProperties: false,
			},
		},
	})
	equalsOrIn := named("EqualsOrIn", &huma.Schema{
		Description: `Matches one value exactly, written "value" or {"$eq": "value"}, or any of up to 50 values, ` +
			`written {"$in": ["a", "b"]}. An object holds one of $eq and $in.`,
		OneOf: []*huma.Schema{
			text(),
			{
				Type: huma.TypeObject,
				Properties: map[string]*huma.Schema{
					"$eq": text(),
					"$in": {Type: huma.TypeArray, Items: text(), MinItems: new(1), MaxItems: new(maxFilterValues)},
				},
				AdditionalProperties: false, MinProperties: new(1), MaxProperties: new(1),
			},
		},
	})
	scoreMatch := named("ReviewScoreMatch", &huma.Schema{
		Description: "Matches a review score from 1 to 5: a number, which is short for $eq, or an object " +
			"of $eq, $gt, $gte, $lt and $lte, every one of which has to hold.",
		OneOf: []*huma.Schema{
			score(),
			{
				Type: huma.TypeObject, AdditionalProperties: false, MinProperties: new(1),
				Properties: map[string]*huma.Schema{
					"$eq": score(), "$gt": score(), "$gte": score(), "$lt": score(), "$lte": score(),
				},
			},
		},
	})
	textMatch := named("TextMatch", &huma.Schema{
		Type:        huma.TypeObject,
		Description: "A full-text match, Stream's $q.",
		Properties: map[string]*huma.Schema{"$q": {
			Type: huma.TypeString, MinLength: new(1),
			Description: "Quoted phrases and bare words both work, and punctuation is taken rather than refused.",
		}},
		Required:             []string{"$q"},
		AdditionalProperties: false,
	})
	state := SessionState("").Schema(registry)
	stateIs := named("SessionStateEquals", &huma.Schema{
		Description: `Matches live or ended: "live" is short for {"$eq": "live"}.`,
		OneOf: []*huma.Schema{
			state,
			{
				Type:                 huma.TypeObject,
				Properties:           map[string]*huma.Schema{"$eq": state},
				Required:             []string{"$eq"},
				AdditionalProperties: false,
			},
		},
	})
	instant := func() *huma.Schema { return &huma.Schema{Type: huma.TypeString, Format: "date-time"} }
	window := named("TimeWindow", &huma.Schema{
		Type: huma.TypeObject,
		Description: "A window of time: $gte is its start, which it holds, and $lt its end, which it does " +
			"not. Either may be left out, and $gte is before $lt.",
		Properties:           map[string]*huma.Schema{"$gte": instant(), "$lt": instant()},
		AdditionalProperties: false, MinProperties: new(1),
	})

	return named("SessionFilter", &huma.Schema{
		Type: huma.TypeObject,
		Description: "Which sessions to list. The keys of one object are ANDed. $and and $or each take " +
			"1 to 10 filters and nest up to 3 deep; a whole filter holds at most 30 field conditions. " +
			"An empty filter lists every session the caller may see. Any other key, and any other " +
			"operator, is refused with a 400 that names it and lists the fields that can be filtered.",
		AdditionalProperties: false,
		Properties: map[string]*huma.Schema{
			"$and": nested("Every one of these filters holds."),
			"$or": nested("At least one of these filters holds. It chooses among the sessions the " +
				"caller may see and never adds to them. text and user_id are refused inside it."),
			"project_id": described(equalsOrIn, "The project the session belongs to."),
			"agent":      described(equals, "The agent name the session was opened against."),
			"user_id": described(equals, "Whose sessions to list. Only a server-side caller may set it: "+
				"an end user is narrowed to their own whatever they ask for. Only at the top level of "+
				"the filter or in a top-level $and."),
			"review_score": described(scoreMatch, "The score the review gave the conversation. A session with "+
				"no review matches no condition on it. Only a server-side caller may use it."),
			"text": described(textMatch, "Full text over the title, description, project and agent name, "+
				"sorted by relevance. Once per filter, at its top level or in a top-level $and, and never "+
				"inside $or."),
			"config_id": described(equals, "The agent config the session ran under, which stays the same "+
				"when the agent is renamed."),
			"state":      described(stateIs, "Whether the agent is still in the call."),
			"created_at": described(window, "When the session began."),
			"custom": {
				Type: huma.TypeObject,
				Description: "Labels the session's custom object holds, every one of them: " +
					`{"origin": "test"} matches a session whose custom has origin set to the string "test".`,
				AdditionalProperties: &huma.Schema{Type: huma.TypeString},
				MinProperties:        new(1),
			},
		},
	})
}

// parsedFilter is a session filter read and checked: the expression the store takes, the
// words of its text search, and which of the fields only a backend may use it named.
type parsedFilter struct {
	where *store.SessionExpr
	// text is the words searched for, and searched whether there is a text condition.
	text     string
	searched bool
	// reviewed and named are whether review_score and user_id appear anywhere in it.
	reviewed bool
	named    bool

	conditions int
}

// filterPlace is where in a filter an object sits, which decides whether text and user_id
// may be in it.
type filterPlace int

const (
	// placeTop is the filter itself.
	placeTop filterPlace = iota
	// placeTopAnd is an item of the filter's own $and.
	placeTopAnd
	// placeNested is anywhere deeper that no $or is above.
	placeNested
	// placeOr is anywhere an $or is above.
	placeOr
)

// sessionFilterOf reads a filter, refusing a key or an operator it does not take and a
// filter past the limits. Nothing is queried before it has answered.
func sessionFilterOf(sent *SessionFilter) (parsedFilter, error) {
	var parsed parsedFilter
	if sent == nil || len(sent.raw) == 0 || string(sent.raw) == "null" {
		return parsed, nil
	}
	expr, err := parsed.object(sent.raw, 0, placeTop)
	if err != nil {
		return parsedFilter{}, err
	}
	if len(expr.Children) > 0 {
		parsed.where = &expr
	}
	return parsed, nil
}

// object reads one filter object: its field conditions and its $and and $or, ANDed.
func (p *parsedFilter) object(raw json.RawMessage, depth int, place filterPlace) (store.SessionExpr, error) {
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil || keys == nil {
		return store.SessionExpr{}, stack.Wrap(errors.New("a filter is an object of fields, $and and $or"))
	}

	var all []store.SessionExpr
	// In the order of the keys' names, so the same filter is always refused for the same thing.
	for _, key := range slices.Sorted(maps.Keys(keys)) {
		value := keys[key]
		switch key {
		case "$and", "$or":
			group, err := p.group(key, value, depth, place)
			if err != nil {
				return store.SessionExpr{}, err
			}
			all = append(all, group)
			continue
		case "project_id", "agent", "user_id", "review_score", "text", "config_id", "state", "created_at", "custom":
		default:
			return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%q cannot be filtered on: %s", key, sessionFilterFields))
		}

		if p.conditions++; p.conditions > maxFilterConditions {
			return store.SessionExpr{}, stack.Wrap(fmt.Errorf("a filter holds at most %d field conditions", maxFilterConditions))
		}
		switch key {
		case "text":
			words, err := p.words(value, place)
			if err != nil {
				return store.SessionExpr{}, err
			}
			p.text, p.searched = words, true
		case "review_score":
			conditions, err := scoreConditions(value)
			if err != nil {
				return store.SessionExpr{}, err
			}
			p.reviewed = true
			all = append(all, conditions...)
		case "created_at":
			conditions, err := createdAtConditions(value)
			if err != nil {
				return store.SessionExpr{}, err
			}
			all = append(all, conditions...)
		case "custom":
			var labels map[string]string
			if json.Unmarshal(value, &labels) != nil || len(labels) == 0 {
				return store.SessionExpr{}, stack.Wrap(errors.New("custom takes an object of at least one label, each a string"))
			}
			all = append(all, store.CustomHolds(labels))
		case "state":
			condition, err := textCondition(key, value)
			if err != nil {
				return store.SessionExpr{}, err
			}
			// The API says live and ended; a row keeps running and closed.
			switch SessionState(condition.Strings[0]) {
			case Live:
				condition.Strings = []string{store.SessionRunning}
			case Ended:
				condition.Strings = []string{store.SessionClosed}
			default:
				return store.SessionExpr{}, stack.Wrap(errors.New("state is live or ended"))
			}
			all = append(all, condition)
		default:
			if key == "user_id" {
				if place != placeTop && place != placeTopAnd {
					return store.SessionExpr{}, stack.Wrap(errors.New(
						"user_id may only be at the top level of the filter or in a top-level $and, never inside $or"))
				}
				p.named = true
			}
			condition, err := textCondition(key, value)
			if err != nil {
				return store.SessionExpr{}, err
			}
			all = append(all, condition)
		}
	}
	return store.AllOf(all...), nil
}

// group reads the filters of one $and or $or.
func (p *parsedFilter) group(operator string, raw json.RawMessage, depth int, place filterPlace) (store.SessionExpr, error) {
	if depth >= maxFilterDepth {
		return store.SessionExpr{}, stack.Wrap(fmt.Errorf("$and and $or nest at most %d deep", maxFilterDepth))
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 {
		return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s takes a list of 1 to %d filters", operator, maxFilterItems))
	}
	if len(items) > maxFilterItems {
		return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s takes at most %d filters, and was sent %d", operator, maxFilterItems, len(items)))
	}

	within := placeNested
	switch {
	case operator == "$or" || place == placeOr:
		within = placeOr
	case place == placeTop:
		within = placeTopAnd
	}
	children := make([]store.SessionExpr, 0, len(items))
	for _, item := range items {
		child, err := p.object(item, depth+1, within)
		if err != nil {
			return store.SessionExpr{}, err
		}
		children = append(children, child)
	}
	if operator == "$or" {
		return store.AnyOf(children...), nil
	}
	return store.AllOf(children...), nil
}

// words reads a text condition, which a filter holds once and never under an $or: a search
// ranks the whole list, so it cannot be one of several alternatives.
func (p *parsedFilter) words(raw json.RawMessage, place filterPlace) (string, error) {
	switch {
	case place == placeOr:
		return "", stack.Wrap(errors.New("text cannot be used inside $or: put it at the top level of the filter or in a top-level $and"))
	case place == placeNested:
		return "", stack.Wrap(errors.New("text may only be at the top level of the filter or in a top-level $and"))
	case p.searched:
		return "", stack.Wrap(errors.New("text may appear only once in a filter"))
	}
	shape := stack.Wrap(errors.New(`text takes {"$q": "words"}, with at least one character`))
	if !strings.HasPrefix(strings.TrimSpace(string(raw)), "{") {
		return "", shape
	}
	operators, err := operatorsOf("text", raw, "$q")
	if err != nil {
		return "", err
	}
	var words string
	if json.Unmarshal(operators["$q"], &words) != nil || words == "" {
		return "", shape
	}
	return words, nil
}

// operatorsOf reads a condition written as an object of operators, refusing one the field
// does not take.
func operatorsOf(field string, raw json.RawMessage, allowed ...string) (map[string]json.RawMessage, error) {
	var operators map[string]json.RawMessage
	if err := json.Unmarshal(raw, &operators); err != nil || len(operators) == 0 {
		return nil, stack.Wrap(fmt.Errorf("%s takes a value or an object of %s", field, strings.Join(allowed, ", ")))
	}
	for _, operator := range slices.Sorted(maps.Keys(operators)) {
		if !slices.Contains(allowed, operator) {
			return nil, stack.Wrap(fmt.Errorf("%s does not take %q: its operators are %s", field, operator, strings.Join(allowed, ", ")))
		}
	}
	return operators, nil
}

// textCondition reads a condition on project_id, agent or user_id: a bare string, $eq, or
// for project_id $in.
func textCondition(key string, raw json.RawMessage) (store.SessionExpr, error) {
	field := store.SessionField(key)
	var bare string
	if json.Unmarshal(raw, &bare) == nil {
		if bare == "" {
			return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s cannot be empty", key))
		}
		return store.SessionIs(field, bare), nil
	}

	allowed := []string{"$eq"}
	if field == store.SessionFieldProject {
		allowed = append(allowed, "$in")
	}
	operators, err := operatorsOf(key, raw, allowed...)
	if err != nil {
		return store.SessionExpr{}, err
	}
	if len(operators) != 1 {
		return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s takes one of %s, not both", key, strings.Join(allowed, " or ")))
	}
	if sent, ok := operators["$eq"]; ok {
		var value string
		if json.Unmarshal(sent, &value) != nil || value == "" {
			return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s $eq takes a string that is not empty", key))
		}
		return store.SessionIs(field, value), nil
	}
	var values []string
	if json.Unmarshal(operators["$in"], &values) != nil || len(values) == 0 || len(values) > maxFilterValues ||
		slices.Contains(values, "") {
		return store.SessionExpr{}, stack.Wrap(fmt.Errorf("%s $in takes a list of 1 to %d strings that are not empty", key, maxFilterValues))
	}
	return store.SessionIsOneOf(field, values...), nil
}

// scoreConditions reads a condition on review_score: a bare number, or an object of $eq,
// $gt, $gte, $lt and $lte, every one of which has to hold.
func scoreConditions(raw json.RawMessage) ([]store.SessionExpr, error) {
	if score, ok := scoreOf(raw); ok {
		return []store.SessionExpr{store.ReviewScoreIs(store.SessionEq, score)}, nil
	}
	if trimmed := strings.TrimSpace(string(raw)); !strings.HasPrefix(trimmed, "{") {
		return nil, stack.Wrap(errors.New("review_score is a whole number from 1 to 5, or an object of $eq, $gt, $gte, $lt, $lte"))
	}
	operators, err := operatorsOf("review_score", raw, "$eq", "$gt", "$gte", "$lt", "$lte")
	if err != nil {
		return nil, err
	}
	conditions := make([]store.SessionExpr, 0, len(operators))
	for _, operator := range slices.Sorted(maps.Keys(operators)) {
		score, ok := scoreOf(operators[operator])
		if !ok {
			return nil, stack.Wrap(fmt.Errorf("review_score %s takes a whole number from 1 to 5", operator))
		}
		conditions = append(conditions, store.ReviewScoreIs(store.SessionOperator(operator), score))
	}
	return conditions, nil
}

// createdAtConditions reads a window on when a session began: an object of $gte and $lt,
// each an RFC 3339 time.
func createdAtConditions(raw json.RawMessage) ([]store.SessionExpr, error) {
	var sent map[string]json.RawMessage
	if json.Unmarshal(raw, &sent) != nil || len(sent) == 0 {
		return nil, stack.Wrap(errors.New("created_at takes an object of $gte, $lt"))
	}
	operators, err := operatorsOf("created_at", raw, "$gte", "$lt")
	if err != nil {
		return nil, err
	}
	// $gte sorts before $lt, so a window with both is its start then its end.
	conditions := make([]store.SessionExpr, 0, len(operators))
	for _, operator := range slices.Sorted(maps.Keys(operators)) {
		var written string
		at, err := time.Time{}, json.Unmarshal(operators[operator], &written)
		if err == nil {
			at, err = time.Parse(time.RFC3339Nano, written)
		}
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("created_at %s takes an RFC 3339 time", operator))
		}
		conditions = append(conditions, store.CreatedAtIs(store.SessionOperator(operator), at.UTC()))
	}
	if len(conditions) == 2 && !conditions[0].Time.Before(conditions[1].Time) {
		return nil, stack.Wrap(errors.New("created_at $gte must be before $lt, or the window holds nothing"))
	}
	return conditions, nil
}

// scoreOf reads a review score: a whole number from 1 to 5.
func scoreOf(raw json.RawMessage) (int, bool) {
	score, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	return score, err == nil && score >= 1 && score <= 5
}

// SessionSort is one key a session query is ordered by.
type SessionSort struct {
	Field     SessionSortField `json:"field"`
	Direction int              `json:"direction,omitempty" enum:"-1,1" default:"-1" doc:"-1 is descending and 1 ascending. pinned and relevance only sort descending."`
}

func (*SessionSort) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// SessionSortField is a key a session query sorts on.
type SessionSortField string

const (
	SessionSortLastMessageAt SessionSortField = "last_message_at"
	SessionSortPinned        SessionSortField = "pinned"
	SessionSortReviewScore   SessionSortField = "review_score"
	SessionSortRelevance     SessionSortField = "relevance"
)

func (SessionSortField) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SessionSortField",
		"last_message_at is when the agent last answered, or when the session began if it has not. "+
			"pinned lists the pinned sessions first: it sorts descending only, comes first, and is "+
			"followed by last_message_at, descending when left out. review_score is the review's "+
			"score, sessions without one last either way, and is a server-side caller's only. "+
			"relevance is the best match first, and is the one sort of a text search.",
		string(SessionSortLastMessageAt), string(SessionSortPinned), string(SessionSortReviewScore),
		string(SessionSortRelevance))
}

// sessionSorts is said in every refusal of a sort.
const sessionSorts = `sort is one of [{"field":"last_message_at"}], ` +
	`[{"field":"pinned"},{"field":"last_message_at"}], [{"field":"review_score"}] for a server-side ` +
	`caller, or [{"field":"relevance"}] for a text search; direction is -1 (descending, the default) ` +
	`or 1, and pinned and relevance only sort descending`

// sessionOrderOf reads a sort as the order the store takes, refusing any that is not one of
// the orders the list can be paged in.
func sessionOrderOf(sort []SessionSort, searched, server bool) (store.SessionOrder, error) {
	refused := func(why string) (store.SessionOrder, error) {
		return store.SessionOrder{}, stack.Wrap(errors.New(why + ": " + sessionSorts))
	}
	ascending := make([]bool, len(sort))
	for i, key := range sort {
		switch key.Direction {
		case 0, -1:
		case 1:
			ascending[i] = true
		default:
			return refused(fmt.Sprintf("a sort direction is 1 or -1, not %d", key.Direction))
		}
		switch key.Field {
		case SessionSortLastMessageAt, SessionSortPinned, SessionSortReviewScore, SessionSortRelevance:
		case "updated_at":
			return refused("updated_at is no longer a sort field")
		default:
			return refused(fmt.Sprintf("%q is not a field sessions sort on", key.Field))
		}
	}

	if searched {
		if len(sort) == 0 || (len(sort) == 1 && sort[0].Field == SessionSortRelevance && !ascending[0]) {
			return store.SessionOrder{By: store.SessionSortRelevance}, nil
		}
		return refused("a text search is sorted by relevance, descending")
	}
	switch {
	case len(sort) == 0:
		return store.SessionOrder{}, nil
	case sort[0].Field == SessionSortRelevance:
		return refused("only a text search is sorted by relevance")
	case len(sort) == 1 && sort[0].Field == SessionSortLastMessageAt:
		return store.SessionOrder{Ascending: ascending[0]}, nil
	case len(sort) == 1 && sort[0].Field == SessionSortReviewScore:
		if !server {
			return refused("only a server-side caller may sort by review_score")
		}
		return store.SessionOrder{By: store.SessionSortReviewScore, Ascending: ascending[0]}, nil
	case sort[0].Field == SessionSortPinned && ascending[0]:
		return refused("pinned only sorts descending")
	case len(sort) == 1 && sort[0].Field == SessionSortPinned:
		return store.SessionOrder{PinnedFirst: true}, nil
	case len(sort) == 2 && sort[0].Field == SessionSortPinned && sort[1].Field == SessionSortLastMessageAt:
		return store.SessionOrder{PinnedFirst: true, Ascending: ascending[1]}, nil
	}
	return refused("that combination of sort fields is not offered")
}

// sortName is an order as a cursor carries it, so a cursor made for one order is told from
// one made for another.
func sortName(order store.SessionOrder) string {
	name := string(SessionSortLastMessageAt)
	switch order.By {
	case store.SessionSortRelevance:
		name = string(SessionSortRelevance)
	case store.SessionSortReviewScore:
		name = string(SessionSortReviewScore)
	}
	if order.PinnedFirst {
		name = string(SessionSortPinned) + "," + name
	}
	if order.Ascending {
		return name + ":1"
	}
	return name + ":-1"
}

type querySessionsRequest struct {
	Body *SessionQuery
}

type querySessionsResponse struct {
	Body SessionPage
}

func (s *Server) registerSessionQuery(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "querySessions",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/query",
		Summary:     "List or search the caller's sessions",
		Description: "Lists the caller's sessions, the ones still running and the ones that ended, " +
			"narrowed by a filter and in the order of a sort.\n\n" +
			"A filter is an object of fields. The keys of one object are ANDed, and `$and` and `$or` " +
			"each take a list of filters, so conditions combine and nest:\n\n" +
			"- `project_id`: a value, `{\"$eq\": value}` or `{\"$in\": [values]}`\n" +
			"- `agent`: a value or `{\"$eq\": value}`, the agent name the session was opened against\n" +
			"- `text`: `{\"$q\": \"words\"}`, full text over the title, description, project and agent name\n" +
			"- `review_score`: a number from 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt` and `$lte`; a server-side caller's only\n" +
			"- `user_id`: a value or `{\"$eq\": value}`; a server-side caller's only\n" +
			"- `config_id`: a value or `{\"$eq\": value}`, the agent config the session ran under, which a rename leaves alone\n" +
			"- `state`: `live` or `ended`, as a value or `{\"$eq\": value}`\n" +
			"- `created_at`: `{\"$gte\": time, \"$lt\": time}`, either bound alone or both, as RFC 3339 times\n" +
			"- `custom`: an object of string labels the session's `custom` holds, every one of them\n\n" +
			"Nothing else can be filtered on: any other key or operator is a 400 that names it. " +
			"`$and` and `$or` take 1 to 10 filters and nest 3 deep, `$in` takes up to 50 values, and a " +
			"filter holds at most 30 field conditions. `text` appears once, at the top level or in a " +
			"top-level `$and`, and never inside `$or`; `user_id` is placed the same way.\n\n" +
			"Every session, pinned first:\n\n" +
			"```json\n{\"sort\": [{\"field\": \"pinned\", \"direction\": -1}, {\"field\": \"last_message_at\", \"direction\": -1}]}\n```\n\n" +
			"One agent's sessions in either of two projects:\n\n" +
			"```json\n{\"filter\": {\"agent\": \"support\", \"$or\": [{\"project_id\": \"health\"}, {\"project_id\": \"billing\"}]}}\n```\n\n" +
			"A search within one project:\n\n" +
			"```json\n{\"filter\": {\"text\": {\"$q\": \"refund\"}, \"project_id\": \"billing\"}}\n```\n\n" +
			"A backend's badly reviewed sessions and everything in two projects, the lowest score first:\n\n" +
			"```json\n{\"filter\": {\"$or\": [{\"review_score\": {\"$lte\": 2}}, {\"project_id\": {\"$in\": [\"vip\", \"pilot\"]}}]}, " +
			"\"sort\": [{\"field\": \"review_score\", \"direction\": 1}]}\n```\n\n" +
			"`sort` is `last_message_at` (the default, descending: when the agent last answered, or when " +
			"the session began), `pinned` followed by `last_message_at`, `review_score` for a server-side " +
			"caller, with the sessions nobody reviewed last, or `relevance`, which is how a `text` search " +
			"is sorted and the only way it is. A session is pinned and unpinned by updating it.\n\n" +
			"A backend gets its customer's sessions; an end user gets their own, whatever the filter " +
			"says: an `$or` chooses among the sessions the caller may see and never adds to them. An " +
			"anonymous caller who named nobody gets none.\n\n" +
			"The search reads what a person named the conversation, not what was said in it. " +
			"Pages are followed by `next_cursor`, sent back with the same filter and sort. " +
			"There is no total: counting every conversation costs more than the page.",
		Responses: map[string]*huma.Response{
			"200": {Description: "A page of sessions"},
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized},
		Extensions: map[string]any{clientAccessibleExtension: true},
		// The body is checked by sessionQueryOf rather than against the schema: a filter
		// nests, and what it refuses is said in its own words, naming the key and the fields
		// that can be filtered.
		SkipValidateBody: true,
	}, s.querySessions)
}

// querySessions lists or searches the caller's sessions.
func (s *Server) querySessions(ctx context.Context, request *querySessionsRequest) (*querySessionsResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}

	var sent SessionQuery
	if request.Body != nil {
		sent = *request.Body
	}
	// Checked before anything is answered, so a query that is refused is refused whether
	// or not this deployment keeps sessions.
	query, err := sessionQueryOf(ctx, sent)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if s.sessions == nil {
		return &querySessionsResponse{Body: SessionPage{Items: []Session{}}}, nil
	}
	var found []session.Found
	if query.searched {
		found, err = s.sessions.Search(ctx, OwnerFrom(ctx), query.text, query.filter)
	} else {
		found, err = s.sessions.Query(ctx, OwnerFrom(ctx), query.filter)
	}
	if err != nil {
		return nil, err
	}
	return &querySessionsResponse{Body: sessionPageOf(ctx, found, query.filter.Limit, query.filter.Order)}, nil
}

// sessionQuery is a SessionQuery checked and turned into what the manager takes.
type sessionQuery struct {
	text     string
	searched bool
	filter   store.SessionFilter
}

// sessionCursor is a page position and the sort it belongs to, so a cursor from one order
// is refused by another rather than paging it from the wrong place.
type sessionCursor struct {
	store.SessionPosition
	Sort string `json:"s"`
}

// sessionQueryOf reads a query, refusing a filter, a sort or a limit that is not offered,
// and review scores and a user id from anybody but a backend.
//
// The manager narrows an end user to their own sessions again, around the whole filter; two
// checks is the right number for something that decides whose conversations a stranger can
// read.
func sessionQueryOf(ctx context.Context, sent SessionQuery) (sessionQuery, error) {
	if sent.refused != nil {
		return sessionQuery{}, sent.refused
	}
	if sent.Limit < 0 || sent.Limit > 200 {
		return sessionQuery{}, stack.Wrap(errors.New("limit is from 1 to 200"))
	}
	parsed, err := sessionFilterOf(sent.Filter)
	if err != nil {
		return sessionQuery{}, err
	}
	server := KindFrom(ctx) == auth.KindServer
	switch {
	case parsed.named && !server:
		return sessionQuery{}, stack.Wrap(errors.New("only a server-side caller may list another user's sessions"))
	case parsed.reviewed && !server:
		return sessionQuery{}, stack.Wrap(errors.New("only a server-side caller may filter on review_score"))
	}
	order, err := sessionOrderOf(sent.Sort, parsed.searched, server)
	if err != nil {
		return sessionQuery{}, err
	}

	query := sessionQuery{
		text: parsed.text, searched: parsed.searched,
		filter: store.SessionFilter{Where: parsed.where, Order: order, Limit: sent.Limit},
	}
	cursor, err := decodeCursor[sessionCursor](&sent.Cursor)
	if err != nil {
		return sessionQuery{}, err
	}
	if cursor != nil {
		if cursor.Sort != sortName(order) {
			return sessionQuery{}, stack.Wrap(errBadCursor)
		}
		query.filter.Cursor = &cursor.SessionPosition
	}
	return query, nil
}

// sessionPageOf renders a query's results as a page, with the cursor to the next one.
func sessionPageOf(ctx context.Context, found []session.Found, limit int, order store.SessionOrder) SessionPage {
	kept, more := page(found, store.SessionLimit(limit))
	rendered := SessionPage{Items: sessionsOf(ctx, kept), HasMore: more}
	if more {
		rendered.NextCursor = encodeCursor(sessionCursor{kept[len(kept)-1].Position(), sortName(order)})
	}
	return rendered
}

// SessionPage is the SessionPage schema.
type SessionPage struct {
	HasMore    bool      `json:"has_more"`
	Items      []Session `json:"items" nullable:"false"`
	NextCursor *string   "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}
