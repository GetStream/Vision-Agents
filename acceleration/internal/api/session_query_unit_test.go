package api

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// sessionFilter is a filter written in Go, for a test that has no JSON to send.
func sessionFilter(fields map[string]any) *SessionFilter {
	// A map of strings, numbers, lists and maps always marshals.
	raw, _ := json.Marshal(fields)
	return &SessionFilter{raw: raw}
}

// queryAsking is a session query as the endpoint reads it off the wire.
func queryAsking(t *testing.T, body string) SessionQuery {
	t.Helper()
	var query SessionQuery
	require.NoError(t, json.Unmarshal([]byte(body), &query))
	return query
}

// aBackend and aDevice are the two kinds of caller a query is checked for.
func aBackend() context.Context {
	return context.WithValue(context.Background(), kindContextKey{}, auth.KindServer)
}

func aDevice() context.Context {
	return context.WithValue(context.Background(), kindContextKey{}, auth.KindAuthenticated)
}

func TestAFilterIsReadAsTheExpressionTheStoreTakes(t *testing.T) {
	project, agent, user := store.SessionFieldProject, store.SessionFieldAgent, store.SessionFieldUser

	query, err := sessionQueryOf(aBackend(), queryAsking(t, `{}`))
	require.NoError(t, err)
	require.Nil(t, query.filter.Where, "no filter admits every session")
	require.False(t, query.searched)
	query, err = sessionQueryOf(aBackend(), queryAsking(t, `{"filter": {}}`))
	require.NoError(t, err)
	require.Nil(t, query.filter.Where)

	query, err = sessionQueryOf(aBackend(), queryAsking(t, `{"filter": {"project_id": "health"}}`))
	require.NoError(t, err)
	require.Equal(t, store.AllOf(store.SessionIs(project, "health")), *query.filter.Where)

	query, err = sessionQueryOf(aBackend(), queryAsking(t, `{"filter": {
		"agent": {"$eq": "support"},
		"project_id": {"$in": ["a", "b"]},
		"review_score": {"$gte": 2, "$lt": 5},
		"user_id": "jlahey",
		"$or": [{"project_id": "c"}, {"$and": [{"agent": "sales"}, {"review_score": 5}]}]
	}}`))
	require.NoError(t, err)
	require.Equal(t, store.AllOf(
		store.AnyOf(
			store.AllOf(store.SessionIs(project, "c")),
			store.AllOf(store.AllOf(
				store.AllOf(store.SessionIs(agent, "sales")),
				store.AllOf(store.ReviewScoreIs(store.SessionEq, 5)),
			)),
		),
		store.SessionIs(agent, "support"),
		store.SessionIsOneOf(project, "a", "b"),
		store.ReviewScoreIs(store.SessionGte, 2),
		store.ReviewScoreIs(store.SessionLt, 5),
		store.SessionIs(user, "jlahey"),
	), *query.filter.Where)

	// Text is the search's words and not part of the expression, at the top level or in a
	// top-level $and, and now alongside a project.
	query, err = sessionQueryOf(aDevice(), queryAsking(t, `{"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}`))
	require.NoError(t, err)
	require.True(t, query.searched)
	require.Equal(t, "refund", query.text)
	require.Equal(t, store.AllOf(store.SessionIs(project, "billing")), *query.filter.Where)
	require.Equal(t, store.SessionOrder{By: store.SessionSortRelevance}, query.filter.Order)

	query, err = sessionQueryOf(aDevice(), queryAsking(t, `{"filter": {"$and": [{"text": {"$q": "refund"}}, {"agent": "support"}]}}`))
	require.NoError(t, err)
	require.Equal(t, "refund", query.text)
}

func TestAFilterIsRefusedForWhatItMayNotHold(t *testing.T) {
	nested := func(depth int) string {
		filter := `{"project_id": "a"}`
		for range depth {
			filter = `{"$and": [` + filter + `]}`
		}
		return filter
	}
	many := func(n int, item string) string {
		return "[" + strings.TrimSuffix(strings.Repeat(item+",", n), ",") + "]"
	}
	conditions := func(n int) string {
		// Ten $or branches of three conditions each, then some.
		branches := make([]string, 0, n/3)
		for range n / 3 {
			branches = append(branches, `{"project_id": "a", "agent": "b", "review_score": 3}`)
		}
		return `{"$or": [` + strings.Join(branches, ",") + `], "project_id": "x"}`
	}

	refusals := []struct {
		filter string
		says   string
	}{
		{`{"agent_id": "a"}`, `"agent_id" cannot be filtered on`},
		{`{"config_id": "a"}`, `"config_id" cannot be filtered on`},
		{`{"modality": "text"}`, `"modality" cannot be filtered on`},
		{`{"state": "live"}`, `"state" cannot be filtered on`},
		{`{"team": "finance"}`, `"team" cannot be filtered on`},
		{`{"created_at": {"$gte": "2026-10-05T00:00:00Z"}}`, `"created_at" cannot be filtered on`},
		{`{"custom": {"origin": "test"}}`, `"custom" cannot be filtered on`},
		{`{"call_id": "abc"}`, `"call_id" cannot be filtered on`},
		{`{"$nor": [{"project_id": "a"}]}`, `"$nor" cannot be filtered on`},
		{`{"$or": [{"project_id": "a"}, {"$not": {"agent": "b"}}]}`, `"$not" cannot be filtered on`},
		{`{"project_id": {"$gt": "a"}}`, `project_id does not take "$gt": its operators are $eq, $in`},
		{`{"project_id": {"$eq": "a", "$in": ["b"]}}`, `project_id takes one of $eq or $in, not both`},
		{`{"project_id": {"$in": []}}`, `project_id $in takes a list of 1 to 50 strings`},
		{`{"project_id": {"$in": ` + many(51, `"p"`) + `}}`, `project_id $in takes a list of 1 to 50 strings`},
		{`{"project_id": {"$in": "a"}}`, `project_id $in takes a list of 1 to 50 strings`},
		{`{"project_id": 7}`, `project_id takes a value or an object of $eq, $in`},
		{`{"project_id": ""}`, `project_id cannot be empty`},
		{`{"agent": {"$in": ["a"]}}`, `agent does not take "$in": its operators are $eq`},
		{`{"user_id": {"$in": ["a"]}}`, `user_id does not take "$in"`},
		{`{"review_score": {"$ne": 3}}`, `review_score does not take "$ne"`},
		{`{"review_score": {"$in": [1, 2]}}`, `review_score does not take "$in"`},
		{`{"review_score": 0}`, `review_score is a whole number from 1 to 5`},
		{`{"review_score": 6}`, `review_score is a whole number from 1 to 5`},
		{`{"review_score": 2.5}`, `review_score is a whole number from 1 to 5`},
		{`{"review_score": "3"}`, `review_score is a whole number from 1 to 5`},
		{`{"review_score": {"$gte": 9}}`, `review_score $gte takes a whole number from 1 to 5`},
		{`{"review_score": {}}`, `review_score takes a value or an object of`},
		{`{"text": "refund"}`, `text takes {"$q": "words"}`},
		{`{"text": {"$q": ""}}`, `text takes {"$q": "words"}`},
		{`{"text": {"$eq": "refund"}}`, `text does not take "$eq"`},
		{`{"$or": [{"text": {"$q": "refund"}}, {"project_id": "a"}]}`, `text cannot be used inside $or`},
		{`{"$and": [{"$or": [{"$and": [{"text": {"$q": "refund"}}]}]}]}`, `text cannot be used inside $or`},
		{`{"$and": [{"$and": [{"text": {"$q": "refund"}}]}]}`, `text may only be at the top level of the filter or in a top-level $and`},
		{`{"text": {"$q": "a"}, "$and": [{"text": {"$q": "b"}}]}`, `text may appear only once`},
		{`{"$or": [{"user_id": "a"}, {"user_id": "b"}]}`, `user_id may only be at the top level`},
		{`{"$and": [{"$and": [{"user_id": "a"}]}]}`, `user_id may only be at the top level`},
		{nested(4), `$and and $or nest at most 3 deep`},
		{`{"$or": []}`, `$or takes a list of 1 to 10 filters`},
		{`{"$and": {"project_id": "a"}}`, `$and takes a list of 1 to 10 filters`},
		{`{"$or": ` + many(11, `{"project_id": "a"}`) + `}`, `$or takes at most 10 filters, and was sent 11`},
		{`{"$or": ["a"]}`, `a filter is an object`},
		{conditions(30), `a filter holds at most 30 field conditions`},
		{`"project_id"`, `a filter is an object`},
		{`[]`, `a filter is an object`},
	}
	for _, refusal := range refusals {
		_, err := sessionQueryOf(aBackend(), queryAsking(t, `{"filter": `+refusal.filter+`}`))
		require.ErrorContains(t, err, refusal.says, refusal.filter)
		if strings.Contains(refusal.says, "cannot be filtered on") {
			require.ErrorContains(t, err, "the fields that can be filtered are project_id, review_score, text, agent and user_id")
		}
	}

	// What is just inside the limits is taken.
	for _, taken := range []string{
		nested(3),
		`{"$or": ` + many(10, `{"project_id": "a"}`) + `}`,
		`{"project_id": {"$in": ` + many(50, `"p"`) + `}}`,
		conditions(27),
		`{"review_score": {"$gt": 1, "$gte": 1, "$lt": 5, "$lte": 5, "$eq": 3}}`,
	} {
		_, err := sessionQueryOf(aBackend(), queryAsking(t, `{"filter": `+taken+`}`))
		require.NoError(t, err, taken)
	}
}

func TestOnlyABackendFiltersOrSortsOnWhatOnlyABackendReads(t *testing.T) {
	for _, body := range []string{
		`{"filter": {"review_score": 3}}`,
		`{"filter": {"review_score": {"$gte": 1}}}`,
		`{"filter": {"$or": [{"project_id": "a"}, {"review_score": 5}]}}`,
	} {
		_, err := sessionQueryOf(aDevice(), queryAsking(t, body))
		require.ErrorContains(t, err, "only a server-side caller may filter on review_score", body)
		_, err = sessionQueryOf(aBackend(), queryAsking(t, body))
		require.NoError(t, err, body)
	}
	for _, body := range []string{
		`{"filter": {"user_id": "randy"}}`,
		`{"filter": {"user_id": {"$eq": "randy"}}}`,
		`{"filter": {"$and": [{"user_id": "randy"}, {"project_id": "a"}]}}`,
	} {
		_, err := sessionQueryOf(aDevice(), queryAsking(t, body))
		require.ErrorContains(t, err, "only a server-side caller may list another user's sessions", body)
		_, err = sessionQueryOf(aBackend(), queryAsking(t, body))
		require.NoError(t, err, body)
	}
	// Inside an $or a user id is refused for a device as a user id, not for where it is.
	_, err := sessionQueryOf(aDevice(), queryAsking(t, `{"filter": {"$or": [{"project_id": "a"}, {"user_id": "randy"}]}}`))
	require.Error(t, err)

	_, err = sessionQueryOf(aDevice(), queryAsking(t, `{"sort": [{"field": "review_score"}]}`))
	require.ErrorContains(t, err, "only a server-side caller may sort by review_score")
}

func TestASortIsOneOfTheOrdersTheListPagesIn(t *testing.T) {
	orders := []struct {
		body string
		want store.SessionOrder
	}{
		{`{}`, store.SessionOrder{}},
		{`{"sort": []}`, store.SessionOrder{}},
		{`{"sort": [{"field": "last_message_at"}]}`, store.SessionOrder{}},
		{`{"sort": [{"field": "last_message_at", "direction": -1}]}`, store.SessionOrder{}},
		{`{"sort": [{"field": "last_message_at", "direction": 1}]}`, store.SessionOrder{Ascending: true}},
		{`{"sort": [{"field": "pinned"}]}`, store.SessionOrder{PinnedFirst: true}},
		{`{"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}`,
			store.SessionOrder{PinnedFirst: true}},
		{`{"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": 1}]}`,
			store.SessionOrder{PinnedFirst: true, Ascending: true}},
		{`{"sort": [{"field": "review_score", "direction": 1}]}`,
			store.SessionOrder{By: store.SessionSortReviewScore, Ascending: true}},
		{`{"sort": [{"field": "review_score", "direction": -1}]}`, store.SessionOrder{By: store.SessionSortReviewScore}},
		{`{"filter": {"text": {"$q": "refund"}}}`, store.SessionOrder{By: store.SessionSortRelevance}},
		{`{"filter": {"text": {"$q": "refund"}}, "sort": [{"field": "relevance", "direction": -1}]}`,
			store.SessionOrder{By: store.SessionSortRelevance}},
	}
	for _, one := range orders {
		query, err := sessionQueryOf(aBackend(), queryAsking(t, one.body))
		require.NoError(t, err, one.body)
		require.Equal(t, one.want, query.filter.Order, one.body)
	}

	refusals := []struct {
		body string
		says string
	}{
		{`{"sort": [{"field": "updated_at"}]}`, "updated_at is no longer a sort field"},
		{`{"sort": [{"field": "updated_at", "direction": 1}]}`, "updated_at is no longer a sort field"},
		{`{"sort": [{"field": "created_at"}]}`, `"created_at" is not a field sessions sort on`},
		{`{"sort": [{"field": "relevance"}]}`, "only a text search is sorted by relevance"},
		{`{"sort": [{"field": "pinned", "direction": 1}]}`, "pinned only sorts descending"},
		{`{"sort": [{"field": "pinned", "direction": 1}, {"field": "last_message_at"}]}`, "pinned only sorts descending"},
		{`{"sort": [{"field": "last_message_at"}, {"field": "pinned"}]}`, "that combination of sort fields is not offered"},
		{`{"sort": [{"field": "pinned"}, {"field": "review_score"}]}`, "that combination of sort fields is not offered"},
		{`{"sort": [{"field": "pinned"}, {"field": "pinned"}]}`, "that combination of sort fields is not offered"},
		{`{"sort": [{"field": "review_score"}, {"field": "last_message_at"}]}`, "that combination of sort fields is not offered"},
		{`{"sort": [{"field": "pinned"}, {"field": "last_message_at"}, {"field": "review_score"}]}`,
			"that combination of sort fields is not offered"},
		{`{"sort": [{"field": "last_message_at", "direction": 2}]}`, "a sort direction is 1 or -1, not 2"},
		{`{"filter": {"text": {"$q": "refund"}}, "sort": [{"field": "last_message_at"}]}`, "a text search is sorted by relevance"},
		{`{"filter": {"text": {"$q": "refund"}}, "sort": [{"field": "pinned"}]}`, "a text search is sorted by relevance"},
		{`{"filter": {"text": {"$q": "refund"}}, "sort": [{"field": "relevance", "direction": 1}]}`,
			"a text search is sorted by relevance"},
	}
	for _, refusal := range refusals {
		_, err := sessionQueryOf(aBackend(), queryAsking(t, refusal.body))
		require.ErrorContains(t, err, refusal.says, refusal.body)
		require.ErrorContains(t, err, `sort is one of [{"field":"last_message_at"}]`, "a refusal says what is allowed")
	}
}

func TestAQueryIsRefusedForAKeyOrALimitItDoesNotTake(t *testing.T) {
	_, err := sessionQueryOf(aBackend(), queryAsking(t, `{"filters": {"project_id": "a"}}`))
	require.ErrorContains(t, err, `"filters" is not part of a session query, which takes filter, sort, limit and cursor`)
	_, err = sessionQueryOf(aBackend(), queryAsking(t, `{"limit": 201}`))
	require.ErrorContains(t, err, "limit is from 1 to 200")
	_, err = sessionQueryOf(aBackend(), queryAsking(t, `{"limit": -1}`))
	require.ErrorContains(t, err, "limit is from 1 to 200")
	_, err = sessionQueryOf(aBackend(), queryAsking(t, `{"limit": 200}`))
	require.NoError(t, err)
}

func TestACursorCarriesItsSortAndIsRefusedByAnother(t *testing.T) {
	sorts := []string{
		`[]`,
		`[{"field": "last_message_at", "direction": 1}]`,
		`[{"field": "pinned"}]`,
		`[{"field": "pinned"}, {"field": "last_message_at", "direction": 1}]`,
		`[{"field": "review_score"}]`,
		`[{"field": "review_score", "direction": 1}]`,
	}
	position := store.SessionPosition{ID: "abc", Pinned: true, ReviewScore: new(4)}
	for i, made := range sorts {
		query, err := sessionQueryOf(aBackend(), queryAsking(t, `{"sort": `+made+`}`))
		require.NoError(t, err)
		cursor := *encodeCursor(sessionCursor{position, sortName(query.filter.Order)})

		for j, used := range sorts {
			paged, err := sessionQueryOf(aBackend(), queryAsking(t, fmt.Sprintf(`{"sort": %s, "cursor": %q}`, used, cursor)))
			if i == j {
				require.NoError(t, err)
				require.Equal(t, position, *paged.filter.Cursor, "every key of the last row comes back")
				continue
			}
			require.ErrorIs(t, err, errBadCursor, "%s used with %s", made, used)
		}
		_, err = sessionQueryOf(aBackend(), queryAsking(t, fmt.Sprintf(`{"filter": {"text": {"$q": "refund"}}, "cursor": %q}`, cursor)))
		require.ErrorIs(t, err, errBadCursor)
	}

	// The same order written the long way is the same cursor.
	query, err := sessionQueryOf(aBackend(), queryAsking(t, `{"sort": [{"field": "pinned"}]}`))
	require.NoError(t, err)
	cursor := *encodeCursor(sessionCursor{position, sortName(query.filter.Order)})
	_, err = sessionQueryOf(aBackend(), queryAsking(t, fmt.Sprintf(
		`{"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}], "cursor": %q}`, cursor)))
	require.NoError(t, err)
}
