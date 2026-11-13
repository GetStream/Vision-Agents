package store

import (
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/uptrace/bun"
)

func TestASessionFilterBindsEveryValueItWasSent(t *testing.T) {
	hostile := `x' OR '1'='1`
	expr := AllOf(
		SessionIs(SessionFieldAgent, hostile),
		AnyOf(
			SessionIsOneOf(SessionFieldProject, hostile, "b"),
			AllOf(ReviewScoreIs(SessionGte, 2), ReviewScoreIs(SessionLt, 5)),
			SessionIs(SessionFieldUser, hostile),
		),
	)

	condition, args := expr.where()

	require.Equal(t,
		"((agent_name = ?) AND ((project IN (?)) OR ((review_score >= ?) AND (review_score < ?)) OR (user_id = ?)))",
		condition, "only columns, operators and placeholders are written into the statement")
	require.Len(t, args, 5, "one argument for each placeholder")
	require.Equal(t, []any{hostile, 2, 5, hostile}, []any{args[0], args[2], args[3], args[4]})
	require.IsType(t, bun.In([]string{}), args[1], "an $in is bound as a list, not written out")

	// What the store does not know how to compare holds for nothing, never for everything.
	for _, unknown := range []SessionExpr{
		{Field: "custom", Op: SessionEq, Strings: []string{"x"}},
		{Field: SessionFieldProject, Op: "$ne", Strings: []string{"x"}},
		{Field: SessionFieldProject, Op: SessionIn},
		{Field: SessionFieldReviewScore, Op: SessionIn, Number: 3},
		{Field: SessionField("project; DROP TABLE agent_sessions"), Op: SessionEq, Strings: []string{"x"}},
	} {
		condition, args := unknown.where()
		require.Equal(t, "FALSE", condition)
		require.Empty(t, args)
		require.False(t, unknown.Matches(SessionValues{Project: "x", ReviewScore: new(3)}))
	}
}

func TestASessionFilterMatchesInGoAsItDoesInTheStatement(t *testing.T) {
	reviewed := SessionValues{Project: "a", Agent: "docs", UserID: "jlahey", ReviewScore: new(3)}
	unreviewed := SessionValues{Project: "a", Agent: "docs", UserID: "jlahey"}

	var none *SessionExpr
	require.True(t, none.Matches(reviewed), "no filter admits every session")

	matches := []struct {
		expr SessionExpr
		want bool
	}{
		{SessionIs(SessionFieldProject, "a"), true},
		{SessionIs(SessionFieldProject, "b"), false},
		{SessionIsOneOf(SessionFieldProject, "b", "a"), true},
		{SessionIsOneOf(SessionFieldProject, "b", "c"), false},
		{SessionIs(SessionFieldAgent, "docs"), true},
		{SessionIs(SessionFieldUser, "randy"), false},
		{ReviewScoreIs(SessionEq, 3), true},
		{ReviewScoreIs(SessionGt, 3), false},
		{ReviewScoreIs(SessionGte, 3), true},
		{ReviewScoreIs(SessionLt, 3), false},
		{ReviewScoreIs(SessionLte, 3), true},
		{AllOf(), true},
		{AnyOf(), false},
		{AllOf(SessionIs(SessionFieldProject, "a"), ReviewScoreIs(SessionGt, 4)), false},
		{AnyOf(SessionIs(SessionFieldProject, "b"), ReviewScoreIs(SessionLt, 4)), true},
		{AnyOf(SessionIs(SessionFieldProject, "b"), AllOf(SessionIs(SessionFieldAgent, "docs"), AnyOf(ReviewScoreIs(SessionEq, 3)))), true},
	}
	for _, one := range matches {
		require.Equal(t, one.want, one.expr.Matches(reviewed), "%+v", one.expr)
	}
	for _, op := range []SessionOperator{SessionEq, SessionGt, SessionGte, SessionLt, SessionLte} {
		for score := 1; score <= 5; score++ {
			condition := ReviewScoreIs(op, score)
			require.False(t, condition.Matches(unreviewed), "a session with no review matches no score")
		}
	}
}

func TestEveryOrderIsTotalAndItsCursorConditionNamesEveryKey(t *testing.T) {
	base := time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
	positions := []SessionPosition{
		{LastMessageAt: base, ID: "a"},
		{LastMessageAt: base, ID: "b", Pinned: true},
		{LastMessageAt: base.Add(time.Minute), ID: "c", ReviewScore: new(3)},
		{LastMessageAt: base.Add(time.Minute), ID: "d", ReviewScore: new(3), Rank: 0.5},
		{LastMessageAt: base.Add(2 * time.Minute), ID: "e", ReviewScore: new(1), Pinned: true, Rank: 0.5},
		{LastMessageAt: base.Add(-time.Minute), ID: "f", Rank: 0.9},
	}
	orders := []SessionOrder{
		{}, {Ascending: true}, {PinnedFirst: true}, {PinnedFirst: true, Ascending: true},
		{By: SessionSortReviewScore}, {By: SessionSortReviewScore, Ascending: true}, {By: SessionSortRelevance},
	}
	wants := [][]string{
		{"e", "d", "c", "b", "a", "f"},
		{"f", "b", "a", "d", "c", "e"},
		{"e", "b", "d", "c", "a", "f"},
		{"b", "e", "f", "a", "d", "c"},
		{"d", "c", "e", "b", "a", "f"},
		{"e", "d", "c", "b", "a", "f"},
		{"f", "e", "d", "c", "b", "a"},
	}
	for i, order := range orders {
		sorted := slices.Clone(positions)
		slices.SortFunc(sorted, order.Compare)
		listed := make([]string, 0, len(sorted))
		for _, position := range sorted {
			listed = append(listed, position.ID)
		}
		require.Equal(t, wants[i], listed, "%+v", order)

		for _, a := range positions {
			for _, b := range positions {
				require.Equal(t, a.ID == b.ID, order.Compare(a, b) == 0, "only a session ties with itself")
				require.Equal(t, order.Compare(a, b), -order.Compare(b, a))
			}
		}
	}

	// Keys that all run one way are one row comparison, which an index is walked from.
	at := SessionPosition{LastMessageAt: base, ID: "a", Pinned: true}
	condition, args := afterSession(SessionOrder{PinnedFirst: true}.keys("", nil, at))
	require.Equal(t, "((pinned_at IS NOT NULL), COALESCE(last_response_at, created_at), id) < (?, ?, ?)", condition)
	require.Equal(t, []any{true, base, "a"}, args)

	// Keys that run both ways are spelled out, level on each key before the one compared.
	condition, args = afterSession(SessionOrder{Ascending: true}.keys("", nil, at))
	require.Equal(t, "((COALESCE(last_response_at, created_at) > ?) OR "+
		"(COALESCE(last_response_at, created_at) = ? AND id < ?))", condition)
	require.Equal(t, []any{base, base, "a"}, args)

	// A search's rank binds its own words before the cursor's value, every time it is named.
	condition, args = afterSession(SessionOrder{By: SessionSortRelevance}.keys("ts_rank(searchable, q(?))", []any{"words"}, at))
	require.Equal(t, "(ts_rank(searchable, q(?)), COALESCE(last_response_at, created_at), id) < (?::real, ?, ?)", condition)
	require.Equal(t, []any{"words", float32(0), base, "a"}, args)
}
