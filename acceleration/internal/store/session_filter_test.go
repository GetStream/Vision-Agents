//go:build integration

package store

import (
	"context"
	"database/sql"
	"slices"
	"time"

	"github.com/uptrace/bun"
)

// after is the cursor that starts the page after a session.
func after(session AgentSession) *SessionPosition {
	position := session.Position()
	return &position
}

// lastAnswered records that the agent last answered in a session at a time, wherever its row is.
func (s *StoreSuite) lastAnswered(id string, at time.Time) {
	s.changed(id, "last_response_at = ?", at)
}

// changed sets one column of a session's row, in whichever shard holds it.
func (s *StoreSuite) changed(id, set string, value any) {
	affected, err := onEveryShard(s.ctx, s.store, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewUpdate().Model((*AgentSession)(nil)).Set(set, value).Where("id = ?", id).Exec(ctx)
	})
	s.Require().NoError(err)
	s.Require().Positive(affected, "no session %s to change", id)
}

// seven records the sessions the order and filter tests read. Their last messages tie in
// pairs, two are pinned, and three have no review.
//
//	id  last message  pinned  score  project  agent
//	s1  +0m                   3      a        docs
//	s2  +0m           yes            a        docs
//	s3  +1m                   5      a        sales
//	s4  +2m                   3      b        docs
//	s5  +2m (answer)          1      b        sales
//	s6  +3m           yes            (none)   docs
//	s7  +4m                          (none)   docs
func (s *StoreSuite) seven() {
	minutes := func(n int) time.Time { return s.base.Add(time.Duration(n) * time.Minute) }
	s.opened("s1", "app", minutes(0), func(session *AgentSession) { session.Project = "a" })
	s.opened("s2", "app", minutes(0), func(session *AgentSession) { session.Project = "a" })
	s.opened("s3", "app", minutes(1), func(session *AgentSession) { session.Project, session.AgentName = "a", "sales" })
	s.opened("s4", "app", minutes(2), func(session *AgentSession) { session.Project = "b" })
	// Opened long before, and answered in at the same instant s4 began.
	s.opened("s5", "app", minutes(-60), func(session *AgentSession) {
		session.Project, session.AgentName, session.UserID = "b", "sales", "randy"
	})
	s.lastAnswered("s5", minutes(2))
	s.opened("s6", "app", minutes(3), nil)
	s.opened("s7", "app", minutes(4), nil)

	for id, score := range map[string]int{"s1": 3, "s3": 5, "s4": 3, "s5": 1} {
		s.changed(id, "review_score = ?", score)
	}
	for _, id := range []string{"s2", "s6"} {
		_, err := s.store.PinSession(s.ctx, "app", id, true, minutes(30))
		s.Require().NoError(err)
	}
	// Somebody else's, which no query of app's may turn up.
	s.opened("theirs", "somebody-else", minutes(5), func(session *AgentSession) { session.Project = "a" })
}

// walked reads a whole list two sessions at a time, following each page's cursor, and fails
// on a page that is not full or a session listed twice.
func (s *StoreSuite) walked(read func(SessionFilter) ([]AgentSession, error), filter SessionFilter) []string {
	filter.Limit = 2
	var listed []string
	for range 20 {
		rows, err := read(filter)
		s.Require().NoError(err)
		kept := rows
		if len(rows) > filter.Limit {
			kept = rows[:filter.Limit]
		}
		for _, row := range kept {
			s.Require().NotContains(listed, row.ID, "listed twice")
			listed = append(listed, row.ID)
		}
		if len(rows) <= filter.Limit {
			return listed
		}
		filter.Cursor = after(kept[len(kept)-1])
	}
	s.FailNow("the pages never ended")
	return nil
}

func (s *StoreSuite) queried(filter SessionFilter) []string {
	return s.walked(func(filter SessionFilter) ([]AgentSession, error) {
		return s.store.QuerySessions(s.ctx, "app", filter)
	}, filter)
}

func (s *StoreSuite) matching(expr SessionExpr) []string {
	found := s.queried(SessionFilter{Where: &expr})
	slices.Sort(found)
	return found
}

func (s *StoreSuite) TestEveryOrderPagesWithoutSkippingOrRepeating() {
	s.seven()

	orders := []struct {
		name  string
		order SessionOrder
		want  []string
	}{
		{"the last message first, ties by id", SessionOrder{},
			[]string{"s7", "s6", "s5", "s4", "s3", "s2", "s1"}},
		{"the oldest message first, ties still by id descending", SessionOrder{Ascending: true},
			[]string{"s2", "s1", "s3", "s5", "s4", "s6", "s7"}},
		{"pinned first, then the last message", SessionOrder{PinnedFirst: true},
			[]string{"s6", "s2", "s7", "s5", "s4", "s3", "s1"}},
		{"pinned first, then the oldest message", SessionOrder{PinnedFirst: true, Ascending: true},
			[]string{"s2", "s6", "s1", "s3", "s5", "s4", "s7"}},
		{"the best score first, unreviewed last", SessionOrder{By: SessionSortReviewScore},
			[]string{"s3", "s4", "s1", "s5", "s7", "s6", "s2"}},
		{"the worst score first, unreviewed still last", SessionOrder{By: SessionSortReviewScore, Ascending: true},
			[]string{"s5", "s4", "s1", "s3", "s7", "s6", "s2"}},
		{"relevance without words is the last message", SessionOrder{By: SessionSortRelevance},
			[]string{"s7", "s6", "s5", "s4", "s3", "s2", "s1"}},
	}
	for _, one := range orders {
		s.Equal(one.want, s.queried(SessionFilter{Order: one.order}), one.name)

		whole, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Order: one.order, Limit: 200})
		s.Require().NoError(err)
		s.Equal(one.want, sessionIDs(whole), "%s, in one page", one.name)
	}
}

func (s *StoreSuite) TestOrdersPageInsideAFilter() {
	s.seven()
	docs := SessionIs(SessionFieldAgent, "docs")

	s.Equal([]string{"s6", "s2", "s7", "s4", "s1"},
		s.queried(SessionFilter{Where: &docs, Order: SessionOrder{PinnedFirst: true}}))
	s.Equal([]string{"s4", "s1", "s7", "s6", "s2"},
		s.queried(SessionFilter{Where: &docs, Order: SessionOrder{By: SessionSortReviewScore, Ascending: true}}))
	s.Equal([]string{"s2", "s1", "s4", "s6", "s7"},
		s.queried(SessionFilter{Where: &docs, Order: SessionOrder{Ascending: true}}))
}

func (s *StoreSuite) TestACursorHoldsWhenSessionsArriveBetweenPages() {
	s.seven()
	order := SessionOrder{PinnedFirst: true}

	first, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Order: order, Limit: 3})
	s.Require().NoError(err)
	s.Equal([]string{"s6", "s2", "s7", "s5"}, sessionIDs(first))

	// A newer session and a newly pinned one both sort before the cursor, so neither
	// shifts where the next page starts.
	s.opened("s8", "app", s.base.Add(time.Hour), nil)
	_, err = s.store.PinSession(s.ctx, "app", "s8", true, time.Time{})
	s.Require().NoError(err)

	second, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Order: order, Limit: 3, Cursor: after(first[2])})
	s.Require().NoError(err)
	s.Equal([]string{"s5", "s4", "s3", "s1"}, sessionIDs(second))
}

func (s *StoreSuite) TestFilterMatchesEachFieldAndOperator() {
	s.seven()
	project, agent, user := SessionFieldProject, SessionFieldAgent, SessionFieldUser

	s.Equal([]string{"s1", "s2", "s3"}, s.matching(SessionIs(project, "a")))
	s.Equal([]string{"s6", "s7"}, s.matching(SessionIs(project, "")), "the sessions in no project")
	s.Empty(s.matching(SessionIs(project, "nowhere")))
	s.Equal([]string{"s1", "s2", "s3", "s4", "s5"}, s.matching(SessionIsOneOf(project, "a", "b")))
	s.Equal([]string{"s4", "s5"}, s.matching(SessionIsOneOf(project, "b", "nowhere")))
	s.Equal([]string{"s3", "s5"}, s.matching(SessionIs(agent, "sales")))
	s.Equal([]string{"s5"}, s.matching(SessionIs(user, "randy")))

	s.Equal([]string{"s1", "s4"}, s.matching(ReviewScoreIs(SessionEq, 3)))
	s.Equal([]string{"s3"}, s.matching(ReviewScoreIs(SessionGt, 3)))
	s.Equal([]string{"s1", "s3", "s4"}, s.matching(ReviewScoreIs(SessionGte, 3)))
	s.Equal([]string{"s5"}, s.matching(ReviewScoreIs(SessionLt, 3)))
	s.Equal([]string{"s1", "s4", "s5"}, s.matching(ReviewScoreIs(SessionLte, 3)))
}

func (s *StoreSuite) TestFilterMatchesTheConfigStateStartAndLabels() {
	minutes := func(n int) time.Time { return s.base.Add(time.Duration(n) * time.Minute) }
	s.opened("c1", "app", minutes(0), func(session *AgentSession) {
		session.Custom = map[string]any{"origin": "test", "attempt": 2}
	})
	s.opened("c2", "app", minutes(1), func(session *AgentSession) {
		session.ConfigID, session.State = "cfg-sales", SessionClosed
	})
	s.opened("c3", "app", minutes(2), func(session *AgentSession) {
		session.Custom = map[string]any{"origin": "production"}
	})
	test := CustomHolds(map[string]string{"origin": "test"})

	s.Equal([]string{"c1", "c3"}, s.matching(SessionIs(SessionFieldConfig, "cfg-docs")))
	s.Equal([]string{"c1", "c3"}, s.matching(SessionIs(SessionFieldState, SessionRunning)))
	s.Equal([]string{"c2"}, s.matching(SessionIs(SessionFieldState, SessionClosed)))

	s.Equal([]string{"c2", "c3"}, s.matching(CreatedAtIs(SessionGte, minutes(1))), "$gte holds at the instant itself")
	s.Equal([]string{"c1"}, s.matching(CreatedAtIs(SessionLt, minutes(1))), "$lt does not")
	s.Equal([]string{"c2"}, s.matching(AllOf(CreatedAtIs(SessionGte, minutes(1)), CreatedAtIs(SessionLt, minutes(2)))))

	s.Equal([]string{"c1"}, s.matching(test))
	s.Empty(s.matching(CustomHolds(map[string]string{"origin": "test", "case": "a"})), "every pair has to be held")
	s.Empty(s.matching(CustomHolds(map[string]string{"attempt": "2"})), "a number is not the string it prints as")
	s.Equal([]string{"c1", "c2"}, s.matching(AnyOf(test, SessionIs(SessionFieldState, SessionClosed))))
}

func (s *StoreSuite) TestASessionWithNoReviewMatchesNoScoreCondition() {
	s.seven()

	// Every score there is, and still not the three nobody reviewed.
	s.Equal([]string{"s1", "s3", "s4", "s5"}, s.matching(ReviewScoreIs(SessionGte, 1)))
	s.Equal([]string{"s1", "s3", "s4", "s5"}, s.matching(ReviewScoreIs(SessionLte, 5)))
	s.Equal([]string{"s1", "s3", "s4", "s5"},
		s.matching(AnyOf(ReviewScoreIs(SessionLt, 3), ReviewScoreIs(SessionGte, 3))))
	// A range is its bounds ANDed.
	s.Equal([]string{"s1", "s4"}, s.matching(AllOf(ReviewScoreIs(SessionGt, 1), ReviewScoreIs(SessionLt, 5))))
	s.Empty(s.matching(AllOf(ReviewScoreIs(SessionGt, 3), ReviewScoreIs(SessionLt, 3))))
	// An unreviewed session is still found by the other branch of an $or.
	s.Equal([]string{"s5", "s6", "s7"},
		s.matching(AnyOf(ReviewScoreIs(SessionEq, 1), SessionIs(SessionFieldProject, ""))))
}

func (s *StoreSuite) TestFilterCombinesWithAndOrAndNesting() {
	s.seven()
	project, agent := SessionFieldProject, SessionFieldAgent

	s.Equal([]string{"s1", "s2"}, s.matching(AllOf(SessionIs(project, "a"), SessionIs(agent, "docs"))))
	s.Equal([]string{"s1", "s2", "s3", "s5"}, s.matching(AnyOf(SessionIs(project, "a"), SessionIs(agent, "sales"))))
	s.Equal([]string{"s1", "s2", "s5"}, s.matching(AnyOf(
		AllOf(SessionIs(project, "a"), SessionIs(agent, "docs")),
		AllOf(SessionIs(project, "b"), SessionIs(agent, "sales")),
	)))
	s.Equal([]string{"s1", "s4"}, s.matching(AllOf(
		SessionIs(agent, "docs"),
		AnyOf(
			AllOf(SessionIs(project, "a"), ReviewScoreIs(SessionGte, 3)),
			AllOf(SessionIs(project, "b"), AnyOf(ReviewScoreIs(SessionEq, 3), ReviewScoreIs(SessionEq, 1))),
		),
	)))
	s.Len(s.matching(AllOf()), 7, "nothing asked for is every session")
	s.Empty(s.matching(AnyOf()), "one of nothing is none")
	s.Empty(s.matching(SessionExpr{Field: "call_id", Op: SessionEq, Strings: []string{"x"}}),
		"a field the store does not know matches nothing rather than everything")
}

func (s *StoreSuite) TestAnOrNeverReachesPastWhoTheCallerMaySee() {
	s.seven()
	// What an end user's device would send to see more than its own: an $or whose second
	// branch names somebody else, and one that is true of every session.
	widened := []SessionExpr{
		AnyOf(SessionIs(SessionFieldProject, "a"), SessionIs(SessionFieldUser, "randy")),
		AnyOf(SessionIs(SessionFieldProject, "a"), SessionIs(SessionFieldProject, "b")),
		AnyOf(SessionIs(SessionFieldUser, "jlahey"), SessionIs(SessionFieldUser, "randy")),
		AnyOf(AllOf(), SessionIs(SessionFieldUser, "randy")),
	}
	for _, where := range widened {
		mine := SessionFilter{UserID: "jlahey", CallerKinds: []string{"user"}, Where: &where, Limit: 200}
		found, err := s.store.QuerySessions(s.ctx, "app", mine)
		s.Require().NoError(err)
		s.NotEmpty(found)
		for _, session := range found {
			s.Equal("jlahey", session.UserID, "%s is somebody else's", session.ID)
			s.Equal("app", session.CustomerID)
		}

		// The kind of caller narrows around the expression too.
		mine.CallerKinds = []string{"anonymous"}
		found, err = s.store.QuerySessions(s.ctx, "app", mine)
		s.Require().NoError(err)
		s.Empty(found)
	}

	// The same holds for a search.
	s.changed("s5", "title = ?", "refund for randy")
	s.changed("s1", "title = ?", "refund for jim")
	where := AnyOf(SessionIs(SessionFieldProject, "b"), SessionIs(SessionFieldUser, "randy"), SessionIs(SessionFieldProject, "a"))
	found, err := s.store.SearchSessions(s.ctx, "app", "refund", SessionFilter{UserID: "jlahey", Where: &where})
	s.Require().NoError(err)
	s.Equal([]string{"s1"}, sessionIDs(found))
}

func (s *StoreSuite) TestSearchNarrowsToAProjectAndPagesByRelevance() {
	minutes := func(n int) time.Time { return s.base.Add(time.Duration(n) * time.Minute) }
	titled := func(id, title, project string, at time.Time) {
		s.opened(id, "app", at, func(session *AgentSession) { session.Title, session.Project = title, project })
	}
	// Three that match equally, so the search ties on rank and falls to the last message
	// and then the id; one that matches better; and two in another project.
	titled("r1", "refund question", "billing", minutes(0))
	titled("r2", "refund question", "billing", minutes(0))
	titled("r3", "refund question", "billing", minutes(1))
	titled("r4", "refund refund refund", "billing", minutes(-5))
	titled("r5", "refund question", "support", minutes(2))
	titled("r6", "refund refund refund", "support", minutes(3))
	titled("r7", "nothing to see", "billing", minutes(4))

	search := func(filter SessionFilter) ([]AgentSession, error) {
		return s.store.SearchSessions(s.ctx, "app", "refund", filter)
	}
	billing := SessionIs(SessionFieldProject, "billing")
	s.Equal([]string{"r4", "r3", "r2", "r1"}, s.walked(search, SessionFilter{Where: &billing}))
	s.Equal([]string{"r6", "r4", "r5", "r3", "r2", "r1"}, s.walked(search, SessionFilter{}))

	either := AnyOf(SessionIs(SessionFieldProject, "support"), SessionIsOneOf(SessionFieldProject, "nowhere"))
	s.Equal([]string{"r6", "r5"}, s.walked(search, SessionFilter{Where: &either}))

	// A search is sorted by relevance whatever order the filter names.
	s.Equal([]string{"r4", "r3", "r2", "r1"},
		s.walked(search, SessionFilter{Where: &billing, Order: SessionOrder{Ascending: true, PinnedFirst: true}}))
}

func (s *StoreSuite) TestPinningKeepsItsFirstTimeAndMovesNothingElse() {
	s.opened("one", "app", s.base, nil)
	s.lastAnswered("one", s.base.Add(time.Minute))
	before, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Nil(before.PinnedAt)

	first := s.base.Add(time.Hour)
	pinnedAt, err := s.store.PinSession(s.ctx, "app", "one", true, first)
	s.Require().NoError(err)
	s.Require().NotNil(pinnedAt)
	s.True(pinnedAt.Equal(first))

	again, err := s.store.PinSession(s.ctx, "app", "one", true, first.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().NotNil(again)
	s.True(again.Equal(first), "pinning again keeps the time it was first pinned")

	pinned, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().NotNil(pinned.PinnedAt)
	s.True(pinned.PinnedAt.Equal(first))
	s.True(pinned.UpdatedAt.Equal(before.UpdatedAt), "a pin is not a change to the conversation")
	s.Equal(before.LastResponseAt, pinned.LastResponseAt)

	// Saving the row again, as a reopened conversation does, keeps the pin.
	s.opened("one", "app", s.base, nil)
	saved, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().NotNil(saved.PinnedAt)

	unpinned, err := s.store.PinSession(s.ctx, "app", "one", false, time.Time{})
	s.Require().NoError(err)
	s.Nil(unpinned)
	cleared, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Nil(cleared.PinnedAt)

	_, err = s.store.PinSession(s.ctx, "somebody-else", "one", true, time.Time{})
	s.Require().ErrorIs(err, ErrNoSession, "another customer's session is not there to pin")
	_, err = s.store.PinSession(s.ctx, "app", "missing", true, time.Time{})
	s.Require().ErrorIs(err, ErrNoSession)
}

func (s *StoreSuite) TestReviewingACallScoresItsSession() {
	s.opened("reviewed", "app", s.base, nil)
	s.opened("unreviewed", "app", s.base.Add(time.Minute), nil)
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{ID: "reviewed", CustomerID: "app", StartedAt: s.base}))
	// A call with no session row, as an incognito one is, is still reviewed.
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{ID: "no-session", CustomerID: "app", StartedAt: s.base}))

	s.Empty(s.matching(ReviewScoreIs(SessionGte, 1)))

	s.Require().NoError(s.store.ReviewCall(s.ctx, "app", "reviewed", "went fine", new(4), "polite"))
	s.Require().NoError(s.store.ReviewCall(s.ctx, "app", "no-session", "went fine", new(2), ""))

	s.Equal([]string{"reviewed"}, s.matching(ReviewScoreIs(SessionEq, 4)))
	s.Equal([]string{"reviewed"}, s.matching(ReviewScoreIs(SessionGt, 3)))
	s.Empty(s.matching(ReviewScoreIs(SessionLte, 3)))
	s.Equal([]string{"reviewed", "unreviewed"}, s.queried(SessionFilter{Order: SessionOrder{By: SessionSortReviewScore}}))

	// Reviewed again with no score, the session has none again.
	s.Require().NoError(s.store.ReviewCall(s.ctx, "app", "reviewed", "nobody spoke", nil, ""))
	s.Empty(s.matching(ReviewScoreIs(SessionGte, 1)))
}

// Defect: a page that skips or repeats a session because the shards were merged in another
// order than each was read in, or a pin or a score written to one shard only.
func (s *ShardedStoreSuite) TestEveryOrderPagesAcrossShards() {
	minutes := func(n int) time.Time { return s.base.Add(time.Duration(n) * time.Minute) }
	in := func(shard *Store, id string, at time.Time, score int) {
		row := &AgentSession{
			ID: id, CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user",
			State: SessionRunning, CreatedAt: at,
		}
		s.Require().NoError(shard.SaveSession(s.ctx, row))
		s.Require().NoError(shard.StartCall(s.ctx, &Call{ID: id, CustomerID: "app", StartedAt: at}))
		if score > 0 {
			// Reviewed from this shard's router, wherever the row is.
			s.Require().NoError(s.store.ReviewCall(s.ctx, "app", id, "", new(score), ""))
		}
	}
	in(s.store, "s1", minutes(0), 3)
	in(s.other, "s2", minutes(0), 0)
	in(s.store, "s3", minutes(1), 5)
	in(s.other, "s4", minutes(2), 3)
	in(s.store, "s5", minutes(2), 1)
	in(s.other, "s6", minutes(3), 0)
	in(s.other, "s7", minutes(4), 0)
	for _, id := range []string{"s2", "s6"} {
		pinnedAt, err := s.store.PinSession(s.ctx, "app", id, true, minutes(30))
		s.Require().NoError(err)
		s.Require().NotNil(pinnedAt, "a session in the other shard is pinned from this one")
	}

	orders := []struct {
		order SessionOrder
		want  []string
	}{
		{SessionOrder{}, []string{"s7", "s6", "s5", "s4", "s3", "s2", "s1"}},
		{SessionOrder{Ascending: true}, []string{"s2", "s1", "s3", "s5", "s4", "s6", "s7"}},
		{SessionOrder{PinnedFirst: true}, []string{"s6", "s2", "s7", "s5", "s4", "s3", "s1"}},
		{SessionOrder{PinnedFirst: true, Ascending: true}, []string{"s2", "s6", "s1", "s3", "s5", "s4", "s7"}},
		{SessionOrder{By: SessionSortReviewScore}, []string{"s3", "s4", "s1", "s5", "s7", "s6", "s2"}},
		{SessionOrder{By: SessionSortReviewScore, Ascending: true}, []string{"s5", "s4", "s1", "s3", "s7", "s6", "s2"}},
	}
	for _, one := range orders {
		s.Equal(one.want, s.queried(SessionFilter{Order: one.order}), "%+v", one.order)
	}
	s.Equal([]string{"s1", "s3", "s4"}, s.matching(ReviewScoreIs(SessionGte, 3)))
}
