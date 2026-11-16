//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"testing"
	"time"
)

type SessionListSuite struct {
	RouterSuite
}

func TestSessionListSuite(t *testing.T) {
	runSuite(t, new(SessionListSuite))
}

// SetupTest gives every test an app of its own, because a list is everything an app holds.
func (s *SessionListSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionListSuite) TestAUserIsListedTheirOwnSessionsAndTheBackendEveryone() {
	alice := s.client.createSession(textSession(nil))
	bob := s.data.createUser().createSession(textSession(nil))
	server := s.serverClient.createSession(textSession(nil))

	s.Equal([]string{alice.Id}, ids(s.client.querySessions(SessionQuery{}).Items))
	s.ElementsMatch([]string{alice.Id, bob.Id, server.Id}, ids(s.serverClient.querySessions(SessionQuery{}).Items))
}

func (s *SessionListSuite) TestPagesFollowTheCursorTheLastMessageFirstWithoutRepeatsOrGaps() {
	var created []string
	for range 5 {
		created = append(created, s.serverClient.createSession(textSession(nil)).Id)
	}
	slices.Reverse(created)

	var listed []string
	var sizes []int
	query := SessionQuery{Limit: 2}
	for {
		page := s.serverClient.querySessions(query)
		listed = append(listed, ids(page.Items)...)
		sizes = append(sizes, len(page.Items))
		if !page.HasMore {
			s.Nil(page.NextCursor, "the last page has nowhere to go next")
			break
		}
		s.Require().NotNil(page.NextCursor)
		query.Cursor = *page.NextCursor
	}

	s.Equal([]int{2, 2, 1}, sizes)
	s.Equal(created, listed)
}

func (s *SessionListSuite) TestARenamedSessionStaysWhereItWas() {
	first := s.serverClient.createSession(textSession(nil))
	second := s.serverClient.createSession(textSession(nil))

	title := "Renamed long after"
	s.serverClient.updateSession(first.Id, UpdateSessionRequest{Title: &title})

	// A rename is not a message, so the list is still ordered by when each last had one.
	s.Require().Eventually(func() bool {
		listed := s.serverClient.querySessions(SessionQuery{}).Items
		return len(listed) == 2 && value(listed[1].Title) == title
	}, settleFor, 20*time.Millisecond, "the rename was never listed")
	s.Equal([]string{second.Id, first.Id}, ids(s.serverClient.querySessions(SessionQuery{}).Items))
}

// A device listing its sessions is not given their brief, and the backend still is.
func (s *SessionListSuite) TestOnlyTheBackendReadsTheBriefOfASessionItLists() {
	owner := s.data.createUser()
	request, brief := s.data.briefed(textSession(nil))
	s.serverClient.actingFor(owner).createSession(request)

	listed := owner.querySessions(SessionQuery{}).Items
	s.Require().Len(listed, 1)
	s.Nil(listed[0].Instructions)
	listed = s.serverClient.actingFor(owner).querySessions(SessionQuery{}).Items
	s.Require().Len(listed, 1)
	s.Equal(brief, value(listed[0].Instructions))
}

// query is a session query written as the JSON a caller sends.
func query(filter map[string]any, sort ...SessionSort) SessionQuery {
	asked := SessionQuery{Sort: sort}
	if filter != nil {
		asked.Filter = sessionFilter(filter)
	}
	return asked
}

// in is a written session in a project, with a title a search finds it by.
func in(project, title string) CreateSessionRequest {
	request := textSession(nil)
	request.ProjectId = &project
	if title != "" {
		request.Title = &title
	}
	return request
}

// walk reads a whole list a page at a time, failing on a session listed twice.
func (s *SessionListSuite) walk(client *testClient, asked SessionQuery, limit int) []string {
	asked.Limit = limit
	var listed []string
	for range 50 {
		page := client.querySessions(asked)
		s.Require().LessOrEqual(len(page.Items), limit)
		for _, id := range ids(page.Items) {
			s.Require().NotContains(listed, id, "listed twice")
			listed = append(listed, id)
		}
		if !page.HasMore {
			s.Nil(page.NextCursor, "the last page has nowhere to go next")
			return listed
		}
		s.Require().NotNil(page.NextCursor)
		asked.Cursor = *page.NextCursor
	}
	s.FailNow("the pages never ended")
	return nil
}

// reviewed scores a session as the review of its call does, once its rows are written.
func (s *SessionListSuite) reviewed(id string, score int) {
	s.serverClient.querySessions(SessionQuery{})
	s.Require().Eventually(func() bool {
		return s.store.ReviewCall(context.Background(), s.customerID(), id, "", &score, "") == nil
	}, settleFor, 20*time.Millisecond, "the session's call was never there to review")
}

// refuses checks a query is a 400 that says what was wrong with it.
func (s *SessionListSuite) refuses(client *testClient, asked any, says ...string) {
	status, message := client.failure(http.MethodPost, "/v1/agents/sessions/query", asked)
	s.Equal(http.StatusBadRequest, status, "%v: %s", asked, message)
	for _, part := range says {
		s.Contains(message, part, "%v", asked)
	}
}

func (s *SessionListSuite) TestAProjectNarrowsTheList() {
	wanted := s.serverClient.createSession(in("support", ""))
	other := s.serverClient.createSession(in("sales", ""))
	s.serverClient.createSession(in("billing", ""))

	s.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(query(map[string]any{"project_id": "support"})).Items))
	s.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(
		query(map[string]any{"project_id": map[string]any{"$eq": "support"}})).Items),
		"$eq is the long way of writing the same filter")
	s.Equal([]string{other.Id, wanted.Id}, ids(s.serverClient.querySessions(
		query(map[string]any{"project_id": map[string]any{"$in": []string{"support", "sales", "nowhere"}}})).Items))
}

func (s *SessionListSuite) TestAnAgentNarrowsTheList() {
	support, sales := s.data.createAgentConfig(), s.data.createAgentConfig()
	opened := func(config AgentConfig, project string) Session {
		request := in(project, "")
		request.Agent = &config.Name
		return s.serverClient.createSession(request)
	}
	wanted := opened(support, "health")
	elsewhere := opened(support, "billing")
	opened(sales, "health")

	s.Equal([]string{elsewhere.Id, wanted.Id}, ids(s.serverClient.querySessions(query(map[string]any{"agent": support.Name})).Items))
	s.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(
		query(map[string]any{"agent": map[string]any{"$eq": support.Name}, "project_id": "health"})).Items))
}

func (s *SessionListSuite) TestATextSearchFindsWhatTheSessionWasCalled() {
	wanted := s.serverClient.createSession(in("billing", "Refund for a duplicate charge"))
	s.serverClient.createSession(in("billing", "Paginating a channel list"))

	s.Require().Eventually(func() bool {
		found := s.serverClient.querySessions(query(map[string]any{"text": map[string]any{"$q": "refund"}}))
		return slices.Equal([]string{wanted.Id}, ids(found.Items))
	}, settleFor, 20*time.Millisecond, "the search never found the session by its title")
}

func (s *SessionListSuite) TestASearchIsNarrowedToAProject() {
	wanted := s.serverClient.createSession(in("billing", "Refund for a duplicate charge"))
	s.serverClient.createSession(in("support", "Refund policy question"))
	s.serverClient.createSession(in("billing", "Paginating a channel list"))

	searched := query(map[string]any{"text": map[string]any{"$q": "refund"}, "project_id": "billing"})
	s.Require().Eventually(func() bool {
		return slices.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(searched).Items))
	}, settleFor, 20*time.Millisecond, "the search never narrowed to the project")

	// The same search with its text in a top-level $and, sorted by relevance by name.
	and := query(map[string]any{"$and": []any{
		map[string]any{"text": map[string]any{"$q": "refund"}},
		map[string]any{"project_id": map[string]any{"$in": []string{"billing"}}},
	}}, SessionSort{Field: SessionSortRelevance, Direction: -1})
	s.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(and).Items))
}

func (s *SessionListSuite) TestAStoppedSessionIsListedWithWhenItEnded() {
	closing := s.serverClient.createSession(textSession(nil))

	s.serverClient.stopSession(closing.Id)

	s.Require().Eventually(func() bool {
		for _, one := range s.serverClient.querySessions(SessionQuery{}).Items {
			if one.Id == closing.Id {
				return one.ClosedAt != nil && one.State == Ended
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond, "the closed session was never listed as closed")
}

func (s *SessionListSuite) TestTheDocumentedQueriesAnswer() {
	support, sales := s.data.createAgentConfig(), s.data.createAgentConfig()
	opened := func(config AgentConfig, project, title string) Session {
		request := in(project, title)
		request.Agent = &config.Name
		return s.serverClient.createSession(request)
	}
	health := opened(support, "health", "Refund for a duplicate charge")
	billing := opened(support, "billing", "Refund policy question")
	vip := opened(support, "vip", "A greeting")
	opened(sales, "health", "Refund of a deposit")
	pilot := opened(sales, "pilot", "Another greeting")
	s.serverClient.updateSession(billing.Id, UpdateSessionRequest{Pinned: pointerTo(true)})
	s.reviewed(health.Id, 2)
	s.reviewed(vip.Id, 5)

	var page SessionPage
	asks := func(body string) []string {
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/query",
			json.RawMessage(body), &page), body)
		return ids(page.Items)
	}

	// Every session, pinned first.
	listed := asks(`{"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}`)
	s.Require().Len(listed, 5)
	s.Equal(billing.Id, listed[0])
	s.Equal(pilot.Id, listed[1], "then the last message first")

	// One agent's sessions in either of two projects.
	s.Equal([]string{billing.Id, health.Id}, asks(
		`{"filter": {"agent": "`+support.Name+`", "$or": [{"project_id": "health"}, {"project_id": "billing"}]}}`))

	// A search within one project.
	s.Require().Eventually(func() bool {
		return slices.Equal([]string{billing.Id},
			asks(`{"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}`))
	}, settleFor, 20*time.Millisecond, "the search never found the session in its project")

	// A backend's badly reviewed sessions and everything in two projects, the lowest score first.
	s.Equal([]string{health.Id, vip.Id, pilot.Id}, asks(
		`{"filter": {"$or": [{"review_score": {"$lte": 2}}, {"project_id": {"$in": ["vip", "pilot"]}}]}, `+
			`"sort": [{"field": "review_score", "direction": 1}]}`))
}

func (s *SessionListSuite) TestAndOrAndNestingChooseAmongTheSessions() {
	a := s.serverClient.createSession(in("a", ""))
	b := s.serverClient.createSession(in("b", ""))
	c := s.serverClient.createSession(in("c", ""))
	s.reviewed(a.Id, 5)
	s.reviewed(b.Id, 2)

	project := func(name string) map[string]any { return map[string]any{"project_id": name} }
	listed := func(filter map[string]any) []string {
		return ids(s.serverClient.querySessions(query(filter)).Items)
	}

	s.Equal([]string{c.Id, a.Id}, listed(map[string]any{"$or": []any{project("a"), project("c")}}))
	s.Empty(listed(map[string]any{"$and": []any{project("a"), project("c")}}))
	s.Equal([]string{a.Id}, listed(map[string]any{"$and": []any{project("a"), map[string]any{"review_score": 5}}}))
	s.Equal([]string{c.Id, b.Id}, listed(map[string]any{"$or": []any{
		project("c"),
		map[string]any{"$and": []any{project("b"), map[string]any{"review_score": map[string]any{"$gte": 1, "$lt": 5}}}},
		map[string]any{"$and": []any{project("a"), map[string]any{"$or": []any{map[string]any{"review_score": 1}}}}},
	}}))
	s.Equal([]string{c.Id, b.Id, a.Id}, listed(map[string]any{}), "an empty filter is every session")
}

func (s *SessionListSuite) TestAReviewedSessionMatchesScoreFiltersAndSorts() {
	good := s.serverClient.createSession(textSession(nil))
	bad := s.serverClient.createSession(textSession(nil))
	unreviewed := s.serverClient.createSession(textSession(nil))

	score := func(condition any) []string {
		return ids(s.serverClient.querySessions(query(map[string]any{"review_score": condition})).Items)
	}
	s.Empty(score(map[string]any{"$gte": 1}), "nobody has been reviewed yet")

	s.reviewed(good.Id, 5)
	s.reviewed(bad.Id, 2)

	s.Equal([]string{good.Id}, score(5))
	s.Equal([]string{good.Id}, score(map[string]any{"$eq": 5}))
	s.Equal([]string{good.Id}, score(map[string]any{"$gt": 2}))
	s.Equal([]string{bad.Id, good.Id}, score(map[string]any{"$gte": 2}))
	s.Equal([]string{bad.Id}, score(map[string]any{"$lt": 5}))
	s.Equal([]string{bad.Id, good.Id}, score(map[string]any{"$lte": 5}), "a session with no review matches no score")
	s.Empty(score(map[string]any{"$gt": 2, "$lt": 5}), "a range is its bounds ANDed")

	best := s.serverClient.querySessions(query(nil, SessionSort{Field: SessionSortReviewScore, Direction: -1}))
	s.Equal([]string{good.Id, bad.Id, unreviewed.Id}, ids(best.Items))
	worst := s.serverClient.querySessions(query(nil, SessionSort{Field: SessionSortReviewScore, Direction: 1}))
	s.Equal([]string{bad.Id, good.Id, unreviewed.Id}, ids(worst.Items), "a session nobody reviewed is last either way")
	for _, row := range best.Items {
		s.Nil(row.ReviewScore, "a query's rows do not carry the score")
	}
	s.Equal([]string{bad.Id, good.Id, unreviewed.Id},
		s.walk(s.serverClient, query(nil, SessionSort{Field: SessionSortReviewScore, Direction: 1}), 1))
}

func (s *SessionListSuite) TestOnlyABackendFiltersOrSortsOnReviewScores() {
	mine := s.client.createSession(textSession(nil))
	s.reviewed(mine.Id, 1)

	s.refuses(s.client, query(map[string]any{"review_score": 1}), "only a server-side caller may filter on review_score")
	s.refuses(s.client, query(map[string]any{"$or": []any{
		map[string]any{"project_id": "a"}, map[string]any{"review_score": map[string]any{"$lte": 2}}}}),
		"only a server-side caller may filter on review_score")
	s.refuses(s.client, query(nil, SessionSort{Field: SessionSortReviewScore}),
		"only a server-side caller may sort by review_score")

	listed := s.client.querySessions(SessionQuery{}).Items
	s.Require().Len(listed, 1)
	s.Nil(listed[0].ReviewScore)
}

func (s *SessionListSuite) TestABackendListsOneUsersSessionsAndADeviceMayNotName() {
	mine := s.client.createSession(textSession(nil))
	stranger := s.data.createUser()
	theirs := stranger.createSession(textSession(nil))
	s.serverClient.createSession(textSession(nil))

	listed := s.serverClient.querySessions(query(map[string]any{"user_id": s.client.userID}))
	s.Equal([]string{mine.Id}, ids(listed.Items))
	listed = s.serverClient.querySessions(query(map[string]any{"$and": []any{
		map[string]any{"user_id": map[string]any{"$eq": stranger.userID}}}}))
	s.Equal([]string{theirs.Id}, ids(listed.Items))

	for _, named := range []string{stranger.userID, s.client.userID} {
		s.refuses(s.client, query(map[string]any{"user_id": named}),
			"only a server-side caller may list another user's sessions")
	}
	s.refuses(s.serverClient, query(map[string]any{"$or": []any{
		map[string]any{"user_id": s.client.userID}, map[string]any{"user_id": stranger.userID}}}),
		"user_id may only be at the top level")
}

func (s *SessionListSuite) TestAnOrNeverListsAUserSomebodyElsesSessions() {
	stranger := s.data.createUser()
	mineA := s.client.createSession(in("a", "Refund for a duplicate charge"))
	mineB := s.client.createSession(in("b", ""))
	theirsA := stranger.createSession(in("a", "Refund of a deposit"))
	theirsB := stranger.createSession(in("b", ""))
	backendA := s.serverClient.createSession(in("a", "Refund policy question"))
	// Sessions nothing records are listed from memory, by the other half of the matcher.
	unrecorded := func(client *testClient, project string) Session {
		request := in(project, "")
		request.Incognito = pointerTo(true)
		return client.createSession(request)
	}
	mineLive := unrecorded(s.client, "a")
	theirsLive := unrecorded(stranger, "a")
	backendLive := unrecorded(s.serverClient, "b")
	strangers := []string{theirsA.Id, theirsB.Id, backendA.Id, theirsLive.Id, backendLive.Id}

	// Naming somebody else in one branch of an $or is refused outright.
	s.refuses(s.client, query(map[string]any{"$or": []any{
		map[string]any{"project_id": "a"}, map[string]any{"user_id": "someone-else"}}}))
	s.refuses(s.client, query(map[string]any{"$or": []any{
		map[string]any{"project_id": "a"}, map[string]any{"user_id": stranger.userID}}}))

	// An $or that holds for everybody's sessions still lists only the caller's own.
	widened := []map[string]any{
		{"$or": []any{map[string]any{"project_id": "a"}, map[string]any{"project_id": "b"}}},
		{"$or": []any{map[string]any{"project_id": map[string]any{"$in": []string{"a", "b"}}}}},
		{"$or": []any{
			map[string]any{"$and": []any{map[string]any{"project_id": "a"}}},
			map[string]any{"$or": []any{map[string]any{"project_id": "b"}, map[string]any{"project_id": "a"}}},
		}},
		{"project_id": map[string]any{"$in": []string{"a", "b"}}},
		{},
	}
	sorts := [][]SessionSort{
		nil,
		{{Field: SessionSortLastMessageAt, Direction: 1}},
		{{Field: SessionSortPinned}},
		{{Field: SessionSortPinned, Direction: -1}, {Field: SessionSortLastMessageAt, Direction: 1}},
	}
	for _, filter := range widened {
		for _, sort := range sorts {
			listed := s.walk(s.client, query(filter, sort...), 2)
			s.ElementsMatch([]string{mineA.Id, mineB.Id, mineLive.Id}, listed, "%v sorted %v", filter, sort)
			for _, id := range strangers {
				s.NotContains(listed, id)
			}
		}
	}
	s.ElementsMatch([]string{mineA.Id, mineLive.Id},
		ids(s.client.querySessions(query(map[string]any{"$or": []any{map[string]any{"project_id": "a"}}})).Items))

	// A search under an $or is as narrow.
	searched := query(map[string]any{
		"text": map[string]any{"$q": "refund"},
		"$or":  []any{map[string]any{"project_id": "a"}, map[string]any{"project_id": "b"}},
	})
	s.Require().Eventually(func() bool {
		return slices.Equal([]string{mineA.Id}, ids(s.client.querySessions(searched).Items))
	}, settleFor, 20*time.Millisecond, "the search never found the caller's own session")
	s.Require().Eventually(func() bool {
		return len(s.serverClient.querySessions(searched).Items) == 3
	}, settleFor, 20*time.Millisecond, "the backend's search never found all three")
	s.Equal([]string{mineA.Id}, ids(s.client.querySessions(searched).Items))

	// The stranger sees their own and nothing of the caller's.
	s.ElementsMatch([]string{theirsA.Id, theirsB.Id, theirsLive.Id},
		s.walk(stranger, query(widened[0]), 2))
}

func (s *SessionListSuite) TestEveryOrderPagesThroughRecordedAndUnrecordedSessions() {
	var created []string
	for i := range 7 {
		request := in("p", "")
		// Every other session is one nothing records, listed from memory among the rows.
		request.Incognito = pointerTo(i%2 == 1)
		created = append(created, s.serverClient.createSession(request).Id)
	}
	pinned := []string{created[0], created[4]}
	for _, id := range pinned {
		s.serverClient.updateSession(id, UpdateSessionRequest{Pinned: pointerTo(true)})
	}
	s.reviewed(created[2], 4)
	s.reviewed(created[6], 1)
	newest := slices.Clone(created)
	slices.Reverse(newest)
	without := func(listed []string, dropped ...string) []string {
		return slices.DeleteFunc(slices.Clone(listed), func(id string) bool { return slices.Contains(dropped, id) })
	}

	orders := []struct {
		name string
		sort []SessionSort
		want []string
	}{
		{"the last message first", nil, newest},
		{"the last message first, by name", []SessionSort{{Field: SessionSortLastMessageAt, Direction: -1}}, newest},
		{"the oldest message first", []SessionSort{{Field: SessionSortLastMessageAt, Direction: 1}}, created},
		{"pinned first", []SessionSort{{Field: SessionSortPinned}},
			append([]string{created[4], created[0]}, without(newest, pinned...)...)},
		{"pinned first, then the oldest", []SessionSort{{Field: SessionSortPinned, Direction: -1}, {Field: SessionSortLastMessageAt, Direction: 1}},
			append([]string{created[0], created[4]}, without(created, pinned...)...)},
		{"the best score first", []SessionSort{{Field: SessionSortReviewScore, Direction: -1}},
			append([]string{created[2], created[6]}, without(newest, created[2], created[6])...)},
		{"the worst score first", []SessionSort{{Field: SessionSortReviewScore, Direction: 1}},
			append([]string{created[6], created[2]}, without(newest, created[2], created[6])...)},
	}
	for _, one := range orders {
		for _, limit := range []int{1, 2, 3} {
			s.Equal(one.want, s.walk(s.serverClient, query(map[string]any{"project_id": "p"}, one.sort...), limit),
				"%s, %d at a time", one.name, limit)
		}
	}
}

func (s *SessionListSuite) TestAUserPinsAndUnpinsTheirOwnSession() {
	first := s.client.createSession(textSession(nil))
	second := s.client.createSession(textSession(nil))
	s.Nil(first.PinnedAt)

	pinned := s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(true)})
	s.Require().NotNil(pinned.PinnedAt)
	s.WithinDuration(time.Now(), *pinned.PinnedAt, time.Minute)
	s.True(s.client.getSession(first.Id).PinnedAt.Equal(*pinned.PinnedAt), "a get reads the pin back")
	s.True(s.serverClient.actingFor(s.client).getSession(first.Id).PinnedAt.Equal(*pinned.PinnedAt), "the backend reads it too")

	// Pinning again keeps the time it was first pinned.
	again := s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(true)})
	s.Require().NotNil(again.PinnedAt)
	s.True(again.PinnedAt.Equal(*pinned.PinnedAt))
	// A change that says nothing of the pin leaves it, and still reports it.
	renamed := s.client.updateSession(first.Id, UpdateSessionRequest{Title: pointerTo("Kept")})
	s.Require().NotNil(renamed.PinnedAt)
	s.True(renamed.PinnedAt.Equal(*pinned.PinnedAt))

	// The pin does not move the session where the list is sorted by its last message.
	listed := s.client.querySessions(SessionQuery{}).Items
	s.Equal([]string{second.Id, first.Id}, ids(listed))
	s.Require().NotNil(listed[1].PinnedAt)
	s.True(listed[1].PinnedAt.Equal(*pinned.PinnedAt), "a query's rows carry the pin")
	s.Nil(listed[0].PinnedAt)
	s.Equal(first.CreatedAt, listed[1].CreatedAt)
	s.Nil(listed[1].LastResponseAt, "a pin is not an answer")
	oldest := s.client.querySessions(query(nil, SessionSort{Field: SessionSortLastMessageAt, Direction: 1})).Items
	s.Equal([]string{first.Id, second.Id}, ids(oldest))

	// Sorted by pinned, it is first.
	s.Equal([]string{first.Id, second.Id},
		ids(s.client.querySessions(query(nil, SessionSort{Field: SessionSortPinned})).Items))
	s.Equal([]string{first.Id, second.Id}, ids(s.client.querySessions(query(nil,
		SessionSort{Field: SessionSortPinned, Direction: -1}, SessionSort{Field: SessionSortLastMessageAt, Direction: -1})).Items))

	unpinned := s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(false)})
	s.Nil(unpinned.PinnedAt)
	s.Nil(s.client.getSession(first.Id).PinnedAt)
	s.Equal([]string{second.Id, first.Id},
		ids(s.client.querySessions(query(nil, SessionSort{Field: SessionSortPinned})).Items))
	// Unpinning what is not pinned is not a failure.
	s.Nil(s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(false)}).PinnedAt)
}

func (s *SessionListSuite) TestAnEndedSessionIsPinnedAndKeepsItsPlace() {
	first := s.client.createSession(textSession(nil))
	second := s.client.createSession(textSession(nil))
	s.client.stopSession(first.Id)
	s.Require().Eventually(func() bool {
		return s.client.getSession(first.Id).State == Ended
	}, settleFor, 20*time.Millisecond, "the stopped session never read as ended")
	before, err := s.store.StoredSession(context.Background(), s.customerID(), first.Id)
	s.Require().NoError(err)

	pinned := s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(true)})
	s.Require().NotNil(pinned.PinnedAt)
	s.Equal(Ended, pinned.State)
	again := s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(true)})
	s.True(again.PinnedAt.Equal(*pinned.PinnedAt), "pinning again keeps the time it was first pinned")

	row, err := s.store.StoredSession(context.Background(), s.customerID(), first.Id)
	s.Require().NoError(err)
	s.True(row.UpdatedAt.Equal(before.UpdatedAt), "a pin does not touch updated_at")
	s.Equal(before.LastResponseAt, row.LastResponseAt)

	s.Equal([]string{second.Id, first.Id}, ids(s.client.querySessions(SessionQuery{}).Items))
	s.Equal([]string{first.Id, second.Id},
		ids(s.client.querySessions(query(nil, SessionSort{Field: SessionSortPinned})).Items))

	s.Nil(s.client.updateSession(first.Id, UpdateSessionRequest{Pinned: pointerTo(false)}).PinnedAt)
}

func (s *SessionListSuite) TestNobodyPinsSomebodyElsesSessionOrOneNothingRecords() {
	mine := s.client.createSession(textSession(nil))
	stranger := s.data.createUser()

	s.Equal(http.StatusNotFound, stranger.do(http.MethodPatch, "/v1/agents/sessions/"+mine.Id,
		UpdateSessionRequest{Pinned: pointerTo(true)}, nil))
	s.Nil(s.client.getSession(mine.Id).PinnedAt)

	hidden := textSession(nil)
	hidden.Incognito = pointerTo(true)
	incognito := s.client.createSession(hidden)
	status, message := s.client.failure(http.MethodPatch, "/v1/agents/sessions/"+incognito.Id,
		UpdateSessionRequest{Pinned: pointerTo(true), Title: pointerTo("Not applied")})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, "an incognito session is not recorded, so it cannot be pinned")
	s.Nil(s.client.getSession(incognito.Id).Title, "a refused change changes nothing")
}

// opened is a session the backend opens, against an agent config when one is given, and
// recorded unless it is incognito, which is listed from memory.
func (s *SessionListSuite) opened(config *AgentConfig, incognito bool, custom map[string]any) Session {
	request := textSession(nil)
	request.Incognito = &incognito
	if config != nil {
		request.Agent = &config.Name
	}
	if custom != nil {
		request.Custom = &custom
	}
	return s.serverClient.createSession(request)
}

func (s *SessionListSuite) listed(filter map[string]any) []string {
	return ids(s.serverClient.querySessions(query(filter)).Items)
}

func (s *SessionListSuite) TestAConfigIdNarrowsTheList() {
	support, sales := s.data.createAgentConfig(), s.data.createAgentConfig()
	recorded, unrecorded := s.opened(&support, false, nil), s.opened(&support, true, nil)
	s.opened(&sales, false, nil)

	s.ElementsMatch([]string{recorded.Id, unrecorded.Id}, s.listed(map[string]any{"config_id": support.Id}))
	s.ElementsMatch([]string{recorded.Id, unrecorded.Id},
		s.listed(map[string]any{"config_id": map[string]any{"$eq": support.Id}}))
}

func (s *SessionListSuite) TestAStateNarrowsTheListToLiveOrEndedSessions() {
	running, unrecorded := s.opened(nil, false, nil), s.opened(nil, true, nil)
	stopped := s.opened(nil, false, nil)

	s.serverClient.stopSession(stopped.Id)

	s.Require().Eventually(func() bool {
		return slices.Equal([]string{stopped.Id}, s.listed(map[string]any{"state": "ended"}))
	}, settleFor, 20*time.Millisecond, "the stopped session was never listed as ended")
	s.ElementsMatch([]string{running.Id, unrecorded.Id}, s.listed(map[string]any{"state": map[string]any{"$eq": "live"}}))
}

func (s *SessionListSuite) TestACustomLabelNarrowsTheList() {
	support := s.data.createAgentConfig()
	test := map[string]any{"origin": "test", "attempt": 2}
	recorded, unrecorded := s.opened(nil, false, test), s.opened(nil, true, test)
	production := s.opened(nil, false, map[string]any{"origin": "production"})
	unlabelled := s.opened(&support, false, nil)
	tests := map[string]any{"custom": map[string]any{"origin": "test"}}

	s.ElementsMatch([]string{recorded.Id, unrecorded.Id}, s.listed(tests))
	s.Equal([]string{production.Id}, s.listed(map[string]any{"custom": map[string]any{"origin": "production"}}))
	s.Empty(s.listed(map[string]any{"custom": map[string]any{"origin": "test", "case": "a"}}), "every label has to be held")
	s.Empty(s.listed(map[string]any{"custom": map[string]any{"attempt": "2"}}), "a number is not the string it prints as")
	s.ElementsMatch([]string{recorded.Id, unrecorded.Id, unlabelled.Id},
		s.listed(map[string]any{"$or": []any{tests, map[string]any{"config_id": support.Id}}}),
		"an $or chooses between a label and another field")
}

func (s *SessionListSuite) TestACreationWindowNarrowsTheList() {
	first, second := s.opened(nil, false, nil), s.opened(nil, true, nil)
	s.Require().Greater(second.CreatedAt.Sub(first.CreatedAt), 2*time.Microsecond)
	between := first.CreatedAt.Add(second.CreatedAt.Sub(first.CreatedAt) / 2).Format(time.RFC3339Nano)

	s.Equal([]string{second.Id}, s.listed(map[string]any{"created_at": map[string]any{"$gte": between}}))
	s.Equal([]string{first.Id}, s.listed(map[string]any{"created_at": map[string]any{"$lt": between}}))
	s.Empty(s.listed(map[string]any{"created_at": map[string]any{
		"$gte": first.CreatedAt.Add(-time.Hour).Format(time.RFC3339Nano), "$lt": first.CreatedAt.Add(-time.Minute).Format(time.RFC3339Nano),
	}}))
}

func (s *SessionListSuite) TestAFieldThatIsNoLongerFilteredOnIsRefusedByName() {
	removed := map[string]any{
		"agent_id": "abc",
		"modality": "text",
		"team":     "finance",
		"call_id":  "abc",
	}
	for field, condition := range removed {
		s.refuses(s.serverClient, query(map[string]any{field: condition}),
			`"`+field+`" cannot be filtered on`,
			"the fields that can be filtered are project_id, review_score, text, agent, user_id, config_id, state, "+
				"created_at and custom, combined with $and and $or")
		s.refuses(s.client, query(map[string]any{"$or": []any{map[string]any{field: condition}}}),
			`"`+field+`" cannot be filtered on`)
	}
}

func (s *SessionListSuite) TestAnOperatorOrAShapeTheFilterDoesNotTakeIsRefused() {
	project := map[string]any{"project_id": "a"}
	nested := any(project)
	for range 4 {
		nested = map[string]any{"$and": []any{nested}}
	}
	eleven := slices.Repeat([]any{project}, 11)
	fiftyOne := slices.Repeat([]string{"p"}, 51)
	thirtyOne := map[string]any{"project_id": "x", "$or": slices.Repeat([]any{
		map[string]any{"project_id": "a", "agent": "b", "review_score": 3}}, 10)}
	text := map[string]any{"text": map[string]any{"$q": "refund"}}

	refusals := []struct {
		filter any
		says   string
	}{
		{map[string]any{"$nor": []any{project}}, `"$nor" cannot be filtered on`},
		{map[string]any{"project_id": map[string]any{"$gt": "a"}}, `project_id does not take "$gt"`},
		{map[string]any{"agent": map[string]any{"$in": []string{"a"}}}, `agent does not take "$in"`},
		{map[string]any{"review_score": map[string]any{"$ne": 3}}, `review_score does not take "$ne"`},
		{map[string]any{"review_score": 6}, "review_score is a whole number from 1 to 5"},
		{map[string]any{"text": "refund"}, `text takes {"$q": "words"}`},
		{map[string]any{"$or": []any{text, project}}, "text cannot be used inside $or"},
		{map[string]any{"$and": []any{map[string]any{"$or": []any{text}}}}, "text cannot be used inside $or"},
		{map[string]any{"text": map[string]any{"$q": "a"}, "$and": []any{text}}, "text may appear only once"},
		{nested, "$and and $or nest at most 3 deep"},
		{map[string]any{"$or": eleven}, "$or takes at most 10 filters, and was sent 11"},
		{map[string]any{"$and": []any{}}, "$and takes a list of 1 to 10 filters"},
		{map[string]any{"project_id": map[string]any{"$in": fiftyOne}}, "project_id $in takes a list of 1 to 50 strings"},
		{thirtyOne, "a filter holds at most 30 field conditions"},
		{"project_id", "a filter is an object"},
	}
	for _, refusal := range refusals {
		s.refuses(s.serverClient, map[string]any{"filter": refusal.filter}, refusal.says)
	}
}

func (s *SessionListSuite) TestASortThatIsNotOfferedIsRefusedWithTheOnesThatAre() {
	sorted := func(keys ...map[string]any) map[string]any {
		sort := make([]any, 0, len(keys))
		for _, key := range keys {
			sort = append(sort, key)
		}
		return map[string]any{"sort": sort}
	}
	key := func(field string, direction ...int) map[string]any {
		if len(direction) == 0 {
			return map[string]any{"field": field}
		}
		return map[string]any{"field": field, "direction": direction[0]}
	}
	searching := func(sort map[string]any) map[string]any {
		sort["filter"] = map[string]any{"text": map[string]any{"$q": "refund"}}
		return sort
	}

	refusals := []struct {
		query map[string]any
		says  string
	}{
		{sorted(key("updated_at")), "updated_at is no longer a sort field"},
		{sorted(key("updated_at", 1)), "updated_at is no longer a sort field"},
		{sorted(key("created_at")), `"created_at" is not a field sessions sort on`},
		{sorted(key("relevance")), "only a text search is sorted by relevance"},
		{sorted(key("pinned", 1)), "pinned only sorts descending"},
		{sorted(key("last_message_at"), key("pinned")), "that combination of sort fields is not offered"},
		{sorted(key("pinned"), key("review_score")), "that combination of sort fields is not offered"},
		{sorted(key("review_score"), key("last_message_at")), "that combination of sort fields is not offered"},
		{sorted(key("pinned"), key("last_message_at"), key("review_score")), "that combination of sort fields is not offered"},
		{sorted(key("last_message_at", 2)), "a sort direction is 1 or -1, not 2"},
		{searching(sorted(key("last_message_at"))), "a text search is sorted by relevance"},
		{searching(sorted(key("pinned"), key("last_message_at"))), "a text search is sorted by relevance"},
	}
	for _, refusal := range refusals {
		s.refuses(s.serverClient, refusal.query, refusal.says, `sort is one of [{"field":"last_message_at"}]`)
	}
}

func (s *SessionListSuite) TestAQueryWithAKeyOrALimitItDoesNotTakeIsRefused() {
	s.refuses(s.serverClient, map[string]any{"filters": map[string]any{"project_id": "a"}},
		`"filters" is not part of a session query`)
	s.refuses(s.serverClient, map[string]any{"limit": 201}, "limit is from 1 to 200")
	s.refuses(s.serverClient, map[string]any{"limit": "many"})
	s.refuses(s.serverClient, map[string]any{"cursor": "nonsense"}, "cursor is not one this list handed out")
}

func (s *SessionListSuite) TestACursorFromOneOrderIsRefusedByAnother() {
	for range 2 {
		s.serverClient.createSession(textSession(nil))
	}
	page := s.serverClient.querySessions(SessionQuery{Limit: 1})
	s.Require().NotNil(page.NextCursor)

	others := []SessionQuery{
		query(map[string]any{"text": map[string]any{"$q": "refund"}}),
		query(nil, SessionSort{Field: SessionSortLastMessageAt, Direction: 1}),
		query(nil, SessionSort{Field: SessionSortPinned}),
		query(nil, SessionSort{Field: SessionSortReviewScore}),
	}
	for _, other := range others {
		other.Cursor = *page.NextCursor
		s.refuses(s.serverClient, other, "cursor is not one this list handed out")
	}
	// The order it was made for takes it, written out or left to the default.
	same := query(nil, SessionSort{Field: SessionSortLastMessageAt, Direction: -1})
	same.Cursor, same.Limit = *page.NextCursor, 1
	s.Len(s.serverClient.querySessions(same).Items, 1)
}

func (s *SessionListSuite) TestNobodyIsListedAnythingWithoutCredentials() {
	s.serverClient.createSession(textSession(nil))

	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodPost, "/v1/agents/sessions/query", SessionQuery{}, nil))
}
