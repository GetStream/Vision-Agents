//go:build integration

package api

import (
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

func (s *SessionListSuite) TestPagesFollowTheCursorMostRecentlyUpdatedFirstWithoutRepeatsOrGaps() {
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

func (s *SessionListSuite) TestARenamedSessionMovesToTheTop() {
	first := s.serverClient.createSession(textSession(nil))
	s.serverClient.createSession(textSession(nil))

	title := "Picked up again"
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/sessions/"+first.Id,
		UpdateSessionRequest{Title: &title}, nil))

	s.Require().Eventually(func() bool {
		return ids(s.serverClient.querySessions(SessionQuery{}).Items)[0] == first.Id
	}, 5*time.Second, 20*time.Millisecond, "renaming a session did not move it up")
}

func (s *SessionListSuite) TestAProjectNarrowsTheList() {
	support, sales := "support", "sales"
	ticketed := textSession(nil)
	ticketed.ProjectId = &support
	wanted := s.serverClient.createSession(ticketed)

	other := textSession(nil)
	other.ProjectId = &sales
	s.serverClient.createSession(other)

	project := Equals("support")
	s.Equal([]string{wanted.Id},
		ids(s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{ProjectID: &project}}).Items))

	var page SessionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/query",
		map[string]any{"filter": map[string]any{"project_id": map[string]any{"$eq": "support"}}}, &page))
	s.Equal([]string{wanted.Id}, ids(page.Items), "$eq is the long way of writing the same filter")
}

func (s *SessionListSuite) TestAWrittenSessionIsListedAsText() {
	written := s.serverClient.createSession(textSession(nil))
	s.Equal(SessionModalityText, written.Modality)

	text, voice := Equals("text"), Equals("voice")
	listed := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{Modality: &text}}).Items
	s.Require().Equal([]string{written.Id}, ids(listed))
	s.Equal(SessionModalityText, listed[0].Modality)
	s.Empty(s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{Modality: &voice}}).Items)
}

func (s *SessionListSuite) TestATextSearchFindsWhatTheSessionWasCalled() {
	refunds, pages := textSession(nil), textSession(nil)
	refundsTitle, pagesTitle := "Refund for a duplicate charge", "Paginating a channel list"
	refunds.Title, pages.Title = &refundsTitle, &pagesTitle
	wanted := s.serverClient.createSession(refunds)
	s.serverClient.createSession(pages)

	s.Require().Eventually(func() bool {
		found := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{Text: &TextMatch{Q: "refund"}}})
		return slices.Equal([]string{wanted.Id}, ids(found.Items))
	}, 5*time.Second, 20*time.Millisecond, "the search never found the session by its title")
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

func (s *SessionListSuite) TestAStateNarrowsTheListToTheLiveOrTheEnded() {
	running := s.serverClient.createSession(textSession(nil))
	closing := s.serverClient.createSession(textSession(nil))
	s.serverClient.stopSession(closing.Id)

	live, ended := Equals("live"), Equals("ended")
	s.Require().Eventually(func() bool {
		found := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{State: &ended}}).Items
		return slices.Equal([]string{closing.Id}, ids(found)) && found[0].State == Ended
	}, 5*time.Second, 20*time.Millisecond, "the stopped session was never listed as ended")
	listed := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{State: &live}}).Items
	s.Require().Equal([]string{running.Id}, ids(listed))
	s.Equal(Live, listed[0].State)
}

func (s *SessionListSuite) TestAUserAskingForLiveSessionsIsListedOnlyTheirOwn() {
	mine := s.client.createSession(textSession(nil))
	s.data.createUser().createSession(textSession(nil))
	s.serverClient.createSession(textSession(nil))

	live := Equals("live")
	s.Equal([]string{mine.Id}, ids(s.client.querySessions(SessionQuery{Filter: &SessionFilter{State: &live}}).Items))
}

func (s *SessionListSuite) TestLiveSessionsPageToTheEndWithoutRepeatsOrGaps() {
	var created []string
	for range 3 {
		created = append(created, s.serverClient.createSession(textSession(nil)).Id)
	}
	slices.Reverse(created)
	closing := s.serverClient.createSession(textSession(nil))
	s.serverClient.stopSession(closing.Id)

	live := Equals("live")
	var listed []string
	query := SessionQuery{Filter: &SessionFilter{State: &live}, Limit: 1}
	for {
		page := s.serverClient.querySessions(query)
		listed = append(listed, ids(page.Items)...)
		if !page.HasMore {
			break
		}
		s.Require().NotNil(page.NextCursor)
		query.Cursor = *page.NextCursor
	}

	s.Equal(created, listed)
}

func (s *SessionListSuite) TestAnAgentIdNarrowsTheList() {
	agentID := s.utils.uuid()
	named := textSession(nil)
	named.AgentId = &agentID
	wanted := s.serverClient.createSession(named)
	s.serverClient.createSession(textSession(nil))

	equals := Equals(agentID)
	listed := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{AgentID: &equals}}).Items
	s.Require().Equal([]string{wanted.Id}, ids(listed))
	s.Equal(agentID, listed[0].AgentId)
}

func (s *SessionListSuite) TestAConfigNarrowsTheListToTheSessionsItRan() {
	config := s.data.createAgentConfig()
	configured := textSession(nil)
	configured.ConfigId = &config.Id
	wanted := s.serverClient.createSession(configured)
	s.serverClient.createSession(textSession(nil))

	equals := Equals(config.Id)
	listed := s.serverClient.querySessions(SessionQuery{Filter: &SessionFilter{ConfigID: &equals}}).Items
	s.Require().Equal([]string{wanted.Id}, ids(listed))
	s.Require().NotNil(listed[0].ConfigId)
	s.Equal(config.Id, *listed[0].ConfigId)
}

func (s *SessionListSuite) TestALabelNarrowsTheListToTheSessionsCarryingAllOfIt() {
	tested := textSession(nil)
	tested.Custom = &map[string]any{"origin": "test", "suite": "checkout"}
	wanted := s.serverClient.createSession(tested)

	other := textSession(nil)
	other.Custom = &map[string]any{"origin": "test"}
	s.serverClient.createSession(other)

	s.Equal([]string{wanted.Id}, ids(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{Custom: &map[string]string{"origin": "test", "suite": "checkout"}},
	}).Items), "every pair has to be held, not just one of them")

	s.Empty(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{Custom: &map[string]string{"origin": "production"}},
	}).Items)
}

func (s *SessionListSuite) TestACreatedAtWindowLeavesOutWhatStartedOutsideIt() {
	opened := s.serverClient.createSession(textSession(nil))
	started := opened.CreatedAt

	before, soon := started.Add(-time.Minute), started.Add(time.Minute)
	within := &TimeRange{Gte: &before, Lt: &soon}
	s.Contains(ids(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{CreatedAt: within}}).Items), opened.Id)

	after := &TimeRange{Gte: &soon}
	s.NotContains(ids(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{CreatedAt: after}}).Items), opened.Id)
}

func (s *SessionListSuite) TestTheWindowIsHalfOpenSoTwoThatMeetShareNoSession() {
	opened := s.serverClient.createSession(textSession(nil))
	// The session started on the boundary, so the window that ends there leaves it out
	// and the window that starts there takes it: paging over both counts it once.
	boundary := opened.CreatedAt

	s.NotContains(ids(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{CreatedAt: &TimeRange{Lt: &boundary}}}).Items), opened.Id)
	s.Contains(ids(s.serverClient.querySessions(SessionQuery{
		Filter: &SessionFilter{CreatedAt: &TimeRange{Gte: &boundary}}}).Items), opened.Id)
}

func (s *SessionListSuite) TestAWindowThatEndsBeforeItStartsIsRefused() {
	s.assertRefused(map[string]any{"filter": map[string]any{"created_at": map[string]any{
		"$gte": "2026-10-05T00:00:00Z", "$lt": "2026-10-01T00:00:00Z"}}})
}

func (s *SessionListSuite) TestATimeThatIsNotRFC3339IsRefused() {
	s.assertRefused(map[string]any{
		"filter": map[string]any{"created_at": map[string]any{"$gte": "last tuesday"}}})
}

func (s *SessionListSuite) TestAnOperatorTheWindowDoesNotTakeIsRefused() {
	s.assertRefused(map[string]any{
		"filter": map[string]any{"created_at": map[string]any{"$ne": "2026-10-05T00:00:00Z"}}})
}

func (s *SessionListSuite) TestAFieldNobodyMayFilterOnIsRefused() {
	s.assertRefused(map[string]any{"filter": map[string]any{"call_id": "abc"}})
}

func (s *SessionListSuite) TestAStateThereIsNotIsRefused() {
	s.assertRefused(map[string]any{"filter": map[string]any{"state": "closed"}})
}

func (s *SessionListSuite) TestAnOperatorTheFieldDoesNotTakeIsRefused() {
	s.assertRefused(map[string]any{
		"filter": map[string]any{"project_id": map[string]any{"$in": []string{"a"}}}})
}

func (s *SessionListSuite) TestASearchNarrowedToAProjectIsRefused() {
	s.assertRefused(map[string]any{"filter": map[string]any{
		"text": map[string]any{"$q": "refund"}, "project_id": "support"}})
}

func (s *SessionListSuite) TestRelevanceWithoutASearchIsRefused() {
	s.assertRefused(map[string]any{"sort": []any{map[string]any{"field": "relevance"}}})
}

func (s *SessionListSuite) TestASearchInAnyOrderButRelevanceIsRefused() {
	s.assertRefused(map[string]any{
		"filter": map[string]any{"text": map[string]any{"$q": "refund"}},
		"sort":   []any{map[string]any{"field": "updated_at"}}})
}

func (s *SessionListSuite) TestOldestFirstIsRefused() {
	s.assertRefused(map[string]any{
		"sort": []any{map[string]any{"field": "updated_at", "direction": 1}}})
}

func (s *SessionListSuite) TestACursorTheRouterNeverIssuedIsRefused() {
	s.assertRefused(map[string]any{"cursor": "nonsense"})
}

func (s *SessionListSuite) TestAModalityThereIsNotIsRefused() {
	s.assertRefused(map[string]any{"filter": map[string]any{"modality": "smoke signals"}})
}

// assertRefused checks a query the endpoint takes none of.
func (s *SessionListSuite) assertRefused(query map[string]any) {
	s.Equal(http.StatusBadRequest,
		s.serverClient.do(http.MethodPost, "/v1/agents/sessions/query", query, nil))
}

func (s *SessionListSuite) TestACursorFromOneOrderIsRefusedByAnother() {
	for range 2 {
		s.serverClient.createSession(textSession(nil))
	}
	page := s.serverClient.querySessions(SessionQuery{Limit: 1})
	s.Require().NotNil(page.NextCursor)

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/query",
		SessionQuery{Filter: &SessionFilter{Text: &TextMatch{Q: "refund"}}, Cursor: *page.NextCursor}, nil))
}

func (s *SessionListSuite) TestNobodyIsListedAnythingWithoutCredentials() {
	s.serverClient.createSession(textSession(nil))

	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodPost, "/v1/agents/sessions/query", SessionQuery{}, nil))
}
