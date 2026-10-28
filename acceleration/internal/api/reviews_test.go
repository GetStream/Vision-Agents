//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type ReviewsSuite struct {
	RouterSuite

	// base is when the calls in a test happened, in an app of the test's own, so what is
	// read is only what the test reviewed.
	base time.Time
}

func TestReviewsSuite(t *testing.T) {
	runSuite(t, new(ReviewsSuite))
}

func (s *ReviewsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

// reviewed starts a call and writes the review a model made of it.
func (s *ReviewsSuite) reviewed(score int, at time.Duration) string {
	ctx := context.Background()
	call := store.Call{
		ID:         s.utils.uuid(),
		CustomerID: s.customerID(),
		CallID:     s.utils.callID(),
		AgentID:    "agent-" + s.utils.uuid(),
		StartedAt:  s.base.Add(at),
	}
	s.Require().NoError(s.store.StartCall(ctx, &call))
	s.Require().NoError(s.store.ReviewCall(ctx, s.customerID(), call.ID,
		"Asked for a refund.", &score, "The agent did not know the refund policy."))
	return call.ID
}

func (s *ReviewsSuite) analysis(target string) map[string]any {
	body := map[string]any{
		"from": s.base.Format(time.RFC3339),
		"to":   s.base.Add(24 * time.Hour).Format(time.RFC3339),
	}
	if target != "" {
		body["target"] = target
	}
	return body
}

func (s *ReviewsSuite) TestTheCausesExplainOnlyTheBadlyReviewedCallsThatWereRead() {
	bad := s.reviewed(1, time.Minute)
	passedOn := s.reviewed(3, 2*time.Minute)
	s.reviewed(5, 3*time.Minute)

	var analysis ReviewAnalysis
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/reviews/analysis", s.analysis("analysing/analysing-model"), &analysis))

	s.Equal(2, analysis.CallsRead, "the good call is not one to learn from")
	s.Equal("Callers asking about refunds were not helped.", analysis.Summary)
	s.Require().Len(analysis.Causes, 1, "a cause explaining no call that was read is dropped")
	s.Equal("No refund policy", analysis.Causes[0].Title)
	s.Equal("Add the policy.", analysis.Causes[0].Fix)
	s.ElementsMatch([]string{bad, passedOn}, analysis.Causes[0].CallIds)
}

func (s *ReviewsSuite) TestNothingIsAskedWhenNoCallWentBadly() {
	s.reviewed(4, time.Minute)

	// No model by that name exists: asking one would fail the request.
	var analysis ReviewAnalysis
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/reviews/analysis", s.analysis("nobody/none"), &analysis))
	s.Zero(analysis.CallsRead)
	s.Empty(analysis.Causes)
}

func (s *ReviewsSuite) TestOnlyTheAppsBackendMayAnalyseItsReviews() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/calls/reviews/analysis", s.analysis("analysing/analysing-model"), nil)
	})
}

func (s *ReviewsSuite) TestAnotherAppsBadCallsAreNotRead() {
	s.reviewed(1, time.Minute)

	var analysis ReviewAnalysis
	s.Require().Equal(http.StatusOK, s.data.backendOfAnotherApp().do(http.MethodPost,
		"/v1/agents/calls/reviews/analysis", s.analysis("nobody/none"), &analysis))
	s.Zero(analysis.CallsRead)
}
