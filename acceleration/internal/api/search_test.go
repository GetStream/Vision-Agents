//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"testing"
)

// SearchSuite covers the one routed modality with no socket: a question and what came back.
type SearchSuite struct {
	RouterSuite
}

func TestSearchSuite(t *testing.T) {
	runSuite(t, new(SearchSuite))
}

func (s *SearchSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *SearchSuite) TestAQuestionComesBackAnsweredAndWithWhatItWasAnsweredFrom() {
	found := s.search(SearchRequest{Query: "how much does a call cost"})

	s.Equal("stub", found.Provider)
	s.Equal("stub-model", found.Model)
	s.Equal("A call costs a penny.", *found.Answer)
	s.Require().Len(found.Results, 1)
	s.Equal("https://example.test/pricing", found.Results[0].Url)
	s.Equal("Pricing", *found.Results[0].Title)
	s.Equal("how much does a call cost", *found.Results[0].Text, "the question reached the provider")
	s.InDelta(0.9, *found.Results[0].Score, 1e-6)
}

func (s *SearchSuite) TestAQuestionOfNothingIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{Query: "  "})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nothing to look for")
}

func (s *SearchSuite) TestATargetNothingAnswersToIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: pointerTo("nowhere")}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SearchSuite) TestACostLabelThatIsNotOneIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Tags: &map[string]string{"not a key": "x"}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SearchSuite) TestAnybodyHoldingTheAppsKeyMayAskAQuestion() {
	// Search is reachable from a device, unlike the other modalities: a question carries
	// no conversation and answering one is what an agent's own tools do anyway.
	s.assertPosture(anyAppCaller, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/search", SearchRequest{Query: "what is the time"})
		return status
	})
}

func (s *SearchSuite) search(request SearchRequest) SearchAnswer {
	var found SearchAnswer
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", request)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &found))
	return found
}
