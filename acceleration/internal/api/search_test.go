//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

func (s *SearchSuite) TestASearchIsBilledPerRequest() {
	probe := s.utils.uuid()
	s.search(SearchRequest{Query: "how much does a call cost", Tags: &map[string]string{"probe": probe}})

	var recorded store.Request
	s.Require().Eventually(func() bool {
		var rows []store.Request
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("tags->>'probe' = ?", probe).Scan(context.Background())
		if err != nil || len(rows) != 1 {
			return false
		}
		recorded = rows[0]
		return true
	}, 5*time.Second, 50*time.Millisecond)
	s.EqualValues(5_000, recorded.CostMicros, "$5 a thousand searches is half a cent for one")
	s.Equal(pricing.Quantities{pricing.Requests: 1}, recorded.Meters)
	s.True(recorded.Billable)
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

func (s *SearchSuite) TestAVendorThatFailsTheQuestionIsAServerError() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: pointerTo("down/" + failingSearchModels["down"])}})

	s.assertServerError(status, payload)
}

func (s *SearchSuite) TestAVendorNoSessionCanBeOpenedWithIsAServerError() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: pointerTo("unbuildable/" + failingSearchModels["unbuildable"])}})

	s.assertServerError(status, payload)
}

func (s *SearchSuite) TestARefusalAboutMoneyFromTheVendorCallIsStillAPaymentRequired() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: pointerTo("broke/" + failingSearchModels["broke"])}})

	s.Equal(http.StatusPaymentRequired, status, string(payload))
	var failure ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Equal(codeSpendLimitReached, failure.Error.Code)
}

func (s *SearchSuite) TestAModelNamedOutsideTheOptionsIsRefused() {
	for _, model := range []any{"nobody/nothing", "", 5, map[string]any{}, []string{"nobody/nothing"}} {
		request := map[string]any{"query": "how much does a call cost"}
		request["model"] = model
		s.assertTopLevelModelRefused("/v1/search", request)
	}
}

func (s *SearchSuite) TestANullModelIsServedAsNoModel() {
	request := map[string]any{"query": "how much does a call cost"}
	request["model"] = nil
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", request)
	s.Equal(http.StatusOK, status, string(payload))
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

// assertServerError says a search failed at its provider and was answered as a failure of the router's
// own: a 500 that says nothing of what the provider answered.
func (s *SearchSuite) assertServerError(status int, payload []byte) {
	s.Require().Equal(http.StatusInternalServerError, status, string(payload))
	var failure ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Equal(ErrorTypeInternal, failure.Error.Type)
	s.Equal("internal_error", failure.Error.Code)
	s.Equal(somethingWentWrong, failure.Error.Message)
}
