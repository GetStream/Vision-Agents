//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SearchSuite covers the one routed modality with no socket: a question and what came back.
type SearchSuite struct {
	RouterSuite
	logged *lockedLog
}

func TestSearchSuite(t *testing.T) {
	runSuite(t, new(SearchSuite))
}

// SetupSuite keeps the router's log, for what a search the caller hung up on was logged as.
func (s *SearchSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
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

func (s *SearchSuite) TestATermTheNamedModelDoesNotTakeIsRefused() {
	target := "down/" + failingSearchModels["down"]
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: &target, Category: pointerTo("news")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "routing: "+target+" does not take category")
}

func (s *SearchSuite) TestATermNoListedProviderTakesIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Category: pointerTo("news"), Providers: &[]string{
			"down/" + failingSearchModels["down"], "broke/" + failingSearchModels["broke"]}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "routing: no provider can express category")
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

func (s *SearchSuite) TestAVendorThatTimesOutWhileTheCallerWaitsIsAServerError() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{
		Query: "how much does a call cost", Options: &SearchOptions{Target: pointerTo("slow/" + failingSearchModels["slow"])}})

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

func (s *SearchSuite) TestACallerThatHangsUpMidSearchIsNoServerError() {
	id := "search-hang-up-" + s.utils.uuid()
	encoded, err := encode(SearchRequest{Query: "how much does a call cost",
		Options: &SearchOptions{Target: pointerTo("hang/" + failingSearchModels["hang"])}})
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.serverClient.base()+"/v1/search", bytes.NewReader(encoded))
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(RequestIDHeader, id)

	_, err = s.server.Client().Do(request)
	s.Require().ErrorIs(err, context.DeadlineExceeded, "the caller gave up")

	served := regexp.MustCompile(`msg="served a request".* status=(\d+) .*request_id=` + regexp.QuoteMeta(id) + `\b`)
	var status int
	s.Require().Eventually(func() bool {
		found := served.FindStringSubmatch(s.logged.String())
		if found == nil {
			return false
		}
		status, err = strconv.Atoi(found[1])
		return err == nil
	}, 5*time.Second, 20*time.Millisecond, "the search the caller left is logged")
	// router_http_requests_total is classed by the same status (withMetrics), and is not read here:
	// its series has no label of this test's own, and AdmissionSuite's searches answered 503 move it.
	s.Less(status, http.StatusInternalServerError, "a caller hanging up is no outage of the router's")
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
