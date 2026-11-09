//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

// HTTPMetricsRouterSuite reads the request metrics of requests the whole router answers.
// Other suites call the same routes at the same time, so a count is read as a rise.
type HTTPMetricsRouterSuite struct {
	RouterSuite
}

func TestHTTPMetricsRouterSuite(t *testing.T) {
	runSuite(t, new(HTTPMetricsRouterSuite))
}

func (s *HTTPMetricsRouterSuite) SetupTest() {
	s.useFixture("standard")
}

// carrying is every series, of any metric, with a label value that contains value.
func (s *HTTPMetricsRouterSuite) carrying(value string) int {
	return len(seriesOf(s.T(), "", func(labels map[string]string) bool {
		for _, label := range labels {
			if strings.Contains(label, value) {
				return true
			}
		}
		return false
	}))
}

// Defect: Handler's chain without withMetrics, or a route labelled with its path, the
// session id in it.
func (s *HTTPMetricsRouterSuite) TestARequestTheRouterAnswersIsCountedUnderItsRouteTemplate() {
	id := s.utils.uuid()
	answered := map[string]string{"method": http.MethodGet, "route": "/v1/agents/sessions/{id}", "status": "4xx"}
	before := sumOf(s.T(), "router_http_requests_total", answered)

	s.Equal(http.StatusNotFound, s.client.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, nil))

	s.GreaterOrEqual(sumOf(s.T(), "router_http_requests_total", answered), before+1)
	s.Zero(s.carrying(id), "a label holds the session id")
}

// Defect: a caller without credentials, refused by the handler, not counted.
func (s *HTTPMetricsRouterSuite) TestARequestWithoutCredentialsIsCountedAsARefusal() {
	answered := map[string]string{"method": http.MethodGet, "route": "/v1/agents/sessions/{id}", "status": "4xx"}
	before := sumOf(s.T(), "router_http_requests_total", answered)

	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodGet, "/v1/agents/sessions/"+s.utils.uuid(), nil, nil))

	s.GreaterOrEqual(sumOf(s.T(), "router_http_requests_total", answered), before+1)
}

// Defect: Handler passes withMetrics a router that serves nothing, so a request that
// withServerSide refuses before the mux is counted under "unmatched", not its route template.
func (s *HTTPMetricsRouterSuite) TestARefusalBeforeTheMuxIsCountedUnderTheRouteItWasFor() {
	refused := map[string]string{"method": http.MethodGet, "route": "/v1/agents/configs/{id}", "status": "4xx"}
	before := sumOf(s.T(), "router_http_requests_total", refused)

	s.Equal(http.StatusForbidden, s.client.do(http.MethodGet, "/v1/agents/configs/"+s.utils.uuid(), nil, nil))

	s.GreaterOrEqual(sumOf(s.T(), "router_http_requests_total", refused), before+1)
}

// Defect: the session socket timed as a response, so the API's latency holds how long
// conversations last. No correct router ever times a 1xx, so the series stays absent
// whatever other suites open.
func (s *HTTPMetricsRouterSuite) TestASessionSocketIsCountedAndNotTimed() {
	upgraded := map[string]string{"method": http.MethodGet, "route": "/v1/agents/sessions/{id}/events", "status": "1xx"}
	before := sumOf(s.T(), "router_http_requests_total", upgraded)
	opened := s.client.createSession(textSession(nil))

	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().NoError(events.Close())

	s.Require().Eventually(func() bool {
		return sumOf(s.T(), "router_http_requests_total", upgraded) >= before+1
	}, settleFor, 10*time.Millisecond)
	s.Empty(seriesOf(s.T(), "router_http_request_seconds", func(labels map[string]string) bool {
		return labels["route"] == upgraded["route"] && labels["status"] == "1xx"
	}))
	s.Empty(seriesOf(s.T(), "router_http_request_status_seconds", func(labels map[string]string) bool {
		return labels["status"] == "1xx"
	}), "a socket was timed by status class")
	s.Zero(s.carrying(opened.Id), "a label holds the session id")
}
