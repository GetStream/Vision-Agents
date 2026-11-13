package api

import (
	"bufio"
	"context"
	"crypto/rand"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
)

// HTTPMetricsSuite serves routes of its own behind withTrace and withMetrics, as Handler
// wraps the API. Every test registers a route under a fresh prefix, so the series it reads
// are its own: the counters are the process's, shared with every other test.
type HTTPMetricsSuite struct {
	suite.Suite
	prefix string
	mux    *chi.Mux
}

func TestHTTPMetricsSuite(t *testing.T) {
	suite.Run(t, new(HTTPMetricsSuite))
}

func (s *HTTPMetricsSuite) SetupTest() {
	s.prefix = "/metrics-test/" + rand.Text()
	s.mux = chi.NewRouter()
}

// serve answers a request through the chain, the way Handler orders it.
func (s *HTTPMetricsSuite) serve(method, path string) *httptest.ResponseRecorder {
	response := httptest.NewRecorder()
	withTrace(s.mux, withMetrics(s.mux, withTiming(s.mux))).ServeHTTP(response, httptest.NewRequest(method, path, nil))
	return response
}

// series returns the series of the metric named whose route label is route.
func (s *HTTPMetricsSuite) series(name, route string) []*dto.Metric {
	return seriesOf(s.T(), name, func(labels map[string]string) bool { return labels["route"] == route })
}

// count is the value of router_http_requests_total for route and status, or zero.
func (s *HTTPMetricsSuite) count(route, status string) float64 {
	for _, metric := range s.series("router_http_requests_total", route) {
		if labelsOf(metric)["status"] == status {
			return metric.GetCounter().GetValue()
		}
	}
	return 0
}

// timed is router_http_request_seconds for route and status, or nil.
func (s *HTTPMetricsSuite) timed(route, status string) *dto.Histogram {
	for _, metric := range s.series("router_http_request_seconds", route) {
		if labelsOf(metric)["status"] == status {
			return metric.GetHistogram()
		}
	}
	return nil
}

// Defect: a handler's 500 counted as a success, or a 200 counted as a server error.
func (s *HTTPMetricsSuite) TestA5xxIsCountedAsAServerErrorAndA2xxIsNot() {
	route := s.prefix + "/{id}"
	s.mux.Get(route, func(w http.ResponseWriter, r *http.Request) {
		if chi.URLParam(r, "id") == "fails" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, "ok")
	})

	s.serve(http.MethodGet, s.prefix+"/fails")
	s.Equal(1.0, s.count(route, "5xx"))
	s.Zero(s.count(route, "2xx"))

	s.serve(http.MethodGet, s.prefix+"/works")
	s.serve(http.MethodGet, s.prefix+"/works")
	s.Equal(1.0, s.count(route, "5xx"), "a 2xx moved the 5xx series")
	s.Equal(2.0, s.count(route, "2xx"))
	s.EqualValues(1, s.timed(route, "5xx").GetSampleCount())
	s.EqualValues(2, s.timed(route, "2xx").GetSampleCount())
}

// statusSum is the sum of the seconds router_api_latency_seconds holds for a status class.
func statusSum(t *testing.T, class string) float64 {
	t.Helper()
	total := 0.0
	for _, metric := range seriesOf(t, "router_api_latency_seconds", func(labels map[string]string) bool {
		return labels["status"] == class
	}) {
		total += metric.GetHistogram().GetSampleSum()
	}
	return total
}

// statusSamples is the sample count of router_api_latency_seconds for a status class.
func statusSamples(t *testing.T, class string) uint64 {
	t.Helper()
	var total uint64
	for _, metric := range seriesOf(t, "router_api_latency_seconds", func(labels map[string]string) bool {
		return labels["status"] == class
	}) {
		total += metric.GetHistogram().GetSampleCount()
	}
	return total
}

// Defect: the status-class histogram missing, observing under the wrong class, or keeping the
// route or the method as a label, so the series count grows with the routes again. The series
// are the process's, so each class is read as a delta of at least the requests sent.
func (s *HTTPMetricsSuite) TestEveryStatusClassIsTimedUnderItsClassAlone() {
	route := s.prefix + "/{code}"
	s.mux.Handle(route, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		code, err := strconv.Atoi(chi.URLParam(r, "code"))
		s.Require().NoError(err)
		w.WriteHeader(code)
	}))
	for code, class := range map[string]string{"200": "2xx", "204": "2xx", "302": "3xx", "404": "4xx", "503": "5xx"} {
		before := statusSamples(s.T(), class)
		s.serve(http.MethodGet, s.prefix+"/"+code)
		s.GreaterOrEqual(statusSamples(s.T(), class), before+1, "a %s was not timed under %s", code, class)
	}

	classes := []string{"1xx", "2xx", "3xx", "4xx", "5xx"}
	series := seriesOf(s.T(), "router_api_latency_seconds", func(map[string]string) bool { return true })
	s.NotEmpty(series)
	for _, metric := range series {
		labels := labelsOf(metric)
		s.Len(labels, 1, "a label besides status: %v", labels)
		s.Contains(classes, labels["status"])
	}
	s.LessOrEqual(len(series), len(classes))
}

// Defect: the status-class histogram on other buckets than the route-keyed one, so the two
// disagree on how slow is slow.
func (s *HTTPMetricsSuite) TestTheStatusClassHistogramHasTheRouteKeyedBuckets() {
	route := s.prefix + "/buckets"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	s.serve(http.MethodGet, route)

	bounds := func(histogram *dto.Histogram) []float64 {
		var upper []float64
		for _, bucket := range histogram.GetBucket() {
			upper = append(upper, bucket.GetUpperBound())
		}
		return upper
	}
	byStatus := seriesOf(s.T(), "router_api_latency_seconds", func(labels map[string]string) bool {
		return labels["status"] == "2xx"
	})
	s.Require().Len(byStatus, 1)
	s.Equal(bounds(s.timed(route, "2xx")), bounds(byStatus[0].GetHistogram()))
}

// Defect: the raw path in the route label, a series per session id.
func (s *HTTPMetricsSuite) TestRequestsForDifferentIdsShareOneSeries() {
	route := s.prefix + "/sessions/{id}"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	ids := []string{rand.Text(), rand.Text(), rand.Text()}

	for _, id := range ids {
		s.serve(http.MethodGet, s.prefix+"/sessions/"+id)
	}

	s.Len(s.series("router_http_requests_total", route), 1)
	s.Equal(3.0, s.count(route, "2xx"))
	s.EqualValues(3, s.timed(route, "2xx").GetSampleCount())
	for _, id := range ids {
		s.Empty(seriesOf(s.T(), "", func(labels map[string]string) bool {
			for _, value := range labels {
				if strings.Contains(value, id) {
					return true
				}
			}
			return false
		}), "a label holds the id")
	}
}

// Defect: a path no route serves labelled with itself, a series per path a scanner tries.
func (s *HTTPMetricsSuite) TestAPathNoRouteServesIsCountedUnmatched() {
	path := s.prefix + "/nothing-here"
	before := sumOf(s.T(), "router_http_requests_total", map[string]string{"route": unmatchedRoute, "status": "4xx"})

	s.Equal(http.StatusNotFound, s.serve(http.MethodGet, path).Code)

	s.Empty(s.series("router_http_requests_total", path))
	s.GreaterOrEqual(sumOf(s.T(), "router_http_requests_total", map[string]string{"route": unmatchedRoute, "status": "4xx"}), before+1)
}

// Defect: a refusal answered before the mux, such as withQuota's 429, counted as unmatched,
// so a route's refusals vanish among the paths nothing serves.
func (s *HTTPMetricsSuite) TestARefusalBeforeTheMuxIsCountedUnderItsRoute() {
	route := s.prefix + "/sessions/{id}"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	refusing := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, newAPIError(ErrorTypeRateLimited, "slow down"))
	})
	response := httptest.NewRecorder()

	withTrace(s.mux, withMetrics(s.mux, refusing)).ServeHTTP(response,
		httptest.NewRequest(http.MethodGet, s.prefix+"/sessions/"+rand.Text(), nil))

	s.Equal(http.StatusTooManyRequests, response.Code)
	s.Equal(1.0, s.count(route, "4xx"))
}

// Defect: a refused path looked up by its decoded form, which the mux does not route by: an
// id holding an escaped slash would be counted unmatched.
func (s *HTTPMetricsSuite) TestARefusalOfAnEscapedPathIsCountedUnderTheRouteTheMuxServes() {
	route := s.prefix + "/sessions/{id}"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	refusing := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, newAPIError(ErrorTypeRateLimited, "slow down"))
	})
	path := s.prefix + "/sessions/" + rand.Text() + "%2F" + rand.Text()
	s.Equal(http.StatusOK, s.serve(http.MethodGet, path).Code, "the mux serves the escaped path")

	withTrace(s.mux, withMetrics(s.mux, refusing)).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, path, nil))

	s.Equal(1.0, s.count(route, "4xx"))
}

// Defect: the method a caller sent used as the label as is, a series per made-up method.
func (s *HTTPMetricsSuite) TestAMethodOutsideTheStandardOnesIsCountedOther() {
	route := s.prefix + "/brew"
	method := "BREW" + rand.Text()
	s.mux.HandleFunc(route, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })

	s.serve(method, route)
	s.serve(http.MethodPatch, route)

	methods := map[string]bool{}
	for _, metric := range seriesOf(s.T(), "router_http_requests_total", func(labels map[string]string) bool {
		return labels["method"] == method || labels["route"] == route || labels["route"] == unmatchedRoute
	}) {
		methods[labelsOf(metric)["method"]] = true
	}
	s.False(methods[method], "the made-up method is a label value")
	s.True(methods["other"])
	s.True(methods[http.MethodPatch], "a standard method is kept as it is")
}

// Defect: the time measured as zero, or from somewhere other than the request's arrival.
func (s *HTTPMetricsSuite) TestTheTimeIsHowLongTheAnswerTookToBegin() {
	route := s.prefix + "/slow"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		// After the answer began: the caller has its status, and what follows is its body.
		time.Sleep(300 * time.Millisecond)
		_, _ = io.WriteString(w, "done")
	})

	classBefore := statusSum(s.T(), "2xx")
	s.serve(http.MethodGet, route)

	spent := s.timed(route, "2xx")
	s.Require().NotNil(spent)
	s.GreaterOrEqual(spent.GetSampleSum(), 0.05)
	// The class histogram times the same beginning, not the end and not zero.
	classSpent := statusSum(s.T(), "2xx") - classBefore
	s.GreaterOrEqual(classSpent, 0.05)
	s.Less(classSpent, 0.3, "the body's time is counted by status class")
	s.Less(spent.GetSampleSum(), 0.3, "the body's time is counted")
	// The buckets tell an answer in 50 ms from one in 300 ms, the range a slow API moves in.
	s.True(slices.ContainsFunc(spent.GetBucket(), func(bucket *dto.Bucket) bool {
		return bucket.GetUpperBound() >= 0.05 && bucket.GetUpperBound() < 0.3
	}), "no bucket between 50 and 300 ms")
}

// Defect: an answer that opens with its body, not a status, timed to its end.
func (s *HTTPMetricsSuite) TestAnAnswerBegunByItsBodyIsTimedToItsFirstPart() {
	route := s.prefix + "/slow-body"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(50 * time.Millisecond)
		_, _ = io.WriteString(w, "first part, ")
		time.Sleep(300 * time.Millisecond)
		_, _ = io.WriteString(w, "the rest")
	})

	s.serve(http.MethodGet, route)

	spent := s.timed(route, "2xx")
	s.Require().NotNil(spent)
	s.GreaterOrEqual(spent.GetSampleSum(), 0.05)
	s.Less(spent.GetSampleSum(), 0.3, "the rest of the body's time is counted")
}

// Defect: a handler that writes nothing is timed at zero or never counted.
func (s *HTTPMetricsSuite) TestAnAnswerNothingWroteIsTimedToTheHandlersReturn() {
	route := s.prefix + "/silent"
	s.mux.Get(route, func(http.ResponseWriter, *http.Request) { time.Sleep(50 * time.Millisecond) })

	s.serve(http.MethodGet, route)

	s.Equal(1.0, s.count(route, "2xx"))
	s.GreaterOrEqual(s.timed(route, "2xx").GetSampleSum(), 0.05)
}

// Defect: an answer that streams timed as a response, or its flushes no longer reaching the
// connection through the writers between the handler and this one.
func (s *HTTPMetricsSuite) TestAStreamedAnswerIsCountedAndNotTimed() {
	route := s.prefix + "/stream"
	sent := make(chan struct{})
	s.mux.Get(route, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "first\n")
		s.NoError(http.NewResponseController(w).Flush())
		// The caller gone, a flush that never reached it ends the test rather than hangs it.
		select {
		case <-sent:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "second\n")
	})
	server := httptest.NewServer(withTrace(s.mux, withMetrics(s.mux, withTiming((&Server{logger: slog.New(slog.DiscardHandler)}).withRequestLog(s.mux, s.mux)))))
	s.T().Cleanup(server.Close)
	classBefore := statusSamples(s.T(), "2xx")

	// A flush that never reaches the caller would leave the request waiting for its headers.
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Get(server.URL + route)
	s.Require().NoError(err)
	defer response.Body.Close()
	lines := bufio.NewReader(response.Body)
	first, err := lines.ReadString('\n')
	s.Require().NoError(err)
	s.Equal("first\n", first, "the flush reached the caller before the handler finished")
	close(sent)
	_, err = io.ReadAll(lines)
	s.Require().NoError(err)

	s.Require().Eventually(func() bool { return s.count(route, "2xx") == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Empty(s.series("router_http_request_seconds", route))
	// The series is the process's: this holds while no other test answers a 2xx at the same time.
	s.Equal(classBefore, statusSamples(s.T(), "2xx"), "a streamed answer was timed by status class")
}

// Defect: a socket timed as a response, so the latency holds the length of sessions.
func (s *HTTPMetricsSuite) TestASocketIsCountedAsAnUpgradeAndNotTimed() {
	route := s.prefix + "/socket"
	upgrader := websocket.Upgrader{}
	s.mux.Get(route, func(w http.ResponseWriter, r *http.Request) {
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		_ = connection.WriteMessage(websocket.TextMessage, []byte("hello"))
		_ = connection.Close()
	})
	server := httptest.NewServer(withTrace(s.mux, withMetrics(s.mux, withTiming((&Server{logger: slog.New(slog.DiscardHandler)}).withRequestLog(s.mux, s.mux)))))
	s.T().Cleanup(server.Close)

	connection, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+route, nil)
	s.Require().NoError(err)
	_, message, err := connection.ReadMessage()
	s.Require().NoError(err)
	s.Equal("hello", string(message))
	_ = connection.Close()

	s.Require().Eventually(func() bool { return s.count(route, "1xx") == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Empty(s.series("router_http_request_seconds", route))
	s.Zero(statusSamples(s.T(), "1xx"), "a socket was timed by status class")
}

// Defect: a panic is not counted (the deferred count is skipped), or counted as the empty
// 200 net/http sends once Sentry has recovered it.
func (s *HTTPMetricsSuite) TestAPanicNothingAnsweredIsCountedAsAServerError() {
	route := s.prefix + "/panics"
	s.mux.Get(route, func(http.ResponseWriter, *http.Request) { panic("synthetic handler failure") })
	recovered := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = recover() }()
		withTrace(s.mux, withMetrics(s.mux, s.mux)).ServeHTTP(w, r)
	})

	recovered.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, route, nil))

	s.Equal(1.0, s.count(route, "5xx"))
	s.Zero(s.count(route, "2xx"))
}

// Defect: a panic withRequestLog answered counted as anything but the 500 it sent.
func (s *HTTPMetricsSuite) TestAPanicTheRequestLogAnsweredIsCountedAsItsAnswer() {
	route := s.prefix + "/panics-logged"
	s.mux.Get(route, func(http.ResponseWriter, *http.Request) { panic("synthetic handler failure") })
	recovered := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = recover() }()
		withTrace(s.mux, withMetrics(s.mux, (&Server{logger: slog.New(slog.DiscardHandler)}).withRequestLog(s.mux, s.mux))).ServeHTTP(w, r)
	})
	response := httptest.NewRecorder()

	recovered.ServeHTTP(response, httptest.NewRequest(http.MethodGet, route, nil))

	s.Equal(http.StatusInternalServerError, response.Code)
	s.Equal(1.0, s.count(route, "5xx"))
}

// Defect: a request that panicked after its answer began counted as the server error it was
// not told: the caller already had its status.
func (s *HTTPMetricsSuite) TestAPanicAfterTheAnswerBeganIsCountedAsTheAnswer() {
	route := s.prefix + "/panics-late"
	s.mux.Get(route, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		// net/http's own way of dropping a connection mid-answer, as a proxy copying a body does.
		panic(http.ErrAbortHandler)
	})
	recovered := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = recover() }()
		withTrace(s.mux, withMetrics(s.mux, s.mux)).ServeHTTP(w, r)
	})

	recovered.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, route, nil))

	s.Equal(1.0, s.count(route, "2xx"))
	s.Zero(s.count(route, "5xx"))
}

// Defect: a socket that ended in a panic once upgraded counted as a server error: it was
// answered, with the upgrade.
func (s *HTTPMetricsSuite) TestASocketThatPanicsOnceUpgradedIsCountedAsAnUpgrade() {
	route := s.prefix + "/socket-panics"
	upgrader := websocket.Upgrader{}
	s.mux.Get(route, func(w http.ResponseWriter, r *http.Request) {
		if _, err := upgrader.Upgrade(w, r, nil); err != nil {
			return
		}
		panic(http.ErrAbortHandler)
	})
	server := httptest.NewServer(withTrace(s.mux, withMetrics(s.mux, s.mux)))
	s.T().Cleanup(server.Close)

	connection, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+route, nil)
	s.Require().NoError(err)
	_ = connection.Close()

	s.Require().Eventually(func() bool { return s.count(route, "1xx") == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Zero(s.count(route, "5xx"))
}

// Defect: a request the node running its session was handed counted there as well as on the
// node that forwarded it, doubling every forwarded request.
func (s *HTTPMetricsSuite) TestARequestAPeerHandedOverIsCountedOnlyWhereItArrived() {
	// The two nodes match the same path with templates of their own, so each one's count
	// is told apart.
	arrived, handed := s.prefix+"/sessions/{id}", s.prefix+"/{resource}/{id}"
	s.mux.Get(handed, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	credential := []byte(rand.Text())
	owner := httptest.NewServer(node.Serve(withTrace(s.mux, withMetrics(s.mux, s.mux)), [][]byte{credential}))
	s.T().Cleanup(owner.Close)
	forwarder := node.NewForwarder(slog.New(slog.DiscardHandler), credential)
	entry := chi.NewRouter()
	entry.Get(arrived, func(w http.ResponseWriter, r *http.Request) {
		s.NoError(forwarder.Forward(r.Context(), owner.Listener.Addr().String(), w, r))
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	path := s.prefix + "/sessions/" + rand.Text()

	response := httptest.NewRecorder()
	withTrace(entry, withMetrics(entry, entry)).ServeHTTP(response, httptest.NewRequestWithContext(ctx, http.MethodGet, path, nil))

	s.Equal(http.StatusOK, response.Code)
	s.Equal(1.0, s.count(arrived, "2xx"))
	s.Empty(s.series("router_http_requests_total", handed), "the node handed the request counted it")

	direct, err := http.Get(owner.URL + path)
	s.Require().NoError(err)
	_ = direct.Body.Close()
	s.Equal(1.0, s.count(handed, "2xx"), "a request that reached the owner itself is counted there")
}

// seriesOf returns the series of the metric named (any metric when name is empty) in the
// default registry whose labels match.
func seriesOf(t *testing.T, name string, match func(labels map[string]string) bool) []*dto.Metric {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather the default registry: %v", err)
	}
	var found []*dto.Metric
	for _, family := range families {
		if name != "" && family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			if match(labelsOf(metric)) {
				found = append(found, metric)
			}
		}
	}
	return found
}

// sumOf adds up the counter of the metric named over its series carrying every label in want.
func sumOf(t *testing.T, name string, want map[string]string) float64 {
	t.Helper()
	total := 0.0
	for _, metric := range seriesOf(t, name, func(labels map[string]string) bool {
		for key, value := range want {
			if labels[key] != value {
				return false
			}
		}
		return true
	}) {
		total += metric.GetCounter().GetValue()
	}
	return total
}

func labelsOf(metric *dto.Metric) map[string]string {
	labels := map[string]string{}
	for _, pair := range metric.GetLabel() {
		labels[pair.GetName()] = pair.GetValue()
	}
	return labels
}
