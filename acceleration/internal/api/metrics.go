package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// How an inbound connector event ended (receiveEvent), the outcome of connectorEvents.
const (
	// eventHandled: verified, and a signal in it was acted on or an agent answers a message.
	eventHandled = "handled"
	// eventUnhandled: verified and answered 200, with nothing in it the router acted on: no
	// message, every message skipped, or none an agent answers.
	eventUnhandled = "unhandled"
	// eventRefused: answered 401 or 404, the connector known: unverified, or nothing here
	// verifies its events.
	eventRefused = "refused"
	// eventFailed: answered 500, so the provider delivers it again.
	eventFailed = "failed"
	// eventChallenge: a verified handshake, answered with its challenge.
	eventChallenge = "challenge"
)

// connectorEvents counts only events whose connector the router has a definition of: the id a
// request's path names before that is the sender's choice, without a bound.
var connectorEvents = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "connector_events_total",
	Help:      "Inbound connector events, such as a Slack bot's message, by connector and how they ended: handled, unhandled, refused, failed or challenge (api receiveEvent)",
}, []string{"connector", "outcome"})

// countEvent counts one inbound event of connector id.
func countEvent(id, outcome string) {
	connectorEvents.WithLabelValues(store.ConnectorLabel(id), outcome).Inc()
}

// unmatchedRoute is the route label of a request no route served: a 404, a 405 for an unknown
// method, or a CORS preflight, which withCORS answers before the mux.
const unmatchedRoute = "unmatched"

var (
	httpRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "http_requests_total",
		Help: "Requests the API answered, by method, route template (unmatched when no route served it) and status class: " +
			"1xx is a socket upgraded, 5xx also a panic nothing answered for. A request a peer node was handed is counted " +
			"on the node it reached first (api withMetrics)",
	}, []string{"method", "route", "status"})
	httpRequestSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "http_request_seconds",
		Help: "How long the API took to begin its answer, as its Server-Timing header says, by method, route template and " +
			"status class. A socket and a streamed answer are left out: what they last is a connection's time, not a response's",
		// Prometheus' defaults, 5 ms to 10 s: an answer slower than 10 s is slow whatever its exact time.
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "route", "status"})
)

// withMetrics counts each request the API answers in httpRequests and times the answers that
// are a response in httpRequestSeconds.
//
// It sits inside withTrace, which seeds the route context the mux fills in: the route label is
// the template the request matched in routes, since its path carries session and config ids, a
// series each. It sits outside withCustomer, withQuota and onOwningNode, so a refusal and a
// request carried to the node running its session are counted too, the latter once: the node
// it was handed to leaves it to this one.
func withMetrics(routes *chi.Mux, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The entry node counts a forwarded request. A node-to-node /internal/* route is only ever
		// sent forwarded, so no node counts it: it is not the customer API.
		if node.Forwarded(r.Context()) {
			next.ServeHTTP(w, r)
			return
		}
		metered := &meteredResponse{loggedResponse: loggedResponse{ResponseWriter: w}, started: time.Now()}
		finished := false
		// Deferred so a panic is counted as well: withRequestLog answers it and panics again.
		defer func() { metered.observe(routes, r, finished) }()
		next.ServeHTTP(metered, r)
		finished = true
	})
}

// meteredResponse is a loggedResponse that also notes when the answer began and whether it was
// streamed.
type meteredResponse struct {
	loggedResponse
	started  time.Time
	answered time.Duration
	begun    bool
	streamed bool
}

func (m *meteredResponse) WriteHeader(code int) {
	m.begin()
	m.loggedResponse.WriteHeader(code)
}

func (m *meteredResponse) Write(body []byte) (int, error) {
	m.begin()
	return m.loggedResponse.Write(body)
}

// FlushError marks the answer streamed: a handler flushes to send part of an answer before the
// rest exists. http.ResponseController reaches it through the writers' Unwrap.
func (m *meteredResponse) FlushError() error {
	m.streamed = true
	return http.NewResponseController(m.ResponseWriter).Flush()
}

// begin notes the answer starting, the moment withTiming stamps Server-Timing. An answer
// nothing wrote begins when the handler returns, as net/http then sends it.
func (m *meteredResponse) begin() {
	if m.begun {
		return
	}
	m.begun = true
	m.answered = time.Since(m.started)
}

// observe records the request once it has been answered, or once it panicked.
func (m *meteredResponse) observe(routes *chi.Mux, r *http.Request, finished bool) {
	m.begin()
	status := m.status()
	// What panicked outside withRequestLog is answered by nothing, which net/http sends as an
	// empty 200 once withSentry has recovered; it is the server's failure all the same.
	if !finished && m.code == 0 && !m.hijacked {
		status = http.StatusInternalServerError
	}
	method, route, class := methodLabel(r.Method), routeLabel(routes, r), strconv.Itoa(status/100)+"xx"
	httpRequests.WithLabelValues(method, route, class).Inc()
	if m.hijacked || m.streamed {
		return
	}
	httpRequestSeconds.WithLabelValues(method, route, class).Observe(m.answered.Seconds())
}

// routeLabel is the template of the route in routes that served r. A request answered before it
// reached the mux, such as one withQuota refused or onOwningNode could not carry, is looked up
// there, so its refusal is the route's rather than lumped in with the paths nothing serves.
func routeLabel(routes *chi.Mux, r *http.Request) string {
	if pattern := chi.RouteContext(r.Context()).RoutePattern(); pattern != "" {
		return pattern
	}
	// The path the mux itself routes by (chi's Mux.routeHTTP).
	path := r.URL.RawPath
	if path == "" {
		path = r.URL.Path
	}
	if pattern := routes.Find(chi.NewRouteContext(), r.Method, path); pattern != "" {
		return pattern
	}
	return unmatchedRoute
}

// methodLabel is method if it is one of RFC 9110 section 9.1's or PATCH, else "other": a caller
// sends any method it likes, and a label must not grow with it.
func methodLabel(method string) string {
	switch method {
	case http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut, http.MethodDelete,
		http.MethodConnect, http.MethodOptions, http.MethodTrace, http.MethodPatch:
		return method
	}
	return "other"
}
