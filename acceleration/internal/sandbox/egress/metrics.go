package egress

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The values of the decision label: what the proxy did with a request. denied_method is for
// a method a grant's host does not take.
const (
	// decisionToken is a request whose placeholder was put in place of the token.
	decisionToken = "token"
	// decisionGrantedNoToken is a request to a granted host that carried no placeholder.
	decisionGrantedNoToken = "granted_no_token"
	// decisionBlind is a tunnel to an allowed domain.
	decisionBlind = "blind"
	// decisionDeniedHost is a host or a port the sandbox has no grant or allowed domain for.
	decisionDeniedHost = "denied_host"
	// decisionDeniedAuth is a missing, unknown or expired proxy credential (407).
	decisionDeniedAuth = "denied_auth"
	// decisionDeniedMethod is a method refused on a granted host.
	decisionDeniedMethod = "denied_method"
	// decisionDeniedRequest is a request the proxy does not serve: not a CONNECT, a CONNECT
	// without a host and a port, or an upgrade.
	decisionDeniedRequest = "denied_request"
	// decisionError is a request the proxy could not serve: the grants unreadable, a
	// connection it could not take over, or a grant that gives no token.
	decisionError = "error"
)

// otherHost is the host label of a request to no granted host. An allowed domain is the
// customer's to name, so it never adds a value.
const otherHost = "other"

var (
	requests = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "sandbox_egress_requests_total",
		Help: "Requests of sandbox code through the egress proxy, by granted host, method, decision and status class. " +
			"A tunnel opened to a granted host is counted by the requests in it, one to an allowed domain by its CONNECT.",
	}, []string{"host", "method", "decision", "status"})
	// A request is timed to its response header, the proxy's wait for the host; a body's time
	// is the sandbox's reading.
	upstreamSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "sandbox_egress_upstream_seconds",
		Help:      "How long a granted host took to answer a request the proxy sent on, to its response header or its failure",
		// 50 ms to 51.2 s, up to the 60 s header timeout (headerTimeout).
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 11),
	}, []string{"host"})
	tunnelsOpen = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "sandbox_egress_tunnels_open",
		Help:      "Tunnels of sandbox code open now, to granted hosts and to allowed domains",
	})
)

// count adds one request of method to requests. status is a status code.
func count(host, method, decision string, status int) {
	requests.WithLabelValues(host, methodLabel(method), decision, strconv.Itoa(status/100)+"xx").Inc()
}

// refuse answers with status and message, and counts the request.
func refuse(w http.ResponseWriter, host, method, decision string, status int, message string) {
	count(host, method, decision, status)
	http.Error(w, message, status)
}

// counted is w, which keeps the status sent, and the func that counts the request once it is
// answered. Call it deferred, so a response lost mid-body is counted too.
func counted(w http.ResponseWriter, host, method, decision string) (*statusWriter, func()) {
	sent := &statusWriter{ResponseWriter: w}
	return sent, func() { count(host, method, decision, sent.status) }
}

// methodLabel is method if it is one of RFC 9110 section 9.1's, else "other": a sandbox
// sends any method it likes, and a label must not grow with it.
func methodLabel(method string) string {
	switch method {
	case http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut, http.MethodDelete,
		http.MethodConnect, http.MethodOptions, http.MethodTrace, http.MethodPatch:
		return method
	}
	return "other"
}

// statusWriter keeps the last status a handler sent, which is the final one: an informational
// response comes before it.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// timed is a transport that records how long next takes to answer a request to host.
type timed struct {
	next http.RoundTripper
	host string
}

func (t timed) RoundTrip(request *http.Request) (*http.Response, error) {
	start := time.Now()
	response, err := t.next.RoundTrip(request)
	upstreamSeconds.WithLabelValues(t.host).Observe(time.Since(start).Seconds())
	return response, err
}
