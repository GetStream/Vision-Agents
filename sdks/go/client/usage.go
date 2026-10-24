package client

import (
	"log/slog"
	"net/http"
	"strconv"
	"sync"
)

// What the router answers a backend with about money: what the request cost, and what its
// organization has left. An end user's device is never sent them.
const (
	CostMicrosHeader     = "X-Stream-Cost-Micros"
	CostEstimatedHeader  = "X-Stream-Cost-Estimated"
	BalanceMicrosHeader  = "X-Stream-Balance-Micros"
	BalanceWarningHeader = "X-Stream-Balance-Warning"
)

// The warnings X-Stream-Balance-Warning carries.
const (
	BalanceLow       = "low"
	BalanceExhausted = "exhausted"
)

// Usage is what one answer said about money.
type Usage struct {
	// Operation is the request it answered, as its method and path: "POST /v1/search".
	Operation string
	// CostMicros is what the request cost, in millionths of a dollar; zero when it recorded
	// nothing.
	CostMicros int64
	// Estimated says some of the cost is the router's own count, because the vendor never said
	// what the work used.
	Estimated bool
	// BalanceMicros is what the organization has left, less what its work under way holds,
	// which may be below zero. Nil until the router has read the organization's account, and
	// from a router that bills nobody.
	BalanceMicros *int64
	// BalanceWarning is BalanceLow or BalanceExhausted, empty while there is credit enough.
	BalanceWarning string
}

// BalanceWarning is the organization's credit going down into low or exhausted, as an answer
// said it.
type BalanceWarning struct {
	// Operation is the request whose answer said it first.
	Operation string
	// Warning is BalanceLow or BalanceExhausted.
	Warning string
	// BalanceMicros is what the organization had left, which may be below zero.
	BalanceMicros int64
}

// Option changes how a Client is made.
type Option func(*watcher)

// WithUsage calls usage after every answer that says what it cost or what credit is left.
// Every resource method's answers go through it. It runs on the goroutine that made the
// request, before the request returns, so it must not block; requests made at once call it at
// once, so it must be safe for concurrent use.
func WithUsage(usage func(Usage)) Option {
	return func(w *watcher) { w.usage = usage }
}

// WithBalanceWarning calls warned when answers say the credit went down further than they have
// said since they last said there was credit enough: once on going low, once more on going on to
// exhausted, and nothing for going back up from exhausted to low or for any answer after. An
// answer saying there is credit enough re-arms both. With neither it nor WithUsage, the client
// logs one WARN for each instead. Like WithUsage's, warned runs on the goroutine that made the
// request, must not block, and may be called concurrently.
func WithBalanceWarning(warned func(BalanceWarning)) Option {
	return func(w *watcher) { w.warned = warned }
}

// WithLogger is where the client logs a balance warning nobody asked to be called for. Nil, as
// without it, is slog.Default().
func WithLogger(logger *slog.Logger) Option {
	return func(w *watcher) { w.logger = logger }
}

// watcher reads what every answer says about money, for every request the client makes.
type watcher struct {
	usage  func(Usage)
	warned func(BalanceWarning)
	logger *slog.Logger

	mu sync.Mutex
	// told is the furthest down a warning passed on went since an answer last said there was
	// credit enough, empty since then.
	told string
}

// transport wraps base so the watcher reads every answer. It keeps base's other settings.
func (w *watcher) transport(base *http.Client) *http.Client {
	wrapped := &http.Client{}
	if base != nil {
		*wrapped = *base
	}
	next := wrapped.Transport
	if next == nil {
		next = http.DefaultTransport
	}
	wrapped.Transport = watched{next: next, watcher: w}
	return wrapped
}

type watched struct {
	next    http.RoundTripper
	watcher *watcher
}

func (t watched) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := t.next.RoundTrip(request)
	if err == nil {
		t.watcher.read(request.Method+" "+request.URL.Path, response.Header)
	}
	return response, err
}

// read passes on what one answer said about money, if it said anything.
func (w *watcher) read(operation string, header http.Header) {
	said := Usage{Operation: operation, Estimated: header.Get(CostEstimatedHeader) == "true",
		BalanceWarning: header.Get(BalanceWarningHeader)}
	cost := header.Get(CostMicrosHeader)
	if micros, err := strconv.ParseInt(cost, 10, 64); err == nil {
		said.CostMicros = micros
	}
	if micros, err := strconv.ParseInt(header.Get(BalanceMicrosHeader), 10, 64); err == nil {
		said.BalanceMicros = &micros
	}
	if cost == "" && said.BalanceMicros == nil {
		return
	}
	if w.usage != nil {
		w.usage(said)
	}
	if said.BalanceMicros == nil {
		return
	}
	w.mu.Lock()
	crossed := depth(said.BalanceWarning) > depth(w.told)
	if crossed || said.BalanceWarning == "" {
		w.told = said.BalanceWarning
	}
	w.mu.Unlock()
	if !crossed {
		return
	}
	warning := BalanceWarning{Operation: operation, Warning: said.BalanceWarning, BalanceMicros: *said.BalanceMicros}
	switch {
	case w.warned != nil:
		w.warned(warning)
	case w.usage == nil:
		logger := w.logger
		if logger == nil {
			logger = slog.Default()
		}
		logger.Warn("the organization's credit is running out", "warning", warning.Warning,
			"balance_micros", warning.BalanceMicros, "operation", operation)
	}
}

// depth is how far down a warning is: none, low, then exhausted. One the client does not know
// counts as low.
func depth(warning string) int {
	switch warning {
	case "":
		return 0
	case BalanceExhausted:
		return 2
	}
	return 1
}
