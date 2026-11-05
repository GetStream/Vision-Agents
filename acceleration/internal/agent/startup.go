package agent

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The stages a voice session's start-up is split into, in the order it passes them. Each runs
// from the end of the one before, so they add up to the wait from asking for the session to
// the agent being heard in its call, and none counts the time of another.
const (
	// StageCreate runs from session.Manager.Create being called to the agent starting to join
	// the call: what the router reads and opens for the session, its model sessions among them.
	StageCreate = "create"
	// StageJoin is the edge joining the call: for Stream, signalling, subscribing and
	// publishing the agent's track, the join_ms its edge logs.
	StageJoin = "join"
	// StageMedia runs from the end of the join to the agent's outgoing connection to the call
	// being up, the publisher's connected_ms its edge logs less the join.
	StageMedia = "media"
)

// The kinds of session a start-up is counted under.
const (
	StartupWeb   = "web"
	StartupPhone = "phone"
)

// Why a stage ended without the session getting past it.
const (
	// causeError is a session session.Manager.Create gave up on: the stage it was in failed.
	causeError = "error"
	// causeTimeout is a stage that never ended within the time its edge gives it.
	causeTimeout = "timeout"
)

// startupBuckets run 10 ms to 30 s: the router's own part of creating a session is a few
// round trips, a join takes from a few hundred ms to seconds, and 30 s is what Stream's edge
// gives its outgoing connection (streamedge.iceTimeout).
var startupBuckets = []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30}

var (
	// stage, kind and cause are the fixed sets above.
	startupSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_startup_seconds",
		Help:      "How long each stage of a voice session's start-up took, once per session, by stage: create from Manager.Create being called to the agent starting to join the call, join the edge joining it, media from the join to the agent's outgoing connection being up; the stages follow each other, so their sum is the wait until the agent can be heard; by kind, phone for a session on a number, else web. A stage that failed is counted in router_agent_startup_failures_total and not here",
		Buckets:   startupBuckets,
	}, []string{"stage", "kind"})
	startupFailures = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_startup_failures_total",
		Help:      "Voice sessions whose start-up stopped in a stage, by stage, kind and cause: error for a session Manager.Create gave up on, the one counted as create_failed in router_session_ends_total, timeout for an outgoing connection that never came up",
	}, []string{"stage", "kind", "cause"})
)

// The series exist from the start, so a rate over the first failure reads it.
func init() {
	for _, kind := range []string{StartupWeb, StartupPhone} {
		for _, stage := range stages {
			startupSeconds.WithLabelValues(stage, kind)
			startupFailures.WithLabelValues(stage, kind, causeError)
		}
		startupFailures.WithLabelValues(StageMedia, kind, causeTimeout)
	}
}

// Startup times one voice session's way into its call, and counts the stage it stopped in.
// Each stage is observed once, when both its ends are known, whichever end is reported
// first: Stream's edge can see its connection up before the agent sees the join return.
//
// A nil Startup times nothing, which is a text session, a session's agent built again later
// in its life, and an edge joined outside a session.
type Startup struct {
	kind string

	mu sync.Mutex
	// ends is when each stage started, and the last stage ended: the session asked for, the
	// join started, the join ended, the connection up. Zero is not reported yet.
	ends [len(stages) + 1]time.Time
	// observed counts the stages already timed, in their order.
	observed int
	over     bool
}

// stages are the stages in their order, each from ends[i] to ends[i+1].
var stages = [...]string{StageCreate, StageJoin, StageMedia}

// NewStartup times the start-up of a session of kind asked for at began.
func NewStartup(kind string, began time.Time) *Startup {
	startup := &Startup{kind: kind}
	startup.ends[0] = began
	return startup
}

type startupKey struct{}

// WithStartup carries startup to the agent's Join and its edge's.
func WithStartup(ctx context.Context, startup *Startup) context.Context {
	return context.WithValue(ctx, startupKey{}, startup)
}

// StartupOf is the Startup ctx carries, nil for none.
func StartupOf(ctx context.Context) *Startup {
	startup, _ := ctx.Value(startupKey{}).(*Startup)
	return startup
}

// JoinStarted says the agent started joining the call at at.
func (s *Startup) JoinStarted(at time.Time) { s.mark(1, at) }

// JoinEnded says the agent is in the call since at.
func (s *Startup) JoinEnded(at time.Time) { s.mark(2, at) }

// Connected says the agent's outgoing connection to the call came up at at, ending the
// start-up.
func (s *Startup) Connected(at time.Time) { s.mark(3, at) }

// TimedOut counts the stage in progress as one that never ended.
func (s *Startup) TimedOut() { s.stop(causeTimeout) }

// Failed counts the stage in progress as the one the session failed in.
func (s *Startup) Failed() { s.stop(causeError) }

// mark sets ends[end] and times every stage both of whose ends are now known.
func (s *Startup) mark(end int, at time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	// A stage that failed is not timed, though its end comes after: the connection can come
	// up as Create gives up on the session.
	if s.over {
		return
	}
	s.ends[end] = at
	for s.observed < len(stages) && !s.ends[s.observed+1].IsZero() {
		// The connection can come up while the join is still returning: no wait is left.
		took := max(s.ends[s.observed+1].Sub(s.ends[s.observed]), 0)
		startupSeconds.WithLabelValues(stages[s.observed], s.kind).Observe(took.Seconds())
		s.observed++
	}
	s.over = s.observed == len(stages)
}

// stop counts the stage in progress as stopped for cause, and ends the start-up.
func (s *Startup) stop(cause string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.over {
		return
	}
	s.over = true
	startupFailures.WithLabelValues(stages[s.observed], s.kind, cause).Inc()
}
