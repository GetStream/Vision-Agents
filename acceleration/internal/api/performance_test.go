//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PerformanceSuite measures what the router spends on a request of its own, with every
// provider answering in process. What is left over is the overhead: authentication, the
// daily quota, the policies and the handler.
//
// It is a test rather than a Benchmark because it needs the suite's Postgres, Redis, app
// and API key, and a Benchmark has no setup to build those. Run it on its own, so the
// other suites are not competing for the same Postgres:
//
//	go test -tags integration -run TestPerformance -v ./internal/api/
type PerformanceSuite struct {
	RouterSuite

	// spans is every span the run recorded, which is how a measurement reports the reads
	// a request made rather than only how long it took.
	spans *tracetest.SpanRecorder
}

func TestPerformance(t *testing.T) { runSuite(t, new(PerformanceSuite)) }

func (s *PerformanceSuite) SetupSuite() {
	// Installed before the harness, so the store and the Redis client it opens record
	// into this recorder rather than into the no-op provider.
	s.spans = tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(s.spans)))
	s.RouterSuite.SetupSuite()
}

func (s *PerformanceSuite) SetupTest() { s.useFixture("standard") }

const (
	// iterations is how many requests each case is measured over, and warmup how many are
	// made first and thrown away. The warm-up matters: the first requests of a run open
	// Postgres connections and fill whatever caches there are, and it is the steady state
	// a deployment lives in that is worth reporting.
	iterations = 200
	warmup     = 40
	// concurrent is how many callers the parallel case runs at once. It is under
	// suiteConnections, so the numbers say what the router costs rather than how long a
	// request waited for a connection.
	concurrent = 4
)

// The routes each case is measured on. A read is attributed to the case whose route began
// the trace it happened in, so what a case does to settle itself is not counted against it.
const (
	askRoute     = "POST /v1/agents/sessions/{id}/responses"
	sessionRoute = "POST /v1/agents/sessions"
)

func (s *PerformanceSuite) TestTheOverheadOfAskingAnAgentSomething() {
	// One conversation per caller. A conversation answers one turn at a time, so callers
	// sharing one would be measuring each other's turns rather than the router.
	live := make([]string, concurrent)
	for caller := range live {
		opened := s.serverClient.createSession(textSession(nil))
		live[caller] = opened.Id
		s.T().Cleanup(func() { s.serverClient.stopSession(opened.Id) })
	}

	// A session nobody opened, which is refused once the request has been authenticated
	// and counted. It is every middleware and none of the work.
	missing := "/v1/agents/sessions/" + s.utils.uuid() + "/responses"

	// A stored agent, which is what a deployment with configs opens its sessions from and
	// the read the configuration cache is there for.
	agent := store.AgentConfig{CustomerID: s.customerID(), Name: "performance", Mode: "text"}
	s.Require().NoError(s.configs.CreateAgentConfig(context.Background(), &agent))
	configured := textSession(nil)
	configured.ConfigId = &agent.ID

	measurements := []measurement{
		s.measure("authenticate only", askRoute, 1, func(int) time.Duration {
			return timed(func() {
				status, _ := s.serverClient.call(http.MethodPost, missing, CreateResponseRequest{
					Text: "What is the capital of France?",
				})
				s.Require().Equal(http.StatusNotFound, status)
			})
		}),
		s.measure("create a session", sessionRoute, 1, func(int) time.Duration {
			var opened Session
			spent := timed(func() {
				s.Require().Equal(http.StatusCreated, s.serverClient.do(
					http.MethodPost, "/v1/agents/sessions", textSession(nil), &opened))
			})
			s.serverClient.stopSession(opened.Id)
			return spent
		}),
		s.measure("create a session from an agent config", sessionRoute, 1, func(int) time.Duration {
			var opened Session
			spent := timed(func() {
				s.Require().Equal(http.StatusCreated, s.serverClient.do(
					http.MethodPost, "/v1/agents/sessions", configured, &opened))
			})
			s.serverClient.stopSession(opened.Id)
			return spent
		}),
		s.measure("ask a question", askRoute, 1, func(caller int) time.Duration {
			return s.ask(live[caller])
		}),
		s.measure("ask a question, four at a time", askRoute, concurrent, func(caller int) time.Duration {
			return s.ask(live[caller])
		}),
	}

	s.T().Log("\n" + report(measurements))
}

// ask puts one question to a live session and reports how long the caller waited to be
// told the turn had begun, which is the whole of what the router does in front of the
// model.
//
// The answer is waited for before returning, because a conversation answers one turn at a
// time and the next question would be refused while this one is still being written. The
// wait is not part of what is reported, and the polling it does begins traces of its own,
// so the reads it makes are not counted against the question either.
func (s *PerformanceSuite) ask(sessionID string) time.Duration {
	command := s.utils.uuid()
	spent := timed(func() {
		status, body := s.serverClient.call(http.MethodPost,
			"/v1/agents/sessions/"+sessionID+"/responses",
			CreateResponseRequest{CommandId: &command, Text: "What is the capital of France?"})
		s.Require().Equal(http.StatusAccepted, status, string(body))
	})
	s.Require().Eventually(func() bool {
		var receipt CommandReceipt
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
			"/v1/agents/sessions/"+sessionID+"/commands/"+url.PathEscape(command), nil, &receipt))
		return receipt.State == "completed"
	}, settleFor, time.Millisecond)
	return spent
}

// measurement is what one case cost per request.
type measurement struct {
	name string
	// mean, p50 and p99 are how long the caller waited.
	mean, p50, p99 time.Duration
	// postgres and redis are how many of each one request made, counted from the spans
	// under the traces the case's own route began. queries names them, so a run after a
	// change says which read went away rather than only that one did.
	postgres, redis float64
	queries         string
}

// measure runs one case from callers callers at once and reports what one request cost.
func (s *PerformanceSuite) measure(name, route string, callers int, call func(caller int) time.Duration) measurement {
	for range warmup {
		call(0)
	}
	s.settle()
	s.spans.Reset()

	each := iterations / callers
	samples := make([]time.Duration, each*callers)
	var running sync.WaitGroup
	for caller := range callers {
		running.Add(1)
		go func() {
			defer running.Done()
			for index := range each {
				samples[caller*each+index] = call(caller)
			}
		}()
	}
	running.Wait()
	s.settle()

	var total time.Duration
	for _, sample := range samples {
		total += sample
	}
	sort.Slice(samples, func(a, b int) bool { return samples[a] < samples[b] })
	counted := s.reads(route)
	return measurement{
		name:     name,
		mean:     total / time.Duration(len(samples)),
		p50:      samples[len(samples)*50/100],
		p99:      samples[len(samples)*99/100],
		postgres: counted.postgres / float64(len(samples)),
		redis:    counted.redis / float64(len(samples)),
		queries:  counted.breakdown(len(samples)),
	}
}

// settle waits for what a request started and did not wait for, so that work belonging to
// one case is recorded before the next one resets the spans.
func (s *PerformanceSuite) settle() {
	for quiet, seen := 0, len(s.spans.Ended()); quiet < 5; {
		time.Sleep(20 * time.Millisecond)
		if ended := len(s.spans.Ended()); ended != seen {
			quiet, seen = 0, ended
			continue
		}
		quiet++
	}
}

// counts is the reads one case made, in total and by the name of each.
type counts struct {
	postgres, redis float64
	byName          map[string]int
}

// breakdown lists each read and how many of it one request made, busiest first.
func (c counts) breakdown(requests int) string {
	names := make([]string, 0, len(c.byName))
	for name := range c.byName {
		names = append(names, name)
	}
	sort.Slice(names, func(a, b int) bool { return c.byName[names[a]] > c.byName[names[b]] })
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s %.2f", name, float64(c.byName[name])/float64(requests)))
	}
	return strings.Join(parts, ", ")
}

// reads counts the queries and the Redis commands that happened under the traces route
// began. Counting by trace rather than over everything is what keeps a case's own
// bookkeeping -- the sessions it stops, the receipts it polls -- out of its numbers.
func (s *PerformanceSuite) reads(route string) counts {
	ended := s.spans.Ended()
	measured := map[trace.TraceID]bool{}
	for _, span := range ended {
		if !span.Parent().IsValid() && span.Name() == route {
			measured[span.SpanContext().TraceID()] = true
		}
	}

	counted := counts{byName: map[string]int{}}
	for _, span := range ended {
		if !measured[span.SpanContext().TraceID()] {
			continue
		}
		switch span.InstrumentationScope().Name {
		case "github.com/uptrace/bun":
			counted.postgres++
			counted.byName["pg:"+span.Name()]++
		case "github.com/redis/rueidis":
			counted.redis++
			counted.byName["redis:"+span.Name()]++
		}
	}
	return counted
}

func timed(call func()) time.Duration {
	at := time.Now()
	call()
	return time.Since(at)
}

// report lays the measurements out as a table, so a run before a change and a run after it
// can be read side by side.
func report(measurements []measurement) string {
	var table strings.Builder
	fmt.Fprintf(&table, "%-32s %10s %10s %10s %9s %7s\n",
		"case", "mean", "p50", "p99", "postgres", "redis")
	for _, one := range measurements {
		fmt.Fprintf(&table, "%-32s %10s %10s %10s %9.2f %7.2f\n",
			one.name,
			one.mean.Round(time.Microsecond),
			one.p50.Round(time.Microsecond),
			one.p99.Round(time.Microsecond),
			one.postgres, one.redis)
		fmt.Fprintf(&table, "%-32s %s\n", "", one.queries)
	}
	return table.String()
}
