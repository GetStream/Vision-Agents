package mem0

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// callSpans is the provider the package's tracer records into, installed once per process.
// tracer is taken at package initialisation, and the global provider hands a tracer taken
// that early to the first provider installed and to no later one (delegateTraceOnce in
// otel's internal/global/state.go).
var callSpans = sync.OnceValue(func() *sdktrace.TracerProvider {
	provider := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(provider)
	return provider
})

// CallRecordingSuite proves every call to mem0 is counted, traced and logged once.
type CallRecordingSuite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	store    *Store
	scope    memory.Scope
	logs     *bytes.Buffer
	recorded *tracetest.SpanRecorder
}

func TestCallRecordingSuite(t *testing.T) { suite.Run(t, new(CallRecordingSuite)) }

func (s *CallRecordingSuite) SetupSuite() {
	s.recorded = tracetest.NewSpanRecorder()
	callSpans().RegisterSpanProcessor(s.recorded)
}

func (s *CallRecordingSuite) TearDownSuite() { callSpans().UnregisterSpanProcessor(s.recorded) }

func (s *CallRecordingSuite) SetupTest() {
	s.ctx = context.Background()
	s.recorded.Reset()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)
	s.logs = &bytes.Buffer{}

	store, err := New(Options{
		APIKey:  "test-key",
		BaseURL: s.platform.server.URL,
		Logger:  slog.New(slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug})),
	})
	s.Require().NoError(err)
	s.store = store
	s.scope = memory.Scope{AppID: "router", UserID: "acme", AgentID: "support", RunID: "session-1"}
}

// counted is how many calls the counter holds for op, status and outcome.
func counted(op, status, outcome string) float64 {
	return testutil.ToFloat64(calls.WithLabelValues(op, status, outcome))
}

// timed is how many calls the latency histogram holds for op and outcome.
func (s *CallRecordingSuite) timed(op, outcome string) uint64 {
	return s.histogram(op, outcome).GetSampleCount()
}

// timedSum is the seconds the latency histogram holds for op and outcome.
func (s *CallRecordingSuite) timedSum(op, outcome string) float64 {
	return s.histogram(op, outcome).GetSampleSum()
}

func (s *CallRecordingSuite) histogram(op, outcome string) *dto.Histogram {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	for _, family := range families {
		if family.GetName() != "router_mem0_call_seconds" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["op"] == op && labels["outcome"] == outcome {
				return metric.GetHistogram()
			}
		}
	}
	return &dto.Histogram{}
}

// lines is every log line written, decoded.
func (s *CallRecordingSuite) lines() []map[string]any {
	var decoded []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(s.logs.String()), "\n") {
		if line == "" {
			continue
		}
		fields := map[string]any{}
		s.Require().NoError(json.Unmarshal([]byte(line), &fields))
		decoded = append(decoded, fields)
	}
	return decoded
}

func (s *CallRecordingSuite) TestEachCallIsCountedAndTimedUnderItsOperation() {
	before := map[string]float64{}
	timedBefore := map[string]uint64{}
	for _, op := range []string{opSearch, opAdd, opList, opDelete} {
		before[op] = counted(op, "200", outcomeOK)
		timedBefore[op] = s.timed(op, outcomeOK)
	}

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Require().NoError(err)
	_, err = s.store.Remember(s.ctx, s.scope, []llm.Message{{Role: llm.User, Content: "I moved"}})
	s.Require().NoError(err)
	s.platform.replies = []string{`{"results":[{"id":"m1"}]}`, `{"message":"ok"}`, `{"results":[]}`}
	_, err = s.store.ForgetRun(s.ctx, "router", "session-1")
	s.Require().NoError(err)

	want := map[string]float64{opSearch: 1, opAdd: 1, opList: 2, opDelete: 1}
	for op, calls := range want {
		s.Equal(calls, counted(op, "200", outcomeOK)-before[op], op)
		s.Equal(uint64(calls), s.timed(op, outcomeOK)-timedBefore[op], op)
	}
	s.span("mem0.delete")
}

func (s *CallRecordingSuite) TestOnlyADeletionWithNothingToDeleteCountsAnEmptyList() {
	empty := testutil.ToFloat64(emptyLists)

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Require().NoError(err)
	s.platform.replies = []string{`{"results":[{"id":"m1"}]}`, `{"message":"ok"}`, `{"results":[]}`}
	_, err = s.store.ForgetRun(s.ctx, "router", "session-1")
	s.Require().NoError(err)
	s.Equal(0.0, testutil.ToFloat64(emptyLists)-empty,
		"a deletion that found m1 ends on an empty page without counting it; an empty search is not a list")

	_, err = s.store.Truncate(s.ctx, "router", "acme")
	s.Require().NoError(err)
	s.Equal(1.0, testutil.ToFloat64(emptyLists)-empty, "a deletion whose first list is empty is counted once")
}

func (s *CallRecordingSuite) TestTheLatencyOfACallIsRecordedInTheHistogramAndTheLog() {
	s.platform.delay = 50 * time.Millisecond
	sumBefore := s.timedSum(opSearch, outcomeOK)

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Require().NoError(err)

	s.GreaterOrEqual(s.timedSum(opSearch, outcomeOK)-sumBefore, 0.05)
	s.GreaterOrEqual(s.lines()[0]["latency_ms"], float64(50))
}

func (s *CallRecordingSuite) TestA429IsCountedUnderItsStatusAndNamedAsTheQuota() {
	before := counted(opSearch, "429", outcomeHTTPError)
	timedBefore := s.timed(opSearch, outcomeHTTPError)
	s.platform.status = http.StatusTooManyRequests
	s.platform.respond = `{"error":"Usage quota exceeded for this billing period."}`

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

	s.ErrorContains(err, "429 Too Many Requests (mem0 quota or rate limit): ")
	s.ErrorContains(err, "Usage quota exceeded for this billing period.")
	s.Equal(1.0, counted(opSearch, "429", outcomeHTTPError)-before)
	s.Equal(uint64(1), s.timed(opSearch, outcomeHTTPError)-timedBefore)
	lines := s.lines()
	s.Require().Len(lines, 1, "the caller logs the returned error; the store adds only its debug line")
	s.Equal("DEBUG", lines[0]["level"])
	s.Equal("429", lines[0]["status"])
	s.Equal(outcomeHTTPError, lines[0]["outcome"])
}

func (s *CallRecordingSuite) TestAnotherRefusalIsNotNamedAsTheQuota() {
	s.platform.status = http.StatusBadRequest
	s.platform.respond = `{"error":"400 Bad Request"}`

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

	// The whole message as the base revision wrote it.
	s.EqualError(err, `mem0: /v3/memories/search/: 400 Bad Request: {"error":"400 Bad Request"}`)
}

func (s *CallRecordingSuite) TestAMem0ThatNeverAnsweredIsCountedWithNoStatus() {
	before := counted(opSearch, "none", outcomeTransport)
	s.platform.server.Close()

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

	s.Error(err)
	s.Equal(1.0, counted(opSearch, "none", outcomeTransport)-before)
	failed := s.span("mem0.search")
	s.Equal(codes.Error, failed.Status().Code)
	s.NotContains(attributes(failed), attribute.Key("mem0.results"))
	s.Equal("none", s.lines()[0]["status"])
}

func (s *CallRecordingSuite) TestAnAnswerThatIsNotJSONIsCountedAsUndecodable() {
	before := counted(opSearch, "200", outcomeDecode)
	s.platform.respond = `not json`

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

	s.ErrorContains(err, "decode")
	s.Equal(1.0, counted(opSearch, "200", outcomeDecode)-before)
	failed := s.span("mem0.search")
	s.Equal(codes.Error, failed.Status().Code)
	s.NotContains(attributes(failed), attribute.Key("mem0.results"))
}

func (s *CallRecordingSuite) TestEachCallIsLoggedWithWhoseMemoriesItTouched() {
	_, err := s.store.Remember(s.ctx, s.scope, []llm.Message{{Role: llm.User, Content: "I moved"}})
	s.Require().NoError(err)
	_, err = s.store.ForgetRun(s.ctx, "router", "session-2")
	s.Require().NoError(err)
	_, err = s.store.Truncate(s.ctx, "router", "acme")
	s.Require().NoError(err)

	var lines []map[string]any
	for _, line := range s.lines() {
		if line["msg"] == "called mem0" {
			lines = append(lines, line)
		}
	}
	s.Require().Len(lines, 3)
	s.Equal("DEBUG", lines[0]["level"])
	s.Equal(opAdd, lines[0]["op"])
	s.Equal("router", lines[0]["app_id"])
	s.Equal("acme", lines[0]["user_id"])
	s.Equal("session-1", lines[0]["run_id"])
	s.Equal("200", lines[0]["status"])
	s.Contains(lines[0], "latency_ms")
	s.Equal(opList, lines[1]["op"])
	s.Equal("session-2", lines[1]["run_id"])
	s.NotContains(lines[1], "user_id", "forgetting a session names no user")
	s.Equal(float64(0), lines[1]["results"])
	s.Equal(opList, lines[2]["op"])
	s.Equal("acme", lines[2]["user_id"])
	s.NotContains(lines[2], "run_id", "truncating names no session")
	s.NotContains(s.logs.String(), "I moved", "what was said is never logged")
}

func (s *CallRecordingSuite) TestEachCallIsASpanUnderTheCallersSpan() {
	s.platform.respond = `{"results":[{"id":"m1","memory":"Lives in Austin"},{"id":"m2","memory":"Has a dog"}]}`
	ctx, parent := callSpans().Tracer("test").Start(s.ctx, "turn")

	_, err := s.store.Recall(ctx, memory.Query{Scope: s.scope, Text: "where do they live"})
	parent.End()
	s.Require().NoError(err)

	span := s.span("mem0.search")
	s.Equal(parent.SpanContext().SpanID(), span.Parent().SpanID())
	s.Equal(map[attribute.Key]attribute.Value{
		"mem0.op":                   attribute.StringValue(opSearch),
		"http.response.status_code": attribute.IntValue(200),
		"mem0.results":              attribute.IntValue(2),
	}, attributes(span), "never the key, the query or what is remembered")
	s.Equal(codes.Unset, span.Status().Code)
}

func (s *CallRecordingSuite) TestARefusedCallIsAFailedSpanWithoutMem0sWords() {
	s.platform.status = http.StatusTooManyRequests
	s.platform.respond = `{"error":"Usage quota exceeded for user acme"}`

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Require().Error(err)

	span := s.span("mem0.search")
	s.Equal(codes.Error, span.Status().Code)
	s.Equal("Too Many Requests", span.Status().Description)
	s.Equal(attribute.IntValue(429), attributes(span)["http.response.status_code"])
	s.NotContains(attributes(span), attribute.Key("mem0.results"))
	s.Empty(span.Events(), "mem0's body may repeat what it was sent")
}

// span is the one recorded span of that name.
func (s *CallRecordingSuite) span(name string) sdktrace.ReadOnlySpan {
	var found []sdktrace.ReadOnlySpan
	for _, span := range s.recorded.Ended() {
		if span.Name() == name {
			found = append(found, span)
		}
	}
	s.Require().Len(found, 1)
	return found[0]
}

func attributes(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	byKey := map[attribute.Key]attribute.Value{}
	for _, pair := range span.Attributes() {
		byKey[pair.Key] = pair.Value
	}
	return byKey
}

// refusedAlert is the mem0_refused condition, fresh, after a first check that takes in what
// earlier tests counted, so what it holds next is only what the test did. That check is a limit
// before start: an item it sent would otherwise be held back from the test's own checks.
func (s *CallRecordingSuite) refusedAlert(start time.Time) alerts.ItemCheck {
	conditions := Alerts(time.Hour)
	s.Require().Len(conditions, 1)
	s.Equal("mem0_refused", conditions[0].Name)
	s.Equal(alerts.Warn, conditions[0].Level, "a refusal does not page")
	conditions[0].Items(start.Add(-time.Hour))
	return conditions[0].Items
}

func (s *CallRecordingSuite) recall() {
	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Error(err)
}

func (s *CallRecordingSuite) TestARefusalAlertsOncePerOperationAndStatusUntilTheLimitPasses() {
	start := time.Now()
	refused := s.refusedAlert(start)
	s.platform.status = http.StatusTooManyRequests

	s.recall()
	held := refused(start.Add(time.Second))
	s.Require().Contains(held, "search/429")
	s.GreaterOrEqual(held["search/429"]["increase"], 1.0)

	s.recall()
	s.NotContains(refused(start.Add(time.Hour)), "search/429", "held back by the limit")

	s.recall()
	s.Contains(refused(start.Add(time.Hour+2*time.Second)), "search/429", "sent again once the limit passed")
}

func (s *CallRecordingSuite) TestEveryHTTPErrorAlertsUnderItsStatusAndNothingElseDoes() {
	for _, tc := range []struct {
		name   string
		status int
		answer string
		closed bool
		item   string
		alerts bool
	}{
		{name: "spent quota", status: http.StatusTooManyRequests, answer: `{"error":"quota"}`, item: "search/429", alerts: true},
		{name: "revoked key", status: http.StatusUnauthorized, answer: `{"detail":"Invalid API key"}`, item: "search/401", alerts: true},
		{name: "bad request", status: http.StatusBadRequest, answer: `{"error":"bad"}`, item: "search/400", alerts: true},
		{name: "server error", status: http.StatusInternalServerError, answer: `{"error":"down"}`, item: "search/500", alerts: true},
		{name: "answered", status: http.StatusOK, answer: `{"results":[]}`, item: "search/200"},
		{name: "undecodable answer", status: http.StatusOK, answer: `not json`, item: "search/200"},
		{name: "never answered", closed: true, item: "search/none"},
	} {
		s.Run(tc.name, func() {
			s.SetupTest()
			start := time.Now()
			refused := s.refusedAlert(start)
			s.platform.status, s.platform.respond = tc.status, tc.answer
			if tc.closed {
				s.platform.server.Close()
			}

			_, _ = s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

			held := refused(start.Add(time.Second))
			if tc.alerts {
				s.Contains(held, tc.item)
			} else {
				s.NotContains(held, tc.item)
			}
		})
	}
}
