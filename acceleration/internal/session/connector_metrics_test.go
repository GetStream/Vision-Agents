package session

import (
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorMetricsSuite is what dispatcher.record counts of a connector call: the connector,
// a built-in's own id or custom, how the call ended, and how long it took. Each test names a
// connector of its own, so a rerun in the same process starts from zero.
type ConnectorMetricsSuite struct {
	suite.Suite
	d *dispatcher
}

func TestConnectorMetricsSuite(t *testing.T) {
	suite.Run(t, new(ConnectorMetricsSuite))
}

func (s *ConnectorMetricsSuite) SetupTest() {
	s.d = &dispatcher{logger: slog.New(slog.DiscardHandler)}
}

func (s *ConnectorMetricsSuite) TestACallThatAnsweredIsCountedOkUnderItsBuiltIn() {
	connector := builtInConnector()

	s.recordOf(connector, time.Now(), "")

	s.Equal(1.0, callsOf(connector, "ok"))
	s.Equal(uint64(1), timedCalls(connector).GetSampleCount())
}

func (s *ConnectorMetricsSuite) TestAFailedCallIsCountedUnderItsErrorType() {
	for _, kind := range []string{store.InvocationCustomerAuth, store.InvocationExternalServer,
		store.InvocationClientTimeout, store.InvocationOutcomeUnknown, store.InvocationDenied} {
		s.Run(kind, func() {
			connector := builtInConnector()

			s.recordOf(connector, time.Now(), kind)

			s.Equal(1.0, callsOf(connector, kind))
			s.Zero(callsOf(connector, "ok"))
			s.Equal(uint64(1), timedCalls(connector).GetSampleCount(), "a failed call is timed too")
		})
	}
}

func (s *ConnectorMetricsSuite) TestACustomConnectorsCallIsCountedAsCustom() {
	connector := store.CustomPrefix + strings.ReplaceAll(uuid.NewString(), "-", "")
	calls, timed := callsOf("custom", "ok"), timedCalls("custom").GetSampleCount()

	s.recordOf(connector, time.Now(), "")

	s.Equal(calls+1, callsOf("custom", "ok"))
	s.Equal(timed+1, timedCalls("custom").GetSampleCount())
	s.Zero(callsOf(connector, "ok"), "the customer's own id is no label")
}

// An incognito session's row names no session, but the call is as much a call of the connector.
func (s *ConnectorMetricsSuite) TestAnIncognitoSessionsCallIsCounted() {
	s.d.spec.Incognito = true
	connector := builtInConnector()

	s.recordOf(connector, time.Now(), "")

	s.Equal(1.0, callsOf(connector, "ok"))
	s.Equal(uint64(1), timedCalls(connector).GetSampleCount())
}

// The time is in seconds, from when the call started: 1.5 s is in the 1.6 s bucket and not the 0.8 s one.
func (s *ConnectorMetricsSuite) TestACallIsTimedInSecondsFromItsStart() {
	connector := builtInConnector()

	s.recordOf(connector, time.Now().Add(-1500*time.Millisecond), "")

	timed := timedCalls(connector)
	s.InDelta(1.5, timed.GetSampleSum(), 0.5)
	buckets := timed.GetBucket()
	s.Equal(0.025, buckets[0].GetUpperBound(), "the fastest bucket is 25 ms")
	s.Equal(51.2, buckets[len(buckets)-1].GetUpperBound(), "the slowest is past a binding's 30 s timeout")
	s.Equal([]float64{0.8, 1.6}, []float64{buckets[5].GetUpperBound(), buckets[6].GetUpperBound()})
	s.Equal([]uint64{0, 1}, []uint64{buckets[5].GetCumulativeCount(), buckets[6].GetCumulativeCount()})
}

// recordOf records one call of connector that started at started and ended as kind.
func (s *ConnectorMetricsSuite) recordOf(connector string, started time.Time, kind string) {
	s.d.record(route{binding: store.ConnectorBinding{Name: "crm", ConnectorID: connector}, tool: "whoami"},
		llm.ToolCall{Name: "crm__whoami", Arguments: "{}"}, started, failure{kind: kind}, nil)
}

// builtInConnector is an id shaped as a built-in's, which no other test counts under.
func builtInConnector() string {
	return "metrics_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}

// callsOf is how many calls of connector were counted as outcome.
func callsOf(connector, outcome string) float64 {
	return testutil.ToFloat64(connectorCalls.WithLabelValues(connector, outcome))
}

// timedCalls is the histogram of connector's call times.
func timedCalls(connector string) *dto.Histogram {
	var written dto.Metric
	if err := connectorCallSeconds.WithLabelValues(connector).(prometheus.Histogram).Write(&written); err != nil {
		panic(err)
	}
	return written.GetHistogram()
}
