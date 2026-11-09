package sttrouter

import (
	"sync"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// startSpans is the provider the package's tracers record into, installed once per process:
// a tracer taken at package initialisation is handed to the first provider installed and to
// no later one (delegateTraceOnce in otel's internal/global/state.go).
var startSpans = sync.OnceValue(func() *sdktrace.TracerProvider {
	provider := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(provider)
	return provider
})

func (s *STTRouterSuite) TestASessionsTraceShowsTheVendorDialledThenTheLeaseTaken() {
	recorded := tracetest.NewSpanRecorder()
	startSpans().RegisterSpanProcessor(recorded)
	s.T().Cleanup(func() { startSpans().UnregisterSpanProcessor(recorded) })
	router, _ := s.dialled()

	_, err := s.startOn(router, Request{Target: "en-low-latency"})

	s.Require().NoError(err)
	var start sdktrace.ReadOnlySpan
	var under []string
	for _, span := range recorded.Ended() {
		if span.Name() == "stt.start" {
			start = span
		}
	}
	s.Require().NotNil(start)
	for _, span := range recorded.Ended() {
		if span.Parent().SpanID() == start.SpanContext().SpanID() {
			under = append(under, span.Name())
		}
	}
	s.Equal([]string{"routing.provider.start", "stt.lease"}, under)
}
