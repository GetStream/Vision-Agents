//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// PathTokensSuite reads what the router's log and spans say of routes whose secret is in the path.
type PathTokensSuite struct {
	RouterSuite
	logged *lockedLog
	spans  *tracetest.SpanRecorder
}

func TestPathTokens(t *testing.T) {
	runSuite(t, new(PathTokensSuite))
}

func (s *PathTokensSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	// Installed before the harness: otelhttp takes the provider when the handler is built.
	s.spans = tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(s.spans)))
	s.RouterSuite.SetupSuite()
}

func (s *PathTokensSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *PathTokensSuite) requireLogged(want string) {
	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), want) },
		5*time.Second, 20*time.Millisecond, want)
}

func (s *PathTokensSuite) TestATokenRouteIsLoggedWithItsTokenTakenOut() {
	for _, route := range []struct{ method, prefix string }{
		{http.MethodGet, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/status/"},
		{http.MethodPost, "/v1/phone/machine/"},
		{http.MethodPost, "/v1/connectors/mcp-events/"},
		{http.MethodGet, "/v1/agents/channels/hooks/"},
		{http.MethodPost, "/v1/agents/channels/hooks/"},
	} {
		token := s.utils.uuid()
		before := strings.Count(s.logged.String(), "path="+route.prefix+"{token}")
		s.unauthenticatedClient.do(route.method, route.prefix+token, nil, nil)

		// The same line is logged for each method of a route, so wait for this request's own.
		line := "path=" + route.prefix + "{token}"
		want := before + 1
		s.Require().Eventually(func() bool { return strings.Count(s.logged.String(), line) >= want },
			5*time.Second, 20*time.Millisecond, line)
		s.NotContains(s.logged.String(), token)
	}
}

func (s *PathTokensSuite) TestATokenRouteSpanHoldsNoToken() {
	for _, route := range []struct{ method, prefix string }{
		{http.MethodGet, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/status/"},
		{http.MethodPost, "/v1/phone/machine/"},
		{http.MethodPost, "/v1/connectors/mcp-events/"},
		{http.MethodGet, "/v1/agents/channels/hooks/"},
		{http.MethodPost, "/v1/agents/channels/hooks/"},
	} {
		token := s.utils.uuid()
		s.spans.Reset()
		s.unauthenticatedClient.do(route.method, route.prefix+token, nil, nil)

		var path string
		s.Require().Eventually(func() bool {
			for _, span := range s.spans.Ended() {
				if span.SpanKind() != trace.SpanKindServer {
					continue
				}
				for _, attr := range span.Attributes() {
					if attr.Key == "url.path" {
						path = attr.Value.Emit()
					}
				}
			}
			return path != ""
		}, 5*time.Second, 20*time.Millisecond, route.prefix)
		s.Equal(route.prefix+"{token}", path)

		for _, span := range s.spans.Ended() {
			s.NotContains(span.Name(), token)
			for _, attr := range span.Attributes() {
				s.NotContains(attr.Value.Emit(), token, attr.Key)
			}
		}
	}
}

func (s *PathTokensSuite) TestATokenSentWithAMethodTheRouteDoesNotServeIsStillTakenOut() {
	token := s.utils.uuid()

	s.unauthenticatedClient.do(http.MethodGet, "/v1/phone/status/"+token, nil, nil)

	s.requireLogged("path=/v1/phone/status/{token}")
	s.NotContains(s.logged.String(), token)
}

func (s *PathTokensSuite) TestAnEncodedTokenIsTakenOut() {
	token := s.utils.uuid()

	s.unauthenticatedClient.do(http.MethodPost, "/v1/phone/status/"+token+"%2Fx", nil, nil)

	s.requireLogged("path=/v1/phone/status/{token}")
	s.NotContains(s.logged.String(), token)
	s.NotContains(s.logged.String(), "%2Fx")
}

func (s *PathTokensSuite) TestARouteWithoutATokenIsLoggedAsItWasSent() {
	path := "/v1/agents/sessions/" + s.utils.uuid() + "/events"

	s.unauthenticatedClient.do(http.MethodGet, path, nil, nil)

	s.requireLogged("path=" + path)
}
