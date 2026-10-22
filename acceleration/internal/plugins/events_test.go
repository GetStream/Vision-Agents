package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/webhook"
)

type EventsSuite struct {
	suite.Suite
	secret string
}

func TestEventsSuite(t *testing.T) {
	suite.Run(t, new(EventsSuite))
}

func (s *EventsSuite) SetupTest() {
	secret, err := webhook.NewSecret()
	s.Require().NoError(err)
	s.secret = secret
}

func (s *EventsSuite) TestSubscribingSendsTheEventItsFiltersAndTheCallback() {
	var subscribed map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("Bearer secret", r.Header.Get("Authorization"))
		s.Equal(EventsVersion, r.Header.Get("MCP-Protocol-Version"))
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		switch body.Method {
		case "server/discover":
			writeRPC(w, body.ID, map[string]any{
				"supportedVersions": []string{EventsVersion},
				"capabilities":      map[string]any{"tools": map[string]any{}, "events": map[string]any{}},
			})
		case "events/subscribe":
			subscribed, _ = body.Params.(map[string]any)
			writeRPC(w, body.ID, map[string]any{"id": "sub_123", "refreshBefore": "2026-10-04T12:00:00Z", "cursor": nil})
		default:
			s.Fail("unexpected method " + body.Method)
		}
	}))
	defer server.Close()

	granted, err := Subscribe(context.Background(), Connection{PluginID: "sentry", Endpoint: server.URL, AccessToken: "secret"},
		"issue.created", map[string]any{"project": "web"},
		Delivery{URL: "https://router.example/v1/agents/plugins/events/tok", Secret: s.secret}, server.Client())

	s.Require().NoError(err)
	s.Equal("sub_123", granted.ID)
	s.Require().NotNil(granted.RefreshBefore)
	s.Equal(time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC), granted.RefreshBefore.UTC())
	s.Equal("issue.created", subscribed["name"])
	s.Equal(map[string]any{"project": "web"}, subscribed["arguments"])
	s.Equal(map[string]any{
		"mode":   "webhook",
		"url":    "https://router.example/v1/agents/plugins/events/tok",
		"secret": s.secret,
	}, subscribed["delivery"])
}

func (s *EventsSuite) TestAServerWithoutEventsIsNotAskedToSubscribe() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		s.Equal("server/discover", body.Method)
		writeRPC(w, body.ID, map[string]any{"capabilities": map[string]any{"tools": map[string]any{}}})
	}))
	defer server.Close()

	_, err := Subscribe(context.Background(), Connection{PluginID: "sentry", Endpoint: server.URL},
		"issue.created", nil, Delivery{URL: "https://router.example/cb", Secret: s.secret}, server.Client())

	s.ErrorIs(err, ErrNoEvents)
}
