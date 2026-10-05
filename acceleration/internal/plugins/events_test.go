package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type EventsSuite struct {
	suite.Suite
	secret string
}

func TestEventsSuite(t *testing.T) {
	suite.Run(t, new(EventsSuite))
}

func (s *EventsSuite) SetupTest() {
	secret, err := NewWebhookSecret()
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

func (s *EventsSuite) TestTheSignatureIsStandardWebhooks() {
	// The example in the Standard Webhooks specification, which its libraries test against.
	signature, err := SignWebhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", "msg_p5jXN8AQM9LWM0D4loKWxJek",
		time.Unix(1614265330, 0), []byte(`{"test": 2432232314}`))

	s.Require().NoError(err)
	s.Equal("v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=", signature)
}

func (s *EventsSuite) TestADeliverySignedWithTheSecretIsAccepted() {
	body := []byte(`{"eventId":"evt_1","name":"issue.created","data":{}}`)
	header := s.signed("evt_1", time.Now(), body, s.secret)

	s.NoError(VerifyWebhook(s.secret, header, body, time.Now()))
}

func (s *EventsSuite) TestADeliverySignedWithAnotherSecretIsRefused() {
	other, err := NewWebhookSecret()
	s.Require().NoError(err)
	body := []byte(`{"eventId":"evt_1"}`)

	s.ErrorIs(VerifyWebhook(s.secret, s.signed("evt_1", time.Now(), body, other), body, time.Now()), ErrBadSignature)
}

func (s *EventsSuite) TestABodyChangedAfterSigningIsRefused() {
	header := s.signed("evt_1", time.Now(), []byte(`{"eventId":"evt_1"}`), s.secret)

	s.ErrorIs(VerifyWebhook(s.secret, header, []byte(`{"eventId":"evt_2"}`), time.Now()), ErrBadSignature)
}

func (s *EventsSuite) TestAnOldSignatureIsRefused() {
	body := []byte(`{"eventId":"evt_1"}`)
	header := s.signed("evt_1", time.Now().Add(-time.Hour), body, s.secret)

	s.ErrorIs(VerifyWebhook(s.secret, header, body, time.Now()), ErrBadSignature)
}

func (s *EventsSuite) TestEitherSignatureOfARotationIsEnough() {
	old, err := NewWebhookSecret()
	s.Require().NoError(err)
	body := []byte(`{"eventId":"evt_1"}`)
	now := time.Now()
	stale, err := SignWebhook(old, "evt_1", now, body)
	s.Require().NoError(err)
	header := s.signed("evt_1", now, body, s.secret)
	header.Set("webhook-signature", stale+" "+header.Get("webhook-signature"))

	s.NoError(VerifyWebhook(s.secret, header, body, now))
}

func (s *EventsSuite) signed(id string, at time.Time, body []byte, secret string) http.Header {
	signature, err := SignWebhook(secret, id, at, body)
	s.Require().NoError(err)
	header := http.Header{}
	header.Set("webhook-id", id)
	header.Set("webhook-timestamp", strconv.FormatInt(at.Unix(), 10))
	header.Set("webhook-signature", signature)
	return header
}
