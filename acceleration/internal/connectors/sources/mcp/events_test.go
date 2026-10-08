package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

// EventsSuite is the source's MCP Events client against the fake provider, which offers events
// (fakeprovider.MCPEvents) behind a bearer token from its client credentials grant, and checks a
// callback with a signed challenge before it subscribes. The callback is a local server that
// echoes the challenge once the signature checks out with the secret the test subscribed with.
type EventsSuite struct {
	suite.Suite
	ctx      context.Context
	fake     *fakeprovider.Server
	resolver *memoryResolver
	source   *Source
	callback *httptest.Server
	secret   string
	// echoes is whether the callback echoes the challenge; verified counts the ones it did.
	echoes   atomic.Bool
	verified atomic.Int32
}

func TestEventsSuite(t *testing.T) {
	suite.Run(t, new(EventsSuite))
}

func (s *EventsSuite) SetupTest() {
	s.ctx = context.Background()
	s.fake = fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.MCPEvents)
	s.resolver = newResolver(s.issue())
	s.source = New()
	s.secret = fakeprovider.NewWebhookSecret()
	s.echoes.Store(true)
	s.verified.Store(0)
	s.callback = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var envelope struct {
			Challenge string `json:"challenge"`
		}
		if plugins.VerifyWebhook(s.secret, r.Header, body, time.Now()) != nil || json.Unmarshal(body, &envelope) != nil {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if !s.echoes.Load() {
			_, _ = w.Write([]byte(`{}`))
			return
		}
		s.verified.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]string{"challenge": envelope.Challenge})
	}))
	s.T().Cleanup(s.callback.Close)
}

func (s *EventsSuite) TestSubscribingSendsTheEventItsFiltersTheCallbackAndTheSecret() {
	granted, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())

	s.Require().NoError(err)
	held := s.fake.EventSubscriptions()
	s.Require().Len(held, 1)
	s.Equal(held[0].ID, granted.ID)
	s.Equal("issue.created", held[0].Name)
	s.Equal(map[string]any{"project": "web"}, held[0].Arguments)
	s.Equal(s.callback.URL+"/tok", held[0].URL)
	s.Equal(s.secret, held[0].Secret)
	s.Require().NotNil(granted.RefreshBefore)
	s.True(granted.RefreshBefore.Equal(held[0].RefreshBefore))
	s.Equal(int32(1), s.verified.Load(), "the server checked the callback with a challenge signed with the secret")
}

func (s *EventsSuite) TestSubscribingAgainRefreshesTheSameSubscription() {
	_, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())
	s.Require().NoError(err)

	_, err = s.source.Subscribe(s.ctx, s.binding(), s.subscription())

	s.Require().NoError(err)
	held := s.fake.EventSubscriptions()
	s.Require().Len(held, 1)
	s.Equal(2, held[0].Subscribes)
}

func (s *EventsSuite) TestAServerThatOffersNoEventsIsNotAskedToSubscribe() {
	s.fake.Use(fakeprovider.ClientCredentials)

	_, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())

	s.ErrorIs(err, core.ErrNoEvents)
	s.Empty(s.fake.EventSubscriptions())
	s.Zero(s.verified.Load())
}

func (s *EventsSuite) TestACallbackThatDoesNotEchoTheChallengeIsNotSubscribed() {
	s.echoes.Store(false)

	_, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())

	s.ErrorContains(err, "-32015")
	s.Empty(s.fake.EventSubscriptions())
}

func (s *EventsSuite) TestUnsubscribingStopsIt() {
	_, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())
	s.Require().NoError(err)
	stop := s.subscription()
	stop.Secret = ""

	s.Require().NoError(s.source.Unsubscribe(s.ctx, s.binding(), stop))

	s.Empty(s.fake.EventSubscriptions())
}

// TestAResolverRefusalSendsNothing: a connection with no credential asks the server nothing,
// not even server/discover.
func (s *EventsSuite) TestAResolverRefusalSendsNothing() {
	s.resolver.disconnect()
	before := s.fake.Hits(fakeprovider.PathMCP)

	_, err := s.source.Subscribe(s.ctx, s.binding(), s.subscription())

	s.ErrorIs(err, errNotConnected)
	s.Equal(before, s.fake.Hits(fakeprovider.PathMCP))
}

// binding reaches the fake's MCP endpoint through core.Transports with the bearer scheme.
func (s *EventsSuite) binding() core.ResolvedBinding {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: requestTimeout, NewClient: loopback(s.fake.Client())})
	s.Require().NoError(err)
	return core.ResolvedBinding{
		Manifest: manifest(s.fake.URL + fakeprovider.PathMCP),
		HTTP:     transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "crm-1"}, bearer.New()),
	}
}

func (s *EventsSuite) subscription() core.EventSubscription {
	return core.EventSubscription{Name: "issue.created", Arguments: map[string]any{"project": "web"},
		URL: s.callback.URL + "/tok", Secret: s.secret}
}

// issue is an access token from the fake's client credentials grant (RFC 6749 section 4.4).
func (s *EventsSuite) issue() string {
	form := url.Values{"grant_type": {"client_credentials"}}
	request, err := http.NewRequest(http.MethodPost, s.fake.URL+fakeprovider.PathToken, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(s.fake.ClientID, s.fake.ClientSecret)
	response, err := s.fake.Client().Do(request)
	s.Require().NoError(err)
	defer func() { _ = response.Body.Close() }()
	var token struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&token))
	return token.AccessToken
}
