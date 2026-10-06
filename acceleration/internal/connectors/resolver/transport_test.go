//go:build integration

package resolver_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// requestTimeout bounds one tool call through core.Transports: cmd/router's
// connectorHTTPTimeout (10 s, the prototype's), so a hung fake fails the test.
const requestTimeout = 10 * time.Second

// toolCall is an MCP tools/call the fake answers 200 to with a token it takes.
const toolCall = `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`

// TransportSuite runs core.Transports over the real resolver, oauth2_code and the fake
// provider's MCP endpoint: what a 401 does to the connection in Postgres, and when the call
// goes again.
type TransportSuite struct {
	suite.Suite
	dsn string
	db  *store.Store
	f   *fixture
}

func TestTransportSuite(t *testing.T) {
	suite.Run(t, new(TransportSuite))
}

func (s *TransportSuite) SetupSuite() {
	s.dsn, s.db = database(s.T())
}

func (s *TransportSuite) TearDownSuite() {
	if s.db != nil {
		s.Require().NoError(s.db.Close())
	}
}

func (s *TransportSuite) SetupTest() {
	s.f = newFixture(s.T(), s.dsn, s.db)
}

// TestATokenTheProviderEndedEarlyIsRefreshedAndTheCallRetried: the fake's clock moves past
// the token's expiry and the router's does not, so the provider refuses a token the router
// believes live. The grant still works: the token is refreshed, the call goes once more,
// and the connection stays connected.
func (s *TransportSuite) TestATokenTheProviderEndedEarlyIsRefreshedAndTheCallRetried() {
	ref := s.f.connected()
	client := s.transports(s.f.srv.Client().Transport).Client(ref, s.f.scheme(s.f.srv.Client()))
	s.Equal(http.StatusOK, s.call(client))
	revision, refreshes := s.f.stored(ref).Revision, s.f.srv.Refreshes()

	s.f.srv.Advance(fakeprovider.AccessTTL + time.Second)

	s.Equal(http.StatusOK, s.call(client))
	connection := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, connection.Status)
	s.Equal(revision+1, connection.Revision)
	s.Equal(refreshes+1, s.f.srv.Refreshes())
	s.Equal(3, s.f.srv.Hits(fakeprovider.PathMCP), "the first call, the refused one and its retry")
}

// TestARefusedRefreshMovesTheConnectionToNeedsReauthorization: the provider refuses the token
// and then the refresh (invalid_grant). Only a reconnect helps, and the call is not sent again.
func (s *TransportSuite) TestARefusedRefreshMovesTheConnectionToNeedsReauthorization() {
	ref := s.f.connected()
	client := s.transports(s.f.srv.Client().Transport).Client(ref, s.f.scheme(s.f.srv.Client()))
	s.Equal(http.StatusOK, s.call(client))
	s.f.srv.Use(fakeprovider.InvalidGrant)

	s.f.srv.Advance(fakeprovider.AccessTTL + time.Second)

	s.Equal(http.StatusUnauthorized, s.call(client))
	s.Equal(store.ConnectionNeedsReauthorization, s.f.stored(ref).Status)
	s.Equal(2, s.f.srv.Hits(fakeprovider.PathMCP))
}

// TestATokenAnotherRouterRenewedIsRetriedWithTheRenewedOne: while the call is on its way,
// another router renews the token and the fake stops taking the old one. The refusal names a
// revision the stored credentials have moved past, so the connection stays connected, and
// the call goes once more with the renewed token.
func (s *TransportSuite) TestATokenAnotherRouterRenewedIsRetriedWithTheRenewedOne() {
	ref := s.f.connected()
	other := s.f.router(s.f.srv.Client())
	var once sync.Once
	renewElsewhere := func() {
		once.Do(func() {
			s.f.srv.Advance(fakeprovider.AccessTTL + time.Second)
			_, err := other.Resolve(s.f.ctx, ref, core.CredentialRequest{Deadline: s.f.clock.Now().Add(2 * fakeprovider.AccessTTL)})
			s.Require().NoError(err)
		})
	}
	revision := s.f.stored(ref).Revision
	refreshes := s.f.srv.Refreshes()
	base := &beforeMCP{base: s.f.srv.Client().Transport, run: renewElsewhere}
	client := s.transports(base).Client(ref, s.f.scheme(s.f.srv.Client()))

	s.Equal(http.StatusOK, s.call(client))

	connection := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, connection.Status)
	s.Equal(revision+1, connection.Revision, "one renewal, by the other router")
	s.Equal(refreshes+1, s.f.srv.Refreshes())
	s.Equal(2, s.f.srv.Hits(fakeprovider.PathMCP))
}

// TestAClientHeldPastADeleteSendsNothing: the resolver reads the connection's row on every
// request, so a client a source still holds stops at the router once the connection is
// deleted, before Transports.Close reaches it.
func (s *TransportSuite) TestAClientHeldPastADeleteSendsNothing() {
	ref := s.f.connected()
	client := s.transports(s.f.srv.Client().Transport).Client(ref, s.f.scheme(s.f.srv.Client()))
	s.Equal(http.StatusOK, s.call(client))
	s.Require().NoError(s.f.db.DeleteConnectorConnection(s.f.ctx, ref.CustomerID, ref.ConnectionID))

	request, err := http.NewRequest(http.MethodPost, s.f.srv.URL+fakeprovider.PathMCP, strings.NewReader(toolCall))
	s.Require().NoError(err)
	_, err = client.Do(request)

	s.ErrorIs(err, store.ErrNoConnectorConnection)
	s.Equal(1, s.f.srv.Hits(fakeprovider.PathMCP))
}

// transports reaches the fake through base, with egress's redirect policy: egress itself
// refuses loopback, where the fake listens (fakeprovider/AGENTS.md, «Egress»).
func (s *TransportSuite) transports(base http.RoundTripper) *core.Transports {
	policy := egress.NewClient(0, nil).CheckRedirect
	transports, err := core.NewTransports(core.TransportsConfig{
		Resolver: s.f.router(s.f.srv.Client()),
		Timeout:  requestTimeout,
		NewClient: func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
			if wrap != nil {
				base = wrap(base)
			}
			return &http.Client{Timeout: timeout, Transport: base, CheckRedirect: policy}
		},
	})
	s.Require().NoError(err)
	return transports
}

// call sends one tool call and returns the status it got.
func (s *TransportSuite) call(client *http.Client) int {
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		s.f.srv.URL+fakeprovider.PathMCP, strings.NewReader(toolCall))
	s.Require().NoError(err)
	response, err := client.Do(request)
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// beforeMCP runs run when a request for the MCP endpoint reaches the transport, after the
// scheme applied the token and before it is sent.
type beforeMCP struct {
	base http.RoundTripper
	run  func()
}

func (b *beforeMCP) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Path == fakeprovider.PathMCP {
		b.run()
	}
	return b.base.RoundTrip(r)
}
