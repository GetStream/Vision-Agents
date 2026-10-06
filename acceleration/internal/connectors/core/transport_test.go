package core_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
)

// requestTimeout bounds every request a test sends, so a broken transport fails the test
// instead of hanging it. Ten times what a loopback round trip takes; a choice.
const requestTimeout = 5 * time.Second

// closedWithin is how long a test waits for the server to see a connection the client
// closed: the close travels over loopback, so it is milliseconds. A choice.
const closedWithin = 2 * time.Second

// sha256Header is where the signing scheme puts the digest of the body it signed.
const sha256Header = "X-Content-Sha256"

// body is what every POST carries.
const body = `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo"}}`

// TransportsSuite runs Transports against a TLS server on loopback that plays the provider,
// with a scheme that signs the final request and a resolver that keeps one connection in
// memory.
type TransportsSuite struct {
	suite.Suite
	resolver *memoryResolver
	scheme   *signingScheme
	provider *provider
	ref      core.ConnectionRef
}

func TestTransportsSuite(t *testing.T) {
	suite.Run(t, new(TransportsSuite))
}

func (s *TransportsSuite) SetupTest() {
	s.resolver = &memoryResolver{token: "token-1", revision: 1, connected: true}
	s.scheme = &signingScheme{}
	s.provider = newProvider(s.T())
	s.ref = core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-1"}
}

// TestAPrivateDestinationIsRefusedAfterTheSchemeSignedTheFinalRequest is the router's own
// client, egress.NewClient: the scheme signs the request as it will leave, Content-Length
// and body digest included, and the egress transport under it refuses to dial loopback, so
// the provider receives nothing.
func (s *TransportsSuite) TestAPrivateDestinationIsRefusedAfterTheSchemeSignedTheFinalRequest() {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: requestTimeout})
	s.Require().NoError(err)

	_, err = s.post(transports.Client(s.ref, s.scheme), s.provider.URL+"/mcp")

	s.Require().Error(err)
	s.Contains(err.Error(), "non-public address")
	signed := s.scheme.signed()
	s.Require().Len(signed, 1, "the scheme saw the request before egress refused it")
	s.Equal(int64(len(body)), signed[0].contentLength)
	s.Equal(digest([]byte(body)), signed[0].digest)
	s.Empty(s.provider.received(), "nothing left the router")
}

func (s *TransportsSuite) TestTheProviderReceivesTheBodyTheSchemeSigned() {
	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	received := s.provider.received()
	s.Require().Len(received, 1)
	s.True(received[0].signatureMatches, "the digest the scheme signed is the digest of the bytes on the wire")
	s.Equal(int64(len(body)), received[0].contentLength)
}

func (s *TransportsSuite) TestEachRequestCarriesTheCredentialTheResolverHandsOutThen() {
	client := s.client()
	_, err := s.post(client, s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.resolver.renew("token-2")
	s.provider.accept("token-2")

	_, err = s.post(client, s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
}

// TestTheRequestsDeadlineIsWhatTheCredentialMustOutlive: the resolver renews a credential
// that expires before the call's deadline, as the real one does (RetrieveOptions.ValidUntil).
// The deadline is the request context's, or, without one, the client's Timeout from now.
func (s *TransportsSuite) TestTheRequestsDeadlineIsWhatTheCredentialMustOutlive() {
	s.resolver.expires = time.Now().Add(requestTimeout / 2)
	s.provider.accept("renewed")
	client := s.client()

	short, cancel := context.WithTimeout(context.Background(), requestTimeout/5)
	defer cancel()
	_, err := s.postWith(short, client, s.provider.URL+"/mcp")
	s.Require().NoError(err)
	_, err = s.post(client, s.provider.URL+"/mcp")
	s.Require().NoError(err)

	s.Equal([]string{"token-1", "renewed"}, s.provider.tokens())
}

func (s *TransportsSuite) TestAResolverRefusalSendsNothing() {
	s.resolver.disconnect()

	_, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().ErrorIs(err, errNotConnected)
	s.Empty(s.provider.received())
}

// TestARefusedCredentialIsRenewedAndTheRequestSentOnceMore: the provider ended token-1
// before the expiry the router knows (its clock runs ahead, or it revoked the access token
// alone). The grant still works, so the resolver renews past token-1 and the request goes
// again, with its body, and the connection stays connected.
func (s *TransportsSuite) TestARefusedCredentialIsRenewedAndTheRequestSentOnceMore() {
	s.resolver.refresh = "token-2"
	s.provider.accept("token-2")
	s.provider.refuse("token-1")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.True(s.resolver.isConnected())
	received := s.provider.received()
	s.Require().Len(received, 2)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
	s.Equal(body, received[1].body)
	s.True(received[1].signatureMatches)
}

// TestARefusedCredentialNothingRenewsNeedsAReconnect: a static key, or a token without a
// refresh token, comes back the same, so only a new grant helps.
func (s *TransportsSuite) TestARefusedCredentialNothingRenewsNeedsAReconnect() {
	s.provider.refuse("token-1")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal("refused", s.read(response), "the caller reads the provider's whole answer")
	s.False(s.resolver.isConnected(), "the resolver was told which credential the provider refused")
	s.Len(s.provider.received(), 1)
}

// TestARefusedRenewalIsTheAnswer: the refresh itself is refused (invalid_grant at the token
// endpoint), which the resolver acts on; the provider's 401 is the answer.
func (s *TransportsSuite) TestARefusedRenewalIsTheAnswer() {
	s.resolver.refreshRefused = true
	s.provider.refuse("token-1")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal("refused", s.read(response))
	s.False(s.resolver.isConnected())
	s.Len(s.provider.received(), 1)
}

// TestARefusalOfACredentialAnotherRouterRenewedIsRetriedOnceWithTheRenewedOne: the request
// left with token-1 while another router renewed it to token-2, so the provider refuses
// token-1. The resolver hands out token-2 without renewing again, and the request goes again
// with its body.
func (s *TransportsSuite) TestARefusalOfACredentialAnotherRouterRenewedIsRetriedOnceWithTheRenewedOne() {
	s.provider.accept("token-2")
	s.provider.onRequest(func() { s.resolver.renew("token-2") })
	s.provider.refuse("token-1")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.True(s.resolver.isConnected())
	received := s.provider.received()
	s.Require().Len(received, 2)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
	s.Equal(body, received[1].body)
	s.True(received[1].signatureMatches)
}

// TestARefusedRequestIsSentAgainOnlyOnce: every request finds its token renewed elsewhere and
// refused, so a retry could always find a newer one. It goes twice.
func (s *TransportsSuite) TestARefusedRequestIsSentAgainOnlyOnce() {
	renewals := 1
	s.provider.onRequest(func() {
		renewals++
		s.resolver.renew("token-" + strconv.Itoa(renewals))
	})
	s.provider.refuse("token-1", "token-2", "token-3")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
}

func (s *TransportsSuite) TestARefusalOfTheRenewedCredentialNeedsAReconnect() {
	s.resolver.refresh = "token-2"
	s.provider.refuse("token-1", "token-2")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
	s.False(s.resolver.isConnected(), "the provider refused the credential just renewed")
}

// TestABodyThatCannotBeReadAgainIsNotSentAgain: the credential is still renewed, so the next
// call carries the renewed one.
func (s *TransportsSuite) TestABodyThatCannotBeReadAgainIsNotSentAgain() {
	s.resolver.refresh = "token-2"
	s.provider.accept("token-2")
	s.provider.refuse("token-1")
	client := s.client()
	// A reader http.NewRequest does not know, so the request has no GetBody.
	request, err := http.NewRequest(http.MethodPost, s.provider.URL+"/mcp", io.MultiReader(strings.NewReader(body)))
	s.Require().NoError(err)

	response, err := client.Do(request)

	s.Require().NoError(err)
	defer response.Body.Close()
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Len(s.provider.received(), 1)
	next, err := s.post(client, s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.Equal(http.StatusOK, next.StatusCode)
	s.Equal([]string{"token-1", "token-2"}, s.provider.tokens())
}

func (s *TransportsSuite) TestA401TheSchemeFindsNothingInLeavesTheConnectionAlone() {
	s.provider.refuseWithoutReason()

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.True(s.resolver.isConnected())
	s.Len(s.provider.received(), 1)
}

// TestACrossOriginRedirectNeverCarriesTheCredential: the egress client's redirect policy
// refuses a redirect to another origin, so the second hop never reaches the scheme and the
// other server never sees the credential.
func (s *TransportsSuite) TestACrossOriginRedirectNeverCarriesTheCredential() {
	other := newProvider(s.T())
	s.provider.redirect(other.URL + "/mcp")

	_, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().Error(err)
	s.Contains(err.Error(), "redirect to another host refused")
	s.Empty(other.received())
	s.Len(s.scheme.signed(), 1)
}

func (s *TransportsSuite) TestASameOriginRedirectCarriesTheCredential() {
	s.provider.redirect(s.provider.URL + "/moved")

	response, err := s.post(s.client(), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"token-1", "token-1"}, s.provider.tokens())
}

func (s *TransportsSuite) TestOneConnectionSharesOnePoolAndAnotherHasItsOwn() {
	transports := s.transports()
	other := core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-2"}

	for _, client := range []*http.Client{transports.Client(s.ref, s.scheme), transports.Client(s.ref, s.scheme)} {
		response, err := s.post(client, s.provider.URL+"/mcp")
		s.Require().NoError(err)
		s.read(response)
	}
	s.Equal(1, s.provider.opened())
	response, err := s.post(transports.Client(other, s.scheme), s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.read(response)

	s.Equal(2, s.provider.opened())
}

func (s *TransportsSuite) TestClosingAConnectionClosesItsIdleConnections() {
	transports := s.transports()
	response, err := s.post(transports.Client(s.ref, s.scheme), s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.read(response)
	s.Zero(s.provider.closed())

	transports.Close(s.ref)

	s.Eventually(func() bool { return s.provider.closed() == 1 }, closedWithin, 10*time.Millisecond)
}

// TestReplacingTheTransportOfOneClientLeavesTheConnectionsOwn: a caller that swaps its
// copy's Transport bypasses the credential and egress for itself only.
func (s *TransportsSuite) TestReplacingTheTransportOfOneClientLeavesTheConnectionsOwn() {
	transports := s.transports()
	replaced := transports.Client(s.ref, s.scheme)
	replaced.Transport = http.DefaultTransport

	response, err := s.post(transports.Client(s.ref, s.scheme), s.provider.URL+"/mcp")

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"token-1"}, s.provider.tokens())
}

// TestAClientUnusedForIdleAfterIsDropped: connection-1 is asked for and sent with at t0,
// connection-3 sent with again at t0+80s. Building a client for connection-2 at t0+95s drops
// connection-1's, which closes its idle connection, and keeps connection-3's pool.
func (s *TransportsSuite) TestAClientUnusedForIdleAfterIsDropped() {
	clock := &clock{now: time.Now()}
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: requestTimeout,
		NewClient: loopback(s.provider.Client()), Now: clock.Now})
	s.Require().NoError(err)
	kept := transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-3"}, s.scheme)
	for _, client := range []*http.Client{transports.Client(s.ref, s.scheme), kept} {
		response, err := s.post(client, s.provider.URL+"/mcp")
		s.Require().NoError(err)
		s.read(response)
	}
	clock.Add(80 * time.Second)
	response, err := s.post(kept, s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.read(response)
	s.Equal(2, s.provider.opened())
	clock.Add(15 * time.Second)

	transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-2"}, s.scheme)

	s.Eventually(func() bool { return s.provider.closed() == 1 }, closedWithin, 10*time.Millisecond)
	response, err = s.post(kept, s.provider.URL+"/mcp")
	s.Require().NoError(err)
	s.read(response)
	s.Equal(2, s.provider.opened(), "connection-3 still has its idle connection")
}

func (s *TransportsSuite) TestTransportsNeedAResolver() {
	_, err := core.NewTransports(core.TransportsConfig{})

	s.Error(err)
}

// transports is Transports whose clients reach s.provider (see loopback).
func (s *TransportsSuite) transports() *core.Transports {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: requestTimeout,
		NewClient: loopback(s.provider.Client())})
	s.Require().NoError(err)
	return transports
}

func (s *TransportsSuite) client() *http.Client {
	return s.transports().Client(s.ref, s.scheme)
}

func (s *TransportsSuite) post(client *http.Client, url string) (*http.Response, error) {
	return s.postWith(context.Background(), client, url)
}

func (s *TransportsSuite) postWith(ctx context.Context, client *http.Client, url string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	s.Require().NoError(err)
	response, err := client.Do(request)
	if err == nil {
		s.T().Cleanup(func() { _ = response.Body.Close() })
	}
	return response, err
}

func (s *TransportsSuite) read(response *http.Response) string {
	raw, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return string(raw)
}

// loopback is egress.NewClient for a test: the same redirect policy, but the TLS server's
// own transport instead of egress's, because egress refuses loopback, where httptest
// listens. Like egress's client, it forwards CloseIdleConnections past the wrap to the
// transport, which a wrap hides from http.Client.
func loopback(server *http.Client) core.NewClientFunc {
	policy := egress.NewClient(0, nil).CheckRedirect
	return func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
		transport := server.Transport.(*http.Transport).Clone()
		var next http.RoundTripper = transport
		if wrap != nil {
			next = wrap(transport)
		}
		return &http.Client{Timeout: timeout, CheckRedirect: policy,
			Transport: idleCloser{RoundTripper: next, transport: transport}}
	}
}

type idleCloser struct {
	http.RoundTripper
	transport *http.Transport
}

func (c idleCloser) CloseIdleConnections() {
	c.transport.CloseIdleConnections()
}

var errNotConnected = errors.New("the connection is not connected")

// memoryResolver is one connection's credential state in memory, with the semantics of
// core.Resolver: Resolve renews a credential that expires before the call's deadline, and
// one the provider refused (CredentialRequest.Refused) while it is still the current one;
// Invalidate moves the connection out of connected only when the refused credential is the
// current one and had not expired.
type memoryResolver struct {
	mu        sync.Mutex
	token     string
	revision  int
	expires   time.Time
	connected bool
	// refresh is the token a renewal of a refused credential gets; empty when nothing
	// renews it, so it comes back the same.
	refresh string
	// refreshRefused makes that renewal fail as a refused refresh does: needs_reauthorization.
	refreshRefused bool
}

func (r *memoryResolver) Resolve(_ context.Context, _ core.ConnectionRef, req core.CredentialRequest) (core.AccessCredential, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.connected {
		return core.AccessCredential{}, errNotConnected
	}
	if req.Refused != nil && req.Refused.Revision == r.revision {
		switch {
		case r.refreshRefused:
			r.connected = false
			return core.AccessCredential{}, errNotConnected
		case r.refresh != "":
			r.token, r.revision = r.refresh, r.revision+1
		}
	}
	if !r.expires.IsZero() && !req.Deadline.IsZero() && !req.Deadline.Before(r.expires) {
		r.token, r.revision, r.expires = "renewed", r.revision+1, req.Deadline.Add(time.Hour)
	}
	secret, _ := json.Marshal(r.token)
	credential := core.NewAccessCredential(signingName, r.expires, secret)
	credential.Revision = r.revision
	return credential, nil
}

func (r *memoryResolver) Invalidate(_ context.Context, _ core.ConnectionRef, rejected core.AccessCredential, why core.Outcome) error {
	if why.Kind != core.OutcomeInvalidGrant && why.Kind != core.OutcomeScopeRequired {
		return errors.New("only invalid_grant and scope_required invalidate")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !rejected.ExpiresAt.IsZero() && !time.Now().Before(rejected.ExpiresAt) {
		return nil
	}
	if r.revision == rejected.Revision {
		r.connected = false
	}
	return nil
}

// renew is another router renewing the stored credentials.
// Revoke ends the grant whatever revision it is at, as a verified provider event does.
func (r *memoryResolver) Revoke(context.Context, core.ConnectionRef, core.SignalKind) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.connected = false
	return nil
}

func (r *memoryResolver) renew(token string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.token, r.revision = token, r.revision+1
}

func (r *memoryResolver) disconnect() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.connected = false
}

func (r *memoryResolver) isConnected() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.connected
}

// clock is the time Transports judges unused clients by, moved by the test.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

const signingName = "signing"

// signingScheme signs each request as a request-signing scheme does (SigV4, for one): a
// digest of the body it is handed and the token, in headers. It keeps what it signed.
type signingScheme struct {
	mu   sync.Mutex
	seen []signedRequest
}

type signedRequest struct {
	contentLength int64
	digest        string
}

func (*signingScheme) Name() string { return signingName }

func (*signingScheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

func (*signingScheme) Complete(context.Context, core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	return core.StoredCredentials{Scheme: signingName}, core.AccountInfo{}, nil
}

func (*signingScheme) Retrieve(context.Context, core.StoredCredentials, core.ResolvedManifest, core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	return core.AccessCredential{}, core.StoredCredentials{}, errors.New("signing: the test's resolver hands out credentials")
}

func (s *signingScheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var token string
	if err := json.Unmarshal(c.Secret(), &token); err != nil {
		panic(err)
	}
	return roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		out := r.Clone(r.Context())
		raw := []byte{}
		if r.Body != nil {
			var err error
			if raw, err = io.ReadAll(r.Body); err != nil {
				return nil, err
			}
			_ = r.Body.Close()
			out.Body = io.NopCloser(bytes.NewReader(raw))
		}
		out.Header.Set("Authorization", "Bearer "+token)
		out.Header.Set(sha256Header, digest(raw))
		s.mu.Lock()
		s.seen = append(s.seen, signedRequest{contentLength: out.ContentLength, digest: digest(raw)})
		s.mu.Unlock()
		return base.RoundTrip(out)
	})
}

// Classify finds a refused credential where RFC 6750 section 3.1 puts it: a Bearer challenge
// with error="invalid_token".
func (*signingScheme) Classify(resp *http.Response, _ []byte, _ error) core.Outcome {
	if resp != nil && strings.Contains(resp.Header.Get("WWW-Authenticate"), `error="invalid_token"`) {
		return core.Outcome{Kind: core.OutcomeInvalidGrant}
	}
	return core.Outcome{Kind: core.OutcomeOK}
}

func (*signingScheme) Revoke(context.Context, core.StoredCredentials, core.ResolvedManifest) error {
	return nil
}

func (s *signingScheme) signed() []signedRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]signedRequest(nil), s.seen...)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// provider is a TLS server on loopback that takes the tokens it accepts, refuses the ones it
// refuses with RFC 6750's invalid_token, and keeps every request it received and every
// connection it saw.
type provider struct {
	*httptest.Server
	mu        sync.Mutex
	accepted  map[string]bool
	refused   map[string]bool
	noReason  bool
	moveTo    string
	before    func()
	requests  []receivedRequest
	connsSeen map[net.Conn]bool
	connsGone int
}

type receivedRequest struct {
	token            string
	body             string
	contentLength    int64
	signatureMatches bool
}

func newProvider(t *testing.T) *provider {
	p := &provider{accepted: map[string]bool{"token-1": true}, refused: map[string]bool{}, connsSeen: map[net.Conn]bool{}}
	p.Server = httptest.NewUnstartedServer(http.HandlerFunc(p.serve))
	p.Config.ConnState = func(conn net.Conn, state http.ConnState) {
		p.mu.Lock()
		defer p.mu.Unlock()
		switch state {
		case http.StateNew:
			p.connsSeen[conn] = true
		case http.StateClosed:
			p.connsGone++
		}
	}
	p.StartTLS()
	t.Cleanup(p.Close)
	return p
}

func (p *provider) serve(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	p.mu.Lock()
	p.requests = append(p.requests, receivedRequest{token: token, body: string(raw), contentLength: r.ContentLength,
		signatureMatches: r.Header.Get(sha256Header) == digest(raw)})
	before, moveTo, refused, noReason, accepted := p.before, p.moveTo, p.refused[token], p.noReason, p.accepted[token]
	p.moveTo = ""
	p.mu.Unlock()
	if before != nil {
		before()
	}
	switch {
	case moveTo != "":
		// RFC 9110 section 15.4.8: 307 keeps the method and the body.
		http.Redirect(w, r, moveTo, http.StatusTemporaryRedirect)
	case noReason:
		w.Header().Set("WWW-Authenticate", "Bearer")
		w.WriteHeader(http.StatusUnauthorized)
	case refused || !accepted:
		w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("refused"))
	default:
		_, _ = w.Write([]byte("ok"))
	}
}

func (p *provider) accept(token string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.accepted[token] = true
}

func (p *provider) refuse(tokens ...string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, token := range tokens {
		p.refused[token] = true
	}
}

func (p *provider) refuseWithoutReason() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.noReason = true
}

// redirect answers the next request with a 307 to url.
func (p *provider) redirect(url string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.moveTo = url
}

// onRequest runs fn when each request arrives, before it is answered.
func (p *provider) onRequest(fn func()) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.before = fn
}

func (p *provider) received() []receivedRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]receivedRequest(nil), p.requests...)
}

func (p *provider) tokens() []string {
	var tokens []string
	for _, r := range p.received() {
		tokens = append(tokens, r.token)
	}
	return tokens
}

func (p *provider) opened() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.connsSeen)
}

func (p *provider) closed() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.connsGone
}

func digest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
