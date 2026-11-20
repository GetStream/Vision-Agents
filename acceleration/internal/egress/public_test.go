package egress

import (
	"bufio"
	"context"
	"encoding/csv"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type EgressSuite struct {
	suite.Suite
}

func TestEgressSuite(t *testing.T) {
	suite.Run(t, new(EgressSuite))
}

func (s *EgressSuite) TestAPublicHTTPSURLPasses() {
	for _, raw := range []string{
		"https://8.8.8.8/mcp",
		"https://8.8.8.8:8443/mcp",
		"https://8.8.8.8:65535/mcp",
		"https://[2606:4700:4700::1111]/mcp",
	} {
		s.NoErrorf(ValidatePublicHTTPSURL(s.T().Context(), raw), "%s should pass", raw)
	}
}

func (s *EgressSuite) TestAURLThatIsNotHTTPSIsRefused() {
	s.requireRefused("http://8.8.8.8/mcp", "javascript:alert(1)", "ftp://8.8.8.8/mcp", "//8.8.8.8/mcp")
}

func (s *EgressSuite) TestUserinfoQueryAndFragmentAreRefused() {
	s.requireRefused(
		"https://user:password@8.8.8.8/mcp",
		"https://8.8.8.8/mcp?token=secret",
		"https://8.8.8.8/mcp?",
		"https://8.8.8.8/mcp#fragment",
	)
}

func (s *EgressSuite) TestAPortOutsideTheTCPRangeIsRefused() {
	for _, raw := range []string{"https://8.8.8.8:0/mcp", "https://8.8.8.8:65536/mcp", "https://8.8.8.8:99999/mcp"} {
		s.ErrorContainsf(ValidatePublicHTTPSURL(s.T().Context(), raw), "port", "%s should be refused", raw)
		_, err := NewClient(5*time.Second, nil).Get(raw)
		s.ErrorContainsf(err, "port", "the client should refuse %s", raw)
	}
}

// Spellings netip does not parse but some resolvers read as IPv4: the cgo resolver turns
// 2130706433 into 127.0.0.1. They are refused before any resolver sees them.
func (s *EgressSuite) TestANumericHostThatIsNotAnIPAddressIsRefused() {
	for _, raw := range []string{
		"https://2130706433/mcp",
		"https://127.1/mcp",
		"https://0x7f.0.0.1/mcp",
		"https://0x7f000001/mcp",
		"https://010.0.0.1/mcp",
		"https://example.123/mcp",
	} {
		s.ErrorContainsf(ValidatePublicHTTPSURL(s.T().Context(), raw), "neither an IP address", "%s should be refused", raw)
		_, err := NewClient(5*time.Second, nil).Get(raw)
		s.ErrorContainsf(err, "neither an IP address", "the client should refuse %s", raw)
	}
}

func (s *EgressSuite) TestPrivateAddressesAreRefused() {
	s.requireRefused(
		"https://10.2.3.4/mcp",
		"https://172.16.0.1/mcp",
		"https://192.168.1.1/mcp",
		"https://100.64.0.1/mcp",
		"https://[fc00::1]/mcp",
		"https://[fd12::1]/mcp",
	)
}

func (s *EgressSuite) TestLoopbackIsRefusedInEveryForm() {
	s.requireRefused(
		"https://127.0.0.1/mcp",
		"https://[::1]/mcp",
		"https://[::ffff:127.0.0.1]/mcp",
		"https://[::127.0.0.1]/mcp",
		"https://[::ffff:0:127.0.0.1]/mcp",
		"https://[64:ff9b::127.0.0.1]/mcp",
		"https://0.0.0.0/mcp",
		"https://[::]/mcp",
	)
}

func (s *EgressSuite) TestLinkLocalAndMetadataAddressesAreRefused() {
	s.requireRefused(
		"https://169.254.169.254/latest/meta-data",
		"https://[::ffff:169.254.169.254]/latest/meta-data",
		"https://[fd00:ec2::254]/latest/meta-data",
		"https://100.100.100.200/latest/meta-data",
		"https://192.0.0.192/latest/meta-data",
		"https://169.254.1.1/mcp",
		"https://[fe80::1]/mcp",
	)
}

// These pass netip's IsGlobalUnicast and IsPrivate checks, so only nonPublicPrefixes
// refuses them: deleting the line that covers one turns this test red.
func (s *EgressSuite) TestAddressesOnlyThePrefixListRefusesAreRefused() {
	s.requireRefused(
		"https://0.0.0.1/mcp",
		"https://192.0.0.192/mcp",
		"https://192.0.2.1/mcp",
		"https://192.88.99.1/mcp",
		"https://198.18.0.1/mcp",
		"https://198.51.100.1/mcp",
		"https://203.0.113.1/mcp",
		"https://240.0.0.1/mcp",
		"https://255.255.255.255/mcp",
		"https://[2001::1]/mcp",
		"https://[2001:db8::1]/mcp",
		"https://[3fff::1]/mcp",
	)
}

// An IPv6 address that carries an IPv4 one inside it, in space that is otherwise global.
func (s *EgressSuite) TestAnIPv6AddressCarryingAPrivateIPv4IsRefused() {
	s.requireRefused(
		"https://[2002:7f00:1::]/mcp",       // 6to4 of 127.0.0.1
		"https://[2002:a9fe:a9fe::]/mcp",    // 6to4 of 169.254.169.254
		"https://[2001:0:4136:e378::1]/mcp", // Teredo
		"https://[64:ff9b::a9fe:a9fe]/mcp",  // NAT64 of 169.254.169.254
	)
}

// Every block the IANA special-purpose registries mark as not globally reachable is
// refused at its first and its last address. The snapshots in testdata were fetched on
// 2026-10-01 from https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry-1.csv
// and https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry-1.csv,
// with line endings normalised to LF. Refresh them to pick up a new registry entry: a
// block nonPublicPrefixes does not cover then fails here by name.
func (s *EgressSuite) TestEveryBlockIANAMarksNotGloballyReachableIsRefused() {
	checked := 0
	for _, path := range []string{"testdata/iana-ipv4-special-registry.csv", "testdata/iana-ipv6-special-registry.csv"} {
		for _, prefix := range s.notGloballyReachable(path) {
			for _, ip := range []netip.Addr{prefix.Addr(), lastAddress(prefix)} {
				s.Falsef(isPublic(ip), "%s in %s (%s) should be refused", ip, prefix, path)
			}
			checked++
		}
	}
	s.GreaterOrEqual(checked, 30, "the registry snapshots should name at least 30 blocks")
}

func (s *EgressSuite) TestAZonedAddressIsRefused() {
	s.requireRefused(
		"https://[fe80::1%25en0]/mcp",
		"https://[64:ff9b::127.0.0.1%25en0]/mcp",
		"https://[2606:4700:4700::1111%25en0]/mcp",
	)
}

func (s *EgressSuite) TestAHostNameIsCheckedAfterResolution() {
	err := ValidatePublicHTTPSURL(s.T().Context(), "https://localhost/mcp")

	s.ErrorContains(err, "resolved to a non-public address")
}

func (s *EgressSuite) TestTheClientRefusesToConnectToALoopbackServer() {
	var hits atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	client := NewClient(5*time.Second, nil)

	for _, target := range []string{server.URL, strings.Replace(server.URL, "127.0.0.1", "localhost", 1)} {
		_, err := client.Get(target)
		s.ErrorContainsf(err, "non-public address", "%s should be refused", target)
	}
	s.Zero(hits.Load())
}

// DNS rebinding: the name is public when it is validated and loopback when it is dialed.
func (s *EgressSuite) TestANameThatRebindsToLoopbackAfterValidationIsRefused() {
	var hits atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	resolver := &answers{byHost: map[string][]string{"api.example.com": {"8.8.8.8", "127.0.0.1"}}}
	policy := testPolicy(server, resolver)

	s.Require().NoError(policy.validate(s.T().Context(), "https://api.example.com/mcp"))
	_, err := policy.client(5*time.Second, nil).Get("https://api.example.com/mcp")

	s.ErrorContains(err, "non-public address")
	s.Zero(hits.Load())
}

// The socket goes to the address that was checked, never back to the name, which a second
// lookup at connect time could answer differently.
func (s *EgressSuite) TestTheClientDialsTheCheckedAddressNotTheName() {
	for target, want := range map[string]string{
		"https://api.example.com/mcp":      "8.8.8.8:443",
		"https://api.example.com:8443/mcp": "8.8.8.8:8443",
	} {
		server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		policy := testPolicy(server, publicAnswers())
		var mu sync.Mutex
		var dialed []string
		connect := policy.dial
		policy.dial = func(ctx context.Context, network, address string) (net.Conn, error) {
			mu.Lock()
			dialed = append(dialed, address)
			mu.Unlock()
			return connect(ctx, network, address)
		}

		response, err := policy.client(5*time.Second, nil).Get(target)
		s.Require().NoErrorf(err, "%s should be reached", target)
		_ = response.Body.Close()

		mu.Lock()
		s.Equalf([]string{want}, dialed, "%s should dial the checked address", target)
		mu.Unlock()
		server.Close()
	}
}

// The URL is refused before the scheme's wrapper runs, so no credential is applied to it.
func (s *EgressSuite) TestTheClientRefusesPlainHTTPBeforeTheWrapperRuns() {
	var hits, wrapped atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	counting := func(next http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			wrapped.Add(1)
			return next.RoundTrip(request)
		})
	}

	_, err := NewClient(5*time.Second, counting).Get(server.URL)

	s.ErrorContains(err, "only HTTPS")
	s.Zero(wrapped.Load())
	s.Zero(hits.Load())
}

func (s *EgressSuite) TestTheClientFollowsARedirectWithinTheOrigin() {
	for _, location := range []string{
		"/mcp/",
		"https://api.example.com/mcp/",
		"https://api.example.com:443/mcp/",
		"https://API.Example.COM/mcp/",
	} {
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/mcp" {
				http.Redirect(w, r, location, http.StatusMovedPermanently)
				return
			}
			_, _ = io.WriteString(w, r.URL.Path)
		}))
		client := testPolicy(server, publicAnswers()).client(5*time.Second, nil)

		response, err := client.Get("https://api.example.com/mcp")
		s.Require().NoErrorf(err, "a redirect to %s should be followed", location)
		body, err := io.ReadAll(response.Body)
		s.Require().NoError(err)
		s.Equalf("/mcp/", string(body), "a redirect to %s should land on /mcp/", location)
		_ = response.Body.Close()
		server.Close()
	}
}

// The server certificate covers *.example.com, so a refused hop here is the redirect
// policy, not TLS.
func (s *EgressSuite) TestTheClientRefusesARedirectOutOfTheOrigin() {
	for _, location := range []string{
		"https://other.example.com/stolen",
		"https://api.example.com:8443/stolen",
		"http://api.example.com/stolen",
	} {
		var hits atomic.Int32
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/stolen" {
				hits.Add(1)
				return
			}
			http.Redirect(w, r, location, http.StatusFound)
		}))
		client := testPolicy(server, publicAnswers()).client(5*time.Second, nil)

		_, err := client.Get("https://api.example.com/mcp")

		s.ErrorIsf(err, errCrossHostRedirect, "a redirect to %s should be refused", location)
		s.Zerof(hits.Load(), "nothing should reach %s", location)
		server.Close()
	}
}

func (s *EgressSuite) TestSameOriginTreatsEquivalentSpellingsAsOne() {
	for _, pair := range [][2]string{
		{"https://example.com/mcp", "https://example.com:443/mcp/"},
		{"https://Example.COM/mcp", "https://example.com/mcp/"},
		{"https://[2606:4700:4700::1111]/mcp", "https://[2606:4700:4700:0:0:0:0:1111]/mcp/"},
		{"https://8.8.8.8/mcp", "https://[::ffff:8.8.8.8]/mcp/"},
	} {
		s.Truef(sameOrigin(mustURL(pair[0]), mustURL(pair[1])), "%s and %s should be one origin", pair[0], pair[1])
	}
	s.False(sameOrigin(mustURL("https://example.com:8443/mcp"), mustURL("https://example.com/mcp")))
	s.False(sameOrigin(mustURL("https://example.com/mcp"), mustURL("https://example.org/mcp")))
}

// net/http turns a POST into a GET on 301, 302 and 303 and drops the body; 307 and 308
// keep both.
func (s *EgressSuite) TestARedirectThatWouldTurnAPostIntoAGetIsRefused() {
	for _, status := range []int{http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther} {
		var hits atomic.Int32
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/mcp/" {
				hits.Add(1)
				return
			}
			http.Redirect(w, r, "/mcp/", status)
		}))
		client := testPolicy(server, publicAnswers()).client(5*time.Second, nil)

		_, err := client.Post("https://api.example.com/mcp", "application/json", strings.NewReader(`{"id":1}`))

		s.ErrorIsf(err, errMethodRedirect, "a POST answered with %d should not be followed", status)
		s.Zero(hits.Load())
		server.Close()
	}
}

func (s *EgressSuite) TestAPostFollowsATemporaryRedirectWithItsBody() {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/mcp" {
			http.Redirect(w, r, "/mcp/", http.StatusTemporaryRedirect)
			return
		}
		body, _ := io.ReadAll(r.Body)
		_, _ = io.WriteString(w, r.Method+" "+string(body))
	}))
	defer server.Close()
	client := testPolicy(server, publicAnswers()).client(5*time.Second, nil)

	response, err := client.Post("https://api.example.com/mcp", "application/json", strings.NewReader(`{"id":1}`))
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)

	s.Require().NoError(err)
	s.Equal(`POST {"id":1}`, string(body))
}

// What wrap adds rides every hop within the origin and never leaves it.
func (s *EgressSuite) TestWhatWrapAddsStaysWithTheOrigin() {
	var stolen atomic.Int32
	seen := make(chan string, 2)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Host == "other.example.com":
			stolen.Add(1)
		case r.URL.Path == "/mcp":
			seen <- r.Header.Get("Authorization")
			http.Redirect(w, r, "/mcp/", http.StatusTemporaryRedirect)
		case r.URL.Path == "/mcp/":
			seen <- r.Header.Get("Authorization")
			http.Redirect(w, r, "https://other.example.com/stolen", http.StatusTemporaryRedirect)
		}
	}))
	defer server.Close()
	bearer := func(next http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			request = request.Clone(request.Context())
			request.Header.Set("Authorization", "Bearer secret")
			return next.RoundTrip(request)
		})
	}
	client := testPolicy(server, publicAnswers()).client(5*time.Second, bearer)

	_, err := client.Get("https://api.example.com/mcp")

	s.ErrorIs(err, errCrossHostRedirect)
	s.Equal("Bearer secret", <-seen)
	s.Equal("Bearer secret", <-seen)
	s.Zero(stolen.Load())
}

// The wrapper does not forward CloseIdleConnections, and the client still reaches the
// inner transport.
func (s *EgressSuite) TestCloseIdleConnectionsClosesThemThroughAWrapper() {
	var closed atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateClosed {
			closed.Add(1)
		}
	}
	server.StartTLS()
	defer server.Close()
	passThrough := func(next http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(next.RoundTrip)
	}
	client := testPolicy(server, publicAnswers()).client(5*time.Second, passThrough)
	response, err := client.Get("https://api.example.com/mcp")
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()

	client.CloseIdleConnections()

	s.Eventually(func() bool { return closed.Load() == 1 }, 2*time.Second, 10*time.Millisecond)
}

// An HTTP/1.1 origin may close a connection it kept alive just as the client sends its next request on it. Go
// sends a GET again on a new connection, but not a POST, whose body it cannot send twice; so a POST never rides a
// connection that has carried a request before.
func (s *EgressSuite) TestAPostNeverRidesAConnectionTheOriginMayBeClosing() {
	origin := s.startOrigin(closeAtReuse, 0, false)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)
	_, err := s.send(client, http.MethodPost, strings.NewReader(`{"id":1}`))
	s.Require().NoError(err)

	status, err := s.send(client, http.MethodPost, strings.NewReader(`{"id":2}`))

	s.Require().NoError(err)
	s.Equal(http.StatusOK, status)
	s.Equal(int32(2), origin.read.Load())
}

// Go itself sends a GET again when the connection it reused turns out closed, so GETs keep sharing connections.
func (s *EgressSuite) TestAGetOnAConnectionTheOriginClosesIsSentAgainByGo() {
	origin := s.startOrigin(closeAtReuse, 0, false)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)
	_, err := s.send(client, http.MethodGet, nil)
	s.Require().NoError(err)

	status, err := s.send(client, http.MethodGet, nil)

	s.Require().NoError(err)
	s.Equal(http.StatusOK, status)
	s.Equal(int32(2), origin.read.Load())
	s.Equal(int32(2), origin.conns.Load())
}

// A POST the origin read and then dropped without an answer may have been applied, so it is never sent again.
func (s *EgressSuite) TestAPostTheOriginReadAndDroppedIsNotSentAgain() {
	origin := s.startOrigin(0, dropAfterRead, false)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)
	_, err := s.send(client, http.MethodGet, nil)
	s.Require().NoError(err)

	_, err = s.send(client, http.MethodPost, strings.NewReader(`{"id":1}`))

	s.Require().Error(err)
	s.Equal(int32(2), origin.read.Load(), "the GET, and the POST once")
}

// GETs to an HTTP/1.1 origin share one connection.
func (s *EgressSuite) TestGetsShareOneConnection() {
	origin := s.startOrigin(0, 0, false)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)

	for range 3 {
		_, err := s.send(client, http.MethodGet, nil)
		s.Require().NoError(err)
	}

	s.Equal(int32(1), origin.conns.Load())
}

// Go sends again what it can replay, so those requests keep sharing a connection.
func (s *EgressSuite) TestRequestsGoCanSendAgainShareAConnection() {
	for _, method := range []string{"", http.MethodHead, http.MethodOptions, http.MethodTrace} {
		s.True(s.reusesAConnection(&http.Request{Method: method, Header: http.Header{}}), "method %q", method)
	}
	s.True(s.reusesAConnection(&http.Request{Method: http.MethodGet, Header: http.Header{}, Body: http.NoBody}), "no body")
	for _, key := range []string{"Idempotency-Key", "X-Idempotency-Key"} {
		request := &http.Request{Method: http.MethodPost, Header: http.Header{key: {"k"}}}
		request.Body, request.GetBody = io.NopCloser(strings.NewReader("{}")), func() (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader("{}")), nil
		}
		s.True(s.reusesAConnection(request), key)
	}
}

// Go sends nothing again whose body it cannot send twice, such as the body the sandbox proxy forwards.
func (s *EgressSuite) TestARequestWithABodyGoCannotSendTwiceRidesANewConnection() {
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		request := &http.Request{Method: method, Header: http.Header{"Idempotency-Key": {"k"}}}
		request.Body = io.NopCloser(strings.NewReader("{}"))
		s.False(s.reusesAConnection(request), method)
	}
}

// An HTTP/2 origin closes a connection with GOAWAY, which tells the client what it did not process, so POSTs to
// it share one connection.
func (s *EgressSuite) TestPostsToAnHTTP2OriginShareOneConnection() {
	origin := s.startOrigin(0, 0, true)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)

	for range 3 {
		_, err := s.send(client, http.MethodPost, strings.NewReader(`{"id":1}`))
		s.Require().NoError(err)
	}

	s.Equal(int32(1), origin.conns.Load())
	client.CloseIdleConnections()
	s.Eventually(func() bool { return origin.closed.Load() == 1 }, 2*time.Second, 10*time.Millisecond)
}

// A client from NewClient keeps every caller's pooling: two POSTs to an HTTP/1.1 origin share one connection, as on
// base 8e68a6c5c, so only the sandbox egress proxy pays a handshake per POST.
func (s *EgressSuite) TestTheDefaultClientSendsPostsOnOneConnection() {
	origin := s.startOrigin(0, 0, false)
	defer origin.close()
	client := testPolicy(origin.server, publicAnswers()).client(5*time.Second, nil)

	for range 3 {
		_, err := s.send(client, http.MethodPost, strings.NewReader(`{"id":1}`))
		s.Require().NoError(err)
	}

	s.Equal(int32(1), origin.conns.Load())
}

// NewUnpooledClient is the opt-in: the same policy as NewClient, with POSTs on a connection each. The test swaps
// defaultPolicy for one that dials the test origin.
func (s *EgressSuite) TestNewUnpooledClientSendsEachPostOnANewConnectionAndNewClientDoesNot() {
	origin := s.startOrigin(0, 0, false)
	defer origin.close()
	saved := defaultPolicy
	defaultPolicy = testPolicy(origin.server, publicAnswers())
	defer func() { defaultPolicy = saved }()

	for range 2 {
		_, err := s.send(NewUnpooledClient(5*time.Second, nil), http.MethodPost, strings.NewReader(`{"id":1}`))
		s.Require().NoError(err)
	}
	s.Equal(int32(2), origin.conns.Load(), "each unpooled client sends on its own connection")
	unpooled := NewUnpooledClient(5*time.Second, nil)
	for range 3 {
		_, err := s.send(unpooled, http.MethodPost, strings.NewReader(`{"id":1}`))
		s.Require().NoError(err)
	}
	s.Equal(int32(5), origin.conns.Load(), "three POSTs of one unpooled client: three connections")
	pooled := NewClient(5*time.Second, nil)
	for range 3 {
		_, err := s.send(pooled, http.MethodPost, strings.NewReader(`{"id":1}`))
		s.Require().NoError(err)
	}
	s.Equal(int32(6), origin.conns.Load(), "three POSTs of one NewClient client: one connection")
}

// reusesAConnection sends a GET and then request to a fresh HTTP/1.1 origin, and reports whether request went on
// the GET's connection.
func (s *EgressSuite) reusesAConnection(request *http.Request) bool {
	origin := s.startOrigin(0, 0, false)
	defer origin.close()
	client := unpooledPolicy(origin.server).client(5*time.Second, nil)
	_, err := s.send(client, http.MethodGet, nil)
	s.Require().NoError(err)
	request.URL = mustURL("https://api.example.com/mcp")
	response, err := client.Do(request)
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	return origin.conns.Load() == 1
}

const (
	// closeAtReuse is an origin that closes a kept-alive connection as the next request on it arrives, unread: its
	// keep-alive timer fired as the client reused the connection.
	closeAtReuse = iota + 1
	// dropAfterRead is an origin that reads the second request in full and closes the connection without an answer.
	dropAfterRead
)

// origin is an origin that answers "ok". read counts the requests it read in full, conns its connections and
// closed those it closed.
type origin struct {
	server              *httptest.Server
	read, conns, closed atomic.Int32
	mu                  sync.Mutex
	held                []net.Conn
}

// startOrigin starts an HTTP/1.1 origin, or an HTTP/2 one with http2. With reuse set, the first connection answers
// its first request and then does what reuse names with the next request on it. With second set, the second
// request the origin gets is read in full and the connection closed without an answer.
func (s *EgressSuite) startOrigin(reuse, second int, http2 bool) *origin {
	o := &origin{}
	var first atomic.Bool
	o.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		n := o.read.Add(1)
		switch {
		case reuse == closeAtReuse && first.CompareAndSwap(false, true):
			conn, buffered := o.hijack(w)
			_, _ = io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
			// The next request's first bytes are on the connection: the client is reusing it.
			_, _ = buffered.Peek(1)
			_ = conn.Close()
		case second == dropAfterRead && n == 2:
			conn, _ := o.hijack(w)
			_ = conn.Close()
		default:
			_, _ = io.WriteString(w, "ok")
		}
	}))
	o.server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			o.conns.Add(1)
		case http.StateClosed:
			o.closed.Add(1)
		}
	}
	o.server.EnableHTTP2 = http2
	o.server.StartTLS()
	return o
}

func (o *origin) hijack(w http.ResponseWriter) (net.Conn, *bufio.Reader) {
	conn, buffered, err := http.NewResponseController(w).Hijack()
	if err != nil {
		panic(err)
	}
	o.mu.Lock()
	o.held = append(o.held, conn)
	o.mu.Unlock()
	return conn, buffered.Reader
}

// close stops the origin and the connections it took over, which the server no longer tracks.
func (o *origin) close() {
	o.server.Close()
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, conn := range o.held {
		_ = conn.Close()
	}
}

// send sends one request to the origin and reads its answer to the end, which returns an HTTP/1.1 connection to
// the pool.
func (s *EgressSuite) send(client *http.Client, method string, body io.Reader) (int, error) {
	request, err := http.NewRequest(method, "https://api.example.com/mcp", body)
	s.Require().NoError(err)
	response, err := client.Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	_, err = io.Copy(io.Discard, response.Body)
	return response.StatusCode, err
}

// A generic or empty User-Agent is what hosts such as Wikimedia refuse; theirs asks for a
// client name and a way to reach whoever runs it.
func (s *EgressSuite) TestTheUserAgentNamesTheRouterAndWhereToReachIt() {
	s.Regexp(`^vision-agents \(https://\S+\)$`, UserAgent)
}

func (s *EgressSuite) requireRefused(raws ...string) {
	for _, raw := range raws {
		s.Errorf(ValidatePublicHTTPSURL(s.T().Context(), raw), "%s should be refused", raw)
	}
}

// notGloballyReachable reads the blocks a registry snapshot marks "Globally Reachable"
// False. A cell can hold several blocks and footnote markers such as "[2]".
func (s *EgressSuite) notGloballyReachable(path string) []netip.Prefix {
	file, err := os.Open(path)
	s.Require().NoError(err)
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	s.Require().NoError(err)
	header := rows[0]
	blockColumn, reachableColumn := -1, -1
	for i, name := range header {
		switch name {
		case "Address Block":
			blockColumn = i
		case "Globally Reachable":
			reachableColumn = i
		}
	}
	s.Require().NotEqual(-1, blockColumn)
	s.Require().NotEqual(-1, reachableColumn)
	var prefixes []netip.Prefix
	for _, row := range rows[1:] {
		if !strings.HasPrefix(row[reachableColumn], "False") {
			continue
		}
		for _, block := range strings.Split(row[blockColumn], ",") {
			block, _, _ = strings.Cut(strings.TrimSpace(block), " ")
			prefix, err := netip.ParsePrefix(block)
			s.Require().NoErrorf(err, "block %q in %s", block, path)
			prefixes = append(prefixes, prefix)
		}
	}
	return prefixes
}

func lastAddress(prefix netip.Prefix) netip.Addr {
	bytes := prefix.Addr().AsSlice()
	for bit := prefix.Bits(); bit < len(bytes)*8; bit++ {
		bytes[bit/8] |= 0x80 >> (bit % 8)
	}
	last, _ := netip.AddrFromSlice(bytes)
	return last
}

// testPolicy answers names from resolver and connects every checked address to server,
// whose certificate covers example.com and *.example.com. Everything else, the redirect
// policy, the per-request check and the dial-time address check, is the real one.
func testPolicy(server *httptest.Server, resolver lookup) policy {
	return policy{
		resolver: resolver,
		dial: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
		tlsConfig: server.Client().Transport.(*http.Transport).TLSClientConfig.Clone(),
	}
}

func unpooledPolicy(server *httptest.Server) policy {
	policy := testPolicy(server, publicAnswers())
	policy.unpooled = true
	return policy
}

func publicAnswers() *answers {
	return &answers{byHost: map[string][]string{"api.example.com": {"8.8.8.8"}, "other.example.com": {"8.8.4.4"}}}
}

// answers resolves each host to its listed addresses in turn, repeating the last one.
// Names match without regard to case, as they do in DNS.
type answers struct {
	mu     sync.Mutex
	byHost map[string][]string
	asked  map[string]int
}

func (a *answers) LookupNetIP(_ context.Context, _ string, host string) ([]netip.Addr, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	host = strings.ToLower(host)
	list, ok := a.byHost[host]
	if !ok {
		return nil, &net.DNSError{Err: "no such host", Name: host, IsNotFound: true}
	}
	if a.asked == nil {
		a.asked = map[string]int{}
	}
	i := min(a.asked[host], len(list)-1)
	a.asked[host]++
	return []netip.Addr{netip.MustParseAddr(list[i])}, nil
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func mustURL(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	if err != nil {
		panic(err)
	}
	return parsed
}

// the unpooled route dials only a checked public address.
func (s *EgressSuite) TestAPostOnTheUnpooledRouteNeverDialsALoopbackAddress() {
	var hits atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	policy := testPolicy(server, &answers{byHost: map[string][]string{"api.example.com": {"127.0.0.1"}}})
	policy.unpooled = true

	_, err := policy.client(5*time.Second, nil).Post("https://api.example.com/mcp", "application/json", strings.NewReader(`{"id":1}`))

	s.ErrorContains(err, "non-public address")
	s.Zero(hits.Load())
}

// the URL check after the wrapper runs before the unpooled route.
func (s *EgressSuite) TestAWrapperRewriteToPlainHTTPIsRefusedOnTheUnpooledRoute() {
	var hits atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	policy := testPolicy(server, publicAnswers())
	policy.unpooled = true
	downgrade := func(next http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			request = request.Clone(request.Context())
			request.URL.Scheme = "http"
			return next.RoundTrip(request)
		})
	}

	_, err := policy.client(5*time.Second, downgrade).Post("https://api.example.com/mcp", "application/json", strings.NewReader(`{"id":1}`))

	s.ErrorContains(err, "only HTTPS")
	s.Zero(hits.Load())
}

// a client from NewUnpooledClient keeps the redirect policy.
func (s *EgressSuite) TestNewUnpooledClientRefusesARedirectOutOfTheOrigin() {
	var stolen atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Host == "other.example.com" {
			stolen.Add(1)
			return
		}
		http.Redirect(w, r, "https://other.example.com/stolen", http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	saved := defaultPolicy
	defaultPolicy = testPolicy(server, publicAnswers())
	defer func() { defaultPolicy = saved }()

	_, err := NewUnpooledClient(5*time.Second, nil).Post("https://api.example.com/mcp", "application/json", strings.NewReader(`{"id":1}`))

	s.ErrorIs(err, errCrossHostRedirect)
	s.Zero(stolen.Load())
}

// neither route sends a request through an environment proxy, which would skip dialPublic.
// The field is read directly: the test is in package egress.
func (s *EgressSuite) TestNeitherRouteUsesAnEnvironmentProxy() {
	pooled := NewClient(5*time.Second, nil).Transport.(clientRoundTripper)
	unpooled := NewUnpooledClient(5*time.Second, nil).Transport.(clientRoundTripper)

	s.Nil(pooled.transport.Proxy)
	s.Nil(unpooled.unpooled.Proxy)
}
