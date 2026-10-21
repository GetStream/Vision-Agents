//go:build integration

package egress_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ProxySuite runs the attacks A1-A14 a proxy can be put through without a sandbox: its hosts
// are httptest TLS servers behind their own CA, and the sandbox is an http.Client whose
// HTTPS_PROXY is the proxy URL with the sandbox's credential.
type ProxySuite struct {
	suite.Suite
	ctx   context.Context
	redis rueidis.Client
	store *egress.Store
	// caCert and caKey are the router's CA, which the sandbox trusts; upstream is the
	// hosts' CA, which the proxy trusts.
	caCert, caKey []byte
	trusted       *x509.CertPool
	upstream      *x509.CertPool
	granted       *host
	outside       *host

	proxy *httptest.Server
	logs  *lockedBuffer
	mu    sync.Mutex
	// token is what the connection gives now, failure the error it gives instead, asked how
	// many times it was asked.
	token   string
	failure error
	asked   int
	// handed are every token the test gave out, and read every body the sandbox read: none
	// may reach a log line or the sandbox.
	handed []string
	read   []string
	// addresses are every host and port the proxy dialed.
	addresses []string

	session, credential, placeholder string
}

func TestProxySuite(t *testing.T) {
	suite.Run(t, new(ProxySuite))
}

func (s *ProxySuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	s.ctx = context.Background()
	var err error
	s.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.redis.Close)
	s.store = egress.NewStore(s.redis)

	routerCA, caCert, caKey := newCA(s.T(), "router sandbox CA")
	s.caCert, s.caKey = caCert, keyPEM(s.T(), caKey)
	s.trusted = x509.NewCertPool()
	s.trusted.AddCert(routerCA)
	upstreamCA, _, upstreamKey := newCA(s.T(), "hosts CA")
	s.upstream = x509.NewCertPool()
	s.upstream.AddCert(upstreamCA)
	s.granted = newHost(s.T(), upstreamCA, upstreamKey, "*.github.test", "github.test")
	s.outside = newHost(s.T(), upstreamCA, upstreamKey, "outside.test")
}

func (s *ProxySuite) SetupTest() {
	s.granted.forget()
	s.outside.forget()
	s.logs = &lockedBuffer{}
	s.token, s.failure, s.asked, s.handed, s.read, s.addresses = "", nil, 0, nil, nil, nil
	s.connect("gho_first-token-of-the-connection")
	s.session, s.credential, s.placeholder = "session-"+uuid.NewString(), egress.NewCredential(), "va_ph_"+uuid.NewString()
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")}})
	s.proxy = httptest.NewServer(s.newProxy(s.store))
	s.T().Cleanup(s.proxy.Close)
}

// TearDownTest is A13: no token reached a log line or anything the sandbox read.
func (s *ProxySuite) TearDownTest() {
	for _, token := range s.handed {
		s.NotContains(s.logs.String(), token, "a log line")
		for _, body := range s.read {
			s.NotContains(body, token, "what the sandbox read")
		}
	}
}

func (s *ProxySuite) newProxy(store *egress.Store) *egress.Proxy {
	dial := &net.Dialer{}
	proxy, err := egress.New(egress.Options{Store: store, Tokens: s.tokens, CACert: s.caCert, CAKey: s.caKey,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: s.upstream},
			DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
				s.mu.Lock()
				s.addresses = append(s.addresses, address)
				s.mu.Unlock()
				to := s.granted
				if strings.HasPrefix(address, "outside.test:") {
					to = s.outside
				}
				return dial.DialContext(ctx, network, to.server.Listener.Addr().String())
			},
		},
		Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	return proxy
}

func (s *ProxySuite) tokens(_ context.Context, grant egress.Grant) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.asked++
	if grant.Binding != "github" || s.failure != nil {
		return "", errors.Join(s.failure, errors.New("egress test: no token for "+grant.Binding))
	}
	return s.token, nil
}

// connect has the connection give token from now on.
func (s *ProxySuite) connect(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.token, s.handed = token, append(s.handed, token)
}

func (s *ProxySuite) dialed() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.addresses...)
}

func (s *ProxySuite) fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failure = err
}

func (s *ProxySuite) grant(placeholder string, hosts ...string) egress.Grant {
	return egress.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-1"}, Binding: "github",
		Scheme: "bearer", Placeholder: placeholder, Hosts: hosts}
}

func (s *ProxySuite) put(credential string, record egress.Record) {
	s.Require().NoError(s.store.Put(s.ctx, credential, record, time.Minute))
	s.T().Cleanup(func() { _ = s.store.Delete(context.Background(), credential) })
}

// sandbox is code in the sandbox: a client with HTTPS_PROXY set to the proxy and the
// router's CA trusted, keeping Authorization on a redirect as curl --location-trusted does.
func (s *ProxySuite) sandbox(credential string) *http.Client {
	proxyURL, err := url.Parse(s.proxy.URL)
	s.Require().NoError(err)
	proxyURL.User = url.UserPassword("sbx", credential)
	return &http.Client{
		Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL), TLSClientConfig: &tls.Config{RootCAs: s.trusted}},
		Timeout:   10 * time.Second,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			request.Header.Set("Authorization", via[0].Header.Get("Authorization"))
			return nil
		},
	}
}

// send is one request of the sandbox's, its answer read whole.
func (s *ProxySuite) send(client *http.Client, request *http.Request) (*http.Response, string, error) {
	response, err := client.Do(request)
	if err != nil {
		return nil, "", err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	header, err := json.Marshal(response.Header)
	s.Require().NoError(err)
	s.read = append(s.read, string(body), string(header))
	return response, string(body), nil
}

func (s *ProxySuite) request(method, target, authorization, body string) *http.Request {
	request, err := http.NewRequestWithContext(s.ctx, method, target, strings.NewReader(body))
	s.Require().NoError(err)
	if authorization != "" {
		request.Header.Set("Authorization", authorization)
	}
	return request
}

// get is a GET to the granted host's echo with authorization, by the test's sandbox.
func (s *ProxySuite) get(authorization string) (*http.Response, string) {
	response, body, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://api.github.test/echo", authorization, ""))
	s.Require().NoError(err)
	return response, body
}

// again is the status of a GET with the placeholder on client, in the tunnel it keeps open.
func (s *ProxySuite) again(client *http.Client) int {
	response, _, err := s.send(client, s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, ""))
	s.Require().NoError(err)
	return response.StatusCode
}

// connectStatus is the proxy's answer to a CONNECT to target with proxyAuthorization.
func (s *ProxySuite) connectStatus(target, proxyAuthorization string) *http.Response {
	conn, err := net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = conn.Close() })
	_, err = fmt.Fprintf(conn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\nProxy-Authorization: %s\r\n\r\n", target, target, proxyAuthorization)
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	s.Require().NoError(err)
	return response
}

func basic(user, password string) string {
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(user+":"+password))
}

// TestABearerPlaceholderReachesTheGrantedHostAsTheToken is A1 and A5: the host gets the
// token, and its echo of it, in a header and in the gzip body it was asked for, comes back
// to the sandbox as the placeholder.
func (s *ProxySuite) TestABearerPlaceholderReachesTheGrantedHostAsTheToken() {
	response, body := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Require().Len(s.granted.requests(), 1)
	seen := s.granted.requests()[0]
	s.Equal("Bearer gho_first-token-of-the-connection", seen.Authorization)
	s.Equal("api.github.test", seen.Host)
	s.Equal("Bearer "+s.placeholder, response.Header.Get("X-Echo-Authorization"))
	s.Contains(body, `"authorization":"Bearer `+s.placeholder+`"`)
	s.Contains(s.logs.String(), "binding=github")
}

// TestAnEchoOfAnotherLengthIsSentWhole: a host that does not compress sends a
// Content-Length, which the rewrite changes.
func (s *ProxySuite) TestAnEchoOfAnotherLengthIsSentWhole() {
	response, body, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://api.github.test/plain", "Bearer "+s.placeholder, ""))

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.JSONEq(`{"authorization":"Bearer `+s.placeholder+`"}`, body)
}

// TestHostIsReadWithoutCase: a name's case is no part of it (RFC 4343).
func (s *ProxySuite) TestHostIsReadWithoutCase() {
	request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
	request.Host = "API.GitHub.test"

	response, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"Bearer gho_first-token-of-the-connection"}, s.granted.authorizations())
}

// TestTheHostGetsTheTunnelsHostWhateverHostSays: the request goes where CONNECT went, port
// 443, whatever port Host names.
func (s *ProxySuite) TestTheHostGetsTheTunnelsHostWhateverHostSays() {
	request := s.request(http.MethodGet, "https://api.github.test/echo", "", "")
	request.Host = "api.github.test:9999"
	_, _, err := s.send(s.sandbox(s.credential), request)
	s.Require().NoError(err)

	s.Equal([]string{"api.github.test"}, s.granted.hosts())
	s.Equal([]string{"api.github.test:443"}, s.dialed())
}

// TestAGrantedHostOnAnotherPortIsRefusedAndNeverReached: a grant is for port 443 only.
func (s *ProxySuite) TestAGrantedHostOnAnotherPortIsRefusedAndNeverReached() {
	for _, target := range []string{"api.github.test:8443", "api.github.test:22", "api.github.test:80", "api.github.test:0443"} {
		s.Equal(http.StatusForbidden, s.connectStatus(target, basic("sbx", s.credential)).StatusCode, target)
	}
	_, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://api.github.test:8443/echo", "Bearer "+s.placeholder, ""))

	s.Error(err)
	s.Empty(s.dialed())
	s.Empty(s.granted.authorizations())
	s.Zero(s.asked)
}

// TestTheTokenSchemeAndABasicPasswordAreSubstitutedToo is A12 for what a proxy takes: the
// token in GitHub's token form and in git's Basic password, whose encoded echo comes back
// as the placeholder's.
func (s *ProxySuite) TestTheTokenSchemeAndABasicPasswordAreSubstitutedToo() {
	_, body := s.get("token " + s.placeholder)
	response, _ := s.get(basic("x-access-token", s.placeholder))

	s.Contains(body, `"authorization":"token `+s.placeholder+`"`)
	s.Equal([]string{"token gho_first-token-of-the-connection", basic("x-access-token", "gho_first-token-of-the-connection")},
		s.granted.authorizations())
	s.Equal(basic("x-access-token", s.placeholder), response.Header.Get("X-Echo-Authorization"))
}

// TestAPlaceholderInAnyOtherFormIsSentAsItIs is A12: no other form of Authorization is
// read, an encoded or split placeholder included.
func (s *ProxySuite) TestAPlaceholderInAnyOtherFormIsSentAsItIs() {
	sent := []string{
		"Bearer " + s.placeholder + "x",
		"Bearer  " + s.placeholder,
		"Digest " + s.placeholder,
		"Bearer " + base64.StdEncoding.EncodeToString([]byte(s.placeholder)),
		basic(s.placeholder, "x"),
		basic("x", s.placeholder) + "!",
	}
	for _, authorization := range sent {
		s.get(authorization)
	}

	s.Equal(sent, s.granted.authorizations())
	s.Zero(s.asked)
}

// TestThePlaceholderInTheBodyQueryOrAnotherHeaderIsSentAsItIs is A3 and A4.
func (s *ProxySuite) TestThePlaceholderInTheBodyQueryOrAnotherHeaderIsSentAsItIs() {
	request := s.request(http.MethodPost, "https://api.github.test/echo?key="+s.placeholder, "Bearer "+s.placeholder, "token="+s.placeholder)
	request.Header.Set("X-Token", s.placeholder)

	_, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Require().Len(s.granted.requests(), 1)
	seen := s.granted.requests()[0]
	s.Equal("key="+s.placeholder, seen.Query)
	s.Equal("token="+s.placeholder, seen.Body)
	s.Equal(s.placeholder, seen.Token)
	s.Equal("Bearer gho_first-token-of-the-connection", seen.Authorization)
}

// TestAHostOutsideTheGrantsIsRefusedAndNeverReached is A2.
func (s *ProxySuite) TestAHostOutsideTheGrantsIsRefusedAndNeverReached() {
	_, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://outside.test/echo", "Bearer "+s.placeholder, ""))

	s.Error(err)
	for _, target := range []string{"outside.test:443", "evilapi.github.test:443", "uploads.api.github.test:443"} {
		s.Equal(http.StatusForbidden, s.connectStatus(target, basic("sbx", s.credential)).StatusCode, target)
	}
	s.Empty(s.outside.authorizations())
	s.Zero(s.asked)
}

// TestAWildcardGrantTakesTheDomainAndTheNamesUnderItOnly: *. as core.SandboxRule reads it.
func (s *ProxySuite) TestAWildcardGrantTakesTheDomainAndTheNamesUnderItOnly() {
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "*.github.test")}})
	status := func(target string) int { return s.connectStatus(target, basic("sbx", s.credential)).StatusCode }

	s.Equal(http.StatusOK, status("uploads.github.test:443"))
	s.Equal(http.StatusOK, status("a.b.github.test:443"))
	s.Equal(http.StatusOK, status("GitHub.test:443"))
	s.Equal(http.StatusForbidden, status("evilgithub.test:443"))
	s.Equal(http.StatusForbidden, status("github.test.outside.test:443"))
}

// TestARedirectToAnOutsideHostCarriesNoToken is A6: the sandbox follows with the
// placeholder, and the proxy opens no tunnel to the outside host.
func (s *ProxySuite) TestARedirectToAnOutsideHostCarriesNoToken() {
	_, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://api.github.test/redirect", "Bearer "+s.placeholder, ""))

	s.Error(err)
	s.Len(s.granted.authorizations(), 1)
	s.Empty(s.outside.authorizations())
}

// TestAnotherHostNameInsideAGrantedTunnelIsRefused is A7: Host names another site than the
// one CONNECT opened.
func (s *ProxySuite) TestAnotherHostNameInsideAGrantedTunnelIsRefused() {
	request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
	request.Host = "outside.test"

	response, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusMisdirectedRequest, response.StatusCode)
	s.Empty(s.granted.authorizations())
	s.Empty(s.outside.authorizations())
}

// TestThePlaceholderOfAGrantForAnotherHostIsSentAsItIs: the sandbox's own placeholder for
// another host's connection gets no token here.
func (s *ProxySuite) TestThePlaceholderOfAGrantForAnotherHostIsSentAsItIs() {
	elsewhere := "va_ph_" + uuid.NewString()
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test"),
		s.grant(elsewhere, "uploads.github.test")}})

	s.get("Bearer " + elsewhere)

	s.Equal([]string{"Bearer " + elsewhere}, s.granted.authorizations())
	s.Zero(s.asked)
}

// TestAnotherSessionsPlaceholderIsSentAsItIs is A8.
func (s *ProxySuite) TestAnotherSessionsPlaceholderIsSentAsItIs() {
	other, otherPlaceholder := egress.NewCredential(), "va_ph_"+uuid.NewString()
	s.put(other, egress.Record{Session: "session-other", Grants: []egress.Grant{s.grant(otherPlaceholder, "api.github.test")}})

	s.get("Bearer " + otherPlaceholder)

	s.Equal([]string{"Bearer " + otherPlaceholder}, s.granted.authorizations())
	s.Zero(s.asked)
}

// TestAGrantThatEndsEndsItsOpenTunnelToo is A9 and A14: a grant withdrawn or deleted
// mid-tunnel stops the next request on it, and the credential opens no tunnel after Delete.
func (s *ProxySuite) TestAGrantThatEndsEndsItsOpenTunnelToo() {
	client := s.sandbox(s.credential)
	s.Require().Equal(http.StatusOK, s.again(client))

	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "uploads.github.test")}})
	withdrawn := s.again(client)
	s.Require().NoError(s.store.Delete(s.ctx, s.credential))
	deleted := s.again(client)

	s.Equal(http.StatusForbidden, withdrawn)
	s.Equal(http.StatusForbidden, deleted)
	s.Len(s.granted.authorizations(), 1)
	s.Equal(http.StatusProxyAuthRequired, s.connectStatus("api.github.test:443", basic("sbx", s.credential)).StatusCode)
}

// TestACredentialThatIsNotASandboxsIsAskedForOne is A14: none, one never issued, one in
// another scheme and one expired get 407 with the challenge RFC 9110 asks for.
func (s *ProxySuite) TestACredentialThatIsNotASandboxsIsAskedForOne() {
	expired := egress.NewCredential()
	s.Require().NoError(s.store.Put(s.ctx, expired, egress.Record{Session: s.session,
		Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")}}, time.Millisecond))
	s.Eventually(func() bool {
		return s.connectStatus("api.github.test:443", basic("sbx", expired)).StatusCode == http.StatusProxyAuthRequired
	}, time.Second, 10*time.Millisecond)

	for _, authorization := range []string{"", basic("sbx", egress.NewCredential()), basic("sbx", s.credential) + "!",
		"Bearer " + base64.StdEncoding.EncodeToString([]byte("sbx:"+s.credential))} {
		response := s.connectStatus("api.github.test:443", authorization)
		s.Equal(http.StatusProxyAuthRequired, response.StatusCode, authorization)
		s.Equal(`Basic realm="sandbox egress"`, response.Header.Get("Proxy-Authenticate"))
	}
	s.Equal(http.StatusOK, s.connectStatus("api.github.test:443", basic("anyone", s.credential)).StatusCode)

	// No credential is none, even if one were stored for it.
	s.put("", egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")}})
	s.Equal(http.StatusProxyAuthRequired, s.connectStatus("api.github.test:443", "").StatusCode)
	s.Equal(http.StatusProxyAuthRequired, s.connectStatus("api.github.test:443", basic("sbx", "")).StatusCode)
}

// TestRedisHoldsNoCredential: what Redis keeps opens no tunnel.
func (s *ProxySuite) TestRedisHoldsNoCredential() {
	keys, err := s.redis.Do(s.ctx, s.redis.B().Keys().Pattern("sandbox:egress:*").Build()).AsStrSlice()
	s.Require().NoError(err)
	s.Require().NotEmpty(keys)
	for _, key := range keys {
		value, err := s.redis.Do(s.ctx, s.redis.B().Get().Key(key).Build()).ToString()
		s.Require().NoError(err)
		s.NotContains(key+value, s.credential)
	}
}

// TestTheTokenIsAskedForOnEveryRequest is A10: a new token is used on the next request of
// an open tunnel, and a connection that gives none is a 401 the host never sees.
func (s *ProxySuite) TestTheTokenIsAskedForOnEveryRequest() {
	client := s.sandbox(s.credential)
	s.again(client)
	s.connect("gho_second-token-of-the-connection")
	s.again(client)

	s.fail(fmt.Errorf("%w: egress test: disconnected", sandbox.ErrNoCredential))
	disconnected := s.again(client)
	s.fail(errors.New("egress test: the provider could not renew just now"))
	unavailable := s.again(client)

	s.Equal([]string{"Bearer gho_first-token-of-the-connection", "Bearer gho_second-token-of-the-connection"}, s.granted.authorizations())
	s.Equal(4, s.asked)
	s.Equal(http.StatusUnauthorized, disconnected)
	s.Equal(http.StatusBadGateway, unavailable)
}

// TestAnIPLiteralIsRefusedEvenWhenAGrantNamesIt is A11: an address is never a granted host.
func (s *ProxySuite) TestAnIPLiteralIsRefusedEvenWhenAGrantNamesIt() {
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "127.0.0.1", "::1")}})

	s.Equal(http.StatusForbidden, s.connectStatus("127.0.0.1:443", basic("sbx", s.credential)).StatusCode)
	s.Equal(http.StatusForbidden, s.connectStatus("[::1]:443", basic("sbx", s.credential)).StatusCode)
}

// TestOnlyConnectIsProxied: plain HTTP would carry the placeholder, and the token, in the clear.
func (s *ProxySuite) TestOnlyConnectIsProxied() {
	response, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "http://api.github.test/echo", "Bearer "+s.placeholder, ""))

	s.Require().NoError(err)
	s.Equal(http.StatusMethodNotAllowed, response.StatusCode)
	s.Empty(s.granted.authorizations())
}

// TestAnUpgradeIsRefused: an upgraded connection would skip the response rewrite.
func (s *ProxySuite) TestAnUpgradeIsRefused() {
	request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")

	response, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusNotImplemented, response.StatusCode)
	s.Empty(s.granted.authorizations())
}

// TestATLSHelloSentWithTheConnectIsKept: a client that does not wait for the 200 still
// gets its handshake.
func (s *ProxySuite) TestATLSHelloSentWithTheConnectIsKept() {
	conn, err := net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	defer conn.Close()
	s.Require().NoError(conn.SetDeadline(time.Now().Add(5 * time.Second)))
	eager := &eagerConn{Conn: conn, connect: fmt.Sprintf("CONNECT api.github.test:443 HTTP/1.1\r\nHost: api.github.test:443\r\n"+
		"Proxy-Authorization: %s\r\n\r\n", basic("sbx", s.credential))}
	eager.reader = bufio.NewReader(conn)

	s.Require().NoError(tls.Client(eager, &tls.Config{ServerName: "api.github.test", RootCAs: s.trusted}).HandshakeContext(s.ctx))
}

// TestAGrantsUnreadableIsAnErrorNotARefusalOfTheSandbox: Redis down is 503, not 407.
func (s *ProxySuite) TestAGrantsUnreadableIsAnErrorNotARefusalOfTheSandbox() {
	closed, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	closed.Close()
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(closed)))
	s.T().Cleanup(s.proxy.Close)

	s.Equal(http.StatusServiceUnavailable, s.connectStatus("api.github.test:443", basic("sbx", s.credential)).StatusCode)
}

// TestGrantsUnreadableInAnOpenTunnelAreAnError: Redis lost after CONNECT is 503 too, and
// the host is not reached.
func (s *ProxySuite) TestGrantsUnreadableInAnOpenTunnelAreAnError() {
	lost, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(lost)))
	s.T().Cleanup(s.proxy.Close)
	client := s.sandbox(s.credential)
	s.Require().Equal(http.StatusOK, s.again(client))

	lost.Close()

	s.Equal(http.StatusServiceUnavailable, s.again(client))
	s.Len(s.granted.authorizations(), 1)
}

func (s *ProxySuite) TestAGrantWithoutAPlaceholderOrAHostIsNotStored() {
	s.Error(s.store.Put(s.ctx, egress.NewCredential(), egress.Record{Grants: []egress.Grant{s.grant("", "api.github.test")}}, time.Minute))
	s.Error(s.store.Put(s.ctx, egress.NewCredential(), egress.Record{Grants: []egress.Grant{s.grant(s.placeholder)}}, time.Minute))
}

func (s *ProxySuite) TestAProxyNeedsACAToSignWith() {
	_, leafCert, leafKey := newLeaf(s.T())
	_, err := egress.New(egress.Options{CACert: leafCert, CAKey: leafKey})
	s.ErrorContains(err, "not a CA's")
	_, err = egress.New(egress.Options{CACert: s.caCert, CAKey: leafKey})
	s.Error(err)
}

// host is a granted or outside host: what it saw, and an echo of it.
type host struct {
	server *httptest.Server
	mu     sync.Mutex
	seen   []seen
}

type seen struct {
	Host, Authorization, Token, Query, Body string
}

func newHost(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey, names ...string) *host {
	h := &host{}
	h.server = httptest.NewUnstartedServer(http.HandlerFunc(h.serve))
	h.server.TLS = &tls.Config{Certificates: []tls.Certificate{sign(t, ca, caKey, names...)}}
	h.server.StartTLS()
	t.Cleanup(h.server.Close)
	return h
}

// serve echoes the request back, gzipped when the client asks except on /plain, and
// redirects /redirect to the outside host.
func (h *host) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	got := seen{Host: r.Host, Authorization: r.Header.Get("Authorization"), Token: r.Header.Get("X-Token"),
		Query: r.URL.RawQuery, Body: string(body)}
	h.mu.Lock()
	h.seen = append(h.seen, got)
	h.mu.Unlock()
	if r.URL.Path == "/redirect" {
		http.Redirect(w, r, "https://outside.test/landing", http.StatusFound)
		return
	}
	w.Header().Set("X-Echo-Authorization", got.Authorization)
	echo, _ := json.Marshal(map[string]string{"authorization": got.Authorization})
	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") && r.URL.Path != "/plain" {
		w.Header().Set("Content-Encoding", "gzip")
		zipped := gzip.NewWriter(w)
		_, _ = zipped.Write(echo)
		_ = zipped.Close()
		return
	}
	_, _ = w.Write(echo)
}

func (h *host) forget() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.seen = nil
}

func (h *host) authorizations() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var all []string
	for _, got := range h.seen {
		all = append(all, got.Authorization)
	}
	return all
}

func (h *host) hosts() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var all []string
	for _, got := range h.seen {
		all = append(all, got.Host)
	}
	return all
}

func (h *host) requests() []seen {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]seen(nil), h.seen...)
}

// eagerConn sends the CONNECT with the TLS hello in one write, and reads the proxy's 200
// before the handshake's answer.
type eagerConn struct {
	net.Conn
	connect string
	reader  *bufio.Reader
	sent    bool
	opened  bool
}

func (c *eagerConn) Write(p []byte) (int, error) {
	if c.sent {
		return c.Conn.Write(p)
	}
	c.sent = true
	if _, err := c.Conn.Write(append([]byte(c.connect), p...)); err != nil {
		return 0, err
	}
	return len(p), nil
}

func (c *eagerConn) Read(p []byte) (int, error) {
	if !c.opened {
		c.opened = true
		response, err := http.ReadResponse(c.reader, nil)
		if err != nil {
			return 0, err
		}
		if response.StatusCode != http.StatusOK {
			return 0, errors.New("egress test: CONNECT answered " + response.Status)
		}
	}
	return c.reader.Read(p)
}

// newCA is a CA certificate and its key, in PEM too.
func newCA(t *testing.T, name string) (*x509.Certificate, []byte, *ecdsa.PrivateKey) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: name},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return cert, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), key
}

// keyPEM is key in PEM.
func keyPEM(t *testing.T, key *ecdsa.PrivateKey) []byte {
	der, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: der})
}

// sign is a server certificate for names, signed by ca.
func sign(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey, names ...string) tls.Certificate {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(2), Subject: pkix.Name{CommonName: names[0]}, DNSNames: names,
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}
	der, err := x509.CreateCertificate(rand.Reader, template, ca, &key.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

// newLeaf is a certificate that is no CA's, and its key, in PEM.
func newLeaf(t *testing.T) (tls.Certificate, []byte, []byte) {
	ca, _, caKey := newCA(t, "issuer")
	leaf := sign(t, ca, caKey, "leaf.test")
	return leaf, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: leaf.Certificate[0]}),
		keyPEM(t, leaf.PrivateKey.(*ecdsa.PrivateKey))
}

// lockedBuffer is a log the proxy's goroutines write to while the test reads it.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
