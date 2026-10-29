//go:build integration

package egress_test

import (
	"bufio"
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"maps"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/textproto"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
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
	upstreamCA    *x509.Certificate
	upstreamKey   *ecdsa.PrivateKey
	granted       *host
	outside       *host
	// answer, when set, is the address every host's request is sent to instead.
	answer string

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
	s.upstreamCA, _, s.upstreamKey = newCA(s.T(), "hosts CA")
	s.upstream = x509.NewCertPool()
	s.upstream.AddCert(s.upstreamCA)
	s.granted = newHost(s.T(), s.upstreamCA, s.upstreamKey, "*.github.test", "github.test")
	s.outside = newHost(s.T(), s.upstreamCA, s.upstreamKey, "outside.test")
}

func (s *ProxySuite) SetupTest() {
	s.granted.forget()
	s.outside.forget()
	s.logs = &lockedBuffer{}
	s.token, s.failure, s.asked, s.handed, s.read, s.addresses, s.answer = "", nil, 0, nil, nil, nil, ""
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

// newProxy is a proxy over store; change, when given, changes its options first. Its transport
// and its tunnels to an allowed domain dial the test's hosts.
func (s *ProxySuite) newProxy(store *egress.Store, change ...func(*egress.Options)) *egress.Proxy {
	dialer := &net.Dialer{}
	dial := func(ctx context.Context, network, address string) (net.Conn, error) {
		s.mu.Lock()
		s.addresses = append(s.addresses, address)
		to := s.answer
		s.mu.Unlock()
		switch {
		case to != "":
		case strings.HasPrefix(address, "outside.test:"):
			to = s.outside.server.Listener.Addr().String()
		default:
			to = s.granted.server.Listener.Addr().String()
		}
		return dialer.DialContext(ctx, network, to)
	}
	options := egress.Options{Store: store, Tokens: s.tokens, CACert: s.caCert, CAKey: s.caKey,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: s.upstream}, DialContext: dial}, Dial: dial,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil))}
	for _, change := range change {
		change(&options)
	}
	proxy, err := egress.New(options)
	s.Require().NoError(err)
	return proxy
}

// answering has every host's request answered by handler, over TLS as api.github.test.
func (s *ProxySuite) answering(handler http.HandlerFunc) {
	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{Certificates: []tls.Certificate{sign(s.T(), s.upstreamCA, s.upstreamKey, "*.github.test", "github.test")}}
	server.StartTLS()
	s.T().Cleanup(server.Close)
	s.answerAt(server.Listener.Addr().String())
}

// answeringRaw has every host's request answered with the bytes answer gives for the
// Authorization it carried, then the connection closed: an answer no HTTP server would send.
func (s *ProxySuite) answeringRaw(answer func(authorization string) string) {
	listener, err := tls.Listen("tcp", "127.0.0.1:0",
		&tls.Config{Certificates: []tls.Certificate{sign(s.T(), s.upstreamCA, s.upstreamKey, "*.github.test", "github.test")}})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = listener.Close() })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				request, err := http.ReadRequest(bufio.NewReader(conn))
				if err != nil {
					return
				}
				_, _ = io.WriteString(conn, answer(request.Header.Get("Authorization")))
			}()
		}
	}()
	s.answerAt(listener.Addr().String())
}

func (s *ProxySuite) answerAt(address string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.answer = address
}

// key is where Redis keeps credential's record: its SHA-256.
func key(credential string) string {
	sum := sha256.Sum256([]byte(credential))
	return "sandbox:egress:" + hex.EncodeToString(sum[:])
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

// TestABasicEchoComesBackAsThePlaceholderForAnyUser: the Base64 of user:token is not aligned like
// the Base64 of the token alone, and git's user name is the remote's (x-access-token, git or a
// short one), so the echo of each is put back as the placeholder's own.
func (s *ProxySuite) TestABasicEchoComesBackAsThePlaceholderForAnyUser() {
	token := "gho_first-token-of-the-connection"
	for _, user := range []string{"git", "a"} {
		response, _ := s.get(basic(user, s.placeholder))

		s.Equal(basic(user, s.placeholder), response.Header.Get("X-Echo-Authorization"), user)
		s.NotContains(response.Header.Get("X-Echo-Authorization"), base64.StdEncoding.EncodeToString([]byte(user+":"+token)), user)
	}
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
		basic("x", "a"+s.placeholder),
		basic("x", s.placeholder+"a"),
	}
	for _, authorization := range sent {
		s.get(authorization)
	}

	s.Equal(sent, s.granted.authorizations())
	s.Zero(s.asked)
}

// TestThePlaceholderInTheBodyQueryOrAnotherHeaderIsSentAsItIs is A3 and A4, in a POST the
// grant reads with.
func (s *ProxySuite) TestThePlaceholderInTheBodyQueryOrAnotherHeaderIsSentAsItIs() {
	grant := s.grant(s.placeholder, "api.github.test")
	grant.Reads = []core.SandboxRead{{Host: "api.github.test", Method: http.MethodPost, Path: "/echo"}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
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

// TestAGrantedHostTakesOnlyAGetOrAHead: code in the sandbox reads a granted host. Any other
// method, with the placeholder or without it, reaches no host and asks for no token, and its
// answer tells the model to write with the connector's tools. A method is case-sensitive (RFC
// 9110 section 9.1), so get is not GET.
func (s *ProxySuite) TestAGrantedHostTakesOnlyAGetOrAHead() {
	for _, method := range []string{http.MethodGet, http.MethodHead} {
		response, _, err := s.send(s.sandbox(s.credential), s.request(method, "https://api.github.test/echo", "Bearer "+s.placeholder, ""))
		s.Require().NoError(err, method)
		s.Equal(http.StatusOK, response.StatusCode, method)
	}
	s.Equal([]string{"Bearer gho_first-token-of-the-connection", "Bearer gho_first-token-of-the-connection"}, s.granted.authorizations())
	s.granted.forget()
	asked := s.asked
	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete, http.MethodOptions, "get"} {
		for _, authorization := range []string{"Bearer " + s.placeholder, ""} {
			response, body, err := s.send(s.sandbox(s.credential), s.request(method, "https://api.github.test/repos/o/r/issues", authorization, "{}"))

			s.Require().NoError(err, method)
			s.Equal(http.StatusForbidden, response.StatusCode, method)
			s.Equal("egress: "+method+" api.github.test/repos/o/r/issues refused: code in the sandbox only reads through a connector; "+
				"write with the connector's tools\n", body, method)
		}
	}
	s.Empty(s.granted.requests())
	s.Equal(asked, s.asked)
	s.Contains(s.logs.String(), "refused a request that may write")
}

// TestARefusedMethodIsCountedAndLoggedBounded: a method a granted host does not take is
// counted as denied_method under its bounded label, and the log keeps only its first 32 bytes.
func (s *ProxySuite) TestARefusedMethodIsCountedAndLoggedBounded() {
	long := strings.Repeat("WRITE", 200)
	posted := s.requestCount("api.github.test", "POST", "denied_method", "4xx")
	odd := s.requestCount("api.github.test", "other", "denied_method", "4xx")

	for _, method := range []string{http.MethodPost, long} {
		response, _, err := s.send(s.sandbox(s.credential), s.request(method, "https://api.github.test/echo", "", "{}"))
		s.Require().NoError(err)
		s.Equal(http.StatusForbidden, response.StatusCode)
	}

	s.Equal(posted+1, s.requestCount("api.github.test", "POST", "denied_method", "4xx"))
	s.Equal(odd+1, s.requestCount("api.github.test", "other", "denied_method", "4xx"))
	s.Contains(s.logs.String(), long[:32]+"…")
	s.NotContains(s.logs.String(), long[:33])
}

// TestGitPushIsRefusedAtItsFirstRequest: git push first GETs info/refs for git-receive-pack,
// and git shows the model the body of a refusal there; a clone's GET for git-upload-pack
// reads.
func (s *ProxySuite) TestGitPushIsRefusedAtItsFirstRequest() {
	for target, status := range map[string]int{
		"https://api.github.test/o/r.git/info/refs?service=git-receive-pack":                         http.StatusForbidden,
		"https://api.github.test/o/r/info/refs?service=git-receive-pack":                             http.StatusForbidden,
		"https://api.github.test/o/r.git/info/refs?service=git-upload-pack&service=git-receive-pack": http.StatusForbidden,
		"https://api.github.test/o/r.git/info/refs?service=git-upload-pack":                          http.StatusOK,
		"https://api.github.test/o/r.git/info/refs":                                                  http.StatusOK,
		"https://api.github.test/echo?service=git-receive-pack":                                      http.StatusOK,
	} {
		s.granted.forget()

		response, body, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, target, "Bearer "+s.placeholder, ""))

		s.Require().NoError(err, target)
		s.Equal(status, response.StatusCode, target)
		if status == http.StatusForbidden {
			s.Empty(s.granted.requests(), target)
			s.Contains(body, "refused: code in the sandbox only reads through a connector; write with the connector's tools", target)
		} else {
			s.Len(s.granted.requests(), 1, target)
		}
	}
}

// TestAReadTakesItsMethodToItsPathOnItsHostAlone: a grant's read lets one more method through, to
// the paths its pattern matches on its host, as the host gets them: a path spelled another way
// (an escape, a dot segment, an empty segment) is refused even where its pattern would match.
func (s *ProxySuite) TestAReadTakesItsMethodToItsPathOnItsHostAlone() {
	grant := s.grant(s.placeholder, "api.github.test", "*.github.test")
	grant.Reads = []core.SandboxRead{{Host: "api.github.test", Method: http.MethodPost, Path: "/*/*/git-upload-pack"}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	for target, status := range map[string]int{
		"POST https://api.github.test/o/r.git/git-upload-pack":     http.StatusOK,
		"POST https://api.github.test/o/r/git-upload-pack":         http.StatusOK,
		"POST https://api.github.test/o/r.git/git-receive-pack":    http.StatusForbidden,
		"PUT https://api.github.test/o/r.git/git-upload-pack":      http.StatusForbidden,
		"POST https://api.github.test/o/r.git/git-upload-pack/x":   http.StatusForbidden,
		"POST https://api.github.test/o/x/r.git/git-upload-pack":   http.StatusForbidden,
		"POST https://other.github.test/o/r.git/git-upload-pack":   http.StatusForbidden,
		"POST https://api.github.test/o/r%2Egit/git-upload-pack":   http.StatusForbidden,
		"POST https://api.github.test/o/../git-upload-pack":        http.StatusForbidden,
		"POST https://api.github.test//r.git/git-upload-pack":      http.StatusForbidden,
		"POST https://api.github.test/o/r.git/git-upload-pack?x=1": http.StatusOK,
		"post https://api.github.test/o/r.git/git-upload-pack":     http.StatusForbidden,
	} {
		method, target, _ := strings.Cut(target, " ")
		s.granted.forget()

		response, _, err := s.send(s.sandbox(s.credential), s.request(method, target, "Bearer "+s.placeholder, "0000"))

		s.Require().NoError(err, target)
		s.Equal(status, response.StatusCode, method+" "+target)
		if status == http.StatusOK {
			s.Equal([]string{"Bearer gho_first-token-of-the-connection"}, s.granted.authorizations(), target)
		} else {
			s.Empty(s.granted.requests(), target)
		}
	}
}

// TestAWriteToAnAllowedDomainIsTunnelledAsItIs: the proxy reads nothing in a tunnel to an
// allowed domain, so it takes any method there, as before (a POST reached outside.test on the
// base, 37d4ed61b).
func (s *ProxySuite) TestAWriteToAnAllowedDomainIsTunnelledAsItIs() {
	s.allow("outside.test")

	response, _, err := s.send(s.direct(), s.request(http.MethodPost, "https://outside.test/echo", "", "written"))

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Require().Len(s.outside.requests(), 1)
	s.Equal("written", s.outside.requests()[0].Body)
}

// graphQLRequest is one request of code in the sandbox to a GraphQL read, and the status the
// proxy answers it with.
type graphQLRequest struct {
	name, method, target, contentType, encoding, body string
	status                                            int
}

// graphQLCase is a JSON POST of body to the GraphQL read, answered with status.
func graphQLCase(name, body string, status int) graphQLRequest {
	return graphQLRequest{name: name, method: http.MethodPost, target: "https://api.github.test/graphql",
		contentType: "application/json", body: body, status: status}
}

// readGraphQL gives the test's grant a GraphQL read of POST api.github.test/graphql, and reads,
// when given, beside it.
func (s *ProxySuite) readGraphQL(reads ...core.SandboxRead) {
	grant := s.grant(s.placeholder, "api.github.test")
	grant.Reads = append([]core.SandboxRead{{Host: "api.github.test", Path: "/graphql", GraphQL: true}}, reads...)
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
}

// sendGraphQL sends each request by the sandbox. One the proxy lets through reaches the host
// with the token and its body byte for byte; one it refuses reaches no host, asks for no token,
// and gets read-only's answer.
func (s *ProxySuite) sendGraphQL(requests []graphQLRequest) {
	for _, c := range requests {
		s.granted.forget()
		asked := s.asked
		request := s.request(c.method, c.target, "Bearer "+s.placeholder, c.body)
		if c.contentType != "" {
			request.Header.Set("Content-Type", c.contentType)
		}
		if c.encoding != "" {
			request.Header.Set("Content-Encoding", c.encoding)
		}

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, c.name)
		s.Equal(c.status, response.StatusCode, c.name)
		if c.status != http.StatusOK {
			s.Empty(s.granted.requests(), c.name)
			s.Equal(asked, s.asked, c.name)
			s.Equal("egress: "+c.method+" api.github.test/graphql refused: code in the sandbox only reads through a connector; "+
				"write with the connector's tools\n", body, c.name)
			continue
		}
		s.Require().Len(s.granted.requests(), 1, c.name)
		s.Equal("Bearer gho_first-token-of-the-connection", s.granted.requests()[0].Authorization, c.name)
		s.Equal(c.body, s.granted.requests()[0].Body, c.name)
	}
}

// TestAGraphQLReadTakesADocumentOfQueriesAlone: a POST to a GraphQL read reaches the host only
// when every operation of its document is a query (GraphQL over HTTP,
// https://http-spec.graphql.org/draft/, 4.4 POST). A parser reads the document, so a comment, a
// string or a fragment that says mutation is no mutation, and an operation named query is.
func (s *ProxySuite) TestAGraphQLReadTakesADocumentOfQueriesAlone() {
	s.readGraphQL()

	s.sendGraphQL([]graphQLRequest{
		graphQLCase("a query", `{"query":"query { viewer { login } }"}`, http.StatusOK),
		graphQLCase("the shorthand", `{"query":"{ viewer { login } }"}`, http.StatusOK),
		graphQLCase("operations named, one chosen", `{"query":"query A { a } query B { b }","operationName":"B","variables":{"x":1}}`, http.StatusOK),
		graphQLCase("operationName and variables null", `{"query":"{ a }","operationName":null,"variables":null}`, http.StatusOK),
		graphQLCase("a fragment", `{"query":"query { ...F } fragment F on Query { viewer { login } }"}`, http.StatusOK),
		graphQLCase("mutation in a comment", `{"query":"# mutation { addStar }\nquery { a }"}`, http.StatusOK),
		graphQLCase("mutation in a string", `{"query":"query { search(query: \"mutation { addStar }\") { n } }"}`, http.StatusOK),
		graphQLCase("mutation in a block string", `{"query":"query { search(query: \"\"\"\nmutation { addStar }\"\"\") { n } }"}`, http.StatusOK),
		graphQLCase("a fragment named mutation", `{"query":"query { ...mutation } fragment mutation on Query { a }"}`, http.StatusOK),
		graphQLCase("a key escaped", `{"\u0071uery":"{ a }"}`, http.StatusOK),
		graphQLCase("a key escaped, a mutation", `{"\u0071uery":"mutation { b }"}`, http.StatusForbidden),
		graphQLCase("a mutation", `{"query":"mutation { addStar(input: {starrableId: \"x\"}) { clientMutationId } }"}`, http.StatusForbidden),
		graphQLCase("a mutation named query", `{"query":"mutation query { a }"}`, http.StatusForbidden),
		graphQLCase("a subscription", `{"query":"subscription { a }"}`, http.StatusForbidden),
		graphQLCase("a query and a mutation, the query chosen", `{"query":"query A { a } mutation B { b }","operationName":"A"}`, http.StatusForbidden),
		graphQLCase("the shorthand and a mutation", `{"query":"{ a } mutation { b }"}`, http.StatusForbidden),
		graphQLCase("fragments alone", `{"query":"fragment F on Query { a }"}`, http.StatusForbidden),
		graphQLCase("an empty document", `{"query":""}`, http.StatusForbidden),
		graphQLCase("invalid GraphQL", `{"query":"query { a "}`, http.StatusForbidden),
		graphQLCase("a document of another kind", `{"query":"type Query { a: String }"}`, http.StatusForbidden),
	})
}

// TestAGraphQLReadTakesOneJSONRequestAlone: the body is one JSON request with query and only the
// members the proxy reads, once each. Anything else could hold an operation the host reads
// another way than the proxy: a persisted document's id, a key the host takes the first of, a
// second JSON value. A batch is refused, as GitHub refuses it (400 «Body should be a JSON
// object», live probe 2026-10-27).
func (s *ProxySuite) TestAGraphQLReadTakesOneJSONRequestAlone() {
	s.readGraphQL()

	s.sendGraphQL([]graphQLRequest{
		graphQLCase("a batch of queries", ` [{"query":"{ a }"},{"query":"query { b }"}]`, http.StatusForbidden),
		graphQLCase("a batch with a mutation", `[{"query":"{ a }"},{"query":"mutation { b }"}]`, http.StatusForbidden),
		graphQLCase("an empty batch", `[]`, http.StatusForbidden),
		graphQLCase("a batch of strings", `["{ a }"]`, http.StatusForbidden),
		graphQLCase("a batch of batches", `[[{"query":"{ a }"}]]`, http.StatusForbidden),
		graphQLCase("a batch of a key and a value", `[["query","{ a }"]]`, http.StatusForbidden),
		graphQLCase("an empty body", ``, http.StatusForbidden),
		graphQLCase("GraphQL as it is", `query { a }`, http.StatusForbidden),
		graphQLCase("null", `null`, http.StatusForbidden),
		graphQLCase("a string", `"{ a }"`, http.StatusForbidden),
		graphQLCase("an empty object", `{}`, http.StatusForbidden),
		graphQLCase("a null query", `{"query":null}`, http.StatusForbidden),
		graphQLCase("a number for a query", `{"query":1}`, http.StatusForbidden),
		graphQLCase("an array for a query", `{"query":["{ a }"]}`, http.StatusForbidden),
		graphQLCase("a persisted query's hash", `{"query":"{ a }","extensions":{"persistedQuery":{"sha256Hash":"x"}}}`, http.StatusForbidden),
		graphQLCase("a document's id", `{"query":"{ a }","id":"m1"}`, http.StatusForbidden),
		graphQLCase("query twice, a mutation last", `{"query":"{ a }","query":"mutation { b }"}`, http.StatusForbidden),
		graphQLCase("query twice, a mutation first", `{"query":"mutation { b }","query":"{ a }"}`, http.StatusForbidden),
		graphQLCase("Query beside query", `{"Query":"mutation { b }","query":"{ a }"}`, http.StatusForbidden),
		graphQLCase("a second JSON value", `{"query":"{ a }"}{"query":"mutation { b }"}`, http.StatusForbidden),
		graphQLCase("a JSON object cut short", `{"query":"{ a }"`, http.StatusForbidden),
	})
	s.Contains(s.logs.String(), `why="a GraphQL request that is no JSON object"`)
}

// TestAGraphQLReadTakesAJSONBodyOfAMiBAtMost: the proxy reads a body as application/json in UTF-8
// (4.4 POST), of 1 MiB at most, with no URL query, which a host could read an operation from
// too. A GET is a read as before, with its query in the URL.
func (s *ProxySuite) TestAGraphQLReadTakesAJSONBodyOfAMiBAtMost() {
	s.readGraphQL()
	query := `{"query":"{ a }"}`
	// 1 MiB is egress.maxGraphQLBody; JSON takes whitespace after its value (RFC 8259 section 2).
	full := query + strings.Repeat(" ", 1<<20-len(query))
	withType := func(contentType string, status int) graphQLRequest {
		c := graphQLCase(contentType, query, status)
		c.contentType = contentType
		return c
	}
	zipped := &bytes.Buffer{}
	writer := gzip.NewWriter(zipped)
	_, _ = writer.Write([]byte(query))
	s.Require().NoError(writer.Close())

	s.sendGraphQL([]graphQLRequest{
		withType("application/json; charset=utf-8", http.StatusOK),
		withType("application/json; charset=UTF-8", http.StatusOK),
		withType("Application/JSON", http.StatusOK),
		withType("application/json; charset=iso-8859-1", http.StatusForbidden),
		withType("application/graphql", http.StatusForbidden),
		withType("text/plain", http.StatusForbidden),
		withType("application/x-www-form-urlencoded", http.StatusForbidden),
		withType("", http.StatusForbidden),
		withType("application/json; charset", http.StatusForbidden),
		graphQLCase("a body of 1 MiB", full, http.StatusOK),
		graphQLCase("a byte over 1 MiB", full+" ", http.StatusForbidden),
		graphQLCase("gzip", zipped.String(), http.StatusForbidden),
		{name: "JSON sent as gzip", method: http.MethodPost, target: "https://api.github.test/graphql", contentType: "application/json",
			encoding: "gzip", body: query, status: http.StatusForbidden},
		{name: "JSON sent as identity and gzip", method: http.MethodPost, target: "https://api.github.test/graphql", contentType: "application/json",
			encoding: "identity, gzip", body: query, status: http.StatusForbidden},
		{name: "JSON sent as identity", method: http.MethodPost, target: "https://api.github.test/graphql", contentType: "application/json",
			encoding: "Identity", body: query, status: http.StatusOK},
		{name: "a URL query on a POST", method: http.MethodPost, target: "https://api.github.test/graphql?query=mutation%7Bb%7D",
			contentType: "application/json", body: query, status: http.StatusForbidden},
		{name: "a PUT", method: http.MethodPut, target: "https://api.github.test/graphql", contentType: "application/json", body: query,
			status: http.StatusForbidden},
		{name: "a GET with a query in the URL", method: http.MethodGet, target: "https://api.github.test/graphql?query=%7Ba%7D",
			status: http.StatusOK},
	})
	s.Contains(s.logs.String(), `why="a GraphQL POST over the size the proxy reads"`)
	s.Contains(s.logs.String(), `why="a GraphQL POST with a content coding"`)
}

// TestAGraphQLReadReachesTheHostAsSentWhenItsLengthIsUnknown: a body sent chunked is read
// whole and reaches the host with the same bytes.
func (s *ProxySuite) TestAGraphQLReadReachesTheHostAsSentWhenItsLengthIsUnknown() {
	s.readGraphQL()
	query := `{"query":"query { viewer { login } }"}`
	request := s.request(http.MethodPost, "https://api.github.test/graphql", "Bearer "+s.placeholder, "")
	request.Header.Set("Content-Type", "application/json")
	request.Body, request.ContentLength = io.NopCloser(struct{ io.Reader }{strings.NewReader(query)}), -1

	response, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Require().Len(s.granted.requests(), 1)
	s.Equal(query, s.granted.requests()[0].Body)
}

// TestAGraphQLReadHoldsBesideAnotherReadOfItsPath: a GraphQL read beside a read of the same
// method and path still takes queries alone, and the refusal's log names why, never the body.
func (s *ProxySuite) TestAGraphQLReadHoldsBesideAnotherReadOfItsPath() {
	s.readGraphQL(core.SandboxRead{Host: "api.github.test", Method: http.MethodPost, Path: "/*"})

	s.sendGraphQL([]graphQLRequest{
		graphQLCase("a query", `{"query":"{ a }"}`, http.StatusOK),
		graphQLCase("a mutation", `{"query":"mutation { secretField }"}`, http.StatusForbidden),
	})
	s.Contains(s.logs.String(), `why="a GraphQL mutation"`)
	s.NotContains(s.logs.String(), "secretField")
}

// TestAMethodOverrideHeaderIsRefusedOnEveryRead: Google runs a GET with X-HTTP-Method-Override:
// POST as the POST (live probe 2026-10-27T20:42Z, AI-2120), so a GET or a GraphQL query that
// carries an override header, under any spelling of its name and with any value, reaches no host
// and asks for no token. Without the header both read (200 on the base, e1a71dcc4).
func (s *ProxySuite) TestAMethodOverrideHeaderIsRefusedOnEveryRead() {
	s.readGraphQL()
	reads := []graphQLRequest{
		{name: "a GET", method: http.MethodGet, target: "https://api.github.test/echo"},
		{name: "a GET with no placeholder", method: http.MethodGet, target: "https://api.github.test/echo"},
		graphQLCase("a GraphQL query", `{"query":"{ a }"}`, http.StatusOK),
	}
	overrides := map[string][]string{
		"X-HTTP-Method-Override": {"POST"},
		"X-HTTP-Method":          {"DELETE"},
		"X-Method-Override":      {"PUT"},
		"x-http-method-override": {"post"},
		"X-HTTP-METHOD":          {"GET"},
		"x-method-override":      {""},
		"X-Http-Method-Override": {"GET", "POST"},
	}
	for _, read := range reads {
		for name, values := range overrides {
			for _, header := range []map[string][]string{nil, {name: values}} {
				s.granted.forget()
				asked := s.asked
				refused := s.requestCount("api.github.test", read.method, "denied_method", "4xx")
				authorization := "Bearer " + s.placeholder
				if read.name == "a GET with no placeholder" {
					authorization = ""
				}
				request := s.request(read.method, read.target, authorization, read.body)
				if read.contentType != "" {
					request.Header.Set("Content-Type", read.contentType)
				}
				// A name not in canonical form is sent as it is written.
				maps.Copy(request.Header, header)

				response, body, err := s.send(s.sandbox(s.credential), request)

				what := fmt.Sprintf("%s %v", read.name, header)
				s.Require().NoError(err, what)
				if header == nil {
					s.Equal(http.StatusOK, response.StatusCode, what)
					s.Len(s.granted.requests(), 1, what)
					continue
				}
				s.Equal(http.StatusForbidden, response.StatusCode, what)
				s.Equal("egress: "+read.method+" "+strings.TrimPrefix(read.target, "https://")+" refused: code in the sandbox "+
					"only reads through a connector; write with the connector's tools\n", body, what)
				s.Empty(s.granted.requests(), what)
				s.Equal(asked, s.asked, what)
				s.Equal(refused+1, s.requestCount("api.github.test", read.method, "denied_method", "4xx"), what)
			}
		}
	}
	for _, name := range []string{"X-Http-Method-Override", "X-Http-Method", "X-Method-Override"} {
		s.Contains(s.logs.String(), `why="a method override header: `+name+`"`)
	}
}

// TestAMethodOverrideTrailerIsRefusedOnEveryRead: a chunked request's trailer arrives once its
// body is read to the end, and ReverseProxy sends it on, so a GraphQL query or a GET that
// carries an override header there, declared in Trailer or not, reaches no host. The same
// request without the trailer reads (200 on the base, e1a71dcc4).
func (s *ProxySuite) TestAMethodOverrideTrailerIsRefusedOnEveryRead() {
	s.readGraphQL()
	authorization := "Authorization: Bearer " + s.placeholder + "\r\n"
	query := `{"query":"{ a }"}`
	chunked := func(method, path, declared, trailer string) string {
		head := method + " " + path + " HTTP/1.1\r\nHost: api.github.test\r\n" + authorization +
			"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
		if declared != "" {
			head += "Trailer: " + declared + "\r\n"
		}
		return head + "\r\n" + fmt.Sprintf("%x\r\n%s\r\n0\r\n", len(query), query) + trailer + "\r\n"
	}
	for _, name := range []string{"X-HTTP-Method-Override", "X-HTTP-Method", "X-Method-Override"} {
		for _, read := range []struct{ method, path string }{{http.MethodPost, "/graphql"}, {http.MethodGet, "/echo"}} {
			for _, declared := range []string{"", name} {
				what := fmt.Sprintf("%s %s with %s, declared %q", read.method, read.path, name, declared)
				for _, trailer := range []string{"", name + ": DELETE\r\n"} {
					s.granted.forget()
					asked := s.asked
					refused := s.requestCount("api.github.test", read.method, "denied_method", "4xx")

					conn := s.tunnel(&tls.Config{})
					_, err := io.WriteString(conn, chunked(read.method, read.path, declared, trailer))
					s.Require().NoError(err, what)
					response, err := http.ReadResponse(bufio.NewReader(conn), nil)
					s.Require().NoError(err, what)
					_, _ = io.Copy(io.Discard, response.Body)
					_ = response.Body.Close()

					if trailer == "" {
						// A name declared in Trailer is refused even unsent, so only the plain request is the control.
						if declared == "" {
							s.Equal(http.StatusOK, response.StatusCode, what+", no trailer")
							s.Len(s.granted.requests(), 1, what+", no trailer")
						}
						continue
					}
					if read.method == http.MethodGet && declared == "" {
						// The body of a GET is not read here, and the transport sends no trailer
						// the request did not declare (probe: the host's r.Trailer is empty).
						s.Equal(http.StatusOK, response.StatusCode, what)
						continue
					}
					s.Equal(http.StatusForbidden, response.StatusCode, what)
					s.Empty(s.granted.requests(), what)
					s.Equal(asked, s.asked, what)
					s.Equal(refused+1, s.requestCount("api.github.test", read.method, "denied_method", "4xx"), what)
				}
			}
		}
		s.Contains(s.logs.String(), `why="a method override trailer: `+strings.Replace(name, "HTTP", "Http", 1)+`"`)
	}
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

// allow puts the test's record again with the grant for api.github.test and allowed.
func (s *ProxySuite) allow(allowed ...string) {
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")},
		Allowed: allowed})
}

// direct is code in the sandbox that trusts the hosts' own CA and not the router's: a tunnel
// it can use is the host's, not one the proxy answers.
func (s *ProxySuite) direct() *http.Client {
	client := s.sandbox(s.credential)
	client.Transport.(*http.Transport).TLSClientConfig = &tls.Config{RootCAs: s.upstream}
	return client
}

// TestAnAllowedDomainIsTunnelledAsItIs: the host's own certificate, the placeholder as the
// sandbox sent it, no token asked for, and nothing of the request in the logs.
func (s *ProxySuite) TestAnAllowedDomainIsTunnelledAsItIs() {
	s.allow("outside.test")

	response, body, err := s.send(s.direct(), s.request(http.MethodGet, "https://outside.test/echo", "Bearer "+s.placeholder, ""))

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("hosts CA", response.TLS.PeerCertificates[0].Issuer.CommonName)
	s.Equal([]string{"Bearer " + s.placeholder}, s.outside.authorizations())
	s.Contains(body, s.placeholder)
	s.Zero(s.asked)
	s.Equal([]string{"outside.test:443"}, s.dialed())
	s.NotContains(s.logs.String(), s.placeholder)
	_, _, err = s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://outside.test/echo", "", ""))
	s.ErrorContains(err, "certificate", "the router's CA signs nothing for an allowed domain")
}

// TestAHostBothGrantedAndAllowedIsAGrants: the proxy answers it and puts the token in.
func (s *ProxySuite) TestAHostBothGrantedAndAllowedIsAGrants() {
	s.allow("*.github.test", "api.github.test")

	response, _ := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"Bearer gho_first-token-of-the-connection"}, s.granted.authorizations())
}

// TestOnlyAnAllowedDomainOnPort443IsTunnelled: the allowed domains are read as grants' hosts
// are, and another port, an IP literal or another host is refused and never dialed.
func (s *ProxySuite) TestOnlyAnAllowedDomainOnPort443IsTunnelled() {
	s.allow("outside.test", "*.pypi.test", "127.0.0.1")
	status := func(target string) int { return s.connectStatus(target, basic("sbx", s.credential)).StatusCode }

	s.Equal(http.StatusForbidden, status("outside.test:8443"))
	s.Equal(http.StatusForbidden, status("evil.outside.test:443"))
	s.Equal(http.StatusForbidden, status("outside.test.evil.test:443"))
	s.Equal(http.StatusForbidden, status("127.0.0.1:443"))
	s.Equal(http.StatusForbidden, status("*.pypi.test:443"))
	s.Empty(s.dialed())
	s.Equal(http.StatusOK, status("files.pypi.test:443"))
	s.Equal(http.StatusOK, status("Outside.test:443"))
	s.Equal([]string{"files.pypi.test:443", "outside.test:443"}, s.dialed())
}

// TestAnAllowedDomainResolvingToANonPublicAddressIsNotDialed is A11 for a tunnel: the dial the
// router uses refuses it.
func (s *ProxySuite) TestAnAllowedDomainResolvingToANonPublicAddressIsNotDialed() {
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.Dial = nil }))
	s.T().Cleanup(s.proxy.Close)
	s.allow("localhost")

	s.Equal(http.StatusBadGateway, s.connectStatus("localhost:443", basic("sbx", s.credential)).StatusCode)
	s.Contains(s.logs.String(), "resolved to a non-public address")
}

// TestATunnelToAnAllowedDomainKeepsTheTLSHelloAndEndsWithTheHost: the hello sent with the
// CONNECT reaches the host, and the host closing the connection closes the sandbox's.
func (s *ProxySuite) TestATunnelToAnAllowedDomainKeepsTheTLSHelloAndEndsWithTheHost() {
	s.allow("outside.test")
	conn, err := net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	defer conn.Close()
	s.Require().NoError(conn.SetDeadline(time.Now().Add(5 * time.Second)))
	eager := &eagerConn{Conn: conn, connect: fmt.Sprintf("CONNECT outside.test:443 HTTP/1.1\r\nHost: outside.test:443\r\n"+
		"Proxy-Authorization: %s\r\n\r\n", basic("sbx", s.credential))}
	eager.reader = bufio.NewReader(conn)
	client := tls.Client(eager, &tls.Config{ServerName: "outside.test", RootCAs: s.upstream})
	s.Require().NoError(client.HandshakeContext(s.ctx))
	_, err = io.WriteString(client, "GET /echo HTTP/1.1\r\nHost: outside.test\r\nConnection: close\r\n\r\n")
	s.Require().NoError(err)

	answer, err := io.ReadAll(client)

	s.Require().NoError(err)
	s.Contains(string(answer), "200 OK")
	// TLS ended with the host's close_notify; the proxy then closes the sandbox's connection.
	_, err = eager.reader.ReadByte()
	s.ErrorIs(err, io.EOF)
}

// TestATunnelToAnAllowedDomainEndsWithTheSandbox: the host side sees the end when the sandbox
// closes its connection, and the sandbox's bytes reach the host as they are.
func (s *ProxySuite) TestATunnelToAnAllowedDomainEndsWithTheSandbox() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	defer listener.Close()
	s.answerAt(listener.Addr().String())
	s.allow("outside.test")
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, _ := listener.Accept()
		accepted <- conn
	}()
	conn, err := net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	defer conn.Close()
	_, err = fmt.Fprintf(conn, "CONNECT outside.test:443 HTTP/1.1\r\nHost: outside.test:443\r\nProxy-Authorization: %s\r\n\r\n",
		basic("sbx", s.credential))
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	upstream := <-accepted
	s.Require().NotNil(upstream)
	defer upstream.Close()
	_, err = io.WriteString(conn, "hello")
	s.Require().NoError(err)
	s.Require().NoError(upstream.SetReadDeadline(time.Now().Add(5 * time.Second)))
	got := make([]byte, 5)
	_, err = io.ReadFull(upstream, got)
	s.Require().NoError(err)
	s.Equal("hello", string(got))

	s.Require().NoError(conn.Close())

	_, err = upstream.Read(got)
	s.ErrorIs(err, io.EOF)
}

// blindTunnel opens a tunnel to an allowed domain on a proxy whose idle timeout is idle, to a host
// that is a bare listener: the sandbox's end and the host's.
func (s *ProxySuite) blindTunnel(idle time.Duration) (sandbox, host net.Conn) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = listener.Close() })
	s.answerAt(listener.Addr().String())
	s.allow("outside.test")
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.IdleTimeout = idle }))
	s.T().Cleanup(s.proxy.Close)
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, _ := listener.Accept()
		accepted <- conn
	}()
	sandbox, err = net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sandbox.Close() })
	_, err = fmt.Fprintf(sandbox, "CONNECT outside.test:443 HTTP/1.1\r\nHost: outside.test:443\r\nProxy-Authorization: %s\r\n\r\n",
		basic("sbx", s.credential))
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(sandbox), nil)
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	host = <-accepted
	s.Require().NotNil(host)
	s.T().Cleanup(func() { _ = host.Close() })
	return sandbox, host
}

// closedWithin is the time conn took to end, the connection closed by its peer and not a read
// that ran out of time: the wait is bounded by a few seconds, past any idle timeout a test gives.
func (s *ProxySuite) closedWithin(conn net.Conn, since time.Time) time.Duration {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, err := io.Copy(io.Discard, conn)
	if err != nil {
		var timeout net.Error
		s.False(errors.As(err, &timeout) && timeout.Timeout(), "the tunnel was not closed in time: %v", err)
	}
	return time.Since(since)
}

// TestATunnelToAnAllowedDomainThatCarriesNothingIsClosedAtTheIdleTimeout: no byte either way,
// and both ends are closed, not before the idle timeout.
func (s *ProxySuite) TestATunnelToAnAllowedDomainThatCarriesNothingIsClosedAtTheIdleTimeout() {
	const idle = 300 * time.Millisecond
	sandbox, host := s.blindTunnel(idle)
	opened := time.Now()

	sandboxEnd := s.closedWithin(sandbox, opened)
	hostEnd := s.closedWithin(host, opened)

	s.GreaterOrEqual(sandboxEnd, idle)
	s.GreaterOrEqual(hostEnd, idle)
}

// TestATunnelToAnAllowedDomainThatKeepsCarryingBytesStaysOpen: past the idle timeout, bytes in one
// direction or the other keep it, though the other direction is silent; it closes the idle
// timeout after the last byte. Both at once is the -race probe of the shared deadline.
func (s *ProxySuite) TestATunnelToAnAllowedDomainThatKeepsCarryingBytesStaysOpen() {
	const idle = 400 * time.Millisecond
	const every = idle / 4
	for _, direction := range []struct {
		name                    string
		sandboxSends, hostSends bool
	}{{"the sandbox sends", true, false}, {"the host sends", false, true}, {"both send", true, true}} {
		s.Run(direction.name, func() {
			sandbox, host := s.blindTunnel(idle)
			var sent, received atomic.Int64
			var drained sync.WaitGroup
			drain := func(conn net.Conn) {
				drained.Add(1)
				go func() {
					defer drained.Done()
					n, _ := io.Copy(io.Discard, conn)
					received.Add(n)
				}()
			}
			drain(sandbox)
			drain(host)
			var sending sync.WaitGroup
			var last atomic.Pointer[time.Time]
			send := func(conn net.Conn) {
				sending.Add(1)
				go func() {
					defer sending.Done()
					// Three idle timeouts of bytes, each well inside one.
					for end := time.Now().Add(3 * idle); time.Now().Before(end); time.Sleep(every) {
						now := time.Now()
						if _, err := conn.Write([]byte("x")); err != nil {
							s.Failf("the tunnel closed while it carried bytes", "%v", err)
							return
						}
						sent.Add(1)
						last.Store(&now)
					}
				}()
			}
			if direction.sandboxSends {
				send(sandbox)
			}
			if direction.hostSends {
				send(host)
			}
			sending.Wait()
			s.Require().NotNil(last.Load())

			closed := s.closedWithin(sandbox, *last.Load())
			drained.Wait()

			s.GreaterOrEqual(closed, idle, "closed before the idle timeout past the last byte")
			s.Equal(sent.Load(), received.Load(), "a byte was lost")
		})
	}
}

// fillUntilBlocked writes conn small chunks until `stalls` writes in a row time out having accepted
// no byte at all. One such write can be a proxy copy that has not run yet, which then moves
// everything on and goes back to a read that the idle timeout ends; a proxy that takes nothing
// for several stalls has both its buffers and its write full, and its copy is stuck in the write.
// It returns an error, not a failed test: the callers run it in goroutines.
func fillUntilBlocked(conn net.Conn, stall time.Duration) error {
	// stalls: chosen, not measured: three stalls outlast the longest pause of a proxy copy seen
	// on this machine under a load of 115 (a single stall was passed by a copy that had not run).
	const stalls = 3
	chunk := make([]byte, 32<<10)
	// 1 GiB is far past any loopback buffer chain; a tunnel that takes it all is not blocked.
	for written, blocked := 0, 0; written < 1<<30; {
		if err := conn.SetWriteDeadline(time.Now().Add(stall)); err != nil {
			return err
		}
		n, err := conn.Write(chunk)
		written += n
		if err != nil && !errors.Is(err, os.ErrDeadlineExceeded) {
			return fmt.Errorf("the tunnel closed while it was being filled: %w", err)
		}
		if n > 0 {
			blocked = 0
			continue
		}
		if blocked++; blocked == stalls {
			return nil
		}
	}
	return errors.New("the tunnel took a gigabyte nobody reads")
}

// TestATunnelToAnAllowedDomainWhoseEndReadsNothingIsClosedAtTheIdleTimeout: bytes fill the
// buffers and a copy blocks in its write. With one copy stuck, the other has a read pending to
// time out; with both stuck, each end's write deadline is the only thing that ends the tunnel.
// The test touches its ends no more once they are full: a byte it sent would count as progress
// and restart the idle timeout, so the close is read from the open-tunnels gauge. That a tunnel
// is not closed before the idle timeout is held by
// TestATunnelToAnAllowedDomainThatCarriesNothingIsClosedAtTheIdleTimeout, not here.
func (s *ProxySuite) TestATunnelToAnAllowedDomainWhoseEndReadsNothingIsClosedAtTheIdleTimeout() {
	const (
		// idle: chosen, not measured: long enough that the fill (stall) and the gauge reads fit
		// inside it on a loaded machine, short enough that the test waits seconds, not 90 s.
		idle = 2 * time.Second
		// stall: how long a write may make no progress before the buffers count as full. Chosen,
		// not measured: many scheduler turns on a loaded machine (load 115 here), and below idle.
		stall = 200 * time.Millisecond
		// margin: what the close may take past the idle timeout. Chosen: the idle timeout again,
		// for the close and a loaded machine; there is no other timer on a hijacked tunnel.
		margin = idle
		// poll: the interval of the gauge reads, as in TestOpenTunnelsAreCountedUntilTheyClose.
		poll = 20 * time.Millisecond
	)
	for _, stalled := range []struct {
		name                    string
		sandboxSends, hostSends bool
	}{{"the host reads nothing", true, false}, {"the sandbox reads nothing", false, true}, {"neither reads", true, true}} {
		s.Run(stalled.name, func() {
			before := s.tunnelsOpen()
			sandbox, host := s.blindTunnel(idle)
			s.Require().Eventually(func() bool { return s.metric("router_sandbox_egress_tunnels_open", nil) == before+1 },
				// Chosen, not measured: a bound only (Eventually returns when the gauge moves); a tunnel opens
				// in milliseconds, 5 s covers a loaded machine.
				5*time.Second, poll, "the tunnel opened")
			// Both directions at once: a fill that waited for the other would leave a copy idle
			// with a read pending, and that read's timeout would close the tunnel early.
			var senders []net.Conn
			if stalled.sandboxSends {
				senders = append(senders, sandbox)
			}
			if stalled.hostSends {
				senders = append(senders, host)
			}
			filled := make(chan error, len(senders))
			for _, sender := range senders {
				go func() { filled <- fillUntilBlocked(sender, stall) }()
			}
			for range senders {
				s.Require().NoError(<-filled)
			}

			// Every buffer is full and nothing moves from here: the last byte moved no later than
			// now, so the tunnel is closed by idle after it at the latest.
			s.Eventually(func() bool { return s.metric("router_sandbox_egress_tunnels_open", nil) == before },
				idle+margin, poll, "the tunnel was not closed in time")
		})
	}
}

// TestAHostWithATrailingDotIsTheHostWithout: matched and dialed as the host without it, and a
// wildcard that names no domain, which no broker stores, takes nothing.
func (s *ProxySuite) TestAHostWithATrailingDotIsTheHostWithout() {
	status := func(target string) int { return s.connectStatus(target, basic("sbx", s.credential)).StatusCode }
	s.allow("*.", "pypi.test")

	s.Equal(http.StatusForbidden, status("evil.test.:443"))
	s.Equal(http.StatusForbidden, status("evil.test:443"))
	s.Equal(http.StatusForbidden, status(".:443"))
	s.Empty(s.dialed())
	s.Equal(http.StatusOK, status("PyPI.test.:443"))
	s.Equal([]string{"pypi.test:443"}, s.dialed())
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
	s.Equal(http.StatusForbidden, status("*.github.test:443"))
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

// TestAGrantWithdrawnOnAnAllowedHostEndsItsOpenTunnel: the host stays allowed, so a new CONNECT
// to it is a blind tunnel, but a tunnel the grant opened refuses its next request.
func (s *ProxySuite) TestAGrantWithdrawnOnAnAllowedHostEndsItsOpenTunnel() {
	s.allow("api.github.test")
	client := s.sandbox(s.credential)
	s.Require().Equal(http.StatusOK, s.again(client))

	s.put(s.credential, egress.Record{Session: s.session, Allowed: []string{"api.github.test"}})

	s.Equal(http.StatusForbidden, s.again(client))
	s.Equal(http.StatusOK, s.connectStatus("api.github.test:443", basic("sbx", s.credential)).StatusCode)
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
	value, err := s.redis.Do(s.ctx, s.redis.B().Get().Key(key(s.credential)).Build()).ToString()

	s.Require().NoError(err)
	s.Contains(value, s.session)
	s.NotContains(value, s.credential)
}

// TestARecordLivesForTheTTLItIsPutWith: the session's lifetime, no more.
func (s *ProxySuite) TestARecordLivesForTheTTLItIsPutWith() {
	credential := egress.NewCredential()
	s.Require().NoError(s.store.Put(s.ctx, credential, egress.Record{Session: s.session,
		Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")}}, 800*time.Millisecond))
	s.T().Cleanup(func() { _ = s.store.Delete(context.Background(), credential) })

	left, err := s.redis.Do(s.ctx, s.redis.B().Pttl().Key(key(credential)).Build()).AsInt64()

	s.Require().NoError(err)
	s.Positive(left)
	s.LessOrEqual(left, int64(800))
}

// TestACorruptRecordIsAnErrorNotAGrant: a record that does not read is 503, and nothing is
// tunnelled.
func (s *ProxySuite) TestACorruptRecordIsAnErrorNotAGrant() {
	s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Set().Key(key(s.credential)).Value("{not json").Px(time.Minute).Build()).Error())

	s.Equal(http.StatusServiceUnavailable, s.connectStatus("api.github.test:443", basic("sbx", s.credential)).StatusCode)
	s.Empty(s.dialed())
}

// TestACredentialNearTheSandboxsIsAskedForOne is A14: only the credential itself opens a
// tunnel, not one a character off, in another case, cut short or longer.
func (s *ProxySuite) TestACredentialNearTheSandboxsIsAskedForOne() {
	c := s.credential
	flipped := "A"
	if strings.HasSuffix(c, "A") {
		flipped = "B"
	}
	for _, near := range []string{c[:len(c)-1] + flipped, strings.ToLower(c), c[:len(c)-1], c[:20], c + "A", " " + c} {
		s.Equal(http.StatusProxyAuthRequired, s.connectStatus("api.github.test:443", basic("sbx", near)).StatusCode, near)
	}
	s.Equal(http.StatusOK, s.connectStatus("api.github.test:443", basic("sbx", c)).StatusCode)
}

// TestTheCredentialIsBasicsPasswordOnly: the scheme is read without case (RFC 9110 section
// 11.1), and the user name is never the credential.
func (s *ProxySuite) TestTheCredentialIsBasicsPasswordOnly() {
	encoded := func(raw string) string { return base64.StdEncoding.EncodeToString([]byte(raw)) }
	for header, want := range map[string]int{
		"basic " + encoded("sbx:"+s.credential): http.StatusOK,
		"BASIC " + encoded("sbx:"+s.credential): http.StatusOK,
		"Basic " + encoded(":"+s.credential):    http.StatusOK,
		"Basic " + encoded(s.credential+":"):    http.StatusProxyAuthRequired,
		"Basic " + encoded(s.credential):        http.StatusProxyAuthRequired,
		"Basic " + encoded(s.credential+":x"):   http.StatusProxyAuthRequired,
	} {
		s.Equal(want, s.connectStatus("api.github.test:443", header).StatusCode, header)
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

// TestAnUpgradeIsRefused: an upgraded connection, to any protocol, would skip the response
// rewrite.
func (s *ProxySuite) TestAnUpgradeIsRefused() {
	for _, protocol := range []string{"websocket", "h2c", "anything/1"} {
		request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
		request.Header.Set("Connection", "Upgrade")
		request.Header.Set("Upgrade", protocol)

		response, _, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err)
		s.Equal(http.StatusNotImplemented, response.StatusCode, protocol)
	}
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

// TestTheProductionTransportDialsNoInternalAddress is A11 past CONNECT: a granted name that
// resolves to a local address is never dialed by the transport the router uses.
func (s *ProxySuite) TestTheProductionTransportDialsNoInternalAddress() {
	proxy, err := egress.New(egress.Options{Store: s.store, Tokens: s.tokens, CACert: s.caCert, CAKey: s.caKey,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(proxy)
	s.T().Cleanup(s.proxy.Close)
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "localhost")}})

	response, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, "https://localhost/echo", "Bearer "+s.placeholder, ""))

	s.Require().NoError(err)
	s.Equal(http.StatusBadGateway, response.StatusCode)
	s.Contains(s.logs.String(), "resolved to a non-public address")
}

// TestTheSandboxGetsOneHostsCertificateOverHTTP11AndTLS12: the certificate names the
// CONNECT host only, is no CA's, is the router CA's and lives at most two and a half hours; h2 is not
// negotiated, and TLS before 1.2 is refused.
func (s *ProxySuite) TestTheSandboxGetsOneHostsCertificateOverHTTP11AndTLS12() {
	// The handshake verifies the leaf against the router's CA alone (tunnel's RootCAs).
	state := s.handshake(&tls.Config{NextProtos: []string{"h2", "http/1.1"}})

	s.Equal("http/1.1", state.NegotiatedProtocol)
	leaf := state.PeerCertificates[0]
	s.Equal([]string{"api.github.test"}, leaf.DNSNames)
	s.False(leaf.IsCA)
	s.Equal("router sandbox CA", leaf.Issuer.CommonName)
	s.LessOrEqual(leaf.NotAfter.Sub(leaf.NotBefore), 2*time.Hour+halfValidity)
	for _, version := range []uint16{tls.VersionTLS10, tls.VersionTLS11} {
		s.Error(s.tunnel(&tls.Config{MinVersion: tls.VersionTLS10, MaxVersion: version}).Handshake(), version)
	}
}

// handshake is the TLS state the sandbox gets in a tunnel to api.github.test with config.
func (s *ProxySuite) handshake(config *tls.Config) tls.ConnectionState {
	conn := s.tunnel(config)
	s.Require().NoError(conn.Handshake())
	return conn.ConnectionState()
}

// tunnel is a CONNECT to api.github.test:443, and TLS in it with config.
func (s *ProxySuite) tunnel(config *tls.Config) *tls.Conn {
	return s.tunnelTo("api.github.test", config)
}

// tunnelTo is tunnel to host.
func (s *ProxySuite) tunnelTo(host string, config *tls.Config) *tls.Conn {
	conn, err := net.Dial("tcp", s.proxy.Listener.Addr().String())
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = conn.Close() })
	s.Require().NoError(conn.SetDeadline(time.Now().Add(5 * time.Second)))
	_, err = fmt.Fprintf(conn, "CONNECT %s:443 HTTP/1.1\r\nHost: %s:443\r\nProxy-Authorization: %s\r\n\r\n",
		host, host, basic("sbx", s.credential))
	s.Require().NoError(err)
	// A one-byte reader keeps no byte of the handshake from tls.Client.
	response, err := http.ReadResponse(bufio.NewReaderSize(conn, 16), nil)
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	config.ServerName, config.RootCAs = host, s.trusted
	return tls.Client(conn, config)
}

// TestASilentOrIdleTunnelIsClosed: a sandbox that sends no handshake, or no next request,
// holds no tunnel past its timeout.
func (s *ProxySuite) TestASilentOrIdleTunnelIsClosed() {
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) {
		options.HandshakeTimeout, options.IdleTimeout = 200*time.Millisecond, 200*time.Millisecond
	}))
	s.T().Cleanup(s.proxy.Close)
	silent := s.tunnel(&tls.Config{}).NetConn()
	idle := s.tunnel(&tls.Config{})
	_, err := io.WriteString(idle, "GET /echo HTTP/1.1\r\nHost: api.github.test\r\n\r\n")
	s.Require().NoError(err)
	reader := bufio.NewReader(idle)
	response, err := http.ReadResponse(reader, nil)
	s.Require().NoError(err)
	_, err = io.ReadAll(response.Body)
	s.Require().NoError(err)

	_, silentEnd := silent.Read(make([]byte, 1))
	_, idleEnd := reader.ReadByte()

	s.ErrorIs(silentEnd, io.EOF)
	s.ErrorIs(idleEnd, io.EOF)
}

// TestAnEncodedResponseToATokenIsRefused is A5: a host that encodes its echo in a way the
// transport does not undo would hand the sandbox the token to decode. Each encoding gets
// its own proxy, so no connection to an earlier host is reused.
func (s *ProxySuite) TestAnEncodedResponseToATokenIsRefused() {
	for _, encoding := range []string{"deflate", "x-gzip", "br", "identity, deflate"} {
		s.proxy = httptest.NewServer(s.newProxy(s.store))
		s.T().Cleanup(s.proxy.Close)
		served := 0
		s.answering(func(w http.ResponseWriter, r *http.Request) {
			served++
			w.Header().Set("Content-Encoding", encoding)
			_, _ = io.WriteString(w, r.Header.Get("Authorization"))
		})

		response, body := s.get("Bearer " + s.placeholder)

		s.Equal(1, served, encoding)
		s.Equal(http.StatusBadGateway, response.StatusCode, encoding)
		s.Empty(body, encoding)
	}
	s.Contains(s.logs.String(), "refused an encoded response")
}

// TestEveryContentEncodingLineIsRead is A5: net/http's transport, had it asked for gzip,
// decodes the first of two Content-Encoding lines and removes both, so the sandbox would
// get a body still encoded once with no header saying so. Over HTTP/1.1 and HTTP/2.
func (s *ProxySuite) TestEveryContentEncodingLineIsRead() {
	gzipped := func(raw []byte) []byte {
		var out bytes.Buffer
		zipped := gzip.NewWriter(&out)
		_, _ = zipped.Write(raw)
		_ = zipped.Close()
		return out.Bytes()
	}
	deflated := func(raw []byte) []byte {
		var out bytes.Buffer
		writer, _ := flate.NewWriter(&out, flate.BestCompression)
		_, _ = writer.Write(raw)
		_ = writer.Close()
		return out.Bytes()
	}
	for _, http2 := range []bool{false, true} {
		for _, lines := range [][]string{{"gzip", "gzip"}, {"gzip", "deflate"}, {"identity", "deflate"}} {
			name := fmt.Sprint(lines, " http2=", http2)
			s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) {
				options.Transport.(*http.Transport).ForceAttemptHTTP2 = http2
			}))
			s.T().Cleanup(s.proxy.Close)
			served := 0
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				served++
				body := []byte(r.Header.Get("Authorization"))
				for i := len(lines) - 1; i >= 0; i-- {
					switch lines[i] {
					case "gzip":
						body = gzipped(body)
					case "deflate":
						body = deflated(body)
					}
				}
				w.Header()["Content-Encoding"] = lines
				w.Header().Set("X-Proto", r.Proto)
				_, _ = w.Write(body)
			}))
			server.EnableHTTP2 = http2
			server.TLS = &tls.Config{Certificates: []tls.Certificate{sign(s.T(), s.upstreamCA, s.upstreamKey, "*.github.test")}}
			server.StartTLS()
			s.T().Cleanup(server.Close)
			s.answerAt(server.Listener.Addr().String())

			response, body := s.get("Bearer " + s.placeholder)

			s.Equal(1, served, name)
			s.Equal(http.StatusBadGateway, response.StatusCode, name)
			s.Empty(body, name)
		}
	}
}

// TestAnIdentityEncodedResponseIsRead: identity is no encoding.
func (s *ProxySuite) TestAnIdentityEncodedResponseIsRead() {
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Encoding", "identity")
		_, _ = io.WriteString(w, r.Header.Get("Authorization"))
	})

	response, body := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("Bearer "+s.placeholder, body)
}

// TestATrailerOrAnInformationalResponseIsNotSentOn is A5: ReverseProxy sends both on past
// the response rewrite.
func (s *ProxySuite) TestATrailerOrAnInformationalResponseIsNotSentOn() {
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Link", "<"+r.Header.Get("Authorization")+">; rel=preload")
		w.WriteHeader(http.StatusEarlyHints)
		w.Header().Del("Link")
		w.Header().Set("Trailer", "X-Echo")
		_, _ = io.WriteString(w, "body")
		w.Header().Set("X-Echo", r.Header.Get("Authorization"))
	})
	var informational []string
	request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
	request = request.WithContext(httptrace.WithClientTrace(s.ctx, &httptrace.ClientTrace{
		Got1xxResponse: func(status int, header textproto.MIMEHeader) error {
			informational = append(informational, fmt.Sprint(status, header))
			return nil
		}}))

	response, body, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal("body", body)
	s.Empty(informational)
	s.Empty(response.Trailer)
}

// TestAnEncodedEchoComesBackAsThePlaceholder is A5 for a host that escapes what it echoes:
// in a JSON string, URL-encoded or in base64.
func (s *ProxySuite) TestAnEncodedEchoComesBackAsThePlaceholder() {
	const token = "gho/first+tok!en&x<y"
	s.connect(token)
	inJSON := func(value string) string { out, _ := json.Marshal(value); return strings.Trim(string(out), `"`) }
	forms := []func(string) string{
		inJSON,
		func(value string) string { return strings.ReplaceAll(value, "/", `\/`) },
		url.QueryEscape,
		url.PathEscape,
		func(value string) string { return base64.StdEncoding.EncodeToString([]byte(value)) },
		func(value string) string { return base64.RawURLEncoding.EncodeToString([]byte(value)) },
	}
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		sent := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		for _, form := range forms {
			w.Header().Add("X-Echo", form(sent))
		}
		for _, form := range forms {
			_, _ = io.WriteString(w, form(sent)+"\n")
		}
	})

	response, body := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusOK, response.StatusCode)
	for _, form := range forms {
		s.NotContains(body, form(token))
		s.Contains(body, form(s.placeholder)+"\n")
	}
	for i, form := range forms {
		s.Equal(form(s.placeholder), response.Header.Values("X-Echo")[i])
	}
}

// TestABase64EchoOfTheWholeAuthorizationComesBackAsThePlaceholder is A5: base64 of
// "Bearer <token>" holds the token at another alignment than the token's own base64. The
// four base64 forms differ only when the value is not a multiple of 3 bytes long or its
// base64 holds + or /, so the tokens give every length mod 3 for both schemes, each with a
// byte that makes +, / in the standard form.
func (s *ProxySuite) TestABase64EchoOfTheWholeAuthorizationComesBackAsThePlaceholder() {
	forms := []*base64.Encoding{base64.StdEncoding, base64.URLEncoding, base64.RawStdEncoding, base64.RawURLEncoding}
	for _, token := range []string{"gho/first+tok!en&x<y", "gho_d3>>?~~", "gho_d3>>?~~~", "gho_d3>>?~"} {
		s.Run(token, func() {
			s.connect(token)
			s.answering(func(w http.ResponseWriter, r *http.Request) {
				for _, form := range forms {
					w.Header().Add("X-Echo", form.EncodeToString([]byte(r.Header.Get("Authorization"))))
				}
				for _, form := range forms {
					_, _ = io.WriteString(w, form.EncodeToString([]byte(r.Header.Get("Authorization")))+"\n")
				}
			})

			for _, authorization := range []string{"Bearer " + s.placeholder, basic("x-access-token", s.placeholder)} {
				response, body := s.get(authorization)

				s.Equal(http.StatusOK, response.StatusCode, authorization)
				echoed := append(strings.Fields(body), response.Header.Values("X-Echo")...)
				s.Len(echoed, 2*len(forms), authorization)
				for _, echo := range echoed {
					for _, form := range forms {
						decoded, _ := form.DecodeString(echo)
						s.NotContains(string(decoded), token, authorization)
					}
				}
				s.Equal(base64.StdEncoding.EncodeToString([]byte(authorization)), echoed[0], authorization)
			}
		})
	}
}

// TestBytesPastAResponseReachNoProcessLog is A13: a host that sends more than its
// response, a HEAD answered with a body or a body past its Content-Length, leaves bytes
// net/http's transport would print to the process's log on the connection's next use; the
// router's main sends that log to its own.
func (s *ProxySuite) TestBytesPastAResponseReachNoProcessLog() {
	previous := log.Writer()
	s.T().Cleanup(func() { log.SetOutput(previous) })
	for _, method := range []string{http.MethodHead, http.MethodGet} {
		var global lockedBuffer
		log.SetOutput(&global)
		s.answeringRaw(func(authorization string) string {
			echo := `{"authorization":"` + authorization + `"}`
			if method == http.MethodHead {
				return "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n" + echo
			}
			return "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok" + echo
		})

		response, _, err := s.send(s.sandbox(s.credential), s.request(method, "https://api.github.test/echo", "Bearer "+s.placeholder, ""))
		s.Require().NoError(err, method)
		s.Equal(http.StatusOK, response.StatusCode, method)
		// The transport reads an idle connection's next bytes as soon as it is idle.
		time.Sleep(300 * time.Millisecond)

		log.SetOutput(previous)
		s.NotContains(global.String(), "gho_first-token-of-the-connection", method)
	}
}

// TestEveryValueOfAnEchoedHeaderIsRewritten: a host may send a header more than once.
func (s *ProxySuite) TestEveryValueOfAnEchoedHeaderIsRewritten() {
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-Echo", "first")
		w.Header().Add("X-Echo", r.Header.Get("Authorization"))
	})

	response, _ := s.get("Bearer " + s.placeholder)

	s.Equal([]string{"first", "Bearer " + s.placeholder}, response.Header.Values("X-Echo"))
}

// TestARangeIsNotSentWithAToken is A5: a range of an echo hands the token over in pieces no
// rewrite recognises.
func (s *ProxySuite) TestARangeIsNotSentWithAToken() {
	var ranges []string
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		ranges = append(ranges, r.Header.Get("Range")+r.Header.Get("If-Range"))
		http.ServeContent(w, r, "echo.txt", time.Time{}, strings.NewReader("auth="+r.Header.Get("Authorization")))
	})
	request := s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, "")
	request.Header.Set("Range", "bytes=12-18")
	request.Header.Set("If-Range", `"tag"`)

	response, body, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("auth=Bearer "+s.placeholder, body)
	s.Equal([]string{""}, ranges)
}

// TestTheSchemeIsReadWithoutCase: RFC 9110 section 11.1; the host gets it as it was sent.
func (s *ProxySuite) TestTheSchemeIsReadWithoutCase() {
	sent := []string{"bearer", "BEARER", "Token"}
	for _, scheme := range sent {
		_, body := s.get(scheme + " " + s.placeholder)
		s.Contains(body, `"authorization":"`+scheme+" "+s.placeholder+`"`)
	}

	s.Equal([]string{"bearer gho_first-token-of-the-connection", "BEARER gho_first-token-of-the-connection",
		"Token gho_first-token-of-the-connection"}, s.granted.authorizations())
}

// TestTheSandboxReadsNoTokenErrorText: the 401 and 502 say which binding gave no token, not
// why, and the log holds none of the error either: a provider's may quote its answer.
func (s *ProxySuite) TestTheSandboxReadsNoTokenErrorText() {
	const canary = "canary-from-the-resolver"
	s.fail(fmt.Errorf("%w: %s", sandbox.ErrNoCredential, canary))
	disconnected, disconnectedBody := s.get("Bearer " + s.placeholder)
	s.fail(errors.New(canary))
	unavailable, unavailableBody := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusUnauthorized, disconnected.StatusCode)
	s.Equal(http.StatusBadGateway, unavailable.StatusCode)
	s.Equal("egress: the connection gives no token: github\n", disconnectedBody)
	s.Equal("egress: the connection gives no token: github\n", unavailableBody)
	s.NotContains(s.logs.String(), canary)
	s.Contains(s.logs.String(), "class=no_credential")
}

// metric reads one series of the process's metrics, as /metrics shows it: a counter's or a
// gauge's value, a histogram's sample count. labels is the whole label set; none is a series
// that has not been touched, which reads 0.
func (s *ProxySuite) metric(name string, labels map[string]string) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
	series:
		for _, metric := range family.GetMetric() {
			if len(metric.GetLabel()) != len(labels) {
				continue
			}
			for _, label := range metric.GetLabel() {
				if labels[label.GetName()] != label.GetValue() {
					continue series
				}
			}
			switch {
			case metric.GetHistogram() != nil:
				return float64(metric.GetHistogram().GetSampleCount())
			case metric.GetGauge() != nil:
				return metric.GetGauge().GetValue()
			}
			return metric.GetCounter().GetValue()
		}
	}
	return 0
}

// requestCount is the proxy's request counter for one label set.
func (s *ProxySuite) requestCount(host, method, decision, status string) float64 {
	return s.metric("router_sandbox_egress_requests_total",
		map[string]string{"host": host, "method": method, "decision": decision, "status": status})
}

// labelValues are every label value of the proxy's metrics.
func (s *ProxySuite) labelValues() []string {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	var values []string
	for _, family := range families {
		if !strings.HasPrefix(family.GetName(), "router_sandbox_egress_") {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				values = append(values, label.GetValue())
			}
		}
	}
	return values
}

// tunnelsOpen is the open-tunnels gauge once it has stopped moving: the tunnels an earlier
// test's sandbox held close a moment after it.
func (s *ProxySuite) tunnelsOpen() float64 {
	last := s.metric("router_sandbox_egress_tunnels_open", nil)
	for stable := 0; stable < 5; {
		time.Sleep(20 * time.Millisecond)
		if now := s.metric("router_sandbox_egress_tunnels_open", nil); now != last {
			last, stable = now, 0
		} else {
			stable++
		}
	}
	return last
}

// TestARequestWithAPlaceholderIsCountedAsAToken: and the host's wait for the answer is timed.
func (s *ProxySuite) TestARequestWithAPlaceholderIsCountedAsAToken() {
	counted := s.requestCount("api.github.test", "GET", "token", "2xx")
	timed := s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "api.github.test"})

	s.get("Bearer " + s.placeholder)

	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "token", "2xx"))
	s.Equal(timed+1, s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "api.github.test"}))
}

// TestARequestToAGrantedHostWithNoPlaceholderIsCountedAsGrantedNoToken: it is sent as it is.
func (s *ProxySuite) TestARequestToAGrantedHostWithNoPlaceholderIsCountedAsGrantedNoToken() {
	counted := s.requestCount("api.github.test", "GET", "granted_no_token", "2xx")
	asked := s.requestCount("api.github.test", "GET", "token", "2xx")

	s.get("")

	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "granted_no_token", "2xx"))
	s.Equal(asked, s.requestCount("api.github.test", "GET", "token", "2xx"))
}

// TestAnAllowedDomainTunnelIsCountedAsBlindUnderOther: the domain is the customer's to name,
// so it is not a label value.
func (s *ProxySuite) TestAnAllowedDomainTunnelIsCountedAsBlindUnderOther() {
	s.allow("outside.test")
	counted := s.requestCount("other", "CONNECT", "blind", "2xx")

	_, _, err := s.send(s.direct(), s.request(http.MethodGet, "https://outside.test/echo", "", ""))

	s.Require().NoError(err)
	s.Equal(counted+1, s.requestCount("other", "CONNECT", "blind", "2xx"))
	s.NotContains(s.labelValues(), "outside.test")
}

// TestABlindTunnelThatCannotBeDialedIsCountedAsBlindWithAServerStatus.
func (s *ProxySuite) TestABlindTunnelThatCannotBeDialedIsCountedAsBlindWithAServerStatus() {
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.Dial = nil }))
	s.T().Cleanup(s.proxy.Close)
	s.allow("localhost")
	counted := s.requestCount("other", "CONNECT", "blind", "5xx")

	s.connectStatus("localhost:443", basic("sbx", s.credential))

	s.Equal(counted+1, s.requestCount("other", "CONNECT", "blind", "5xx"))
}

// TestAHostOrAPortWithoutAGrantIsCountedAsDeniedHost: a granted host on another port keeps its
// label, any other host is "other", an allowed domain on another port included.
func (s *ProxySuite) TestAHostOrAPortWithoutAGrantIsCountedAsDeniedHost() {
	s.allow("allowed.test")
	outside := s.requestCount("other", "CONNECT", "denied_host", "4xx")
	port := s.requestCount("api.github.test", "CONNECT", "denied_host", "4xx")

	s.connectStatus("outside.test:443", basic("sbx", s.credential))
	s.connectStatus("127.0.0.1:443", basic("sbx", s.credential))
	s.connectStatus("api.github.test:8443", basic("sbx", s.credential))
	s.connectStatus("allowed.test:8443", basic("sbx", s.credential))

	s.NotContains(s.labelValues(), "allowed.test")
	s.Equal(outside+3, s.requestCount("other", "CONNECT", "denied_host", "4xx"))
	s.Equal(port+1, s.requestCount("api.github.test", "CONNECT", "denied_host", "4xx"))
}

// TestAnotherHostInAGrantedTunnelIsCountedAsDeniedHost.
func (s *ProxySuite) TestAnotherHostInAGrantedTunnelIsCountedAsDeniedHost() {
	counted := s.requestCount("api.github.test", "GET", "denied_host", "4xx")

	request := s.request(http.MethodGet, "https://api.github.test/echo", "", "")
	request.Host = "other.github.test"
	s.send(s.sandbox(s.credential), request)

	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "denied_host", "4xx"))
}

// TestAGrantThatEndedIsCountedAsDeniedHost.
func (s *ProxySuite) TestAGrantThatEndedIsCountedAsDeniedHost() {
	client := s.sandbox(s.credential)
	s.Require().Equal(http.StatusOK, s.again(client))
	counted := s.requestCount("api.github.test", "GET", "denied_host", "4xx")

	s.Require().NoError(s.store.Delete(s.ctx, s.credential))

	s.Equal(http.StatusForbidden, s.again(client))
	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "denied_host", "4xx"))
}

// TestACredentialThatIsNotASandboxsIsCountedAsDeniedAuth.
func (s *ProxySuite) TestACredentialThatIsNotASandboxsIsCountedAsDeniedAuth() {
	counted := s.requestCount("other", "CONNECT", "denied_auth", "4xx")

	s.connectStatus("api.github.test:443", "")
	s.connectStatus("api.github.test:443", basic("sbx", egress.NewCredential()))

	s.Equal(counted+2, s.requestCount("other", "CONNECT", "denied_auth", "4xx"))
}

// TestARequestTheProxyDoesNotServeIsCountedAsDeniedRequest: plain HTTP, an upgrade and a
// CONNECT with no port.
func (s *ProxySuite) TestARequestTheProxyDoesNotServeIsCountedAsDeniedRequest() {
	plain := s.requestCount("other", "GET", "denied_request", "4xx")
	upgrade := s.requestCount("api.github.test", "GET", "denied_request", "5xx")
	noPort := s.requestCount("other", "CONNECT", "denied_request", "4xx")

	s.send(s.sandbox(s.credential), s.request(http.MethodGet, "http://api.github.test/echo", "", ""))
	request := s.request(http.MethodGet, "https://api.github.test/echo", "", "")
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	s.send(s.sandbox(s.credential), request)
	s.connectStatus("api.github.test", basic("sbx", s.credential))

	s.Equal(plain+1, s.requestCount("other", "GET", "denied_request", "4xx"))
	s.Equal(upgrade+1, s.requestCount("api.github.test", "GET", "denied_request", "5xx"))
	s.Equal(noPort+1, s.requestCount("other", "CONNECT", "denied_request", "4xx"))
}

// TestGrantsThatCannotBeReadAreCountedAsAnError: Redis lost before the CONNECT.
func (s *ProxySuite) TestGrantsThatCannotBeReadAreCountedAsAnError() {
	closed, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	closed.Close()
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(closed)))
	s.T().Cleanup(s.proxy.Close)
	counted := s.requestCount("other", "CONNECT", "error", "5xx")

	s.connectStatus("api.github.test:443", basic("sbx", s.credential))

	s.Equal(counted+1, s.requestCount("other", "CONNECT", "error", "5xx"))
}

// TestGrantsThatCannotBeReadInAnOpenTunnelAreCountedAsAnError: Redis lost after the CONNECT.
func (s *ProxySuite) TestGrantsThatCannotBeReadInAnOpenTunnelAreCountedAsAnError() {
	lost, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(lost)))
	s.T().Cleanup(s.proxy.Close)
	client := s.sandbox(s.credential)
	s.Require().Equal(http.StatusOK, s.again(client))
	counted := s.requestCount("api.github.test", "GET", "error", "5xx")

	lost.Close()

	s.Equal(http.StatusServiceUnavailable, s.again(client))
	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "error", "5xx"))
}

// TestEveryStandardMethodIsItsOwnLabel: RFC 9110 section 9.1's methods, in a tunnel, each one
// the grant reads with.
func (s *ProxySuite) TestEveryStandardMethodIsItsOwnLabel() {
	grant := s.grant(s.placeholder, "api.github.test")
	for _, method := range []string{"POST", "PUT", "DELETE", "OPTIONS", "TRACE", "PATCH"} {
		grant.Reads = append(grant.Reads, core.SandboxRead{Host: "api.github.test", Method: method, Path: "/echo"})
	}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	client := s.sandbox(s.credential)
	for _, method := range []string{"GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "TRACE", "PATCH"} {
		counted := s.requestCount("api.github.test", method, "granted_no_token", "2xx")

		_, _, err := s.send(client, s.request(method, "https://api.github.test/echo", "", ""))

		s.Require().NoError(err, method)
		s.Equal(counted+1, s.requestCount("api.github.test", method, "granted_no_token", "2xx"), method)
	}
}

// TestAConnectionThatGivesNoTokenIsCountedAsAnError: the placeholder is not replaced, so it is
// not counted as a token.
func (s *ProxySuite) TestAConnectionThatGivesNoTokenIsCountedAsAnError() {
	s.fail(sandbox.ErrNoCredential)
	counted := s.requestCount("api.github.test", "GET", "error", "4xx")
	replaced := s.requestCount("api.github.test", "GET", "token", "4xx")

	response, _ := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal(counted+1, s.requestCount("api.github.test", "GET", "error", "4xx"))
	s.Equal(replaced, s.requestCount("api.github.test", "GET", "token", "4xx"))
}

// TestTheStatusIsTheClassTheSandboxGot: a host's 404 and a host that cannot be reached, both
// timed.
func (s *ProxySuite) TestTheStatusIsTheClassTheSandboxGot() {
	s.answering(func(w http.ResponseWriter, _ *http.Request) { http.NotFound(w, nil) })
	missing := s.requestCount("api.github.test", "GET", "token", "4xx")
	timed := s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "api.github.test"})

	s.get("Bearer " + s.placeholder)

	s.Equal(missing+1, s.requestCount("api.github.test", "GET", "token", "4xx"))
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.answerAt(closed.Addr().String())
	s.Require().NoError(closed.Close())
	unreachable := s.requestCount("api.github.test", "GET", "token", "5xx")

	s.get("Bearer " + s.placeholder)

	s.Equal(unreachable+1, s.requestCount("api.github.test", "GET", "token", "5xx"))
	s.Equal(timed+2, s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "api.github.test"}))
}

// TestTheLabelsHoldNoNameTheSandboxChoosesNorAnySecret: a wildcard grant is one host label
// whatever name under it is asked for, an unknown method is "other", and no token,
// placeholder, credential or session is a label value. The grant reads with the unknown method.
func (s *ProxySuite) TestTheLabelsHoldNoNameTheSandboxChoosesNorAnySecret() {
	grant := s.grant(s.placeholder, "*.github.test")
	grant.Reads = []core.SandboxRead{{Host: "*.github.test", Method: "BREW", Path: "/echo"}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	wildcard := s.requestCount("*.github.test", "GET", "token", "2xx")
	odd := s.requestCount("*.github.test", "other", "granted_no_token", "2xx")
	client := s.sandbox(s.credential)

	for _, target := range []string{"https://a.github.test/echo", "https://b.github.test/echo", "https://github.test/echo"} {
		_, _, err := s.send(client, s.request(http.MethodGet, target, "Bearer "+s.placeholder, ""))
		s.Require().NoError(err)
	}
	_, _, err := s.send(client, s.request("BREW", "https://a.github.test/echo", "", ""))
	s.Require().NoError(err)

	s.Equal(wildcard+3, s.requestCount("*.github.test", "GET", "token", "2xx"))
	s.Equal(odd+1, s.requestCount("*.github.test", "other", "granted_no_token", "2xx"))
	values := s.labelValues()
	for _, secret := range []string{"gho_first-token-of-the-connection", s.placeholder, s.credential, s.session,
		"a.github.test", "b.github.test", "BREW"} {
		s.NotContains(values, secret)
	}
}

// TestAStreamedResponseReachesTheSandboxBeforeTheHostEnds: the first flushed chunk is not held
// by the counting wrapper, with a placeholder and without.
func (s *ProxySuite) TestAStreamedResponseReachesTheSandboxBeforeTheHostEnds() {
	for _, authorization := range []string{"", "Bearer " + s.placeholder} {
		release := make(chan struct{})
		s.answering(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("first\n"))
			s.NoError(http.NewResponseController(w).Flush())
			select {
			case <-release:
			case <-time.After(10 * time.Second):
			}
			_, _ = w.Write([]byte("second\n"))
		})
		response, err := s.sandbox(s.credential).Do(s.request(http.MethodGet, "https://api.github.test/echo", authorization, ""))
		s.Require().NoError(err)
		read := make(chan string, 1)
		go func() {
			line, _ := bufio.NewReader(response.Body).ReadString('\n')
			read <- line
		}()
		select {
		case line := <-read:
			s.Equal("first\n", line, "authorization %q", authorization)
		case <-time.After(5 * time.Second):
			s.Fail("the first chunk was held until the host ended", "authorization %q", authorization)
		}
		close(release)
		_ = response.Body.Close()
	}
}

// TestAnInformationalResponseDoesNotHideTheFinalStatus: a request with no placeholder gets
// its 1xx sent on, and is counted by the status after it.
func (s *ProxySuite) TestAnInformationalResponseDoesNotHideTheFinalStatus() {
	s.answering(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusEarlyHints)
		w.WriteHeader(http.StatusTeapot)
	})
	teapot := s.requestCount("api.github.test", "GET", "granted_no_token", "4xx")
	early := s.requestCount("api.github.test", "GET", "granted_no_token", "1xx")

	response, _ := s.get("")

	s.Equal(http.StatusTeapot, response.StatusCode)
	s.Equal(teapot+1, s.requestCount("api.github.test", "GET", "granted_no_token", "4xx"))
	s.Equal(early, s.requestCount("api.github.test", "GET", "granted_no_token", "1xx"))
}

// TestAConnectionThatCannotBeTakenOverIsCountedAsAnError: a response writer that cannot hijack,
// as an HTTP/2 stream's, for a granted host and for an allowed domain.
func (s *ProxySuite) TestAConnectionThatCannotBeTakenOverIsCountedAsAnError() {
	s.allow("outside.test")
	proxy := s.newProxy(s.store)
	granted := s.requestCount("api.github.test", "CONNECT", "error", "5xx")
	blind := s.requestCount("other", "CONNECT", "error", "5xx")

	for _, target := range []string{"api.github.test:443", "outside.test:443"} {
		request := httptest.NewRequest(http.MethodConnect, "http://"+target, nil)
		request.Host = target
		request.Header.Set("Proxy-Authorization", basic("sbx", s.credential))
		recorder := httptest.NewRecorder()

		proxy.ServeHTTP(recorder, request)

		s.Equal(http.StatusInternalServerError, recorder.Code, target)
	}

	s.Equal(granted+1, s.requestCount("api.github.test", "CONNECT", "error", "5xx"))
	s.Equal(blind+1, s.requestCount("other", "CONNECT", "error", "5xx"))
}

// TestOpenTunnelsAreCountedUntilTheyClose: a tunnel to a granted host, one to an allowed domain.
func (s *ProxySuite) TestOpenTunnelsAreCountedUntilTheyClose() {
	s.allow("outside.test")
	before := s.tunnelsOpen()
	granted := s.sandbox(s.credential)
	blind := s.direct()
	s.Require().Equal(http.StatusOK, s.again(granted))
	_, _, err := s.send(blind, s.request(http.MethodGet, "https://outside.test/echo", "", ""))
	s.Require().NoError(err)

	s.Eventually(func() bool { return s.metric("router_sandbox_egress_tunnels_open", nil) == before+2 },
		5*time.Second, 20*time.Millisecond, "open")

	granted.CloseIdleConnections()
	s.Eventually(func() bool { return s.metric("router_sandbox_egress_tunnels_open", nil) == before+1 },
		5*time.Second, 20*time.Millisecond, "the granted tunnel closed")
	blind.CloseIdleConnections()
	s.Eventually(func() bool { return s.metric("router_sandbox_egress_tunnels_open", nil) == before },
		5*time.Second, 20*time.Millisecond, "the blind tunnel closed")
}

// TestAHostThatCannotBeReachedIsABadGateway: and the dial error, which holds nothing of a
// host's, is logged.
func (s *ProxySuite) TestAHostThatCannotBeReachedIsABadGateway() {
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.answerAt(closed.Addr().String())
	s.Require().NoError(closed.Close())

	response, body := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusBadGateway, response.StatusCode)
	s.Empty(body)
	s.Contains(s.logs.String(), "connection refused")
}

// TestAHostThatSendsNoHeaderInTimeIsABadGateway: a slow host holds no tunnel past the
// header timeout.
func (s *ProxySuite) TestAHostThatSendsNoHeaderInTimeIsABadGateway() {
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.HeaderTimeout = 200 * time.Millisecond }))
	s.T().Cleanup(s.proxy.Close)
	release := make(chan struct{})
	s.answering(func(http.ResponseWriter, *http.Request) { <-release })
	// Before the host's Close, which waits for its handler.
	s.T().Cleanup(func() { close(release) })

	response, _ := s.get("Bearer " + s.placeholder)

	s.Equal(http.StatusBadGateway, response.StatusCode)
	s.Contains(s.logs.String(), "class=header_timeout")
}

// TestAMalformedAnswerEchoingTheTokenIsNotLogged is A13: an error the transport gives for a
// malformed header line, status line, length or trailer quotes the line; TearDownTest checks
// the token reached no log.
func (s *ProxySuite) TestAMalformedAnswerEchoingTheTokenIsNotLogged() {
	for _, answer := range []func(string) string{
		func(authorization string) string {
			return "HTTP/1.1 200 OK\r\nbad header " + authorization + "\r\n\r\n"
		},
		func(authorization string) string { return "HTTP/1.1 " + authorization + "\r\n\r\n" },
		func(authorization string) string {
			return "HTTP/1.1 200 OK\r\nContent-Length: " + authorization + "\r\n\r\n"
		},
		func(authorization string) string {
			return "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nbody\r\n0\r\nbad trailer " + authorization + "\r\n\r\n"
		},
	} {
		s.answeringRaw(answer)

		response, err := s.sandbox(s.credential).Do(s.request(http.MethodGet, "https://api.github.test/echo", "Bearer "+s.placeholder, ""))
		if err == nil {
			_, _ = io.ReadAll(response.Body)
			_ = response.Body.Close()
		}
	}

	s.Contains(s.logs.String(), "got no answer it could read")
	s.Contains(s.logs.String(), "lost a host's response mid-body")
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
	// BigQuery's dry run of jobs.query names the statement, SCRIPT for more than one (live probe
	// 2026-10-27T19:55Z).
	if strings.HasPrefix(r.URL.Path, "/bigquery/") && strings.Contains(got.Body, `"dryRun":true`) {
		statement := "SELECT"
		if strings.Contains(got.Body, ";") {
			statement = "SCRIPT"
		}
		_, _ = io.WriteString(w, `{"statementType":"`+statement+`"}`)
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

// TestTheListenerTLSHoldsTLS12AndHTTP11: the floor and the protocol a tunnel's TLS has. Go's
// own default floor is TLS 1.2 today, so the setting is read from the config, which a default
// would not show.
func (s *ProxySuite) TestTheListenerTLSHoldsTLS12AndHTTP11() {
	config := s.newProxy(s.store).ListenerTLS("203.0.113.7")

	s.Equal(uint16(tls.VersionTLS12), config.MinVersion)
	s.Equal([]string{"http/1.1"}, config.NextProtos)
}

// leafClock is the time a proxy dates and renews its certificates by, which a test moves.
type leafClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *leafClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *leafClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// TestAHostKeepsItsCertificateUntilHalfOfItsValidityHasPassed: two tunnels to one host within
// the window get one certificate, and a tunnel after it gets a new one. A kept certificate is
// still valid for the client for at least the rest of the half hour, when it is reused.
func (s *ProxySuite) TestAHostKeepsItsCertificateUntilHalfOfItsValidityHasPassed() {
	clock := &leafClock{now: time.Now()}
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.Now = clock.Now }))
	s.T().Cleanup(s.proxy.Close)
	// The client reads the clock the proxy reads, so a certificate it dates is judged by it.
	handshake := func() *x509.Certificate {
		state := s.handshake(&tls.Config{Time: clock.Now})
		return state.PeerCertificates[0]
	}

	first := handshake()
	clock.advance(29*time.Minute + 59*time.Second)
	reused := handshake()
	reusedAt := clock.Now()
	clock.advance(time.Second)
	renewed := handshake()

	s.Equal(first.SerialNumber, reused.SerialNumber, "within half of the validity")
	s.True(reused.NotAfter.After(reusedAt.Add(halfValidity)), "a kept certificate has more than half an hour left: %s", reused.NotAfter)
	s.NotEqual(first.SerialNumber, renewed.SerialNumber, "half of the validity has passed")
	// A certificate's times are whole seconds.
	s.WithinDuration(clock.Now().Add(time.Hour+halfValidity), renewed.NotAfter, time.Second, "the renewed certificate is dated by the clock")
}

// TestHostsKeepTheirOwnCertificates: a certificate kept for one host is never another host's,
// whether the host is a tunnel's name or the listener's address.
func (s *ProxySuite) TestHostsKeepTheirOwnCertificates() {
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test", "uploads.github.test")}})

	api := s.handshake(&tls.Config{}).PeerCertificates[0]
	uploads := s.tunnelTo("uploads.github.test", &tls.Config{})
	s.Require().NoError(uploads.Handshake())
	other := uploads.ConnectionState().PeerCertificates[0]
	again := s.handshake(&tls.Config{}).PeerCertificates[0]

	s.Equal([]string{"api.github.test"}, api.DNSNames)
	s.Equal([]string{"uploads.github.test"}, other.DNSNames)
	s.NotEqual(api.SerialNumber, other.SerialNumber)
	s.Equal(api.SerialNumber, again.SerialNumber)
}

// TestTheListenerKeepsItsCertificateForTheAddress: every handshake the listener answers in the
// window gets one certificate, an IP address's, dated by the clock when none is set. The
// control for a proxy with no Now is time.Now: the certificate leafValidity either side of it.
func (s *ProxySuite) TestTheListenerKeepsItsCertificateForTheAddress() {
	proxy := s.newProxy(s.store)
	config := proxy.ListenerTLS("127.0.0.1")
	before := time.Now()

	first := s.listenerCertificate(config, "127.0.0.1", time.Now)
	second := s.listenerCertificate(config, "127.0.0.1", time.Now)

	s.Equal(first.SerialNumber, second.SerialNumber)
	s.True(first.IPAddresses[0].Equal(net.ParseIP("127.0.0.1")))
	s.WithinDuration(before.Add(-time.Hour), first.NotBefore, 5*time.Second)
	s.WithinDuration(before.Add(time.Hour+halfValidity), first.NotAfter, 5*time.Second)
}

// listenerCertificate is the certificate config answers a client that connects to address with.
func (s *ProxySuite) listenerCertificate(config *tls.Config, address string, now func() time.Time) *x509.Certificate {
	server, client := net.Pipe()
	s.T().Cleanup(func() { _ = client.Close(); _ = server.Close() })
	s.Require().NoError(client.SetDeadline(time.Now().Add(5 * time.Second)))
	s.Require().NoError(server.SetDeadline(time.Now().Add(5 * time.Second)))
	served := make(chan error, 1)
	go func() { served <- tls.Server(server, config).Handshake() }()
	conn := tls.Client(client, &tls.Config{ServerName: address, RootCAs: s.trusted, Time: now})
	s.Require().NoError(conn.Handshake())
	s.Require().NoError(<-served)
	return conn.ConnectionState().PeerCertificates[0]
}

// maxKeptLeaves is egress's maxLeaves.
const maxKeptLeaves = 1024

// TestTheKeptCertificatesAreBounded: past the cap a host still gets a certificate, signed for
// the handshake alone; a certificate half a validity old is dropped to make room.
func (s *ProxySuite) TestTheKeptCertificatesAreBounded() {
	clock := &leafClock{now: time.Now()}
	proxy := s.newProxy(s.store, func(options *egress.Options) { options.Now = clock.Now })
	serial := func(address string) string {
		certificate, err := proxy.ListenerTLS(address).GetCertificate(&tls.ClientHelloInfo{})
		s.Require().NoError(err)
		leaf, err := x509.ParseCertificate(certificate.Certificate[0])
		s.Require().NoError(err)
		return leaf.SerialNumber.String()
	}
	address := func(i int) string { return fmt.Sprintf("10.%d.%d.1", i/256, i%256) }
	const hosts = maxKeptLeaves + 50
	first := make([]string, hosts)
	for i := range first {
		first[i] = serial(address(i))
	}

	kept := 0
	for i := range first {
		if serial(address(i)) == first[i] {
			kept++
		}
	}
	clock.advance(halfValidity)
	fresh := serial("192.0.2.1")

	s.Equal(maxKeptLeaves, kept)
	s.Equal(fresh, serial("192.0.2.1"), "the old certificates made room")
}

// TestAKeptCertificateVerifiesForAClockAnHourAhead: a certificate served 29 minutes after it
// was signed is still valid, as a fresh one is, for a sandbox clock 59 minutes ahead.
func (s *ProxySuite) TestAKeptCertificateVerifiesForAClockAnHourAhead() {
	// The suite's CA lives an hour, which is less than the clock skew and the age together.
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	s.Require().NoError(err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "day CA"},
		NotBefore: time.Now().Add(-24 * time.Hour), NotAfter: time.Now().Add(24 * time.Hour), IsCA: true,
		BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	s.Require().NoError(err)
	ca, err := x509.ParseCertificate(der)
	s.Require().NoError(err)
	clock := &leafClock{now: time.Now()}
	proxy, err := egress.New(egress.Options{Store: s.store, Tokens: s.tokens, Now: clock.Now,
		CACert: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), CAKey: keyPEM(s.T(), key)})
	s.Require().NoError(err)
	trusted := x509.NewCertPool()
	trusted.AddCert(ca)
	config := proxy.ListenerTLS("127.0.0.1")
	// handshake is the sandbox's, with a clock 59 minutes ahead of the proxy's.
	handshake := func() *x509.Certificate {
		server, client := net.Pipe()
		s.T().Cleanup(func() { _ = client.Close(); _ = server.Close() })
		s.Require().NoError(client.SetDeadline(time.Now().Add(5 * time.Second)))
		s.Require().NoError(server.SetDeadline(time.Now().Add(5 * time.Second)))
		go func() { _ = tls.Server(server, config).Handshake() }()
		conn := tls.Client(client, &tls.Config{ServerName: "127.0.0.1", RootCAs: trusted,
			Time: func() time.Time { return clock.Now().Add(59 * time.Minute) }})
		s.Require().NoError(conn.Handshake())
		return conn.ConnectionState().PeerCertificates[0]
	}
	first := handshake()
	clock.advance(29 * time.Minute)

	kept := handshake()

	s.Equal(first.SerialNumber, kept.SerialNumber, "the certificate served is the first one")
}

// blockingRandom is crypto/rand's reader that holds a read once armed, until released, which
// is how a test holds a certificate's signing, the first thing it reads randomness for.
type blockingRandom struct {
	reader  io.Reader
	armed   atomic.Bool
	entered chan struct{}
	release chan struct{}
}

func (r *blockingRandom) Read(p []byte) (int, error) {
	if r.armed.CompareAndSwap(true, false) {
		close(r.entered)
		<-r.release
	}
	return r.reader.Read(p)
}

// TestAHostWithACertificateIsServedWhileAnotherHostIsSigned: a handshake for a host with a
// kept certificate completes while a handshake for a new host is held in its CA signature.
func (s *ProxySuite) TestAHostWithACertificateIsServedWhileAnotherHostIsSigned() {
	proxy := s.newProxy(s.store)
	_, err := proxy.ListenerTLS("10.0.0.1").GetCertificate(&tls.ClientHelloInfo{})
	s.Require().NoError(err)
	random := &blockingRandom{reader: rand.Reader, entered: make(chan struct{}), release: make(chan struct{})}
	original := rand.Reader
	rand.Reader = random
	released := false
	release := func() {
		if !released {
			released = true
			close(random.release)
		}
	}
	s.T().Cleanup(func() { release(); rand.Reader = original })
	signed := make(chan error, 1)
	random.armed.Store(true)
	go func() {
		_, err := proxy.ListenerTLS("10.0.0.2").GetCertificate(&tls.ClientHelloInfo{})
		signed <- err
	}()
	<-random.entered

	served := make(chan error, 1)
	go func() {
		_, err := proxy.ListenerTLS("10.0.0.1").GetCertificate(&tls.ClientHelloInfo{})
		served <- err
	}()

	select {
	case err := <-served:
		s.NoError(err)
	case <-time.After(5 * time.Second):
		s.Fail("a host with a certificate waits for another host's signature")
	}
	release()
	s.NoError(<-signed)
}

// halfValidity is half of egress's leafValidity.
const halfValidity = 30 * time.Minute

// TestConcurrentHandshakesShareTheKeptCertificates is the -race probe: handshakes for the same
// and for different hosts at once, while the clock passes the renewal point, get one
// certificate per host and period.
func (s *ProxySuite) TestConcurrentHandshakesShareTheKeptCertificates() {
	clock := &leafClock{now: time.Now()}
	proxy := s.newProxy(s.store, func(options *egress.Options) { options.Now = clock.Now })
	addresses := []string{"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"}
	serials := func() map[string]map[string]bool {
		var mu sync.Mutex
		seen := map[string]map[string]bool{}
		var wg sync.WaitGroup
		for range 8 {
			for _, address := range addresses {
				wg.Add(1)
				go func() {
					defer wg.Done()
					certificate, err := proxy.ListenerTLS(address).GetCertificate(&tls.ClientHelloInfo{})
					if !s.NoError(err) {
						return
					}
					mu.Lock()
					defer mu.Unlock()
					if seen[address] == nil {
						seen[address] = map[string]bool{}
					}
					seen[address][string(certificate.Certificate[0])] = true
				}()
			}
		}
		wg.Wait()
		return seen
	}

	before := serials()
	clock.advance(halfValidity)
	after := serials()

	all := map[string]bool{}
	for _, address := range addresses {
		s.Len(before[address], 1, address)
		s.Len(after[address], 1, address)
		for certificate := range before[address] {
			s.False(all[certificate], "a certificate is one host's")
			all[certificate] = true
		}
		for certificate := range after[address] {
			s.False(before[address][certificate], "renewed")
		}
	}
	s.Len(all, len(addresses))
}
