//go:build integration

package egress_test

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"log/slog"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/brokertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
)

func TestProxyBrokerContract(t *testing.T) {
	var current *rig
	suite.Run(t, &brokertest.BrokerContract{New: func(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) brokertest.Subject {
		// One proxy per test, shared by its sessions.
		if current == nil || current.test != t.Name() {
			current = newRig(t, tokens, logger)
		}
		r := current
		var mount sandbox.Mount
		broker := r.broker(t, session, tokens, logger)
		return brokertest.Subject{
			Broker:  prepared{Broker: broker, mount: &mount},
			Held:    func(ref string) (string, []string, bool) { return r.held(t, mount, ref) },
			Left:    r.left,
			Refuse:  func() { r.gate.shut(true) },
			Recover: r.recover,
			Live:    true,
		}
	}})
}

// prepared is the broker, keeping what its last Prepare gave the sandbox, as the sandbox does.
type prepared struct {
	*egress.Broker
	mount *sandbox.Mount
}

func (p prepared) Prepare(ctx context.Context, grants []sandbox.Grant) (sandbox.Mount, error) {
	mount, err := p.Broker.Prepare(ctx, grants)
	*p.mount = mount
	return mount, err
}

// BrokerSuite is what the proxy broker gives a sandbox, used as the sandbox would: an
// http.Client built from the env and files it returned, through a real proxy.
type BrokerSuite struct {
	suite.Suite
	ctx context.Context
	rig *rig
	mu  sync.Mutex
	// token is what the connection gives now, failure the error it gives instead.
	token   string
	failure error
	broker  *egress.Broker
	mount   sandbox.Mount
}

func TestBrokerSuite(t *testing.T) {
	suite.Run(t, new(BrokerSuite))
}

var github = sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "app", ConnectionID: "connection-github"},
	Binding: "github", Scheme: "bearer", Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com"}}

func (s *BrokerSuite) SetupTest() {
	s.ctx = context.Background()
	s.token, s.failure = "gho_live-token-of-the-connection", nil
	logger := slog.New(slog.DiscardHandler)
	s.rig = newRig(s.T(), s.tokens, logger)
	s.broker = s.rig.broker(s.T(), "session-1", s.tokens, logger)
	var err error
	s.mount, err = s.broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)
}

func (s *BrokerSuite) tokens(context.Context, sandbox.Grant) (sandbox.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return sandbox.Token{Value: s.token, Revision: 1}, s.failure
}

func (s *BrokerSuite) fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failure = err
}

// get is a request from the sandbox with its placeholder as a bearer token.
func (s *BrokerSuite) get() (*http.Response, string, error) {
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, "https://api.github.com/user", nil)
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+s.mount.Env["GITHUB_TOKEN"])
	response, err := sandboxClient(s.T(), s.mount).Do(request)
	if err != nil {
		return nil, "", err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response, string(body), nil
}

// TestTheSandboxReachesAGrantedHostWithTheTokenInPlaceOfItsPlaceholder: everything the
// sandbox is given holds a placeholder, and the host gets the token.
func (s *BrokerSuite) TestTheSandboxReachesAGrantedHostWithTheTokenInPlaceOfItsPlaceholder() {
	placeholder := s.mount.Env["GITHUB_TOKEN"]
	s.Regexp(`^va_ph_[A-Z2-7]{26}$`, placeholder)

	response, body, err := s.get()

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"Bearer gho_live-token-of-the-connection"}, s.rig.host.authorizations())
	s.Contains(body, placeholder)
	raw, err := json.Marshal(s.mount)
	s.Require().NoError(err)
	s.NotContains(string(raw), s.token)
}

// TestTheEnvNamesTheProxyAndTheCA: every common client finds the proxy, with the sandbox's
// credential, and trusts the proxy's CA from a file that holds its certificate alone.
func (s *BrokerSuite) TestTheEnvNamesTheProxyAndTheCA() {
	proxy := s.mount.Proxy.String()
	credential, _ := s.mount.Proxy.User.Password()
	s.Len(credential, 26, "128 random bits in base32")
	s.Equal(strings.Replace(s.rig.proxy.URL, "http://", "http://sbx:"+credential+"@", 1), proxy)
	for _, name := range []string{"HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"} {
		s.Equal(proxy, s.mount.Env[name], name)
	}
	for _, name := range []string{"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "GIT_SSL_CAINFO"} {
		s.Equal("/tmp/va-egress-ca.pem", s.mount.Env[name], name)
	}
	s.Equal(map[string][]byte{"/tmp/va-egress-ca.pem": s.rig.caCert}, s.mount.Files)
	s.Len(s.mount.Env, 9)
}

// TestClosingEndsTheProxyCredential: the next CONNECT is a 407, and a run after Close does
// not bring the credential back.
func (s *BrokerSuite) TestClosingEndsTheProxyCredential() {
	s.Require().NoError(s.broker.Close(s.ctx))
	s.Require().NoError(s.broker.BeforeRun(s.ctx))

	_, _, err := s.get()

	s.ErrorContains(err, "Proxy Authentication Required")
	s.Empty(s.rig.host.authorizations())
}

// TestARunAfterTheRecordExpiredRenewsIt: a session idle for longer than the record's life
// gets it back on its next run, and each run renews it for longer than a run may take.
func (s *BrokerSuite) TestARunAfterTheRecordExpiredRenewsIt() {
	credential, _ := s.mount.Proxy.User.Password()
	key := key(credential)
	s.Require().NoError(s.rig.redis.Do(s.ctx, s.rig.redis.B().Pexpire().Key(key).Milliseconds(1).Build()).Error())
	s.Require().Eventually(func() bool {
		exists, _ := s.rig.redis.Do(s.ctx, s.rig.redis.B().Exists().Key(key).Build()).AsInt64()
		return exists == 0
	}, time.Second, 10*time.Millisecond)
	_, _, err := s.get()
	s.Require().ErrorContains(err, "Proxy Authentication Required")

	s.Require().NoError(s.broker.BeforeRun(s.ctx))

	response, _, err := s.get()
	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	ttl, err := s.rig.redis.Do(s.ctx, s.rig.redis.B().Pttl().Key(key).Build()).AsInt64()
	s.Require().NoError(err)
	s.Greater(time.Duration(ttl)*time.Millisecond, sandbox.MaxTimeout)
}

// TestADisconnectedConnectionIsRefusedAtTheProxy: revocation needs no run in between.
func (s *BrokerSuite) TestADisconnectedConnectionIsRefusedAtTheProxy() {
	s.fail(sandbox.ErrNoCredential)

	response, body, err := s.get()

	s.Require().NoError(err)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Contains(body, "github")
	s.Empty(s.rig.host.authorizations())
}

// TestAProxyURLItCannotUseIsRefused: the sandbox's own credential is the URL's user, and a
// proxy reached over anything but http would take no Basic credential from it.
func (s *BrokerSuite) TestAProxyURLItCannotUseIsRefused() {
	for _, address := range []string{"", "https://proxy.test:8091", "http://user:pass@proxy.test:8091", "http://:8091", "proxy.test:8091"} {
		_, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: address, CACert: s.rig.caCert, Session: "session-1",
			Tokens: s.tokens})
		s.Error(err, address)
		if err != nil {
			s.NotContains(err.Error(), "pass", address)
		}
	}
	_, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: s.rig.proxy.URL, Session: "session-1", Tokens: s.tokens})
	s.Error(err, "no CA certificate")
	_, err = egress.NewBroker(egress.BrokerOptions{URL: s.rig.proxy.URL, CACert: s.rig.caCert, Session: "session-1", Tokens: s.tokens})
	s.Error(err, "no store")
}

// TestNoGrantWithACredentialGivesNoProxy: the sandbox is created as without grants, and
// nothing is stored.
func (s *BrokerSuite) TestNoGrantWithACredentialGivesNoProxy() {
	s.fail(sandbox.ErrNoCredential)

	mount, err := s.broker.Prepare(s.ctx, []sandbox.Grant{github})

	s.Require().NoError(err)
	s.Equal(sandbox.Mount{}, mount)
	credential, _ := s.mount.Proxy.User.Password()
	exists, err := s.rig.redis.Do(s.ctx, s.rig.redis.B().Exists().Key(key(credential)).Build()).AsInt64()
	s.Require().NoError(err)
	s.Zero(exists, "a second Prepare ends the first one's credential")
	s.Require().NoError(s.rig.redis.Do(s.ctx, s.rig.redis.B().Del().Key(key("")).Build()).Error())
	s.Require().NoError(s.broker.BeforeRun(s.ctx))
	exists, err = s.rig.redis.Do(s.ctx, s.rig.redis.B().Exists().Key(key("")).Build()).AsInt64()
	s.Require().NoError(err)
	s.Zero(exists, "a run without a credential stores nothing")
}

// rig is a real proxy over Redis, whose hosts are one httptest TLS server, and the store the
// brokers write through, behind a gate that can refuse.
type rig struct {
	test   string
	redis  rueidis.Client
	store  *egress.Store
	gated  rueidis.Client
	gate   *redisGate
	caCert []byte
	host   *host
	proxy  *httptest.Server
	// run marks the records of this test's sessions, so left reads no other test's.
	run string
}

func newRig(t *testing.T, tokens sandbox.Tokens, logger *slog.Logger) *rig {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR is not set")
	}
	r := &rig{test: t.Name(), run: uuid.NewString()}
	var err error
	r.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.redis.Close)
	r.gate = newRedisGate(t, address)
	// One connection (PipelineMultiplex -1): by default rueidis spreads keyed commands over
	// four and sends PING on the first only, so recover's PING would not prove the others back
	// (rueidis v1.0.76 rueidis.go:220, mux.go slotfn).
	gated, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{r.gate.listener.Addr().String()}, DisableCache: true,
		DisableRetry: true, PipelineMultiplex: -1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(gated.Close)
	r.store, r.gated = egress.NewStore(gated), gated

	_, caCert, caKey := newCA(t, "router sandbox CA")
	r.caCert = caCert
	upstreamCA, _, upstreamKey := newCA(t, "hosts CA")
	upstream := x509.NewCertPool()
	upstream.AddCert(upstreamCA)
	r.host = newHost(t, upstreamCA, upstreamKey, "api.github.com", "uploads.github.com")
	dial := &net.Dialer{}
	proxy, err := egress.New(egress.Options{Store: egress.NewStore(r.redis), CACert: caCert, CAKey: keyPEM(t, caKey), Logger: logger,
		Tokens: func(ctx context.Context, grant egress.Grant) (string, error) {
			token, err := tokens(ctx, sandbox.Grant{Connection: grant.Connection, Binding: grant.Binding, Hosts: grant.Hosts})
			return token.Value, err
		},
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: upstream},
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return dial.DialContext(ctx, network, r.host.server.Listener.Addr().String())
			}}})
	if err != nil {
		t.Fatal(err)
	}
	r.proxy = httptest.NewServer(proxy)
	t.Cleanup(r.proxy.Close)
	return r
}

func (r *rig) broker(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) *egress.Broker {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: r.store, URL: r.proxy.URL, CACert: r.caCert,
		Session: session + "/" + r.run, Tokens: tokens, Logger: logger})
	if err != nil {
		t.Fatal(err)
	}
	return broker
}

// recover opens the gate and waits for the store's client to have a connection again: the
// first command after a cut may still be sent on the cut one.
func (r *rig) recover() {
	r.gate.shut(false)
	for range 100 {
		if r.gated.Do(context.Background(), r.gated.B().Ping().Build()).Error() == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	panic("egress test: redis does not answer through the gate")
}

// held is the token a request from the sandbox mount was made for gets in place of ref, and
// the hosts the proxy keeps it for.
func (r *rig) held(t *testing.T, mount sandbox.Mount, ref string) (string, []string, bool) {
	if mount.Proxy == nil {
		return "", nil, false
	}
	credential, _ := mount.Proxy.User.Password()
	record, found := r.record(t, key(credential))
	var hosts []string
	for _, grant := range record.Grants {
		if grant.Placeholder == ref {
			hosts = grant.Hosts
		}
	}
	if !found || hosts == nil {
		return "", nil, false
	}
	r.host.forget()
	request, err := http.NewRequest(http.MethodGet, "https://"+hosts[0]+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+ref)
	response, err := sandboxClient(t, mount).Do(request)
	if err != nil {
		return "", nil, false
	}
	_ = response.Body.Close()
	sent := r.host.authorizations()
	if response.StatusCode != http.StatusOK || len(sent) != 1 {
		return "", nil, false
	}
	return strings.TrimPrefix(sent[0], "Bearer "), hosts, true
}

// left are the placeholders in every record of this test's sessions.
func (r *rig) left() []string {
	ctx := context.Background()
	var placeholders []string
	keys, err := r.redis.Do(ctx, r.redis.B().Keys().Pattern("sandbox:egress:*").Build()).AsStrSlice()
	if err != nil {
		panic(err)
	}
	for _, key := range keys {
		raw, err := r.redis.Do(ctx, r.redis.B().Get().Key(key).Build()).AsBytes()
		var record egress.Record
		if err != nil || json.Unmarshal(raw, &record) != nil || !strings.HasSuffix(record.Session, "/"+r.run) {
			continue
		}
		for _, grant := range record.Grants {
			placeholders = append(placeholders, grant.Placeholder)
		}
	}
	return placeholders
}

func (r *rig) record(t *testing.T, key string) (egress.Record, bool) {
	raw, err := r.redis.Do(context.Background(), r.redis.B().Get().Key(key).Build()).AsBytes()
	if rueidis.IsRedisNil(err) {
		return egress.Record{}, false
	}
	var record egress.Record
	if err != nil || json.Unmarshal(raw, &record) != nil {
		t.Fatal(err)
	}
	return record, true
}

// sandboxClient is an HTTP client set up from what the sandbox was created with alone.
func sandboxClient(t *testing.T, mount sandbox.Mount) *http.Client {
	proxy, err := url.Parse(mount.Env["HTTPS_PROXY"])
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(mount.Files[mount.Env["SSL_CERT_FILE"]]) {
		t.Fatal("the sandbox's CA file holds no certificate")
	}
	return &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxy), TLSClientConfig: &tls.Config{RootCAs: roots}}}
}

// redisGate passes connections on to Redis until shut, which cuts every one and refuses the
// next: a store that refuses, then answers again.
type redisGate struct {
	listener net.Listener
	mu       sync.Mutex
	closed   bool
	open     []net.Conn
}

func newRedisGate(t *testing.T, to string) *redisGate {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	g := &redisGate{listener: listener}
	t.Cleanup(func() { _ = listener.Close(); g.shut(true) })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			g.pass(conn, to)
		}
	}()
	return g
}

func (g *redisGate) pass(conn net.Conn, to string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		_ = conn.Close()
		return
	}
	redis, err := net.Dial("tcp", to)
	if err != nil {
		_ = conn.Close()
		return
	}
	g.open = append(g.open, conn, redis)
	go func() { _, _ = io.Copy(redis, conn); _ = redis.Close() }()
	go func() { _, _ = io.Copy(conn, redis); _ = conn.Close() }()
}

func (g *redisGate) shut(closed bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closed = closed
	if closed {
		for _, conn := range g.open {
			_ = conn.Close()
		}
		g.open = nil
	}
}
