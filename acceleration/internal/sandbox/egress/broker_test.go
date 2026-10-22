//go:build integration

package egress_test

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
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
			Push:    func(ref, content string) ([]string, string, []string) { return r.push(t, mount, ref, content) },
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
	ctx  context.Context
	rig  *rig
	logs *bytes.Buffer
	mu   sync.Mutex
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
	s.logs = &bytes.Buffer{}
	logger := slog.New(slog.NewJSONHandler(s.logs, nil))
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
	response, err := s.rig.sandboxClient(s.T(), s.mount).Do(request)
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

// TestEveryRunGoesThroughTheProxyAndTrustsTheCA: the sandbox may reach the proxy alone, and
// every run's env sends each common client there with the sandbox's credential, keeps loopback
// off it, and has each trust the proxy's CA: Node in addition to its own roots, from a file
// that holds the CA's certificate alone, and the rest from a bundle of the sandbox's roots and
// it. The sandbox's own env holds the placeholder alone.
func (s *BrokerSuite) TestEveryRunGoesThroughTheProxyAndTrustsTheCA() {
	credential, _ := s.mount.Proxy.User.Password()
	s.Len(credential, 26, "128 random bits in base32")
	proxy := strings.Replace(publicURL, "http://", "http://sbx:"+credential+"@", 1)
	s.Equal(proxy, s.mount.Proxy.String())
	s.Equal([]string{"GITHUB_TOKEN"}, slices.Collect(maps.Keys(s.mount.Env)))
	s.Regexp(`^va_ph_[A-Z2-7]{26}$`, s.mount.Env["GITHUB_TOKEN"])
	bundle, loopback := "/tmp/va-egress-ca-bundle.pem", "localhost,127.0.0.1,::1"
	s.Equal(map[string]string{"SSL_CERT_FILE": bundle, "REQUESTS_CA_BUNDLE": bundle, "CURL_CA_BUNDLE": bundle,
		"GIT_SSL_CAINFO": bundle, "NODE_EXTRA_CA_CERTS": "/tmp/va-egress-ca.pem", "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy,
		"https_proxy": proxy, "http_proxy": proxy, "NO_PROXY": loopback, "no_proxy": loopback,
		"GIT_HTTP_PROXY_AUTHMETHOD": "basic"}, s.mount.RunEnv)
	s.Equal(map[string][]byte{"/tmp/va-egress-ca.pem": s.rig.caCert}, s.mount.Files)
	s.Len(s.mount.Setup, 1)
}

// TestGitSendsTheProxyCredentialOnItsFirstConnect: each run's env has git send it at once
// rather than wait for a 407 and connect again (tunnel probe 2, 2026-10-22: without it, a
// CONNECT with no credential before each one with it).
func (s *BrokerSuite) TestGitSendsTheProxyCredentialOnItsFirstConnect() {
	github := github
	github.Hosts = append(github.Hosts, "github.com")
	mount, err := s.broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)

	_, _, _ = s.rig.push(s.T(), mount, mount.Env["GITHUB_TOKEN"], "pushed\n")

	s.NotEmpty(s.rig.connects())
	s.NotContains(s.rig.connects(), false, "a CONNECT without the proxy credential")
}

// TestTheBundleHoldsTheSandboxsRootsAndTheCA: a host of the config's allowed domains, which
// the proxy tunnels to as it is, is verified with the image's own roots, from the first file of
// them it has, and an image with none runs no code. Daytona's netleash CA, which nothing passes
// through any more, is never trusted. The setup runs in a shell here, under a directory standing
// for the sandbox's root.
func (s *BrokerSuite) TestTheBundleHoldsTheSandboxsRootsAndTheCA() {
	_, debian, _ := newCA(s.T(), "debian roots")
	_, alpine, _ := newCA(s.T(), "alpine roots")
	_, netleash, _ := newCA(s.T(), "netleash roots")
	setup := func(roots map[string][]byte) ([]string, error) {
		root := s.T().TempDir()
		for name, data := range roots {
			s.Require().NoError(os.MkdirAll(filepath.Dir(root+name), 0o755))
			s.Require().NoError(os.WriteFile(root+name, data, 0o600))
		}
		for name, data := range s.mount.Files {
			s.Require().NoError(os.MkdirAll(filepath.Dir(root+name), 0o755))
			s.Require().NoError(os.WriteFile(root+name, data, 0o600))
		}
		command := strings.NewReplacer("/etc/", root+"/etc/", "/tmp/", root+"/tmp/").Replace(s.mount.Setup[0])
		if err := exec.Command("sh", "-c", command).Run(); err != nil {
			return nil, err
		}
		bundle, err := os.ReadFile(root + s.mount.RunEnv["SSL_CERT_FILE"])
		s.Require().NoError(err)
		return subjects(s.T(), bundle), nil
	}

	held, err := setup(map[string][]byte{"/etc/ssl/certs/ca-certificates.crt": debian, "/etc/ssl/cert.pem": alpine})
	s.Require().NoError(err)
	s.Equal([]string{"debian roots", "router sandbox CA"}, held)
	held, err = setup(map[string][]byte{"/etc/daytona/netleash/ca.crt": netleash, "/etc/ssl/certs/ca-certificates.crt": debian})
	s.Require().NoError(err)
	s.Equal([]string{"debian roots", "router sandbox CA"}, held)
	held, err = setup(map[string][]byte{"/etc/ssl/cert.pem": bytes.TrimSuffix(alpine, []byte("\n"))})
	s.Require().NoError(err)
	s.Equal([]string{"alpine roots", "router sandbox CA"}, held, "a roots file without a last newline")
	_, err = setup(nil)
	s.Error(err)
}

// subjects are the common names of the certificates in bundle, in order.
func subjects(t *testing.T, bundle []byte) []string {
	var names []string
	for block, rest := pem.Decode(bundle); block != nil; block, rest = pem.Decode(rest) {
		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, cert.Subject.CommonName)
	}
	return names
}

// TestTheSandboxGetsTheCAsCertificateAlone: a setting holding the CA's key too gives the
// sandbox the certificate, never the key.
func (s *BrokerSuite) TestTheSandboxGetsTheCAsCertificateAlone() {
	_, cert, caKey := newCA(s.T(), "router sandbox CA")
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: publicURL,
		CACert: append(keyPEM(s.T(), caKey), cert...), Session: "session-1", Tokens: s.tokens})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = broker.Close(s.ctx) })

	mount, err := broker.Prepare(s.ctx, []sandbox.Grant{github})

	s.Require().NoError(err)
	s.Equal(map[string][]byte{"/tmp/va-egress-ca.pem": cert}, mount.Files)
	_, err = egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: publicURL, CACert: keyPEM(s.T(), caKey),
		Session: "session-1", Tokens: s.tokens})
	s.Error(err, "a key and no certificate")
}

// TestNoErrorTextReachesTheLogs: a provider's refresh error may quote its answer, so a
// broker logs its class alone, whether the grant is left out or kept.
func (s *BrokerSuite) TestNoErrorTextReachesTheLogs() {
	const quoted = "gho_quoted-by-the-provider"
	s.fail(fmt.Errorf("refresh answered %s", quoted))
	s.Require().NoError(s.broker.BeforeRun(s.ctx))
	_, err := s.broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)
	s.fail(fmt.Errorf("%w: refresh answered %s", sandbox.ErrNoCredential, quoted))
	s.Require().NoError(s.broker.BeforeRun(s.ctx))
	_, err = s.broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)

	s.Equal(4, strings.Count(s.logs.String(), `"level":"WARN"`))
	s.NotContains(s.logs.String(), quoted)
}

// TestACredentialUnavailableAtTheFirstRunIsKept: the proxy asks for the token on every
// request, so a provider that cannot answer just now leaves no grant out of the sandbox for
// its life, required or not.
func (s *BrokerSuite) TestACredentialUnavailableAtTheFirstRunIsKept() {
	s.fail(errors.New("the provider could not renew the credential just now"))
	required := github
	required.Required = true

	mount, err := s.broker.Prepare(s.ctx, []sandbox.Grant{required})
	s.Require().NoError(err)
	s.mount = mount
	_, _, err = s.get()
	s.Require().NoError(err)
	s.Empty(s.rig.host.authorizations(), "no token reaches the host meanwhile")
	s.fail(nil)
	response, _, err := s.get()

	s.Require().NoError(err)
	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"Bearer gho_live-token-of-the-connection"}, s.rig.host.authorizations())
}

// TestARecordLastsNoLongerThanARunMayTakeAndAMinute: Prepare and each run store it for that
// long, so a crashed router's sandboxes lose their grants soon after.
func (s *BrokerSuite) TestARecordLastsNoLongerThanARunMayTakeAndAMinute() {
	credential, _ := s.mount.Proxy.User.Password()
	ttl := func() time.Duration {
		ms, err := s.rig.redis.Do(s.ctx, s.rig.redis.B().Pttl().Key(key(credential)).Build()).AsInt64()
		s.Require().NoError(err)
		return time.Duration(ms) * time.Millisecond
	}
	s.LessOrEqual(ttl(), sandbox.MaxTimeout+time.Minute, "after Prepare")
	s.Require().NoError(s.rig.redis.Do(s.ctx, s.rig.redis.B().Pexpire().Key(key(credential)).Milliseconds(60_000).Build()).Error())

	s.Require().NoError(s.broker.BeforeRun(s.ctx))

	s.Greater(ttl(), sandbox.MaxTimeout)
	s.LessOrEqual(ttl(), sandbox.MaxTimeout+time.Minute, "after a run")
}

// TestAPrepareThatCannotEndTheLastCredentialFails: a sandbox created again after its first
// creation failed, while the store refuses, is not told it has no grants while the record
// of the first stays.
func (s *BrokerSuite) TestAPrepareThatCannotEndTheLastCredentialFails() {
	s.rig.gate.shut(true)
	s.Require().Error(s.broker.Close(s.ctx))
	s.fail(sandbox.ErrNoCredential)

	_, err := s.broker.Prepare(s.ctx, []sandbox.Grant{github})

	s.Require().Error(err)
	s.rig.recover()
	s.Require().NoError(s.broker.Close(s.ctx))
	_, _, err = s.get()
	s.ErrorContains(err, "Proxy Authentication Required")
}

// TestARunAfterACloseTheStoreRefusedEndsTheCredential: the run deletes the record Close could
// not, rather than store it again.
func (s *BrokerSuite) TestARunAfterACloseTheStoreRefusedEndsTheCredential() {
	s.rig.gate.shut(true)
	s.Require().Error(s.broker.Close(s.ctx))
	s.rig.recover()

	s.Require().NoError(s.broker.BeforeRun(s.ctx))

	_, _, err := s.get()
	s.ErrorContains(err, "Proxy Authentication Required")
}

// TestARunWithEveryGrantWithdrawnEndsTheCredential: nothing is left for it to reach.
func (s *BrokerSuite) TestARunWithEveryGrantWithdrawnEndsTheCredential() {
	credential, _ := s.mount.Proxy.User.Password()
	s.fail(sandbox.ErrNoCredential)

	s.Require().NoError(s.broker.BeforeRun(s.ctx))

	exists, err := s.rig.redis.Do(s.ctx, s.rig.redis.B().Exists().Key(key(credential)).Build()).AsInt64()
	s.Require().NoError(err)
	s.Zero(exists)
}

// TestTheRecordHoldsTheConfigsAllowedDomains: as hosts, read as grants' hosts are, and kept
// once every grant is withdrawn, so code still reaches them.
func (s *BrokerSuite) TestTheRecordHoldsTheConfigsAllowedDomains() {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: publicURL, CACert: s.rig.caCert,
		Session: "session-1", Tokens: s.tokens, Allowed: []string{"PyPI.org.", "*.PythonHosted.org", ".npmjs.org", "**.x.org", "y.org",
			"*", "*.", "**", "..", "*.*", ""}})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = broker.Close(s.ctx) })
	mount, err := broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)
	credential, _ := mount.Proxy.User.Password()
	allowed := []string{"pypi.org", "*.pythonhosted.org", "*.npmjs.org", "*.x.org", "y.org"}
	record, found := s.rig.record(s.T(), key(credential))
	s.Require().True(found)
	s.Equal(allowed, record.Allowed)
	s.fail(sandbox.ErrNoCredential)

	s.Require().NoError(broker.BeforeRun(s.ctx))

	record, found = s.rig.record(s.T(), key(credential))
	s.Require().True(found)
	s.Empty(record.Grants)
	s.Equal(allowed, record.Allowed)
}

// TestClosingEndsTheCredentialOfARecordWithAllowedDomains: the allowed domains outlive no
// Close, so the next CONNECT to one is a 407.
func (s *BrokerSuite) TestClosingEndsTheCredentialOfARecordWithAllowedDomains() {
	broker, mount := s.allowing("session-allowed", "pypi.test")
	credential, _ := mount.Proxy.User.Password()
	_, found := s.rig.record(s.T(), key(credential))
	s.Require().True(found)

	s.Require().NoError(broker.Close(s.ctx))

	_, found = s.rig.record(s.T(), key(credential))
	s.False(found)
	_, err := s.rig.sandboxClient(s.T(), mount).Get("https://pypi.test/")
	s.ErrorContains(err, "Proxy Authentication Required")
}

// allowing is a broker of session with the allowed domains, and the mount it gave the grant.
func (s *BrokerSuite) allowing(session string, allowed ...string) (*egress.Broker, sandbox.Mount) {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: publicURL, CACert: s.rig.caCert,
		Session: session + "/" + s.rig.run, Tokens: s.tokens, Allowed: allowed})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = broker.Close(s.ctx) })
	mount, err := broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)
	return broker, mount
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

// publicURL is the proxy as a sandbox reaches it: a public IPv4 address (TEST-NET-3, RFC 5737).
const publicURL = "http://203.0.113.7:8091"

// unusableURLs are proxy URLs the sandbox cannot use: an address it may not reach or the
// provider's range would reach too much of (unspecified, loopback, RFC 1918, link-local with the
// metadata address, broadcast), a port that is none, or a path, query or fragment.
var unusableURLs = []string{"http://0.0.0.0:8091", "http://127.0.0.1:8091", "http://10.1.2.3:8091", "http://172.16.0.1:8091",
	"http://192.168.1.1:8091", "http://169.254.169.254:8091", "http://255.255.255.255:8091", "http://203.0.113.7",
	"http://203.0.113.7:0", "http://203.0.113.7:99999", "http://203.0.113.7:8091/", "http://203.0.113.7:8091/path",
	"http://203.0.113.7:8091?a=b", "http://203.0.113.7:8091#frag"}

// TestAProxyURLItCannotUseIsRefused: the sandbox's own credential is the URL's user, and the
// sandbox reaches an IPv4 address alone.
func (s *BrokerSuite) TestAProxyURLItCannotUseIsRefused() {
	for _, address := range append([]string{"", "https://proxy.test:8091", "https://user:pass@203.0.113.7:8091", "http://user:pass@proxy.test:8091", "http://:8091", "proxy.test:8091",
		"http://proxy.test:8091", "http://[2001:db8::7]:8091", "http://[::ffff:203.0.113.7]:8091", "http://user:pass@203.0.113.7:8091"}, unusableURLs...) {
		_, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: address, CACert: s.rig.caCert, Session: "session-1",
			Tokens: s.tokens})
		s.Error(err, address)
		if err != nil {
			s.NotContains(err.Error(), "pass", address)
		}
	}
	_, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: publicURL, Session: "session-1", Tokens: s.tokens})
	s.Error(err, "no CA certificate")
	_, err = egress.NewBroker(egress.BrokerOptions{URL: publicURL, CACert: s.rig.caCert, Session: "session-1", Tokens: s.tokens})
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

// TestNoGrantWithACredentialGivesNoProxyWhateverIsAllowed: allowed domains alone make no
// record, and the sandbox is created without a proxy.
func (s *BrokerSuite) TestNoGrantWithACredentialGivesNoProxyWhateverIsAllowed() {
	s.fail(sandbox.ErrNoCredential)
	_, mount := s.allowing("session-allowed", "pypi.test")

	s.Equal(sandbox.Mount{}, mount)
	for _, key := range s.rig.keys() {
		record, _ := s.rig.record(s.T(), key)
		s.NotEqual("session-allowed/"+s.rig.run, record.Session)
	}
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
	// handler is the proxy that proxy serves.
	handler *egress.Proxy
	proxy   *httptest.Server
	// upstream signs the hosts' certificates.
	upstream    *x509.Certificate
	upstreamKey *ecdsa.PrivateKey
	mu          sync.Mutex
	// git is github.com, made by the first push. credentialed is, for each CONNECT the proxy
	// got, whether it carried Proxy-Authorization.
	git          *gitHost
	credentialed []bool
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
	r.upstream, r.upstreamKey = upstreamCA, upstreamKey
	dial := &net.Dialer{}
	proxy, err := egress.New(egress.Options{Store: egress.NewStore(r.redis), CACert: caCert, CAKey: keyPEM(t, caKey), Logger: logger,
		Tokens: func(ctx context.Context, grant egress.Grant) (string, error) {
			token, err := tokens(ctx, sandbox.Grant{Connection: grant.Connection, Binding: grant.Binding, Hosts: grant.Hosts})
			return token.Value, err
		},
		Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: upstream},
			DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
				if git := r.github(); git != nil && address == "github.com:443" {
					return dial.DialContext(ctx, network, git.server.Listener.Addr().String())
				}
				return dial.DialContext(ctx, network, r.host.server.Listener.Addr().String())
			}}})
	if err != nil {
		t.Fatal(err)
	}
	r.handler = proxy
	r.proxy = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method == http.MethodConnect {
			r.mu.Lock()
			r.credentialed = append(r.credentialed, request.Header.Get("Proxy-Authorization") != "")
			r.mu.Unlock()
		}
		proxy.ServeHTTP(w, request)
	}))
	t.Cleanup(r.proxy.Close)
	return r
}

func (r *rig) broker(t *testing.T, session string, tokens sandbox.Tokens, logger *slog.Logger) *egress.Broker {
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: r.store, URL: publicURL, CACert: r.caCert,
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
	response, err := r.sandboxClient(t, mount).Do(request)
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

func (r *rig) github() *gitHost {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.git
}

func (r *rig) connects() []bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.credentialed)
}

// push is git as code in the sandbox mount runs it, with each run's env alone: it commits a
// file holding content and pushes it to github.com with ref as Basic auth's password, which is
// how GitHub takes a token over git
// (https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation:
// «git clone https://x-access-token:TOKEN@github.com/owner/repo.git»). The proxy's public
// address is not one a test can reach, so the proxy env names the rig's proxy in its place, and
// the CA files the env names are under a directory standing for the sandbox's root. It returns
// what github.com got: each request's Authorization, the file, and every other header value and
// each request's URL. The pack is stored uncompressed (pack.compression=0), so a placeholder in
// the file is plain text in the push body.
func (r *rig) push(t *testing.T, mount sandbox.Mount, ref, content string) ([]string, string, []string) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	git := r.github()
	if git == nil {
		git = newGitHost(t, r.upstream, r.upstreamKey)
		r.mu.Lock()
		r.git = git
		r.mu.Unlock()
	}
	root, work := t.TempDir(), t.TempDir()
	// The user's own git config stays out (GIT_CONFIG_NOSYSTEM, a HOME of its own), so no
	// credential helper keeps the placeholder.
	env := []string{"PATH=" + os.Getenv("PATH"), "HOME=" + t.TempDir(), "GIT_CONFIG_NOSYSTEM=1", "GIT_TERMINAL_PROMPT=0"}
	for name, value := range mount.RunEnv {
		switch {
		case slices.Contains([]string{"HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"}, name):
			proxy := *mount.Proxy
			proxy.Host = r.proxy.Listener.Addr().String()
			value = proxy.String()
		case strings.HasPrefix(value, "/tmp/"):
			// The bundle holds the image's roots too; github.com is the rig's, signed by the CA.
			value = root + value
			if err := os.MkdirAll(filepath.Dir(value), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(value, mount.Files[mount.RunEnv["NODE_EXTRA_CA_CERTS"]], 0o600); err != nil {
				t.Fatal(err)
			}
		}
		env = append(env, name+"="+value)
	}
	if err := os.WriteFile(filepath.Join(work, "token.txt"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"init", "--quiet", "--initial-branch=main"}, {"add", "token.txt"},
		{"-c", "user.name=sandbox", "-c", "user.email=sandbox@example.com", "commit", "--quiet", "--message=token"},
		{"-c", "pack.compression=0", "push", "--quiet", "https://x-access-token:" + ref + "@github.com/octo/hello.git", "main"}} {
		command := exec.Command("git", args...)
		command.Dir, command.Env = work, env
		if out, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", args[0], err, out)
		}
	}
	return git.requests(), git.file(t, "token.txt"), git.elsewhere()
}

// gitHost is github.com's git over HTTPS as a stand-in: git http-backend
// (https://git-scm.com/docs/git-http-backend) over one bare repository, octo/hello.git, which
// answers a request with no Basic credential 401 asking for one, as github.com did to git with
// none (probe, 2026-10-22T03:53:09Z). It takes any Basic credential:
// the test reads what each request carried.
type gitHost struct {
	server *httptest.Server
	repo   string
	mu     sync.Mutex
	seen   []string
	rest   []string
}

func newGitHost(t *testing.T, ca *x509.Certificate, caKey *ecdsa.PrivateKey) *gitHost {
	root := t.TempDir()
	g := &gitHost{repo: filepath.Join(root, "octo", "hello.git")}
	git, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"init", "--quiet", "--bare", "--initial-branch=main", g.repo},
		{"--git-dir", g.repo, "config", "http.receivepack", "true"}} {
		if out, err := exec.Command(git, args...).CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", args[0], err, out)
		}
	}
	backend := &cgi.Handler{Path: git, Args: []string{"http-backend"},
		Env: []string{"GIT_PROJECT_ROOT=" + root, "GIT_HTTP_EXPORT_ALL=1", "GIT_CONFIG_NOSYSTEM=1", "HOME=" + root}}
	g.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		g.mu.Lock()
		// Every Authorization value is kept, so a second line holding the token shows; a request with
		// none is kept as "". Go moves Host out of Header, so it is kept apart.
		authorizations := request.Header.Values("Authorization")
		if len(authorizations) == 0 {
			authorizations = []string{""}
		}
		g.seen = append(g.seen, authorizations...)
		g.rest = append(g.rest, request.URL.String(), request.Host)
		for name, values := range request.Header {
			if name != "Authorization" {
				g.rest = append(g.rest, values...)
			}
		}
		g.mu.Unlock()
		if _, _, ok := request.BasicAuth(); !ok {
			w.Header().Set("WWW-Authenticate", `Basic realm="GitHub"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		backend.ServeHTTP(w, request)
	}))
	g.server.TLS = &tls.Config{Certificates: []tls.Certificate{sign(t, ca, caKey, "github.com")}}
	g.server.StartTLS()
	t.Cleanup(g.server.Close)
	return g
}

func (g *gitHost) requests() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.seen)
}

// elsewhere is each request's URL, Host and every header value but Authorization.
func (g *gitHost) elsewhere() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.rest)
}

// file is name on main as the repository keeps it.
func (g *gitHost) file(t *testing.T, name string) string {
	out, err := exec.Command("git", "--git-dir", g.repo, "show", "main:"+name).Output()
	if err != nil {
		t.Fatalf("git show: %v", err)
	}
	return string(out)
}

// keys are every record's key.
func (r *rig) keys() []string {
	keys, err := r.redis.Do(context.Background(), r.redis.B().Keys().Pattern("sandbox:egress:*").Build()).AsStrSlice()
	if err != nil {
		panic(err)
	}
	return keys
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

// sandboxClient is an HTTP client set up from what the sandbox was given alone: each run's env
// names the proxy, with the sandbox's credential, and the CA file. The proxy's public address
// is not one a test can reach, so the client dials the rig's proxy in its place.
func (r *rig) sandboxClient(t *testing.T, mount sandbox.Mount) *http.Client {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(mount.Files[mount.RunEnv["NODE_EXTRA_CA_CERTS"]]) {
		t.Fatal("the sandbox's CA file holds no certificate")
	}
	proxy, err := url.Parse(mount.RunEnv["HTTPS_PROXY"])
	if err != nil {
		t.Fatal(err)
	}
	dialer := &net.Dialer{}
	return &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxy), TLSClientConfig: &tls.Config{RootCAs: roots},
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return dialer.DialContext(ctx, network, r.proxy.Listener.Addr().String())
		}}}
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
