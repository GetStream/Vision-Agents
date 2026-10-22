//go:build integration

package egress_test

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
)

// tlsPublicURL is the proxy as a sandbox reaches it over TLS (TEST-NET-3, RFC 5737).
const tlsPublicURL = "https://203.0.113.7:8091"

// TestAnHTTPProxyURLGivesTheSandboxWhatBaseDid: with an http URL the broker is as before the
// tunnel. The expected env and the setup command's SHA-256 are a probe's on base b16b0e64c.
func (s *BrokerSuite) TestAnHTTPProxyURLGivesTheSandboxWhatBaseDid() {
	credential, _ := s.mount.Proxy.User.Password()
	proxy := "http://sbx:" + credential + "@203.0.113.7:8091"
	bundle := "/tmp/va-egress-ca-bundle.pem"
	s.Equal(map[string]string{"CURL_CA_BUNDLE": bundle, "GIT_SSL_CAINFO": bundle, "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy,
		"NODE_EXTRA_CA_CERTS": "/tmp/va-egress-ca.pem", "NO_PROXY": "localhost,127.0.0.1,::1", "REQUESTS_CA_BUNDLE": bundle,
		"SSL_CERT_FILE": bundle, "http_proxy": proxy, "https_proxy": proxy, "no_proxy": "localhost,127.0.0.1,::1",
		"GIT_HTTP_PROXY_AUTHMETHOD": "basic"}, s.mount.RunEnv)
	s.Equal(proxy, s.mount.Proxy.String())
	s.Equal([]string{"/tmp/va-egress-ca.pem"}, slices.Collect(maps.Keys(s.mount.Files)))
	s.Require().Len(s.mount.Setup, 1)
	sum := sha256.Sum256([]byte(s.mount.Setup[0]))
	s.Equal("05f358debc79cf0dddd46d4578ca1f9a332ea2d97f5176fa7e011824abb72fc9", hex.EncodeToString(sum[:]))
	s.Nil(s.mount.EachRun)
}

// TunnelSuite is the sandbox's TLS tunnel to an https proxy, run as the sandbox runs it: the
// broker's files under a directory standing for the sandbox's root, its commands in a shell,
// and clients set up from its env. The proxy is a real one, served over TLS, behind a wire that
// keeps every byte the forwarder sends it.
type TunnelSuite struct {
	suite.Suite
	ctx   context.Context
	rig   *rig
	mount sandbox.Mount
	// root stands for the sandbox's /, port is the forwarder's port here, and path the PATH
	// its commands run with.
	root, port, path string
	wire             *wire
}

func TestTunnelSuite(t *testing.T) {
	suite.Run(t, new(TunnelSuite))
}

func (s *TunnelSuite) SetupTest() {
	if _, err := exec.LookPath("python3"); err != nil {
		s.T().Skip("python3 is not installed: the forwarder is a Python script")
	}
	s.ctx = context.Background()
	tokens := func(context.Context, sandbox.Grant) (sandbox.Token, error) {
		return sandbox.Token{Value: "gho_live-token-of-the-connection", Revision: 1}, nil
	}
	logger := slog.New(slog.DiscardHandler)
	s.rig = newRig(s.T(), tokens, logger)
	broker, err := egress.NewBroker(egress.BrokerOptions{Store: s.rig.store, URL: tlsPublicURL, CACert: s.rig.caCert,
		Session: "w15tlst/" + s.rig.run, Tokens: tokens, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = broker.Close(context.Background()) })
	s.mount, err = broker.Prepare(s.ctx, []sandbox.Grant{github})
	s.Require().NoError(err)

	free, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	_, s.port, _ = net.SplitHostPort(free.Addr().String())
	s.Require().NoError(free.Close())
	s.root, s.path = s.T().TempDir(), os.Getenv("PATH")
	if _, err := exec.LookPath("setsid"); err != nil {
		// macOS has no setsid; Linux images and CI have util-linux's. The probe ran the real one.
		shim := s.T().TempDir()
		s.Require().NoError(os.WriteFile(filepath.Join(shim, "setsid"), []byte("#!/bin/sh\nexec \"$@\"\n"), 0o755))
		s.path = shim + ":" + s.path
	}
	for name, data := range s.mount.Files {
		s.Require().NoError(os.MkdirAll(filepath.Dir(s.root+name), 0o755))
		s.Require().NoError(os.WriteFile(s.root+name, data, 0o600))
	}
	s.wire = newWire(s, s.serveTLS(s.rig.handler, s.rig.handler.ListenerTLS("127.0.0.1")))
	s.T().Cleanup(func() {
		// A pid file a test wrote may name the test itself.
		if pid, err := s.pid(); err == nil && pid != os.Getpid() {
			_ = syscall.Kill(pid, syscall.SIGKILL)
		}
	})
}

// local is a command or URL of the sandbox as it runs here: under root, with the forwarder on
// port and the proxy at the wire.
func (s *TunnelSuite) local(command string) string {
	_, wirePort, _ := net.SplitHostPort(s.wire.listener.Addr().String())
	return strings.NewReplacer("/tmp/", s.root+"/tmp/", "3128", s.port, "203.0.113.7 8091", "127.0.0.1 "+wirePort).Replace(command)
}

func (s *TunnelSuite) sh(command string) (string, error) {
	cmd := exec.Command("/bin/sh", "-c", s.local(command))
	cmd.Env = append(os.Environ(), "PATH="+s.path)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// setUp runs the sandbox's setup, as Daytona does before any code.
func (s *TunnelSuite) setUp() {
	for _, command := range s.mount.Setup {
		out, err := s.sh(command)
		s.Require().NoError(err, out)
	}
}

func (s *TunnelSuite) pid() (int, error) {
	raw, err := os.ReadFile(s.root + "/tmp/va-egress-tunnel.pid")
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(string(raw))
}

// get is a request from code in the sandbox, through the proxy env it was given, with its
// placeholder as a bearer token. It gives up after 5 s, where one through the forwarder takes
// milliseconds, so a connection the forwarder leaves open fails the test.
func (s *TunnelSuite) get() (int, error) {
	roots := x509.NewCertPool()
	s.Require().True(roots.AppendCertsFromPEM(s.rig.caCert))
	proxy, err := url.Parse(s.local(s.mount.RunEnv["HTTPS_PROXY"]))
	s.Require().NoError(err)
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxy), TLSClientConfig: &tls.Config{RootCAs: roots},
		DisableKeepAlives: true}, Timeout: 5 * time.Second}
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, "https://api.github.com/user", nil)
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+s.mount.Env["GITHUB_TOKEN"])
	response, err := client.Do(request)
	if err != nil {
		return 0, err
	}
	_ = response.Body.Close()
	return response.StatusCode, nil
}

// TestAnHTTPSProxyURLSendsClientsToTheTunnel: every proxy env var names the forwarder on the
// sandbox's loopback, with the credential; node is told to read them; loopback and the CA stay
// as with http. Daytona's allow list still holds the proxy's address.
func (s *TunnelSuite) TestAnHTTPSProxyURLSendsClientsToTheTunnel() {
	credential, _ := s.mount.Proxy.User.Password()
	local := "http://sbx:" + credential + "@127.0.0.1:3128"
	bundle := "/tmp/va-egress-ca-bundle.pem"
	s.Equal(map[string]string{"CURL_CA_BUNDLE": bundle, "GIT_SSL_CAINFO": bundle, "HTTPS_PROXY": local, "HTTP_PROXY": local,
		"NODE_EXTRA_CA_CERTS": "/tmp/va-egress-ca.pem", "NO_PROXY": "localhost,127.0.0.1,::1", "REQUESTS_CA_BUNDLE": bundle,
		"SSL_CERT_FILE": bundle, "http_proxy": local, "https_proxy": local, "no_proxy": "localhost,127.0.0.1,::1",
		"NODE_USE_ENV_PROXY": "1", "GIT_HTTP_PROXY_AUTHMETHOD": "basic"}, s.mount.RunEnv)
	s.Equal("https://sbx:"+credential+"@203.0.113.7:8091", s.mount.Proxy.String())
	s.ElementsMatch([]string{"/tmp/va-egress-ca.pem", "/tmp/va-egress-tunnel.py"}, slices.Collect(maps.Keys(s.mount.Files)))
	s.Require().Len(s.mount.Setup, 2)
	s.Equal([]string{s.mount.Setup[1]}, s.mount.EachRun)
}

// TestAClientReachesTheProxyThroughTheTunnelOverTLSAlone: the request gets through with the
// token in place of its placeholder, and what crosses the network to the proxy is TLS: no
// CONNECT, no Proxy-Authorization, no credential. The forwarder takes no connection from
// another interface.
func (s *TunnelSuite) TestAClientReachesTheProxyThroughTheTunnelOverTLSAlone() {
	s.setUp()

	status, err := s.get()

	s.Require().NoError(err)
	s.Equal(http.StatusOK, status)
	s.Equal([]string{"Bearer gho_live-token-of-the-connection"}, s.rig.host.authorizations())
	sent := s.wire.sent.String()
	credential, _ := s.mount.Proxy.User.Password()
	s.Require().NotEmpty(sent)
	s.Equal(byte(0x16), sent[0], "a TLS handshake record (RFC 8446 section 5.1)")
	for _, clear := range []string{"CONNECT", "Proxy-Authorization", credential, s.mount.Env["GITHUB_TOKEN"]} {
		s.NotContains(sent, clear)
	}
	addresses, err := net.InterfaceAddrs()
	s.Require().NoError(err)
	for _, address := range addresses {
		if ip, ok := address.(*net.IPNet); ok && ip.IP.To4() != nil && !ip.IP.IsLoopback() {
			conn, err := net.Dial("tcp", net.JoinHostPort(ip.IP.String(), s.port))
			if err == nil {
				_ = conn.Close()
			}
			s.Error(err, ip.IP.String())
		}
	}
}

// TestTwentyParallelClientsAllGetThrough: the threaded forwarder the probe tried lost 9 of 20.
func (s *TunnelSuite) TestTwentyParallelClientsAllGetThrough() {
	s.setUp()
	statuses := make([]int, 20)
	errs := make([]error, 20)
	var wg sync.WaitGroup
	for i := range statuses {
		wg.Go(func() { statuses[i], errs[i] = s.get() })
	}
	wg.Wait()

	for i := range statuses {
		s.NoError(errs[i])
		s.Equal(http.StatusOK, statuses[i])
	}
	s.Len(s.rig.host.authorizations(), 20)
}

// TestADeadForwarderIsStartedAgainBeforeTheNextRunAndALiveOneIsLeftAlone: a restart would cut
// a download in flight, so a forwarder that runs and accepts keeps its process.
func (s *TunnelSuite) TestADeadForwarderIsStartedAgainBeforeTheNextRunAndALiveOneIsLeftAlone() {
	s.setUp()
	first, err := s.pid()
	s.Require().NoError(err)

	out, err := s.sh(s.mount.EachRun[0])
	s.Require().NoError(err, out)
	second, err := s.pid()
	s.Require().NoError(err)
	s.Equal(first, second)
	s.Never(func() bool {
		log, _ := os.ReadFile(s.root + "/tmp/va-egress-tunnel.log")
		return len(log) > 0
	}, time.Second, 50*time.Millisecond, "no second forwarder was started")
	s.Empty(s.wire.sent.String(), "a check opens no connection to the proxy")

	s.Require().NoError(syscall.Kill(first, syscall.SIGKILL))
	s.Eventually(func() bool { _, err := s.get(); return err != nil }, 5*time.Second, 10*time.Millisecond)
	out, err = s.sh(s.mount.EachRun[0])
	s.Require().NoError(err, out)
	third, err := s.pid()
	s.Require().NoError(err)
	s.NotEqual(first, third)
	status, err := s.get()
	s.Require().NoError(err)
	s.Equal(http.StatusOK, status)
}

// TestAPidFileOfAnotherProcessStartsTheForwarder: a live process that is not the forwarder,
// such as one that took a dead forwarder's pid, does not pass for it.
func (s *TunnelSuite) TestAPidFileOfAnotherProcessStartsTheForwarder() {
	s.Require().NoError(os.WriteFile(s.root+"/tmp/va-egress-tunnel.pid", []byte(strconv.Itoa(os.Getpid())), 0o600))

	s.setUp()

	status, err := s.get()
	s.Require().NoError(err)
	s.Equal(http.StatusOK, status)
}

// TestAnotherProcessOnTheTunnelsPortFailsTheRun: it would get the sandbox's proxy traffic.
func (s *TunnelSuite) TestAnotherProcessOnTheTunnelsPortFailsTheRun() {
	listener, err := net.Listen("tcp", "127.0.0.1:"+s.port)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = listener.Close() })

	start := time.Now()
	out, err := s.sh(s.mount.EachRun[0])

	s.Require().Error(err)
	s.Contains(out, "the TLS tunnel to the proxy did not start")
	s.Contains(strings.ToLower(out), "address already in use", "the forwarder's own error")
	s.Less(time.Since(start), 5*time.Second, "no waiting once the forwarder exited: 50 tries sleep 5 s")
}

// TestASecondForwarderLeavesThePidFileToTheFirst: one that cannot have the port exits without
// naming itself, so the next check still finds the first.
func (s *TunnelSuite) TestASecondForwarderLeavesThePidFileToTheFirst() {
	s.setUp()
	first, err := s.pid()
	s.Require().NoError(err)

	out, err := s.sh("python3 /tmp/va-egress-tunnel.py 3128 203.0.113.7 8091 /tmp/va-egress-ca.pem /tmp/va-egress-tunnel.pid")

	s.Require().Error(err, out)
	pid, err := s.pid()
	s.Require().NoError(err)
	s.Equal(first, pid)
	out, err = s.sh(s.mount.EachRun[0])
	s.Require().NoError(err, out)
	pid, err = s.pid()
	s.Require().NoError(err)
	s.Equal(first, pid)
}

// exchange sends an HTTP/1.0 request to the forwarder, as a client that may then stop sending,
// and is what it reads back by the deadline.
func (s *TunnelSuite) exchange(halfClose bool) (string, error) {
	conn, err := net.Dial("tcp", "127.0.0.1:"+s.port)
	s.Require().NoError(err)
	defer func() { _ = conn.Close() }()
	_, err = conn.Write([]byte("GET http://stand.in/ HTTP/1.0\r\nHost: stand.in\r\n\r\n"))
	s.Require().NoError(err)
	if halfClose {
		s.Require().NoError(conn.(*net.TCPConn).CloseWrite())
	}
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	got, err := io.ReadAll(conn)
	return string(got), err
}

// standIn is a proxy that answers after the client has had time to stop sending, and closes, as
// Go's server does after an HTTP/1.0 reply.
func (s *TunnelSuite) standIn() {
	s.wire = newWire(s, s.serveTLS(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(200 * time.Millisecond)
		_, _ = w.Write([]byte("answered"))
	}), s.rig.handler.ListenerTLS("127.0.0.1")))
}

// TestAProxysCloseReachesTheClient: a client reading to the end is not left waiting.
func (s *TunnelSuite) TestAProxysCloseReachesTheClient() {
	s.standIn()
	s.setUp()

	got, err := s.exchange(false)

	s.Require().NoError(err)
	s.True(strings.HasPrefix(got, "HTTP/1.0 200 OK"), got)
	s.True(strings.HasSuffix(got, "answered"), got)
}

// TestAClientThatStopsSendingStillGetsTheReply: as it did from the proxy without the tunnel.
func (s *TunnelSuite) TestAClientThatStopsSendingStillGetsTheReply() {
	s.standIn()
	s.setUp()

	got, err := s.exchange(true)

	s.Require().NoError(err)
	s.True(strings.HasPrefix(got, "HTTP/1.0 200 OK"), got)
	s.True(strings.HasSuffix(got, "answered"), got)
}

// TestAnImageWithoutPython3FailsTheSetup: and Daytona runs no code after a failed setup.
func (s *TunnelSuite) TestAnImageWithoutPython3FailsTheSetup() {
	s.path = s.T().TempDir()

	out, err := s.sh(s.mount.Setup[1])

	s.Require().Error(err)
	s.Contains(out, "the sandbox image has no python3")
}

// TestAProxyWithoutTheRouterCAsCertificateGetsNoRequest: the forwarder trusts the router CA
// alone, not the roots SSL_CERT_FILE names, so a server at the proxy's address that cannot show
// its certificate gets nothing past the handshake.
func (s *TunnelSuite) TestAProxyWithoutTheRouterCAsCertificateGetsNoRequest() {
	_, otherCert, otherKey := newCA(s.T(), "another CA")
	roots := filepath.Join(s.T().TempDir(), "roots.pem")
	s.Require().NoError(os.WriteFile(roots, otherCert, 0o600))
	s.T().Setenv("SSL_CERT_FILE", roots)
	other, err := egress.New(egress.Options{Store: egress.NewStore(s.rig.redis), CACert: otherCert, CAKey: keyPEM(s.T(), otherKey),
		Tokens: func(context.Context, egress.Grant) (string, error) { return "", nil }})
	s.Require().NoError(err)
	var mu sync.Mutex
	var got []string
	impostor := s.serveTLS(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, r.Header.Get("Proxy-Authorization"))
	}), other.ListenerTLS("127.0.0.1"))
	s.wire = newWire(s, impostor)
	s.setUp()

	_, err = s.get()

	s.Error(err)
	mu.Lock()
	defer mu.Unlock()
	s.Empty(got)
}

// TestAProxyWithTheRouterCAsCertificateForAnotherAddressGetsNoRequest: the certificate must
// name the proxy's address.
func (s *TunnelSuite) TestAProxyWithTheRouterCAsCertificateForAnotherAddressGetsNoRequest() {
	var mu sync.Mutex
	got := 0
	s.wire = newWire(s, s.serveTLS(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		got++
	}), s.rig.handler.ListenerTLS("10.9.9.9")))
	s.setUp()

	_, err := s.get()

	s.Error(err)
	mu.Lock()
	defer mu.Unlock()
	s.Zero(got)
}

// serveTLS serves handler over TLS on loopback, as cmd/router serves the proxy for an https URL,
// and is its address.
func (s *TunnelSuite) serveTLS(handler http.Handler, config *tls.Config) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	server := &http.Server{Handler: handler}
	go func() { _ = server.Serve(tls.NewListener(listener, config)) }()
	s.T().Cleanup(func() { _ = server.Close() })
	return listener.Addr().String()
}

// wire passes each connection on to a proxy and keeps every byte sent to it.
type wire struct {
	listener net.Listener
	sent     lockedBuffer
}

func newWire(s *TunnelSuite, to string) *wire {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	w := &wire{listener: listener}
	s.T().Cleanup(func() { _ = listener.Close() })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			proxy, err := net.Dial("tcp", to)
			if err != nil {
				_ = conn.Close()
				continue
			}
			go func() { _, _ = io.Copy(io.MultiWriter(proxy, &w.sent), conn); _ = proxy.Close() }()
			go func() { _, _ = io.Copy(conn, proxy); _ = conn.Close() }()
		}
	}()
	return w
}
