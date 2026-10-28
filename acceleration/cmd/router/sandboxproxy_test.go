package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// TestARouterWithoutAProxyAddressServesNoProxy is the control: unset, as on every router
// before the proxy, there is nothing to listen with.
func TestARouterWithoutAProxyAddressServesNoProxy(t *testing.T) {
	proxy, err := newSandboxProxy(config.Defaults().CodeSandbox, nil, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.Nil(t, proxy)
}

// TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart: the grants live in Redis and the
// tokens come from the connections resolver.
func TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart(t *testing.T) {
	settings := config.Defaults().CodeSandbox
	settings.ProxyAddr = ":8091"

	_, err := newSandboxProxy(settings, nil, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.ErrorContains(t, err, "redis.addr")
	_, err = newSandboxProxy(settings, nil, &resolver.Resolver{}, nil, nil, slog.New(slog.DiscardHandler))
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	_, err = newSandboxProxy(settings, redis, nil, nil, nil, slog.New(slog.DiscardHandler))
	require.ErrorContains(t, err, "connectors.enabled")
}

// TestAProxyAddressServesTheProxyThereAlone: the CA and key from the settings sign, the
// listener on the address asked for answers as the proxy (a GET is refused, 405, not served
// by the public API), and stop frees the port.
func TestAProxyAddressServesTheProxyThereAlone(t *testing.T) {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	settings := config.Defaults().CodeSandbox
	settings.ProxyAddr = freeAddress(t)
	settings.ProxyCACert, settings.ProxyCAKey = newCA(t)

	proxy, err := newSandboxProxy(settings, redis, &resolver.Resolver{}, nil, nil, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	require.NotNil(t, proxy)
	stop, _, err := serveSandboxProxy(proxy, settings.ProxyAddr, 0, settings.ProxyPublicURL, false, func(server *http.Server, listener net.Listener) {
		_ = server.Serve(listener)
	}, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	response, err := http.Get("http://" + settings.ProxyAddr + "/")
	require.NoError(t, err)
	_ = response.Body.Close()

	require.Equal(t, http.StatusMethodNotAllowed, response.StatusCode)
	require.Equal(t, http.MethodConnect, response.Header.Get("Allow"))
	require.NoError(t, stop())
	freed, err := net.Listen("tcp", settings.ProxyAddr)
	require.NoError(t, err)
	_ = freed.Close()
}

// TestNoProxyServesNothing: the router without one listens on no proxy address, whatever its
// public URL, and stop is safe.
func TestNoProxyServesNothing(t *testing.T) {
	address := freeAddress(t)
	stop, handOver, err := serveSandboxProxy(nil, address, 0, tlsPublicURL, true, func(*http.Server, net.Listener) {}, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(func() { _ = stop() })
	handOver()

	free, err := net.Listen("tcp", address)

	require.NoError(t, err)
	_ = free.Close()
}

// freeAddress is a loopback address no one listens on.
func freeAddress(t *testing.T) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	address := listener.Addr().String()
	require.NoError(t, listener.Close())
	return address
}

// newCA is a CA certificate and its key in PEM, as code_sandbox.proxy_ca_cert and
// proxy_ca_key take them.
func newCA(t *testing.T) (string, string) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "router sandbox CA"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	require.NoError(t, err)
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})),
		string(pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}))
}

// TestTheDaytonaBrokerAsksNothingOfTheProxy is the control: the default broker gets the
// settings base gave the session manager, needs no Redis and gets no proxy.
func TestTheDaytonaBrokerAsksNothingOfTheProxy(t *testing.T) {
	settings := config.Defaults().CodeSandbox
	settings.SecretPrefix = "stg"
	var options session.ManagerOptions

	err := sandboxBroker(settings, nil, &options)

	require.NoError(t, err)
	require.Equal(t, session.ManagerOptions{SandboxBroker: "daytona", SandboxSecretPrefix: "stg"}, options)
}

// TestTheProxyBrokerNeedsRedisAndIsGivenTheProxy: the session manager is set to the proxy
// broker, whose grants are written where the proxy reads them, and the sandbox is given the
// CA's certificate, never its key.
func TestTheProxyBrokerNeedsRedisAndIsGivenTheProxy(t *testing.T) {
	settings := config.Defaults().CodeSandbox
	settings.Broker, settings.ProxyPublicURL = config.CodeSandboxBrokerProxy, "http://203.0.113.7:8091"
	settings.ProxyCACert, settings.ProxyCAKey = newCA(t)
	var options session.ManagerOptions

	err := sandboxBroker(settings, nil, &options)
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	require.NoError(t, sandboxBroker(settings, redis, &options))
	require.Equal(t, egress.BrokerName, options.SandboxBroker)
	require.NotNil(t, options.SandboxProxy.Store)
	require.Equal(t, "http://203.0.113.7:8091", options.SandboxProxy.URL)
	require.Equal(t, []byte(settings.ProxyCACert), options.SandboxProxy.CACert)
	require.NotContains(t, string(options.SandboxProxy.CACert), "PRIVATE KEY")
}

// TestTheProxyReadsTheGrantsItsBrokerWrites on a router whose shard has its own Redis: both
// use the one every shard shares, which here is the only one that answers.
func TestTheProxyReadsTheGrantsItsBrokerWrites(t *testing.T) {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	shard := forward(t, address)
	redis, err := live.New(live.Options{Address: address, Shard: live.Shard{Address: shard.Addr().String()}})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	require.NoError(t, shard.Close())
	settings := config.Defaults().CodeSandbox
	settings.Broker, settings.ProxyPublicURL = config.CodeSandboxBrokerProxy, "http://203.0.113.7:8091"
	settings.ProxyAddr = ":8091"
	settings.ProxyCACert, settings.ProxyCAKey = newCA(t)
	var options session.ManagerOptions
	require.NoError(t, sandboxBroker(settings, redis, &options))
	proxy, err := newSandboxProxy(settings, redis, &resolver.Resolver{}, nil, nil, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	server := httptest.NewServer(proxy)
	t.Cleanup(server.Close)
	credential := egress.NewCredential()
	require.NoError(t, options.SandboxProxy.Store.Put(context.Background(), credential, egress.Record{Session: "session-1",
		Grants: []egress.Grant{{Binding: "github", Placeholder: "va_ph_1", Hosts: []string{"api.github.com"}}}}, time.Minute))
	t.Cleanup(func() { _ = options.SandboxProxy.Store.Delete(context.Background(), credential) })

	conn, err := net.Dial("tcp", server.Listener.Addr().String())
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	// A proxy reading the cut shard waits on it: fail here rather than at the test's timeout.
	require.NoError(t, conn.SetDeadline(time.Now().Add(10*time.Second)))
	_, err = fmt.Fprintf(conn, "CONNECT other.example:443 HTTP/1.1\r\nHost: other.example:443\r\nProxy-Authorization: Basic %s\r\n\r\n",
		base64.StdEncoding.EncodeToString([]byte("sbx:"+credential)))
	require.NoError(t, err)
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	require.NoError(t, err)
	_ = response.Body.Close()

	require.Equal(t, http.StatusForbidden, response.StatusCode, "the record is found, and names no such host")
}

// forward is a listener that passes each connection on to address until it is closed, which
// cuts them all.
func forward(t *testing.T, address string) net.Listener {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	var mu sync.Mutex
	var open []net.Conn
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				mu.Lock()
				for _, c := range open {
					_ = c.Close()
				}
				mu.Unlock()
				return
			}
			upstream, err := net.Dial("tcp", address)
			if err != nil {
				_ = conn.Close()
				continue
			}
			mu.Lock()
			open = append(open, conn, upstream)
			mu.Unlock()
			go func() { _, _ = io.Copy(upstream, conn); _ = upstream.Close() }()
			go func() { _, _ = io.Copy(conn, upstream); _ = conn.Close() }()
		}
	}()
	t.Cleanup(func() { _ = listener.Close() })
	return listener
}

// tlsPublicURL is a proxy served over TLS at a public IPv4 address (TEST-NET-3, RFC 5737), which
// listenedProxy's clients reach at its loopback listener.
const tlsPublicURL = "https://203.0.113.7:8091"

// listenedProxy is a proxy served as serveSandboxProxy serves it for publicURL, on a loopback
// address, and a sandbox's credential whose record allows example.com, which the proxy
// tunnels to host.
type listenedProxy struct {
	address    string
	credential string
	host       *httptest.Server
	// trusted holds the router CA's certificate and host's, which a sandbox trusts.
	trusted *x509.CertPool
}

func newListenedProxy(t *testing.T, publicURL string) listenedProxy {
	return newLoggedListenedProxy(t, publicURL, slog.New(slog.DiscardHandler))
}

// newLoggedListenedProxy is newListenedProxy whose listener logs to logger.
func newLoggedListenedProxy(t *testing.T, publicURL string, logger *slog.Logger) listenedProxy {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	host := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "hello") }))
	t.Cleanup(host.Close)
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{Store: sandboxStore(redis), CACert: []byte(caCert), CAKey: []byte(caKey),
		Dial: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, host.Listener.Addr().String())
		}, Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)
	credential := "w15tlsl-" + egress.NewCredential()
	require.NoError(t, sandboxStore(redis).Put(context.Background(), credential,
		egress.Record{Session: "w15tlsl-session", Allowed: []string{"example.com"}}, time.Minute))
	t.Cleanup(func() { _ = sandboxStore(redis).Delete(context.Background(), credential) })
	listen := freeAddress(t)
	stop, _, err := serveSandboxProxy(proxy, listen, 0, publicURL, false, func(server *http.Server, listener net.Listener) {
		_ = server.Serve(listener)
	}, logger)
	require.NoError(t, err)
	t.Cleanup(func() { _ = stop() })
	trusted := host.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs.Clone()
	require.True(t, trusted.AppendCertsFromPEM([]byte(caCert)))
	return listenedProxy{address: listen, credential: credential, host: host, trusted: trusted}
}

// client is a sandbox's client with proxyURL, whose credential is the sandbox's, trusting
// roots. It dials the listener wherever proxyURL points.
func (l listenedProxy) client(t *testing.T, proxyURL string, roots *x509.CertPool) *http.Client {
	proxy, err := url.Parse(proxyURL)
	require.NoError(t, err)
	proxy.User = url.UserPassword("sbx", l.credential)
	return &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{Proxy: http.ProxyURL(proxy),
		TLSClientConfig: &tls.Config{RootCAs: roots},
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, l.address)
		}}}
}

// connect sends a plain-HTTP CONNECT with the sandbox's credential to the listener and gives
// the first line it answers.
func (l listenedProxy) connect(t *testing.T) (string, error) {
	conn, err := net.Dial("tcp", l.address)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	require.NoError(t, conn.SetDeadline(time.Now().Add(10*time.Second)))
	_, err = fmt.Fprintf(conn, "CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\nProxy-Authorization: Basic %s\r\n\r\n",
		base64.StdEncoding.EncodeToString([]byte("sbx:"+l.credential)))
	require.NoError(t, err)
	return bufio.NewReader(conn).ReadString('\n')
}

// TestAnHTTPSPublicURLServesTheProxyOverTLS: a sandbox that trusts the router CA sends its
// credential inside TLS and reaches the host through the tunnel.
func TestAnHTTPSPublicURLServesTheProxyOverTLS(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	response, err := proxy.client(t, tlsPublicURL, proxy.trusted).Get("https://example.com/")

	require.NoError(t, err)
	body, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, response.StatusCode)
	require.Equal(t, "hello", string(body))
}

// TestTheTLSProxysCertificateNamesItsPublicAddress: the sandbox reaches the proxy at that IPv4
// address alone, so the certificate the CA signs names it as an IP address.
func TestTheTLSProxysCertificateNamesItsPublicAddress(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	conn, err := tls.Dial("tcp", proxy.address, &tls.Config{ServerName: "203.0.113.7", RootCAs: proxy.trusted})

	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	leaf := conn.ConnectionState().PeerCertificates[0]
	require.Len(t, leaf.IPAddresses, 1)
	require.Equal(t, "203.0.113.7", leaf.IPAddresses[0].String())
	require.Empty(t, leaf.DNSNames)
}

// TestTheTLSProxyNamesOnlyItsPublicAddressWhateverTheClientAsks: a client that sends another
// name as its SNI gets the same certificate, for the IP address alone. Were the certificate
// signed for the name asked, any client on the internet would get one from the router CA.
func TestTheTLSProxyNamesOnlyItsPublicAddressWhateverTheClientAsks(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	conn, err := tls.Dial("tcp", proxy.address, &tls.Config{ServerName: "example.com", InsecureSkipVerify: true})

	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	leaf := conn.ConnectionState().PeerCertificates[0]
	require.Empty(t, leaf.DNSNames)
	require.Len(t, leaf.IPAddresses, 1)
	require.Equal(t, "203.0.113.7", leaf.IPAddresses[0].String())
}

// TestTheTLSProxyHoldsAClientToTLS12AndHTTP11: TLS 1.1 is refused, and a client offering h2 is
// held to HTTP/1.1, as a tunnel does.
func TestTheTLSProxyHoldsAClientToTLS12AndHTTP11(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	_, err := tls.Dial("tcp", proxy.address, &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS10,
		MaxVersion: tls.VersionTLS11})
	require.Error(t, err)

	conn, err := tls.Dial("tcp", proxy.address, &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"h2", "http/1.1"}})
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	require.Equal(t, "http/1.1", conn.ConnectionState().NegotiatedProtocol)
	require.GreaterOrEqual(t, conn.ConnectionState().Version, uint16(tls.VersionTLS12))
}

// TestAFailedTLSHandshakeIsTheRouterLogsLine: not the std log's.
func TestAFailedTLSHandshakeIsTheRouterLogsLine(t *testing.T) {
	var out lockedBuffer
	proxy := newLoggedListenedProxy(t, tlsPublicURL, slog.New(slog.NewTextHandler(&out, nil)))

	conn, err := net.Dial("tcp", proxy.address)
	require.NoError(t, err)
	_, err = io.WriteString(conn, "GET / HTTP/1.1\r\n\r\n")
	require.NoError(t, err)
	_, _ = io.Copy(io.Discard, conn)
	_ = conn.Close()

	require.Eventually(t, func() bool { return strings.Contains(out.String(), "TLS handshake error") },
		5*time.Second, 10*time.Millisecond)
}

// lockedBuffer is a log sink the listener's goroutine writes and the test reads.
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

// TestPlainHTTPToTheTLSProxyGetsNoProxy: a CONNECT sent in the clear opens no tunnel and gets
// no answer.
func TestPlainHTTPToTheTLSProxyGetsNoProxy(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	line, err := proxy.connect(t)

	require.ErrorIs(t, err, io.EOF)
	require.Empty(t, line)
}

// TestASandboxThatDoesNotTrustTheRouterCAReachesNoTLSProxy: the proxy's certificate is the
// router CA's alone.
func TestASandboxThatDoesNotTrustTheRouterCAReachesNoTLSProxy(t *testing.T) {
	proxy := newListenedProxy(t, tlsPublicURL)

	_, err := proxy.client(t, tlsPublicURL, proxy.host.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs).
		Get("https://example.com/")

	var unknown x509.UnknownAuthorityError
	require.ErrorAs(t, err, &unknown)
}

// TestAnHTTPOrNoPublicURLServesThePlainProxyAsOnBase is the control. Each value is what base
// e5d6d8057 answered, with no public URL to give: a plain CONNECT opens the tunnel, a TLS
// handshake is not one, and a client with an http proxy URL reaches the host.
func TestAnHTTPOrNoPublicURLServesThePlainProxyAsOnBase(t *testing.T) {
	for _, publicURL := range []string{"http://203.0.113.7:8091", ""} {
		proxy := newListenedProxy(t, publicURL)

		line, err := proxy.connect(t)
		require.NoError(t, err, publicURL)
		require.Equal(t, "HTTP/1.1 200 Connection established\r\n", line, publicURL)
		_, err = tls.Dial("tcp", proxy.address, &tls.Config{InsecureSkipVerify: true})
		require.EqualError(t, err, "tls: first record does not look like a TLS handshake", publicURL)
		response, err := proxy.client(t, "http://"+proxy.address, proxy.trusted).Get("https://example.com/")
		require.NoError(t, err, publicURL)
		body, err := io.ReadAll(response.Body)
		_ = response.Body.Close()
		require.NoError(t, err)
		require.Equal(t, "200 hello", fmt.Sprintf("%d %s", response.StatusCode, body), publicURL)
	}
}

// serveBareProxy serves on address, as serveSandboxProxy does, a proxy with no store: enough
// to answer a request without a credential, which is all a test of its listener asks.
func serveBareProxy(t *testing.T, address string, overlap bool) (stop func() error, handOver func()) {
	t.Helper()
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{CACert: []byte(caCert), CAKey: []byte(caKey), Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)
	stop, handOver, err = serveSandboxProxy(proxy, address, 0, "", overlap, func(server *http.Server, listener net.Listener) {
		_ = server.Serve(listener)
	}, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	require.Eventually(t, func() bool { return servesProxy(address) }, 5*time.Second, 10*time.Millisecond,
		"the proxy listener never answered")
	return stop, handOver
}

// servesProxy is whether a proxy takes a connection on address: it refuses a GET with 405.
// Each call dials anew, since a connection the proxy took is served after a hand-over.
func servesProxy(address string) bool {
	client := http.Client{Timeout: time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	response, err := client.Get("http://" + address + "/")
	if err != nil {
		return false
	}
	_ = response.Body.Close()
	return response.StatusCode == http.StatusMethodNotAllowed
}

// TestAnAddressThatCannotBeListenedOnStillFailsTheStart: only an address another router holds
// is waited for. 192.0.2.1 is TEST-NET-1 (RFC 5737), on no interface of this host.
func TestAnAddressThatCannotBeListenedOnStillFailsTheStart(t *testing.T) {
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{CACert: []byte(caCert), CAKey: []byte(caKey), Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)

	_, _, err = serveSandboxProxy(proxy, "192.0.2.1:8091", 0, "", true, func(*http.Server, net.Listener) {}, slog.New(slog.DiscardHandler))

	require.ErrorContains(t, err, "192.0.2.1:8091")
}

// TestAHeldAddressFailsTheStartWithoutAnOverlapAsOnBase is the control. Base e5d6d8057 answered
// "listen tcp <address>: bind: address already in use" and did not start; only a router started
// beside the one it replaces, which the inherited listener says, waits.
func TestAHeldAddressFailsTheStartWithoutAnOverlapAsOnBase(t *testing.T) {
	holder, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = holder.Close() })
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{CACert: []byte(caCert), CAKey: []byte(caKey), Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)

	_, _, err = serveSandboxProxy(proxy, holder.Addr().String(), 0, "", false, func(*http.Server, net.Listener) {},
		slog.New(slog.DiscardHandler))

	require.EqualError(t, err, "listen tcp "+holder.Addr().String()+": bind: address already in use")
}

// TestWithoutAnOverlapTheProxyIsServedUntilTheRouterStops is the control: base takes proxy
// connections until the process exits, and a hand-over is for the overlap deploy alone.
func TestWithoutAnOverlapTheProxyIsServedUntilTheRouterStops(t *testing.T) {
	address := freeAddress(t)
	stop, handOver := serveBareProxy(t, address, false)
	t.Cleanup(func() { _ = stop() })

	handOver()

	require.True(t, servesProxy(address), "the proxy stopped taking connections at the hand-over")
	require.NoError(t, stop())
	require.False(t, servesProxy(address))
}

// SandboxProxyHandOverSuite is two routers on one proxy address, as an overlap deploy runs them
// in one pod: the old one serving, the new one started beside it. Each tunnels a sandbox to a
// host of its own, so a body says which router a tunnel goes through.
type SandboxProxyHandOverSuite struct {
	suite.Suite
	redis      *live.Client
	address    string
	credential string
	caCert     string
	caKey      string
	trusted    *x509.CertPool
	// accepted is told of each connection a router's proxy accepts.
	accepted chan struct{}
	// served is what each router's Serve ended with: serve in main takes all but
	// http.ErrServerClosed for a failure that stops the router.
	served chan error
	// devrun is the proxy socket a supervisor holds across routers, as devrun holds the
	// public one; nil, the default, has each router listen on address itself.
	devrun *os.File
}

func TestSandboxProxyHandOverSuite(t *testing.T) {
	suite.Run(t, new(SandboxProxyHandOverSuite))
}

func (s *SandboxProxyHandOverSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.redis = redis
}

func (s *SandboxProxyHandOverSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

// SetupTest: a credential whose record lets the sandbox reach example.com, named with this
// PR's id since the Redis is shared.
func (s *SandboxProxyHandOverSuite) SetupTest() {
	s.address = freeAddress(s.T())
	s.caCert, s.caKey = newCA(s.T())
	s.trusted = x509.NewCertPool()
	s.accepted = make(chan struct{}, 16)
	s.served = make(chan error, 4)
	s.devrun = nil
	s.credential = "w21ho-" + egress.NewCredential()
	store := sandboxStore(s.redis)
	s.Require().NoError(store.Put(context.Background(), s.credential,
		egress.Record{Session: "w21ho-session", Allowed: []string{"example.com"}}, time.Minute))
	s.T().Cleanup(func() { _ = store.Delete(context.Background(), s.credential) })
}

// startRouter is a router's proxy on the suite's address, tunnelling to a host that answers
// body, or the error serveSandboxProxy gives.
func (s *SandboxProxyHandOverSuite) startRouter(body string, logger *slog.Logger) (stop func() error, handOver func(), err error) {
	return s.startRouterAt("", body, logger)
}

// startRouterAt is startRouter for a proxy reached at publicURL.
func (s *SandboxProxyHandOverSuite) startRouterAt(publicURL, body string, logger *slog.Logger) (stop func() error, handOver func(), err error) {
	host := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) }))
	s.T().Cleanup(host.Close)
	s.trusted.AddCert(host.Certificate())
	proxy, err := egress.New(egress.Options{Store: sandboxStore(s.redis), CACert: []byte(s.caCert), CAKey: []byte(s.caKey),
		Dial: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, host.Listener.Addr().String())
		}, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	// This test's channels: a router's Serve may end after the next test has set its own.
	accepted, served := s.accepted, s.served
	// Each router is handed a duplicate of the supervisor's socket, as exec hands a child its
	// ExtraFiles, and closes it once it serves.
	fd := 0
	if s.devrun != nil {
		fd, err = syscall.Dup(int(s.devrun.Fd()))
		s.Require().NoError(err)
	}
	stop, handOver, err = serveSandboxProxy(proxy, s.address, fd, publicURL, true, func(server *http.Server, listener net.Listener) {
		server.ConnState = func(_ net.Conn, state http.ConnState) {
			if state == http.StateNew {
				select {
				case accepted <- struct{}{}:
				default:
				}
			}
		}
		served <- server.Serve(listener)
	}, logger)
	if err == nil {
		s.T().Cleanup(func() { _ = stop() })
	}
	return stop, handOver, err
}

// sandboxTunnel is a sandbox's TLS connection to example.com through the proxy.
type sandboxTunnel struct {
	conn   *tls.Conn
	reader *bufio.Reader
}

// openTunnel sends a CONNECT with the sandbox's credential to whichever router listens on the
// address, and starts TLS with the host through the tunnel it opens.
func (s *SandboxProxyHandOverSuite) openTunnel() (*sandboxTunnel, error) {
	conn, err := net.DialTimeout("tcp", s.address, time.Second)
	if err != nil {
		return nil, err
	}
	s.T().Cleanup(func() { _ = conn.Close() })
	if err := conn.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		return nil, err
	}
	_, err = fmt.Fprintf(conn, "CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\nProxy-Authorization: Basic %s\r\n\r\n",
		base64.StdEncoding.EncodeToString([]byte("sbx:"+s.credential)))
	if err != nil {
		return nil, err
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		return nil, err
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("CONNECT answered %s", response.Status)
	}
	tlsConn := tls.Client(conn, &tls.Config{ServerName: "example.com", RootCAs: s.trusted})
	return &sandboxTunnel{conn: tlsConn, reader: bufio.NewReader(tlsConn)}, nil
}

// get is the body the host answers a GET with, through the tunnel.
func (t *sandboxTunnel) get() (string, error) {
	if _, err := io.WriteString(t.conn, "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"); err != nil {
		return "", err
	}
	response, err := http.ReadResponse(t.reader, nil)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	return string(body), err
}

// throughTunnel is the body a new tunnel gets, or the error that kept it from one.
func (s *SandboxProxyHandOverSuite) throughTunnel() (string, error) {
	tunnel, err := s.openTunnel()
	if err != nil {
		return "", err
	}
	return tunnel.get()
}

// TestTheRouterTakingOverServesOnceTheDrainingOneHandsOver: until the hand-over the old router
// takes every CONNECT; from it the new one does, within the 1s AI-2030 allows, and a tunnel
// open on the old router still carries the sandbox's requests.
func (s *SandboxProxyHandOverSuite) TestTheRouterTakingOverServesOnceTheDrainingOneHandsOver() {
	_, handOverFirst, err := s.startRouter("first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	open, err := s.openTunnel()
	s.Require().NoError(err)
	body, err := open.get()
	s.Require().NoError(err)
	s.Require().Equal("first", body)

	_, _, err = s.startRouter("second", slog.New(slog.DiscardHandler))
	s.Require().NoError(err, "the router taking over did not start on the address the old one holds")
	body, err = s.throughTunnel()
	s.Require().NoError(err)
	s.Require().Equal("first", body, "the router taking over served before the hand-over")
	// An overlap holds the new router through several tries before the old one drains.
	time.Sleep(3 * proxyListenRetry)

	handedOver := time.Now()
	handOverFirst()
	var gap time.Duration
	s.Require().Eventually(func() bool {
		body, err := s.throughTunnel()
		gap = time.Since(handedOver)
		return err == nil && body == "second"
	}, 5*time.Second, 5*time.Millisecond, "the router taking over never served the proxy")
	s.Less(gap, time.Second, "sandboxes went without a proxy for longer than AI-2030 allows")
	s.ErrorIs(<-s.served, net.ErrClosed, "the draining router's Serve did not end with its listener")

	body, err = open.get()
	s.Require().NoError(err, "the hand-over cut a tunnel open on the draining router")
	s.Equal("first", body)
}

// TestASandboxConnectedBeforeTheHandOverStillGetsItsTunnel: the hand-over closes the listener
// alone, so a connection the old router took and whose CONNECT is still on its way is answered.
func (s *SandboxProxyHandOverSuite) TestASandboxConnectedBeforeTheHandOverStillGetsItsTunnel() {
	_, handOver, err := s.startRouter("first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	conn, err := net.DialTimeout("tcp", s.address, time.Second)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = conn.Close() })
	s.Require().NoError(conn.SetDeadline(time.Now().Add(10 * time.Second)))
	select {
	case <-s.accepted:
	case <-time.After(5 * time.Second):
		s.FailNow("the router never took the connection")
	}

	handOver()
	_, err = fmt.Fprintf(conn, "CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\nProxy-Authorization: Basic %s\r\n\r\n",
		base64.StdEncoding.EncodeToString([]byte("sbx:"+s.credential)))
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)

	s.Require().NoError(err, "the hand-over cut a connection the router had taken")
	_ = response.Body.Close()
	s.Equal(http.StatusOK, response.StatusCode)
}

// TestARouterDrainedBeforeItGetsTheAddressNeverTakesIt: a new router that is itself replaced
// while it waits stops waiting, so it never holds the address for a router after it.
func (s *SandboxProxyHandOverSuite) TestARouterDrainedBeforeItGetsTheAddressNeverTakesIt() {
	stopFirst, _, err := s.startRouter("first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	_, handOverSecond, err := s.startRouter("second", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	handOverSecond()
	s.Require().NoError(stopFirst())

	// The address, not s.address: Never's last check may run after the next test sets its own.
	address := s.address
	s.Never(func() bool {
		conn, err := net.DialTimeout("tcp", address, time.Second)
		if err != nil {
			return false
		}
		_ = conn.Close()
		return true
	}, 500*time.Millisecond, 20*time.Millisecond, "a router that had handed over took the address")
}

// TestHandingOverSaysSoOnce, as metrics do, and a router that only stops says nothing of it.
func (s *SandboxProxyHandOverSuite) TestHandingOverSaysSoOnce() {
	// The router serves an inherited socket, whose address is not the proxy address it is
	// given, so the line names the one it stopped serving.
	s.holdProxySocket()
	socket := s.address
	s.address = "127.0.0.1:1"
	var lines lockedBuffer
	stop, handOver, err := s.startRouter("first", slog.New(slog.NewTextHandler(&lines, nil)))
	s.Require().NoError(err)

	handOver()
	handOver()
	s.Require().NoError(stop())

	s.Equal(1, strings.Count(lines.String(), "stopped taking sandbox egress proxy connections for the router taking over"))
	s.Contains(lines.String(), "stopped taking sandbox egress proxy connections for the router taking over address="+socket)
	var stopped lockedBuffer
	stop, handOver, err = s.startRouter("second", slog.New(slog.NewTextHandler(&stopped, nil)))
	s.Require().NoError(err)
	s.Require().NoError(stop())
	handOver()
	s.NotContains(stopped.String(), "router taking over")
}

// TestTheTLSProxyHandsOverToTheRouterTakingOver: the proxy a router serves over TLS is the one
// it waits to serve and hands over, so a sandbox that trusts the router CA reaches the host
// through the new router, once the old one hands over, and through its own open tunnel still.
func (s *SandboxProxyHandOverSuite) TestTheTLSProxyHandsOverToTheRouterTakingOver() {
	s.Require().True(s.trusted.AppendCertsFromPEM([]byte(s.caCert)))
	_, handOverFirst, err := s.startRouterAt(tlsPublicURL, "first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	client := func(keepAlive bool) *http.Client {
		proxyURL, err := url.Parse(tlsPublicURL)
		s.Require().NoError(err)
		proxyURL.User = url.UserPassword("sbx", s.credential)
		return &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL),
			DisableKeepAlives: !keepAlive, TLSClientConfig: &tls.Config{RootCAs: s.trusted},
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, s.address)
			}}}
	}
	get := func(c *http.Client) (string, error) {
		response, err := c.Get("https://example.com/")
		if err != nil {
			return "", err
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		return string(body), err
	}
	kept := client(true)
	body, err := get(kept)
	s.Require().NoError(err)
	s.Require().Equal("first", body)
	_, _, err = s.startRouterAt(tlsPublicURL, "second", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	time.Sleep(3 * proxyListenRetry)

	handedOver := time.Now()
	handOverFirst()
	var gap time.Duration
	s.Require().Eventually(func() bool {
		body, err := get(client(false))
		gap = time.Since(handedOver)
		return err == nil && body == "second"
	}, 5*time.Second, 5*time.Millisecond, "the router taking over never served the proxy over TLS")

	s.Less(gap, time.Second)
	body, err = get(kept)
	s.Require().NoError(err, "the hand-over cut a TLS tunnel open on the draining router")
	s.Equal("first", body)
}

// waitingForTheAddress is a stack frame in a func of serveSandboxProxy: a wait for the proxy
// address. A "created by" line does not match, and the serve goroutines have no such frame.
var waitingForTheAddress = regexp.MustCompile(`(?m)^\S*router\.serveSandboxProxy\.func`)

// waitingGoroutines is how many routers' waits for the proxy address are running.
func waitingGoroutines() int {
	buffer := make([]byte, 1<<20)
	count := 0
	for _, goroutine := range strings.Split(string(buffer[:runtime.Stack(buffer, true)]), "\n\n") {
		if waitingForTheAddress.MatchString(goroutine) {
			count++
		}
	}
	return count
}

// TestTheWaitForTheAddressEndsWithTheRouterWhicheverWay: an address no router ever frees leaves
// the wait running until the router stops or hands over, and then it ends.
func (s *SandboxProxyHandOverSuite) TestTheWaitForTheAddressEndsWithTheRouterWhicheverWay() {
	for _, viaHandOver := range []bool{false, true} {
		holder, err := net.Listen("tcp", "127.0.0.1:0")
		s.Require().NoError(err)
		// Waits other tests ended are done within a retry.
		s.Require().Eventually(func() bool { return waitingGoroutines() == 0 }, 2*time.Second, 20*time.Millisecond)
		before := waitingGoroutines()
		s.address = holder.Addr().String()
		stop, handOver, err := s.startRouter("held", slog.New(slog.DiscardHandler))
		s.Require().NoError(err)
		time.Sleep(3 * proxyListenRetry)
		s.Require().Equal(before+1, waitingGoroutines(), "the router was not waiting for the address")

		if viaHandOver {
			handOver()
		} else {
			s.Require().NoError(stop())
		}

		s.Require().Eventually(func() bool { return waitingGoroutines() == before }, 2*time.Second, 20*time.Millisecond,
			"the wait outlived the router (hand over: %v)", viaHandOver)
		s.Require().NoError(holder.Close())
	}
}

// holdProxySocket is the supervisor's part of an overlap deploy: it listens on the proxy
// address and keeps the socket alone, never accepting on it, so it outlives every router.
func (s *SandboxProxyHandOverSuite) holdProxySocket() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	file, err := listener.(*net.TCPListener).File()
	s.Require().NoError(err)
	s.address = listener.Addr().String()
	s.Require().NoError(listener.Close())
	s.devrun = file
	s.T().Cleanup(func() { _ = file.Close() })
}

// TestRoutersOnTheInheritedSocketRefuseNoSandboxAcrossTheHandOver is AI-2071: a sandbox that
// keeps opening tunnels through an overlap deploy is never refused, and every tunnel opened
// after the old router hands over goes through the new one.
func (s *SandboxProxyHandOverSuite) TestRoutersOnTheInheritedSocketRefuseNoSandboxAcrossTheHandOver() {
	s.holdProxySocket()
	_, handOverFirst, err := s.startRouter("first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	type attempt struct {
		body string
		err  error
		// afterHandOver: the sandbox dialled once the old router had handed over.
		afterHandOver bool
	}
	var handedOver atomic.Bool
	done := make(chan struct{})
	attempts := make(chan attempt, 1<<14)
	var stopping sync.Once
	stopSandbox := func() { stopping.Do(func() { close(done) }) }
	// A failed assertion ends the sandbox too, before the test's other cleanups run.
	ended := make(chan struct{})
	s.T().Cleanup(func() { stopSandbox(); <-ended })
	go func() {
		defer close(ended)
		defer close(attempts)
		for {
			select {
			case <-done:
				return
			default:
			}
			after := handedOver.Load()
			tunnel, err := s.openTunnel()
			body := ""
			if err == nil {
				body, err = tunnel.get()
				_ = tunnel.conn.Close()
			}
			attempts <- attempt{body: body, err: err, afterHandOver: after}
		}
	}()
	// Under load the sandbox is slow to start: the swap waits for its first tunnel.
	var seen []attempt
	next := func() attempt {
		select {
		case a, open := <-attempts:
			s.Require().True(open)
			seen = append(seen, a)
			return a
		case <-time.After(10 * time.Second):
			s.FailNow("the sandbox got no answer")
			return attempt{}
		}
	}
	next()
	_, _, err = s.startRouter("second", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	// Both routers take tunnels for a while before the swap, as they do in a deploy.
	for range 20 {
		next()
	}

	handOverFirst()
	// The old router's Serve has ended: an Accept already under way when the listener closed
	// may still take a connection, so only a dial after this one is the new router's.
	s.ErrorIs(s.servedWithin(), net.ErrClosed, "the draining router's Serve did not end with its listener")
	handedOver.Store(true)
	for after := 0; after < 20; {
		if next().afterHandOver {
			after++
		}
	}
	stopSandbox()
	for a := range attempts {
		seen = append(seen, a)
	}

	for i, a := range seen {
		s.Require().NoError(a.err, "attempt %d of %d failed (after the hand-over: %v)", i, len(seen), a.afterHandOver)
		if a.afterHandOver {
			s.Require().Equal("second", a.body, "attempt %d went to the router that had handed over", i)
		}
	}
}

// servedWithin is what a router's Serve ended with, failing the test if none ends in 5s.
func (s *SandboxProxyHandOverSuite) servedWithin() error {
	select {
	case err := <-s.served:
		return err
	case <-time.After(5 * time.Second):
		s.FailNow("no router stopped serving")
		return nil
	}
}

// TestHandingOverTheInheritedSocketLeavesItListeningAndTheTunnelsOpen: the old router closes its
// own copy alone, so a sandbox that connects while no router accepts waits in the socket's queue
// for the next one, and the tunnel it already had keeps working.
func (s *SandboxProxyHandOverSuite) TestHandingOverTheInheritedSocketLeavesItListeningAndTheTunnelsOpen() {
	s.holdProxySocket()
	_, handOver, err := s.startRouter("first", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	open, err := s.openTunnel()
	s.Require().NoError(err)

	handOver()
	s.Require().ErrorIs(s.servedWithin(), net.ErrClosed, "the router still accepts after its hand-over")
	queued, err := net.DialTimeout("tcp", s.address, time.Second)
	s.Require().NoError(err, "the hand-over closed the supervisor's socket")
	s.T().Cleanup(func() { _ = queued.Close() })
	s.Require().NoError(queued.SetDeadline(time.Now().Add(10 * time.Second)))
	_, err = fmt.Fprintf(queued, "CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\nProxy-Authorization: Basic %s\r\n\r\n",
		base64.StdEncoding.EncodeToString([]byte("sbx:"+s.credential)))
	s.Require().NoError(err)
	_, _, err = s.startRouter("second", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(queued), nil)
	s.Require().NoError(err, "the sandbox queued during the hand-over was never served")
	_ = response.Body.Close()
	s.Equal(http.StatusOK, response.StatusCode)

	body, err := open.get()
	s.Require().NoError(err, "the hand-over cut a tunnel open on the draining router")
	s.Equal("first", body)
}

// TestAnInheritedDescriptorThatIsNoSocketFailsTheStart names the setting, so a supervisor
// passing the wrong descriptor is told which one.
func TestAnInheritedDescriptorThatIsNoSocketFailsTheStart(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "not-a-socket")
	require.NoError(t, err)
	t.Cleanup(func() { _ = file.Close() })
	fd, err := syscall.Dup(int(file.Fd()))
	require.NoError(t, err)
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{CACert: []byte(caCert), CAKey: []byte(caKey), Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)

	_, _, err = serveSandboxProxy(proxy, freeAddress(t), fd, "", true, func(*http.Server, net.Listener) {},
		slog.New(slog.DiscardHandler))

	require.ErrorContains(t, err, fmt.Sprintf("could not serve on the inherited socket %d (code_sandbox.proxy_listen_fd)", fd))
}

// TestAnInheritedProxySocketIsHandedOverEvenWithoutAnInheritedPublicOne: the proxy's socket
// alone says a supervisor runs the next router on it, so the hand-over stops this one accepting.
func TestAnInheritedProxySocketIsHandedOverEvenWithoutAnInheritedPublicOne(t *testing.T) {
	opened, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	file, err := opened.(*net.TCPListener).File()
	require.NoError(t, err)
	t.Cleanup(func() { _ = file.Close() })
	require.NoError(t, opened.Close())
	fd, err := syscall.Dup(int(file.Fd()))
	require.NoError(t, err)
	caCert, caKey := newCA(t)
	proxy, err := egress.New(egress.Options{CACert: []byte(caCert), CAKey: []byte(caKey), Logger: slog.New(slog.DiscardHandler)})
	require.NoError(t, err)
	served := make(chan error, 1)
	stop, handOver, err := serveSandboxProxy(proxy, ":8091", fd, "", false, func(server *http.Server, listener net.Listener) {
		served <- server.Serve(listener)
	}, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(func() { _ = stop() })

	handOver()

	select {
	case err := <-served:
		require.ErrorIs(t, err, net.ErrClosed)
	case <-time.After(5 * time.Second):
		t.Fatal("the router still accepts on the inherited socket after its hand-over")
	}
}
