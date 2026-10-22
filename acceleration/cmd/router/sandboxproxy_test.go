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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// TestARouterWithoutAProxyAddressServesNoProxy is the control: unset, as on every router
// before the proxy, there is nothing to listen with.
func TestARouterWithoutAProxyAddressServesNoProxy(t *testing.T) {
	proxy, err := newSandboxProxy(config.Defaults().CodeSandbox, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.Nil(t, proxy)
}

// TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart: the grants live in Redis and the
// tokens come from the connections resolver.
func TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart(t *testing.T) {
	settings := config.Defaults().CodeSandbox
	settings.ProxyAddr = ":8091"

	_, err := newSandboxProxy(settings, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.ErrorContains(t, err, "redis.addr")
	_, err = newSandboxProxy(settings, nil, &resolver.Resolver{}, nil, slog.New(slog.DiscardHandler))
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	_, err = newSandboxProxy(settings, redis, nil, nil, slog.New(slog.DiscardHandler))
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

	proxy, err := newSandboxProxy(settings, redis, &resolver.Resolver{}, nil, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	require.NotNil(t, proxy)
	stop, err := serveSandboxProxy(proxy, settings.ProxyAddr, settings.ProxyPublicURL, func(server *http.Server, listener net.Listener) {
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
	stop, err := serveSandboxProxy(nil, address, tlsPublicURL, func(*http.Server, net.Listener) {}, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(func() { _ = stop() })

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
	proxy, err := newSandboxProxy(settings, redis, &resolver.Resolver{}, nil, slog.New(slog.DiscardHandler))
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
	stop, err := serveSandboxProxy(proxy, listen, publicURL, func(server *http.Server, listener net.Listener) {
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
