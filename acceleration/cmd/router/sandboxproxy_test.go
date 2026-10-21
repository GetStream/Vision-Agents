package main

import (
	"bufio"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
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
	"os"
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
	stop, err := serveSandboxProxy(proxy, settings.ProxyAddr, func(server *http.Server, listener net.Listener) {
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

// TestNoProxyServesNothing: the router without one listens on no proxy address, and stop
// is safe.
func TestNoProxyServesNothing(t *testing.T) {
	address := freeAddress(t)
	stop, err := serveSandboxProxy(nil, address, func(*http.Server, net.Listener) {}, slog.New(slog.DiscardHandler))
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
	settings.Broker, settings.ProxyPublicURL = config.CodeSandboxBrokerProxy, "http://proxy.stg.example:8091"
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
	require.Equal(t, "http://proxy.stg.example:8091", options.SandboxProxy.URL)
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
	settings.Broker, settings.ProxyPublicURL = config.CodeSandboxBrokerProxy, "http://proxy.stg.example:8091"
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
