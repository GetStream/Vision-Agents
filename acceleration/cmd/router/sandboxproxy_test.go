package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
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

// TestNoProxyServesNothing: the router without one starts no listener, and stop is safe.
func TestNoProxyServesNothing(t *testing.T) {
	served := false
	stop, err := serveSandboxProxy(nil, freeAddress(t), func(*http.Server, net.Listener) { served = true }, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.NoError(t, stop())
	require.False(t, served)
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
