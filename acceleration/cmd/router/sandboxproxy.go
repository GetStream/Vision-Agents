package main

import (
	"crypto/tls"
	"errors"
	"log/slog"
	"net"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/concurrency"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// newSandboxProxy is the sandbox egress proxy code_sandbox.proxy_addr asks for, nil when it
// is unset. It keeps the sandboxes' grants in Redis, so every replica serves every sandbox,
// and asks connections for each token, so it needs both.
func newSandboxProxy(settings config.CodeSandbox, redis *live.Client, connections *resolver.Resolver,
	schemes map[string]core.Scheme, logger *slog.Logger) (*egress.Proxy, error) {
	if settings.ProxyAddr == "" {
		return nil, nil
	}
	if redis == nil || connections == nil {
		return nil, stack.Wrap(errors.New("code_sandbox.proxy_addr is set, and the sandbox egress proxy needs redis.addr " +
			"and connectors.enabled with a database"))
	}
	return egress.New(egress.Options{Store: sandboxStore(redis), Tokens: egress.ResolverTokens(connections, schemes),
		CACert: []byte(settings.ProxyCACert), CAKey: []byte(settings.ProxyCAKey), Logger: logger})
}

// sandboxStore is where the proxy reads each sandbox's grants and the proxy broker writes
// them: the Redis every shard shares, since a sandbox's requests may reach any router's proxy.
func sandboxStore(redis *live.Client) *egress.Store {
	return egress.NewStore(redis.Redis())
}

// sandboxBroker sets the session manager's code sandbox broker from settings, and what the
// proxy broker needs of the router. Only a session with a sandbox binding reads them.
func sandboxBroker(settings config.CodeSandbox, redis *live.Client, options *session.ManagerOptions) error {
	options.SandboxBroker, options.SandboxSecretPrefix = settings.Broker, settings.SecretPrefix
	if settings.Broker != config.CodeSandboxBrokerProxy {
		return nil
	}
	if redis == nil {
		return stack.Wrap(errors.New("code_sandbox.broker is proxy, and the proxy reads each " +
			"sandbox's grants from redis: set redis.addr"))
	}
	options.SandboxProxy = egress.BrokerOptions{Store: sandboxStore(redis), URL: settings.ProxyPublicURL,
		CACert: []byte(settings.ProxyCACert)}
	return nil
}

// serveSandboxProxy has serve run proxy on addr, until stop: over TLS when publicURL, the proxy
// as a sandbox reaches it, is https, and plain HTTP otherwise. Without a proxy it listens on
// nothing.
func serveSandboxProxy(proxy *egress.Proxy, addr, publicURL string, serve func(*http.Server, net.Listener),
	logger *slog.Logger) (stop func() error, err error) {
	if proxy == nil {
		return func() error { return nil }, nil
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	public, valid := config.ProxyURL(publicURL)
	secure := valid && public.Scheme == "https"
	if secure {
		listener = tls.NewListener(listener, proxy.ListenerTLS(public.Hostname()))
	}
	// ErrorLog: a failed handshake is the router's log line, as in a tunnel, not the std log's.
	server := &http.Server{Handler: proxy, ReadHeaderTimeout: readHeaderTimeout,
		ErrorLog: slog.NewLogLogger(logger.Handler(), slog.LevelInfo)}
	logger.Info("serving the sandbox egress proxy", "address", listener.Addr().String(), "setting", "code_sandbox.proxy_addr",
		"tls", secure)
	go serve(server, listener)
	// Close, not a drain: a tunnel is the sandbox's to reopen.
	return server.Close, nil
}

// secretSweeper sweeps the Daytona secrets a crashed router left behind, and is nil where there
// are none to sweep: under the proxy broker, whose grants expire in Redis, and without Daytona
// or a secret prefix, which every secret this router makes is named with. It also needs the
// node directory, the only thing that says whether a session runs on another router.
func secretSweeper(settings config.CodeSandbox, redis *live.Client, directory *node.Directory,
	logger *slog.Logger) (*daytona.Sweeper, error) {
	if settings.Broker != config.CodeSandboxBrokerDaytona || settings.SecretPrefix == "" || !daytona.Configured() {
		return nil, nil
	}
	// newDirectory is nil exactly when redis is.
	if directory == nil {
		logger.Warn("the Daytona secrets a crashed router leaves behind are not swept without redis", "setting", "redis.addr")
		return nil, nil
	}
	return daytona.NewSweeper(daytona.SweepOptions{Prefix: settings.SecretPrefix, Sessions: directory,
		Slots: concurrency.New(redis.Redis(), nil), Logger: logger})
}
