package main

import (
	"errors"
	"log/slog"
	"net"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
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

// serveSandboxProxy has serve run proxy on addr, until stop. Without a proxy it listens on
// nothing.
func serveSandboxProxy(proxy *egress.Proxy, addr string, serve func(*http.Server, net.Listener),
	logger *slog.Logger) (stop func() error, err error) {
	if proxy == nil {
		return func() error { return nil }, nil
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	server := &http.Server{Handler: proxy, ReadHeaderTimeout: readHeaderTimeout}
	logger.Info("serving the sandbox egress proxy", "address", listener.Addr().String(), "setting", "code_sandbox.proxy_addr")
	go serve(server, listener)
	// Close, not a drain: a tunnel is the sandbox's to reopen.
	return server.Close, nil
}
