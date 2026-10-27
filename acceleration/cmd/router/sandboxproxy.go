package main

import (
	"context"
	"crypto/tls"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"syscall"
	"time"

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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// newSandboxProxy is the sandbox egress proxy code_sandbox.proxy_addr asks for, nil when it
// is unset. It keeps the sandboxes' grants in Redis, so every replica serves every sandbox,
// and asks connections for each token, so it needs both. configs, the store connections read,
// is where it checks on each request that the agent config still binds the grant.
func newSandboxProxy(settings config.CodeSandbox, redis *live.Client, connections *resolver.Resolver, configs *store.Store,
	schemes map[string]core.Scheme, logger *slog.Logger) (*egress.Proxy, error) {
	if settings.ProxyAddr == "" {
		return nil, nil
	}
	if redis == nil || connections == nil {
		return nil, stack.Wrap(errors.New("code_sandbox.proxy_addr is set, and the sandbox egress proxy needs redis.addr " +
			"and connectors.enabled with a database"))
	}
	return egress.New(egress.Options{Store: sandboxStore(redis), Tokens: session.SandboxProxyTokens(connections, configs, schemes),
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

// proxyListenRetry is how often a router taking over tries the proxy address again while the
// router before it still holds it. Between the old router's hand-over and the new one's next
// try no router listens, and a sandbox's CONNECT is refused: 100ms keeps that gap a tenth of
// the 1s AI-2030 allows, and a listen that fails is one syscall, so ten a second for the length
// of an overlap deploy costs nothing.
const proxyListenRetry = 100 * time.Millisecond

// serveSandboxProxy has serve run proxy on addr, until stop: over TLS when publicURL, the proxy
// as a sandbox reaches it, is https, and plain HTTP otherwise. Without a proxy it listens on
// nothing.
//
// A router started beside the one it replaces, as an overlap deploy does, finds addr held: it
// keeps trying until the old router hands the address over, or ends. handOver is that hand-over,
// for a router that starts draining: it stops taking connections on addr and leaves the tunnels
// already open, which are the sandboxes' until the router stops. Either may be called more than
// once, and after the other.
func serveSandboxProxy(proxy *egress.Proxy, addr, publicURL string, serve func(*http.Server, net.Listener),
	logger *slog.Logger) (stop func() error, handOver func(), err error) {
	if proxy == nil {
		return func() error { return nil }, func() {}, nil
	}
	public, valid := config.ProxyURL(publicURL)
	secure := valid && public.Scheme == "https"
	// ErrorLog: a failed handshake is the router's log line, as in a tunnel, not the std log's.
	server := &http.Server{Handler: proxy, ReadHeaderTimeout: readHeaderTimeout,
		ErrorLog: slog.NewLogLogger(logger.Handler(), slog.LevelInfo)}
	start := func(listener net.Listener) {
		if secure {
			listener = tls.NewListener(listener, proxy.ListenerTLS(public.Hostname()))
		}
		logger.Info("serving the sandbox egress proxy", "address", listener.Addr().String(), "setting", "code_sandbox.proxy_addr",
			"tls", secure)
		go serve(server, listener)
	}
	done := make(chan struct{})
	listener, err := net.Listen("tcp", addr)
	switch {
	case err == nil:
		start(listener)
	case errors.Is(err, syscall.EADDRINUSE):
		logger.Info("the sandbox egress proxy address is held, waiting for the router before this one to hand it over",
			"address", addr, "setting", "code_sandbox.proxy_addr")
		go func() {
			ticker := time.NewTicker(proxyListenRetry)
			defer ticker.Stop()
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
				}
				// A listener bound just as the router stops is closed by serve: Serve on a
				// closed server closes it.
				if listener, err := net.Listen("tcp", addr); err == nil {
					start(listener)
					return
				}
			}
		}()
	default:
		return nil, nil, stack.Wrap(err)
	}
	var once sync.Once
	finish := func(handingOver bool) {
		once.Do(func() {
			close(done)
			if !handingOver {
				return
			}
			logger.Info("stopped taking sandbox egress proxy connections for the router taking over", "address", addr,
				"setting", "code_sandbox.proxy_addr")
			// Shutdown closes the listener and leaves the tunnels: they are hijacked connections,
			// which an http.Server neither closes nor waits for. It waits only for the CONNECTs
			// still being answered, so it runs on its own; stop ends it.
			go func() { _ = server.Shutdown(context.Background()) }()
		})
	}
	// Close, not a drain: a tunnel is the sandbox's to reopen.
	return func() error { finish(false); return server.Close() }, func() { finish(true) }, nil
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

// startSecretSweeper sweeps, until ctx ends, with the sweeper secretSweeper gives, if any.
func startSecretSweeper(ctx context.Context, settings config.CodeSandbox, redis *live.Client, directory *node.Directory,
	logger *slog.Logger) error {
	sweeper, err := secretSweeper(settings, redis, directory, logger)
	if err != nil || sweeper == nil {
		return err
	}
	go sweeper.Run(ctx)
	return nil
}
