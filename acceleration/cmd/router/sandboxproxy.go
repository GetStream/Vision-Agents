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
// overlap is whether a supervisor starts this router beside the one it replaces, as an overlap
// deploy does (the public listener is then inherited, settings.ListenFD). Only then is addr found
// held by the old router a thing to wait for: the router keeps trying until the old one hands the
// address over, or this one ends. handOver is that hand-over, for a router that starts draining:
// it stops taking connections on addr and leaves the tunnels already open, which are the
// sandboxes' until the router stops. Either may be called more than once, and after the other.
// Without overlap a held address fails the start, as on any listener, and handOver does nothing:
// the proxy takes connections until stop.
//
// fd, when not zero, is the proxy socket a supervisor handed down (code_sandbox.proxy_listen_fd),
// served in place of addr: the router before this one and this one accept from the same queue,
// so there is nothing to wait for. handOver then closes this router's copy of it alone, which
// stops it accepting while the supervisor's copy keeps the socket listening for the next router.
func serveSandboxProxy(proxy *egress.Proxy, addr string, fd int, publicURL string, overlap bool,
	serve func(*http.Server, net.Listener), logger *slog.Logger) (stop func() error, handOver func(), err error) {
	if proxy == nil {
		return func() error { return nil }, func() {}, nil
	}
	public, valid := config.ProxyURL(publicURL)
	secure := valid && public.Scheme == "https"
	// ErrorLog: a failed handshake is the router's log line, as in a tunnel, not the std log's.
	server := &http.Server{Handler: proxy, ReadHeaderTimeout: readHeaderTimeout,
		ErrorLog: slog.NewLogLogger(logger.Handler(), slog.LevelInfo)}
	// mu orders a listen while waiting against the hand-over, so a router done with the
	// address never takes it.
	var mu sync.Mutex
	var serving net.Listener
	finished := false
	setting := "code_sandbox.proxy_addr"
	if fd != 0 {
		setting = "code_sandbox.proxy_listen_fd"
	}
	// start serves on listener; mu is held.
	start := func(listener net.Listener) {
		serving = listener
		if secure {
			listener = tls.NewListener(listener, proxy.ListenerTLS(public.Hostname()))
		}
		logger.Info("serving the sandbox egress proxy", "address", listener.Addr().String(), "setting", setting, "tls", secure)
		go serve(server, listener)
	}
	// retry takes addr unless the router is done with it, and says whether to stop trying.
	retry := func() bool {
		mu.Lock()
		defer mu.Unlock()
		if finished {
			return true
		}
		listener, err := net.Listen("tcp", addr)
		if err != nil {
			return false
		}
		start(listener)
		return true
	}
	listener, err := listen(addr, fd, setting)
	switch {
	case err == nil:
		start(listener)
	case overlap && errors.Is(err, syscall.EADDRINUSE):
		logger.Info("the sandbox egress proxy address is held, waiting for the router before this one to hand it over",
			"address", addr, "setting", "code_sandbox.proxy_addr")
		go func() {
			ticker := time.NewTicker(proxyListenRetry)
			defer ticker.Stop()
			for range ticker.C {
				if retry() {
					return
				}
			}
		}()
	default:
		return nil, nil, stack.Wrap(err)
	}
	finish := func(handingOver bool) {
		mu.Lock()
		defer mu.Unlock()
		if finished {
			return
		}
		finished = true
		if handingOver {
			logger.Info("stopped taking sandbox egress proxy connections for the router taking over", "address", addr,
				"setting", setting)
		}
		// The listener alone: the connections the proxy took are still served, a request
		// they have yet to send included, which a Shutdown would drop. Serve ends with the
		// closed listener's error, which no one reads once the router drains. A stop closes
		// it too, because Serve runs on its own goroutine and the server only closes a
		// listener Serve has already started on: one it has not keeps the address held. An
		// inherited socket's listener is this router's duplicate of it: closing it leaves
		// the socket, and the connections queued on it, to the supervisor and the next router.
		if serving != nil {
			_ = serving.Close()
		}
	}
	// Close, not a drain: a tunnel is the sandbox's to reopen.
	// finish has closed the listener already, so Close finding it closed is no failure.
	stop = func() error {
		finish(false)
		if err := server.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			return err
		}
		return nil
	}
	if !overlap && fd == 0 {
		return stop, func() {}, nil
	}
	return stop, func() { finish(true) }, nil
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
