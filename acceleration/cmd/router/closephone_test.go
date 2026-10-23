package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

type closeRecorder struct {
	phone.Provider
	closed int
}

func (c *closeRecorder) Close(context.Context) error {
	c.closed++
	return nil
}

func TestClosePhoneHangsUpTrunkCallsAndIsSafeToRepeat(t *testing.T) {
	trunks := &closeRecorder{}
	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  phone.NewRegistry(phone.Config{}),
		SIPTrunks: trunks,
	})
	require.NoError(t, err)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	closePhone(service, logger)
	closePhone(service, logger)

	require.Equal(t, 2, trunks.closed)
}

func TestStoppingDrainsHTTPForTheWholeGraceWhenNoTrunkCallIsUp(t *testing.T) {
	started, done := make(chan struct{}), make(chan struct{})
	httpServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			defer close(done)
			close(started)
			time.Sleep(200 * time.Millisecond)
			w.WriteHeader(http.StatusNoContent)
		}),
		ReadHeaderTimeout: time.Second,
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	go func() { _ = httpServer.Serve(listener) }()
	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  phone.NewRegistry(phone.Config{}),
		SIPTrunks: &closeRecorder{},
	})
	require.NoError(t, err)

	status := make(chan int, 1)
	go func() {
		res, err := http.Get("http://" + listener.Addr().String())
		if err != nil {
			status <- 0
			return
		}
		_ = res.Body.Close()
		status <- res.StatusCode
	}()
	<-started

	err = stopServing(stopping{
		public:    httpServer,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: service,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
	select {
	case <-done:
	default:
		t.Fatal("stopServing returned while the request was still running")
	}
	require.Equal(t, http.StatusNoContent, <-status)
}

// testGrace stands in for shutdownGrace, 10 s, so a request outlives it without the test
// waiting that long.
const testGrace = 100 * time.Millisecond

func idlePhone(t *testing.T) *phone.Service {
	t.Helper()
	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  phone.NewRegistry(phone.Config{}),
		SIPTrunks: &closeRecorder{},
	})
	require.NoError(t, err)
	return service
}

// servingAnOpenRequest is a server with one request that does not finish until the test
// ends: a slow request on a router told to stop.
func servingAnOpenRequest(t *testing.T) *http.Server {
	t.Helper()
	opened, release := make(chan struct{}), make(chan struct{})
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			close(opened)
			<-release
			w.WriteHeader(http.StatusNoContent)
		}),
		ReadHeaderTimeout: time.Second,
	}
	t.Cleanup(func() {
		close(release)
		_ = server.Close()
	})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	go func() { _ = server.Serve(listener) }()
	go func() {
		if res, err := http.Get("http://" + listener.Addr().String()); err == nil {
			_ = res.Body.Close()
		}
	}()
	<-opened
	return server
}

func TestARouterWithNoOpenRequestStopsWithoutAnError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }),
		ReadHeaderTimeout: time.Second,
	}
	go func() { _ = server.Serve(listener) }()
	require.True(t, answers(listener.Addr().String()))

	err = stopServing(stopping{
		public:    server,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		grace:     testGrace,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
}

func TestARequestOutlivingTheGraceIsNotAnError(t *testing.T) {
	var logs bytes.Buffer
	err := stopServing(stopping{
		public:    servingAnOpenRequest(t),
		peers:     servingAnOpenRequest(t),
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		grace:     testGrace,
		logger:    slog.New(slog.NewTextHandler(&logs, nil)),
	})

	require.NoError(t, err)
	require.Contains(t, logs.String(), "level=WARN")
	require.Contains(t, logs.String(), "requests were still running when the grace ran out")
	require.Contains(t, logs.String(), "listener=public grace="+testGrace.String())
	require.Contains(t, logs.String(), "listener=peers grace="+testGrace.String())
}

func TestAPeerRequestOutlivingTheGraceAfterADrainIsNotAnError(t *testing.T) {
	var logs bytes.Buffer
	err := stopServing(stopping{
		public:    &http.Server{ReadHeaderTimeout: time.Second},
		peers:     servingAnOpenRequest(t),
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		drain:     time.Hour,
		grace:     testGrace,
		logger:    slog.New(slog.NewTextHandler(&logs, nil)),
	})

	require.NoError(t, err)
	require.Contains(t, logs.String(), "listener=peers grace="+testGrace.String())
}

// servingASlowRequest is a server with one request open that finishes on its own in 200 ms:
// long enough that a shutdown given no grace returns first, and short against shutdownGrace.
// done closes when the request has finished.
func servingASlowRequest(t *testing.T) (server *http.Server, done <-chan struct{}) {
	t.Helper()
	opened, finished := make(chan struct{}), make(chan struct{})
	server = &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			defer close(finished)
			close(opened)
			time.Sleep(200 * time.Millisecond)
			w.WriteHeader(http.StatusNoContent)
		}),
		ReadHeaderTimeout: time.Second,
	}
	t.Cleanup(func() { _ = server.Close() })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	go func() { _ = server.Serve(listener) }()
	go func() {
		if res, err := http.Get("http://" + listener.Addr().String()); err == nil {
			_ = res.Body.Close()
		}
	}()
	<-opened
	return server, finished
}

// servingNothing is a server with no request open, and the address it answers on.
func servingNothing(t *testing.T) (*http.Server, string) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }),
		ReadHeaderTimeout: time.Second,
	}
	t.Cleanup(func() { _ = server.Close() })
	go func() { _ = server.Serve(listener) }()
	require.True(t, answers(listener.Addr().String()))
	return server, listener.Addr().String()
}

func TestARouterWithoutADrainStopsAnsweringItsPeers(t *testing.T) {
	public, _ := servingNothing(t)
	peers, peerAddress := servingNothing(t)

	err := stopServing(stopping{
		public:    public,
		peers:     peers,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		grace:     testGrace,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
	require.False(t, answers(peerAddress), "the peer listener outlived a router stopping without a drain")
}

func TestAPublicRequestOpenWhenADrainStartsFinishesInsideTheGrace(t *testing.T) {
	public, done := servingASlowRequest(t)

	err := stopServing(stopping{
		public:    public,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		drain:     time.Hour,
		grace:     shutdownGrace,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
	select {
	case <-done:
	default:
		t.Fatal("stopServing returned while the public request was still running")
	}
}

func TestAPeerRequestOpenAfterADrainFinishesInsideTheGrace(t *testing.T) {
	public, _ := servingNothing(t)
	peers, done := servingASlowRequest(t)

	err := stopServing(stopping{
		public:    public,
		peers:     peers,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		drain:     time.Hour,
		grace:     shutdownGrace,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
	select {
	case <-done:
	default:
		t.Fatal("stopServing returned while the peer request was still running")
	}
}

// unclosable closes but says it could not. Besides its deadline, a listener that fails to
// close is the one error http.Server.Shutdown returns (closeListenersLocked in net/http),
// and no real listener fails to close on demand.
type unclosable struct{ net.Listener }

var errUnclosable = errors.New("the listener could not close")

func (u unclosable) Close() error {
	_ = u.Listener.Close()
	return errUnclosable
}

func TestAShutdownThatFailsForAnotherReasonIsStillAnError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }),
		ReadHeaderTimeout: time.Second,
	}
	go func() { _ = server.Serve(unclosable{listener}) }()
	require.True(t, answers(listener.Addr().String()))

	err = stopServing(stopping{
		public:    server,
		drainWork: func() {},
		held:      func() int { return 0 },
		telephony: idlePhone(t),
		grace:     testGrace,
		logger:    slog.New(slog.DiscardHandler),
	})

	require.ErrorIs(t, err, errUnclosable)
}

// draining is a router on its way down with a drain, a public, a peer and a metrics
// listener, and a session count the test decides.
type draining struct {
	public, peers, metrics string
	held                   atomic.Int64
	stopped                chan error
}

func startDraining(t *testing.T, drain time.Duration) *draining {
	t.Helper()
	metrics := freeAddress(t)
	stopMetrics, handOverMetrics := serveMetrics(metrics, slog.New(slog.DiscardHandler))
	t.Cleanup(stopMetrics)
	require.Eventually(t, func() bool { return servesMetrics(metrics) }, 5*time.Second, 10*time.Millisecond,
		"the metrics listener never answered")
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	public, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	peers, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	publicServer := &http.Server{Handler: handler, ReadHeaderTimeout: time.Second}
	peerServer := &http.Server{Handler: handler, ReadHeaderTimeout: time.Second}
	go func() { _ = publicServer.Serve(public) }()
	go func() { _ = peerServer.Serve(peers) }()
	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  phone.NewRegistry(phone.Config{}),
		SIPTrunks: &closeRecorder{},
	})
	require.NoError(t, err)

	d := &draining{public: public.Addr().String(), peers: peers.Addr().String(), metrics: metrics, stopped: make(chan error, 1)}
	d.held.Store(1)
	go func() {
		d.stopped <- stopServing(stopping{
			public:          publicServer,
			peers:           peerServer,
			drainWork:       func() {},
			held:            func() int { return int(d.held.Load()) },
			telephony:       service,
			handOverMetrics: handOverMetrics,
			drain:           drain,
			grace:           shutdownGrace,
			logger:          slog.New(slog.DiscardHandler),
		})
	}()
	require.Eventually(t, func() bool { return !answers(d.public) }, 5*time.Second, 10*time.Millisecond,
		"the public listener kept taking connections")
	return d
}

func answers(address string) bool {
	client := http.Client{Timeout: time.Second}
	res, err := client.Get("http://" + address)
	if err != nil {
		return false
	}
	_ = res.Body.Close()
	return res.StatusCode == http.StatusNoContent
}

func servesMetrics(address string) bool {
	client := http.Client{Timeout: time.Second}
	res, err := client.Get("http://" + address + "/metrics")
	if err != nil {
		return false
	}
	_ = res.Body.Close()
	return res.StatusCode == http.StatusOK
}

func TestADrainingRouterFreesItsMetricsAddressForTheRouterTakingOver(t *testing.T) {
	d := startDraining(t, time.Hour)

	require.Eventually(t, func() bool {
		successor, err := net.Listen("tcp", d.metrics)
		if err != nil {
			return false
		}
		_ = successor.Close()
		return true
	}, time.Second, 10*time.Millisecond, "the draining router kept its metrics address")
	require.True(t, answers(d.peers), "a draining router stopped answering for its sessions")
	select {
	case <-d.stopped:
		t.Fatal("the router stopped while it still held a session")
	default:
	}

	d.held.Store(0)
	select {
	case err := <-d.stopped:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("the router kept draining after its last session ended")
	}
}

func TestARouterWithoutADrainServesMetricsUntilItStops(t *testing.T) {
	public, _ := servingNothing(t)
	metrics := freeAddress(t)
	stopMetrics, handOverMetrics := serveMetrics(metrics, slog.New(slog.DiscardHandler))
	t.Cleanup(stopMetrics)
	require.Eventually(t, func() bool { return servesMetrics(metrics) }, 5*time.Second, 10*time.Millisecond,
		"the metrics listener never answered")

	err := stopServing(stopping{
		public:          public,
		drainWork:       func() {},
		held:            func() int { return 0 },
		telephony:       idlePhone(t),
		handOverMetrics: handOverMetrics,
		grace:           testGrace,
		logger:          slog.New(slog.DiscardHandler),
	})

	require.NoError(t, err)
	require.True(t, servesMetrics(metrics), "metrics stopped before the router exited")
	stopMetrics()
	require.False(t, servesMetrics(metrics), "metrics outlived the router")
}

func TestADrainingRouterStillAnswersItsPeersUntilItsLastSessionEnds(t *testing.T) {
	d := startDraining(t, time.Hour)

	require.True(t, answers(d.peers), "a draining router stopped answering for its sessions")
	select {
	case <-d.stopped:
		t.Fatal("the router stopped while it still held a session")
	case <-time.After(3 * drainPoll):
	}

	d.held.Store(0)
	select {
	case err := <-d.stopped:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("the router kept draining after its last session ended")
	}
	require.False(t, answers(d.peers), "the peer listener outlived the drain")
}

func TestADrainThatRunsOutStopsWithSessionsStillRunning(t *testing.T) {
	d := startDraining(t, 500*time.Millisecond)

	select {
	case err := <-d.stopped:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("the router drained past its drain")
	}
}

func TestASecondSignalEndsTheDrain(t *testing.T) {
	d := startDraining(t, time.Hour)

	require.NoError(t, syscall.Kill(os.Getpid(), syscall.SIGINT))

	select {
	case err := <-d.stopped:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("a second signal did not end the drain")
	}
}

func TestARouterServesOnTheSocketItInherited(t *testing.T) {
	opened, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	file, err := opened.(*net.TCPListener).File()
	require.NoError(t, err)
	address := opened.Addr().String()
	require.NoError(t, opened.Close())
	// listen owns the descriptor it is given and closes it, as it does a real inherited one.
	// Give it a duplicate and close our file: handing it file's own descriptor closed that
	// number twice, and a later test's socket that reused it was closed underneath it.
	inheritedFD, err := syscall.Dup(int(file.Fd()))
	require.NoError(t, err)
	require.NoError(t, file.Close())

	inherited, err := listen("", inheritedFD)
	require.NoError(t, err)
	server := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }),
		ReadHeaderTimeout: time.Second,
	}
	go func() { _ = server.Serve(inherited) }()
	t.Cleanup(func() { _ = server.Close() })

	require.True(t, answers(address), "nothing answered on the inherited socket")
}
