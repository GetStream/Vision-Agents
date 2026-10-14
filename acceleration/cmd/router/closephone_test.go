package main

import (
	"context"
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
	started := make(chan struct{})
	httpServer := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
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
	require.Equal(t, http.StatusNoContent, <-status)
}

// draining is a router on its way down with a drain, a public and a peer listener, and a
// session count the test decides.
type draining struct {
	public, peers string
	held          atomic.Int64
	stopped       chan error
}

func startDraining(t *testing.T, drain time.Duration) *draining {
	t.Helper()
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

	d := &draining{public: public.Addr().String(), peers: peers.Addr().String(), stopped: make(chan error, 1)}
	d.held.Store(1)
	go func() {
		d.stopped <- stopServing(stopping{
			public:    publicServer,
			peers:     peerServer,
			drainWork: func() {},
			held:      func() int { return int(d.held.Load()) },
			telephony: service,
			drain:     drain,
			logger:    slog.New(slog.DiscardHandler),
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

	inherited, err := listen("", int(file.Fd()))
	require.NoError(t, err)
	server := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }),
		ReadHeaderTimeout: time.Second,
	}
	go func() { _ = server.Serve(inherited) }()
	t.Cleanup(func() { _ = server.Close() })

	require.True(t, answers(address), "nothing answered on the inherited socket")
}
