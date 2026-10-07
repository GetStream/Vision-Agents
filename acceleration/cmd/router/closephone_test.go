package main

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
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

	err = stopServing(httpServer, service, slog.New(slog.DiscardHandler), time.Second)

	require.NoError(t, err)
	require.Equal(t, http.StatusNoContent, <-status)
}
