package main

import (
	"crypto/ed25519"
	"encoding/base64"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// ChatDeliveryStartupSuite starts the router with ROUTER_STREAM_DELIVERY_KEYS set, as its serve
// command does, with no database, Redis or Stream app.
type ChatDeliveryStartupSuite struct {
	suite.Suite
	settings config.Config
	// stopped is closed when run returns, with what it returned in err.
	stopped chan struct{}
	err     error
}

func TestChatDeliveryStartupSuite(t *testing.T) {
	suite.Run(t, new(ChatDeliveryStartupSuite))
}

func (s *ChatDeliveryStartupSuite) SetupTest() {
	s.settings = config.Defaults()
	s.settings.Auth.Mode = string(auth.NoAuth)
	s.settings.Limits.Mode = config.LimitsOff
}

// A key that does not read stops the router at start, rather than leaving it to refuse
// every delivery Stream sends.
func (s *ChatDeliveryStartupSuite) TestAKeyThatDoesNotReadStopsTheRouter() {
	s.settings.Addr = "127.0.0.1:0"
	s.settings.Stream.DeliveryKeys = []string{"not-a-key"}

	s.start()

	select {
	case <-s.stopped:
		s.ErrorContains(s.err, "delivery key 1 is not")
	case <-time.After(10 * time.Second):
		s.Fail("the router started with a key that does not read")
	}
}

// With a key the route is served: an unsigned delivery is refused 401, where a router with
// no key answers 404.
func (s *ChatDeliveryStartupSuite) TestWithAKeyAnUnsignedDeliveryIsRefused() {
	public, _, err := ed25519.GenerateKey(nil)
	s.Require().NoError(err)
	s.settings.Stream.DeliveryKeys = []string{base64.StdEncoding.EncodeToString(public)}
	opened, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.settings.Addr = opened.Addr().String()
	s.Require().NoError(opened.Close())

	s.start()

	// A transport of its own: run sets up http.DefaultTransport as it starts, and the next
	// run in this process would do so under this request's connection.
	transport := &http.Transport{}
	s.T().Cleanup(transport.CloseIdleConnections)
	client := http.Client{Timeout: time.Second, Transport: transport}
	deadline := time.Now().Add(30 * time.Second)
	for {
		response, err := client.Post("http://"+s.settings.Addr+"/v1/chat/internal/messages",
			"application/json", strings.NewReader(`{"type":"message.new","app_id":1257545}`))
		if err == nil {
			s.Require().NoError(response.Body.Close())
			s.Equal(http.StatusUnauthorized, response.StatusCode)
			return
		}
		s.Require().True(time.Now().Before(deadline), "the router never answered: %v", err)
		time.Sleep(50 * time.Millisecond)
	}
}

// start runs the router as its serve command does. A router still running when the test ends
// is stopped with SIGTERM, which run handles only while it runs.
func (s *ChatDeliveryStartupSuite) start() {
	s.stopped = make(chan struct{})
	go func() {
		s.err = run(s.settings, slog.New(slog.DiscardHandler))
		close(s.stopped)
	}()
	s.T().Cleanup(func() {
		select {
		case <-s.stopped:
			return
		default:
		}
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		select {
		case <-s.stopped:
			s.NoError(s.err)
		case <-time.After(30 * time.Second):
			s.Fail("the router did not stop on SIGTERM")
		}
	})
}
