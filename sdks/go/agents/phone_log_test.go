package agents

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// PhoneLogSuite covers what an agent's log says about the numbers on a call it places.
type PhoneLogSuite struct {
	suite.Suite
}

func TestPhoneLogSuite(t *testing.T) {
	suite.Run(t, new(PhoneLogSuite))
}

// captured is a log a test reads back: everything written, debug included, as JSON.
type captured struct {
	mu   sync.Mutex
	text bytes.Buffer
}

func (c *captured) Write(line []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.text.Write(line)
}

func (c *captured) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.text.String()
}

func (s *PhoneLogSuite) TestACallPlacedIsLoggedByTheLastFourDigitsOfItsNumbers() {
	router := newWorked(s.T(), nil)
	// The stand-in router places no calls, so placing one is answered in front of it.
	placed := "session-a"
	placing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/v1/phone/calls" {
			reply(w, http.StatusAccepted, acceleration.PlacedCall{SessionId: &placed, Status: "queued", VendorCallId: "CA1"})
			return
		}
		router.Config.Handler.ServeHTTP(w, r)
	}))
	s.T().Cleanup(placing.Close)
	logged := &captured{}
	logger := slog.New(slog.NewJSONHandler(logged, &slog.HandlerOptions{Level: slog.LevelDebug}))
	agent, err := New(Options{Name: "jean", Logger: logger, LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: placing.URL, CustomerID: "acme"},
		Logger:  logger,
	})})
	s.Require().NoError(err)

	session, err := agent.StartCall(s.T().Context(), "+15125551234", "+351210000000")
	s.Require().NoError(err)
	s.Require().NoError(session.Close(context.WithoutCancel(s.T().Context())))

	s.NotContains(logged.String(), "+351210000000")
	s.NotContains(logged.String(), "351210000000")
	s.NotContains(logged.String(), "+15125551234")
	s.Contains(logged.String(), `"to":"***0000"`)
	s.Contains(logged.String(), `"from":"***1234"`)
}
