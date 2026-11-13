package stream

import (
	"bytes"
	"context"
	"log/slog"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
)

// DispatchLogSuite covers what a worker's log says about who it is answering.
type DispatchLogSuite struct {
	suite.Suite
}

func TestDispatchLogSuite(t *testing.T) {
	suite.Run(t, new(DispatchLogSuite))
}

// captured is a log a test reads back: everything written at the level or above, as JSON.
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

func (c *captured) logger(level slog.Level) *slog.Logger {
	return slog.New(slog.NewJSONHandler(c, &slog.HandlerOptions{Level: level}))
}

// answered runs a worker on the router until it has reported one piece of work finished, and
// returns what it logged at the level or above.
func (s *DispatchLogSuite) answered(router *pool, level slog.Level) string {
	logged := &captured{}
	worker, err := NewDispatch(DispatchOptions{
		Backend: Backend{URL: router.URL, CustomerID: "acme"},
		Logger:  logged.logger(level),
	})
	s.Require().NoError(err)
	worker.OnCall(func(context.Context, InboundCall) error { return nil })
	worker.OnMessage(func(context.Context, InboundMessage) error { return nil })

	ctx, stop := context.WithCancel(s.T().Context())
	defer stop()
	stopped := run(s.T(), ctx, worker)
	router.told(s.T(), "done")
	stop()
	_ = stopped()
	return logged.String()
}

func (s *DispatchLogSuite) TestACallIsLoggedByTheLastFourDigitsOfItsNumbers() {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "call", "work_id": "work-1", "call_id": "session-1", "session_id": "session-1",
			"called_number": "+13035550100", "caller_number": "+13035550111",
		})
	})

	logged := s.answered(router, slog.LevelDebug)

	s.NotContains(logged, "+13035550111")
	s.NotContains(logged, "3035550111")
	s.NotContains(logged, "+13035550100")
	s.Contains(logged, `"from":"***0111"`)
	s.Contains(logged, `"on":"***0100"`)
}

func (s *DispatchLogSuite) TestWhoWroteAMessageIsNotLoggedUnlessDebugging() {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "message", "work_id": "work-1", "channel_id": "support-42",
			"user_id": "jane-doe-1987", "text": "hello",
		})
	})

	logged := s.answered(router, slog.LevelInfo)

	s.NotContains(logged, "jane-doe-1987")
	s.Contains(logged, `"msg":"answering a message"`)
	s.Contains(logged, `"channel":"support-42"`)
}

func (s *DispatchLogSuite) TestWhoWroteAMessageIsLoggedWhenDebugging() {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "message", "work_id": "work-1", "channel_id": "support-42",
			"user_id": "jane-doe-1987", "text": "hello",
		})
	})

	logged := s.answered(router, slog.LevelDebug)

	s.Contains(logged, `"from":"jane-doe-1987"`)
}
