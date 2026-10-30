package session

import (
	"os"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

// countsSessions gives the suite a manager that lets each app have limit sessions open at once,
// counted in the Redis the environment names, and skips the test without one.
func (s *SessionSuite) countsSessions(limit int64) {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.manages()
	counted, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(counted.Close)
	manager, err := NewManager(ManagerOptions{
		LLM: s.manager.options.LLM, Edge: s.manager.options.Edge, Live: counted, SessionsPerApp: limit,
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { manager.Shutdown() })
	s.manager = manager
}

func (s *SessionSuite) TestAnAppPastItsSessionsAtOnceIsRefused() {
	s.countsSessions(1)
	customer := "app-" + uuid.NewString()
	s.writes(Spec{CustomerID: customer})

	_, err := s.manager.Create(s.ctx, Spec{Text: true, CustomerID: customer, LLMTarget: "en-low-latency"})

	s.ErrorIs(err, ErrTooManySessions)
	s.writes(Spec{CustomerID: "other-" + uuid.NewString()})
}

func (s *SessionSuite) TestASessionThatClosedFreesItsAppsPlace() {
	s.countsSessions(1)
	customer := "app-" + uuid.NewString()
	first := s.writes(Spec{CustomerID: customer})

	s.Require().NoError(first.Close())

	s.writes(Spec{CustomerID: customer})
}

func (s *SessionSuite) TestASessionThatFailedToOpenFreesItsAppsPlace() {
	s.countsSessions(1)
	customer := "app-" + uuid.NewString()
	// Resuming a conversation needs Chat, which this manager has none of.
	_, err := s.manager.Create(s.ctx, Spec{
		Text: true, CustomerID: customer, LLMTarget: "en-low-latency", PersistConversation: true, ConversationID: "c-1",
	})
	s.Require().ErrorContains(err, "Stream Chat credentials are required")

	s.writes(Spec{CustomerID: customer})
}
