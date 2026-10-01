package streamedge

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/stretchr/testify/suite"
)

type ClientsSuite struct {
	suite.Suite
	clients *Clients
	baseURL string
}

func TestClientsSuite(t *testing.T) {
	suite.Run(t, new(ClientsSuite))
}

func (s *ClientsSuite) SetupTest() {
	coordinator := httptest.NewServer(http.NotFoundHandler())
	s.T().Cleanup(coordinator.Close)
	s.baseURL = coordinator.URL
	s.clients = NewClients()
	s.T().Cleanup(s.clients.Close)
}

// client is the SDK client an Edge of userID joins with, and its release.
func (s *ClientsSuite) client(userID string, clients *Clients) (*rtc.Client, func()) {
	edge, err := New(Options{
		CallID: "a-call", User: User{ID: userID}, APIKey: "key", APISecret: "secret",
		BaseURL: s.baseURL, Clients: clients,
		clientOptions: []rtc.Option{rtc.WithoutCoordinatorWS(), rtc.WithoutKeepWarm()},
	})
	s.Require().NoError(err)
	client, release, err := edge.connect()
	s.Require().NoError(err)
	s.T().Cleanup(release)
	return client, release
}

func (s *ClientsSuite) TestTheSameAgentsSessionsShareAClient() {
	first, _ := s.client("agent", s.clients)
	second, _ := s.client("agent", s.clients)
	s.Same(first, second)
}

func (s *ClientsSuite) TestAnotherAgentHasAClientOfItsOwn() {
	first, _ := s.client("agent", s.clients)
	other, _ := s.client("another-agent", s.clients)
	s.NotSame(first, other)
}

func (s *ClientsSuite) TestTheClientOutlivesTheSessionThatOpenedIt() {
	first, release := s.client("agent", s.clients)
	release()
	next, _ := s.client("agent", s.clients)
	s.Same(first, next, "the next session joins on the first one's connections")
}

func (s *ClientsSuite) TestAClientNoSessionUsesIsClosedOnceIdle() {
	s.clients.idleFor = 50 * time.Millisecond
	first, release := s.client("agent", s.clients)
	release()
	s.Require().Eventually(func() bool {
		s.clients.mu.Lock()
		defer s.clients.mu.Unlock()
		return len(s.clients.clients) == 0
	}, 5*time.Second, 10*time.Millisecond)

	next, _ := s.client("agent", s.clients)
	s.NotSame(first, next)
}

func (s *ClientsSuite) TestAClientInUseIsNotClosedWhenAnotherSessionLeaves() {
	s.clients.idleFor = 50 * time.Millisecond
	first, release := s.client("agent", s.clients)
	s.client("agent", s.clients)
	release()
	s.Never(func() bool {
		s.clients.mu.Lock()
		defer s.clients.mu.Unlock()
		return len(s.clients.clients) == 0
	}, 200*time.Millisecond, 10*time.Millisecond)

	next, _ := s.client("agent", s.clients)
	s.Same(first, next)
}

func (s *ClientsSuite) TestWithoutClientsEachSessionHasItsOwn() {
	first, _ := s.client("agent", nil)
	second, _ := s.client("agent", nil)
	s.NotSame(first, second)
}

func (s *ClientsSuite) TestAfterCloseEachSessionHasItsOwn() {
	first, _ := s.client("agent", s.clients)
	s.clients.Close()
	second, _ := s.client("agent", s.clients)
	third, _ := s.client("agent", s.clients)
	s.NotSame(first, second)
	s.NotSame(second, third)
}
