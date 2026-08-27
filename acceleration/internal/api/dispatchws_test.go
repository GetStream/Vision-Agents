package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// DispatchSuite covers the worker side of dispatch: what a worker has to send to be given
// calls, and what it is given. It runs against a real listener rather than a recorder,
// because a recorder cannot be upgraded.
type DispatchSuite struct {
	suite.Suite
	pool   *dispatch.Pool
	server *httptest.Server
}

func TestDispatchSuite(t *testing.T) {
	suite.Run(t, new(DispatchSuite))
}

func (s *DispatchSuite) SetupTest() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	s.pool = dispatch.NewPool()
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.STT: speech},
		Dispatch: s.pool,
	})
	s.Require().NoError(err)

	s.server = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.server.Close)
}

// connect opens the dispatch socket as a worker and waits until it is in the rotation, so a
// test that assigns a call straight afterwards does not race the registration.
func (s *DispatchSuite) connect(customerID, query string) *websocket.Conn {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/dispatch" + query
	header := http.Header{}
	if customerID != "" {
		header.Set(CustomerHeader, customerID)
	}

	connection, response, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().NoError(err)
	s.T().Cleanup(func() { connection.Close() })
	s.Require().Equal(http.StatusSwitchingProtocols, response.StatusCode)

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var ready map[string]any
	s.Require().NoError(connection.ReadJSON(&ready))
	s.Equal("ready", ready["type"])
	s.NotEmpty(ready["worker_id"], "a worker is told what it is called")
	return connection
}

// refused dials without expecting an upgrade, and reports the status instead.
func (s *DispatchSuite) refused(customerID, query string) int {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/dispatch" + query
	header := http.Header{}
	if customerID != "" {
		header.Set(CustomerHeader, customerID)
	}

	connection, response, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().Error(err, "this is meant not to be upgraded")
	if connection != nil {
		connection.Close()
	}
	s.Require().NotNil(response)
	return response.StatusCode
}

func (s *DispatchSuite) TestAWorkerWithoutACustomerCannotWaitForCalls() {
	s.Equal(http.StatusUnauthorized, s.refused("", ""))
}

func (s *DispatchSuite) TestACapacityThatIsNotANumberOfCallsIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused("acme", "?capacity=plenty"))
	s.Equal(http.StatusBadRequest, s.refused("acme", "?capacity=0"))
}

func (s *DispatchSuite) TestAConnectedWorkerIsHandedAnArrivingCall() {
	connection := s.connect("acme", "")

	_, err := s.pool.Assign("acme", dispatch.Call{
		CallID:       "phone-+15125551234",
		CallType:     "default",
		CalledNumber: "+15125551234",
		CallerNumber: "+15550001111",
		Custom:       map[string]string{"line": "support"},
		At:           time.Now().UTC(),
	})
	s.Require().NoError(err)

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var call map[string]any
	s.Require().NoError(connection.ReadJSON(&call))

	s.Equal("call", call["type"])
	s.Equal("phone-+15125551234", call["call_id"])
	s.Equal("default", call["call_type"])
	s.Equal("+15125551234", call["called_number"])
	s.Equal("+15550001111", call["caller_number"])
	s.Equal(map[string]any{"line": "support"}, call["custom"])
	s.NotEmpty(call["at"])
}

func (s *DispatchSuite) TestACallWithNoCustomDataStillCarriesAnObject() {
	connection := s.connect("acme", "")

	_, err := s.pool.Assign("acme", dispatch.Call{CallID: "phone-+15125551234"})
	s.Require().NoError(err)

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var call map[string]any
	s.Require().NoError(connection.ReadJSON(&call))

	s.Equal(map[string]any{}, call["custom"], "a client should not have to tell absent from empty")
}

func (s *DispatchSuite) TestAWorkerOnlyGetsItsOwnCustomersCalls() {
	connection := s.connect("acme", "")

	_, err := s.pool.Assign("globex", dispatch.Call{CallID: "phone-+15125559999"})
	s.Require().ErrorIs(err, dispatch.ErrNoWorkers)

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(300 * time.Millisecond)))
	var call map[string]any
	s.Error(connection.ReadJSON(&call), "nothing should arrive on somebody else's call")
}

func (s *DispatchSuite) TestWhatAWorkerReportsAboutItselfIsReadable() {
	connection := s.connect("acme", "")

	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type":           "load",
		"active_agents":  2,
		"cpu_percent":    37.5,
		"memory_percent": 61.25,
		"latency_ms":     12.5,
	}))

	// The report crosses a socket, so it is waited for rather than assumed to have landed.
	s.Eventually(func() bool {
		waiting := s.pool.Workers("acme")
		return len(waiting) == 1 && waiting[0].Load().ActiveAgents == 2
	}, 5*time.Second, 10*time.Millisecond)

	load := s.pool.Workers("acme")[0].Load()
	s.InDelta(37.5, load.CPUPercent, 0.001)
	s.InDelta(61.25, load.MemoryPercent, 0.001)
	s.InDelta(12.5, load.LatencyMs, 0.001)
}

func (s *DispatchSuite) TestAWorkerCanMeasureItsOwnRoundTrip() {
	connection := s.connect("acme", "")

	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "ping", "at": 1234.5}))

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var pong map[string]any
	s.Require().NoError(connection.ReadJSON(&pong))

	s.Equal("pong", pong["type"])
	s.Equal(1234.5, pong["at"], "the timestamp comes back so the worker can subtract it")
}

func (s *DispatchSuite) TestAMessageTheServerCannotReadDoesNotEndTheConnection() {
	connection := s.connect("acme", "")

	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "who-knows"}))

	// Still in the rotation, and still handed calls.
	_, err := s.pool.Assign("acme", dispatch.Call{CallID: "phone-+15125551234"})
	s.Require().NoError(err)

	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var call map[string]any
	s.Require().NoError(connection.ReadJSON(&call))
	s.Equal("call", call["type"])
}

func (s *DispatchSuite) TestAWorkerThatDisconnectsLeavesTheRotation() {
	connection := s.connect("acme", "")
	s.Require().Len(s.pool.Workers("acme"), 1)

	s.Require().NoError(connection.Close())

	s.Eventually(func() bool {
		return len(s.pool.Workers("acme")) == 0
	}, 5*time.Second, 10*time.Millisecond, "a closed socket must not hold a slot in the rotation")
}

func (s *DispatchSuite) TestTwoWorkersOnOneCustomerShareTheCalls() {
	first := s.connect("acme", "")
	second := s.connect("acme", "")

	for _, id := range []string{"call-1", "call-2"} {
		_, err := s.pool.Assign("acme", dispatch.Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal("call-1", s.nextCall(first))
	s.Equal("call-2", s.nextCall(second))
}

// nextCall reads the id of the next call handed to a worker.
func (s *DispatchSuite) nextCall(connection *websocket.Conn) string {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var call map[string]any
	s.Require().NoError(connection.ReadJSON(&call))
	s.Require().Equal("call", call["type"])
	id, _ := call["call_id"].(string)
	return id
}

func (s *DispatchSuite) TestADeploymentThatDoesNotDispatchSaysSo() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
	})
	s.Require().NoError(err)
	s.server.Close()
	s.server = httptest.NewServer(server.Handler())

	s.Equal(http.StatusNotFound, s.refused("acme", ""))
}
