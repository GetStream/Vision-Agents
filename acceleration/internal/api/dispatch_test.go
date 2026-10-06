//go:build integration

package api

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// DispatchSuite covers the worker side of dispatch: what a worker has to send to be given
// calls, and what it is given.
type DispatchSuite struct {
	RouterSuite
}

func TestDispatchSuite(t *testing.T) {
	runSuite(t, new(DispatchSuite))
}

func (s *DispatchSuite) SetupTest() {
	s.useFixture("standard")

	// A worker the last test opened is dropped a moment after its socket closes, and work
	// handed to that one is work this test never sees.
	s.Require().Eventually(func() bool {
		return len(s.dispatch.Workers(s.customerID())) == 0
	}, settleFor, 10*time.Millisecond, "the last test's worker is still in the pool")
}

func (s *DispatchSuite) TestAWorkerWithoutACredentialCannotWaitForCalls() {
	_, status := s.unauthenticatedClient.watch("/v1/dispatch")

	s.Equal(http.StatusUnauthorized, status)
}

func (s *DispatchSuite) TestACapacityThatIsNotANumberOfCallsIsRefused() {
	_, status := s.serverClient.watch("/v1/dispatch?capacity=plenty")

	s.Equal(http.StatusBadRequest, status)
}

func (s *DispatchSuite) TestAWorkerWithNoRoomForACallIsRefused() {
	_, status := s.serverClient.watch("/v1/dispatch?capacity=0")

	s.Equal(http.StatusBadRequest, status)
}

func (s *DispatchSuite) TestWorkAWorkerCannotAlreadyBeHoldingIsRefused() {
	_, status := s.serverClient.watch("/v1/dispatch?active=-1")

	s.Equal(http.StatusBadRequest, status)
}

func (s *DispatchSuite) TestAKindOfWorkThisServiceDoesNotHandOutIsRefused() {
	// Taking it would leave the worker believing it had opted out of calls when it had not.
	_, status := s.serverClient.watch("/v1/dispatch?handles=telegrams")

	s.Equal(http.StatusBadRequest, status)
}

func (s *DispatchSuite) TestWorkCarriesTheIdItIsReportedFinishedUnder() {
	worker := s.worker()

	s.assign(dispatch.Call{CallID: "phone-+15125551234"})

	s.NotEmpty(s.next(worker)["work_id"], "without it the worker cannot say which call it finished")
}

func (s *DispatchSuite) TestAWorkerIsHeldToTheCapacityItDeclaredUntilItReportsWorkDone() {
	// The queue empties as the frame is written, which is not the worker finishing the
	// call. Without the report, a worker that promised one call would be handed the next.
	worker := s.reportingWorker(1)

	s.assign(dispatch.Call{CallID: "call-1"})
	handed := s.next(worker)

	_, err := s.dispatch.Assign(s.customerID(), dispatch.Call{CallID: "call-2"})
	s.Require().ErrorContains(err, "at capacity")

	s.Require().NoError(worker.WriteJSON(frame{"type": "done", "work_id": handed["work_id"]}))
	s.Require().Eventually(func() bool {
		_, err := s.dispatch.Assign(s.customerID(), dispatch.Call{CallID: "call-2"})
		return err == nil
	}, settleFor, 10*time.Millisecond, "reporting the call finished should give the worker its room back")

	s.Equal("call-2", s.next(worker)["call_id"])
}

func (s *DispatchSuite) TestAWorkerThatAnswersOnlyMessagesIsNotOfferedCalls() {
	connection := s.serverClient.opens("/v1/dispatch?handles=message")
	s.Equal("ready", s.next(connection)["type"])
	s.Require().Eventually(func() bool {
		return len(s.dispatch.Workers(s.customerID())) == 1
	}, settleFor, 10*time.Millisecond)

	_, err := s.dispatch.Assign(s.customerID(), dispatch.Call{CallID: "call-1"})

	s.Require().ErrorContains(err, "handles call")
	_, err = s.dispatch.AssignMessage(s.customerID(),
		dispatch.Message{ChannelID: "chat-1", Text: "hello"})
	s.Require().NoError(err)
	s.Equal("message", s.next(connection)["type"])
}

func (s *DispatchSuite) TestAConnectedWorkerIsHandedAnArrivingCall() {
	worker := s.worker()

	s.assign(dispatch.Call{
		CallID:       "phone-+15125551234",
		CallType:     "default",
		CalledNumber: "+15125551234",
		CallerNumber: "+15550001111",
		Custom:       map[string]string{"line": "support"},
		At:           time.Now().UTC(),
	})

	call := s.next(worker)
	s.Equal("call", call["type"])
	s.Equal("phone-+15125551234", call["call_id"])
	s.Equal("default", call["call_type"])
	s.Equal("+15125551234", call["called_number"])
	s.Equal("+15550001111", call["caller_number"])
	s.Equal(map[string]any{"line": "support"}, call["custom"])
	s.NotEmpty(call["at"])
}

func (s *DispatchSuite) TestACallWithNoCustomDataStillCarriesAnObject() {
	worker := s.worker()

	s.assign(dispatch.Call{CallID: "phone-+15125551234"})

	s.Equal(map[string]any{}, s.next(worker)["custom"],
		"a client should not have to tell absent from empty")
}

func (s *DispatchSuite) TestAWorkerOnlyGetsItsOwnCustomersCalls() {
	worker := s.worker()

	_, err := s.dispatch.Assign("somebody-else", dispatch.Call{CallID: "phone-+15125559999"})
	s.Require().ErrorIs(err, dispatch.ErrNoWorkers)

	s.Require().NoError(worker.SetReadDeadline(time.Now().Add(dropped)))
	var call frame
	s.Error(worker.ReadJSON(&call), "nothing should arrive on somebody else's call")
}

func (s *DispatchSuite) TestWhatAWorkerReportsAboutItselfIsReadable() {
	s.worker().WriteJSON(frame{
		"type": "load", "active_agents": 2, "cpu_percent": 37.5,
		"memory_percent": 61.25, "latency_ms": 12.5,
	})

	// The report crosses a socket, so it is waited for rather than assumed to have landed.
	s.Require().Eventually(func() bool {
		waiting := s.dispatch.Workers(s.customerID())
		return len(waiting) == 1 && waiting[0].Load().ActiveAgents == 2
	}, settleFor, 10*time.Millisecond)

	load := s.dispatch.Workers(s.customerID())[0].Load()
	s.InDelta(37.5, load.CPUPercent, 0.001)
	s.InDelta(61.25, load.MemoryPercent, 0.001)
	s.InDelta(12.5, load.LatencyMs, 0.001)
}

func (s *DispatchSuite) TestAWorkerCanMeasureItsOwnRoundTrip() {
	worker := s.worker()

	s.Require().NoError(worker.WriteJSON(frame{"type": "ping", "at": 1234.5}))

	pong := s.next(worker)
	s.Equal("pong", pong["type"])
	s.Equal(1234.5, pong["at"], "the timestamp comes back so the worker can subtract it")
}

func (s *DispatchSuite) TestPingRepliesAndCallDeliveryShareTheSocket() {
	worker := s.worker()
	s.Require().NoError(worker.SetWriteDeadline(time.Now().Add(settleFor)))
	for i := range 32 {
		s.Require().NoError(worker.WriteJSON(frame{"type": "ping", "at": i}))
	}
	s.assign(dispatch.Call{CallID: "concurrent-call"})

	seen := make(map[float64]bool)
	calls := 0
	for range 33 {
		event := s.next(worker)
		switch event["type"] {
		case "pong":
			at, ok := event["at"].(float64)
			s.Require().True(ok)
			s.False(seen[at], "each ping has exactly one reply")
			seen[at] = true
		case "call":
			s.Equal("concurrent-call", event["call_id"])
			calls++
		default:
			s.FailNow("unexpected dispatch event")
		}
	}
	s.Equal(1, calls)
	for i := range 32 {
		s.True(seen[float64(i)], "all ping timestamps survive call delivery")
	}
}

func (s *DispatchSuite) TestAMessageTheServerCannotReadDoesNotEndTheConnection() {
	worker := s.worker()

	s.Require().NoError(worker.WriteJSON(frame{"type": "who-knows"}))

	// Still in the rotation, and still handed calls.
	s.assign(dispatch.Call{CallID: "phone-+15125551234"})
	s.Equal("call", s.next(worker)["type"])
}

func (s *DispatchSuite) TestAWorkerThatDisconnectsLeavesTheRotation() {
	worker := s.worker()

	s.Require().NoError(worker.Close())

	s.Require().Eventually(func() bool {
		return len(s.dispatch.Workers(s.customerID())) == 0
	}, settleFor, 10*time.Millisecond, "a closed socket must not hold a slot in the rotation")
}

func (s *DispatchSuite) TestTwoWorkersOnOneCustomerShareTheCalls() {
	first, second := s.worker(), s.worker()

	s.assign(dispatch.Call{CallID: "call-1"})
	s.assign(dispatch.Call{CallID: "call-2"})

	s.Equal("call-1", s.next(first)["call_id"])
	s.Equal("call-2", s.next(second)["call_id"])
}

func (s *DispatchSuite) TestHostedToolsAreKeptForTheAgentTheWorkerNames() {
	worker := s.worker()

	s.Require().NoError(worker.WriteJSON(frame{
		"type": "host_tools", "agent_id": "stream-support",
		"tools": []frame{{"name": "investigate_sdk", "description": "Read SDK source"}},
	}))

	hosting := s.next(worker)
	s.Equal("hosting", hosting["type"], "an agent id needs nothing stored to host for")
	s.Equal("stream-support", hosting["agent_id"])

	tools, timeout := s.dispatch.HostedTools(s.customerID(), "stream-support")
	s.Require().Len(tools, 1)
	s.Equal("investigate_sdk", tools[0].Name)
	s.Equal(2*time.Minute, timeout, "a worker that names no timeout takes the default")
}

func (s *DispatchSuite) TestHostedToolsNeedAnAgentToBeFor() {
	worker := s.worker()

	s.Require().NoError(worker.WriteJSON(frame{
		"type":  "host_tools",
		"tools": []frame{{"name": "investigate_sdk", "description": "Read SDK source"}},
	}))

	s.Equal("hosting_refused", s.next(worker)["type"])
	tools, _ := s.dispatch.HostedTools(s.customerID(), "")
	s.Empty(tools)
}

func (s *DispatchSuite) TestAHostedToolCallIsAnsweredOverTheSocket() {
	worker := s.worker()
	s.Require().NoError(s.dispatch.Host(s.dispatch.Workers(s.customerID())[0], "stream-support",
		[]dispatch.Tool{{Name: "investigate_sdk", Description: "Read SDK source"}}, time.Minute))

	type answer struct {
		output string
		err    error
	}
	answered := make(chan answer, 1)
	go func() {
		output, err := s.dispatch.RunHosted(s.T().Context(), s.customerID(), "stream-support",
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1",
				Name: "investigate_sdk", Arguments: `{"sdk":"android"}`})
		answered <- answer{output, err}
	}()

	call := s.next(worker)
	s.Equal("tool_call", call["type"])
	s.Equal("call-1", call["id"])
	s.Equal("session-1", call["session_id"])
	s.Equal(`{"sdk":"android"}`, call["arguments"])

	s.Require().NoError(worker.WriteJSON(frame{
		"type": "tool_result", "id": "call-1", "output": "targetSdkVersion 35"}))

	got := <-answered
	s.Require().NoError(got.err)
	s.Equal("targetSdkVersion 35", got.output)
}

// worker opens the dispatch socket as a backend and waits until it is in the rotation, so a
// test that assigns a call straight afterwards does not race the registration.
func (s *DispatchSuite) worker() *websocket.Conn {
	return s.opensWith("/v1/dispatch")
}

// reportingWorker is a worker that reports each piece of work finished, which is what holds
// it to the capacity it declared.
func (s *DispatchSuite) reportingWorker(capacity int) *websocket.Conn {
	return s.opensWith(fmt.Sprintf("/v1/dispatch?capacity=%d&active=0", capacity))
}

func (s *DispatchSuite) opensWith(address string) *websocket.Conn {
	before := len(s.dispatch.Workers(s.customerID()))
	connection := s.serverClient.opens(address)

	ready := s.next(connection)
	s.Equal("ready", ready["type"])
	s.NotEmpty(ready["worker_id"], "a worker is told what it is called")
	s.Require().Eventually(func() bool {
		return len(s.dispatch.Workers(s.customerID())) == before+1
	}, settleFor, 10*time.Millisecond)
	return connection
}

// assign hands a call to the customer's workers, one of which must take it.
func (s *DispatchSuite) assign(call dispatch.Call) {
	_, err := s.dispatch.Assign(s.customerID(), call)
	s.Require().NoError(err)
}

// next reads the next frame a worker is sent.
func (s *DispatchSuite) next(connection *websocket.Conn) frame {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	var event frame
	s.Require().NoError(connection.ReadJSON(&event))
	return event
}
