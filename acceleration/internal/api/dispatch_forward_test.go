//go:build integration

package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ForwardedDispatchSuite covers hooks that land on a node with no worker connected for the
// customer, while one waits on another node.
//
// That is what a deployment of more than one router looks like: a worker's socket and
// Stream's hook each land wherever the load balancer sends them.
type ForwardedDispatchSuite struct {
	RouterSuite

	// other is the node the workers in a test connect to, and otherWorkers its pool.
	other        *httptest.Server
	otherWorkers *dispatch.Pool
	// elsewhere is how a session on a third node, with no workers of its own, reaches the
	// deployment's workers.
	elsewhere *node.Workers
}

func TestForwardedDispatchSuite(t *testing.T) {
	runSuite(t, new(ForwardedDispatchSuite))
}

func (s *ForwardedDispatchSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.otherWorkers = dispatch.NewPool()
	s.other = s.otherNodeDispatching(s.otherWorkers)
	logger := slog.New(slog.DiscardHandler)
	third := httptest.NewUnstartedServer(nil)
	s.T().Cleanup(third.Close)
	s.elsewhere = s.deploymentWorkers(dispatch.NewPool(), s.nodeDirectory(third, s.shard, logger), s.workBus(logger))
}

func (s *ForwardedDispatchSuite) TestARunningSessionsTextReachesAWorkerOnAnotherNode() {
	worker := s.waitsOnTheOtherNode()

	workerID, err := s.elsewhere.AssignMessage(context.Background(), s.customerID(),
		dispatch.Message{SessionID: "session-1", Text: "where is my order?"})
	s.Require().NoError(err)

	message := s.await(worker, "message")
	s.Equal("session-1", message["session_id"])
	s.Equal("where is my order?", message["text"])
	s.NotEmpty(workerID)
}

func (s *ForwardedDispatchSuite) TestARunningSessionsTextWithNoWorkerAnywhereSaysSo() {
	_, err := s.elsewhere.AssignMessage(context.Background(), s.customerID(),
		dispatch.Message{SessionID: "session-1", Text: "where is my order?"})

	s.ErrorIs(err, dispatch.ErrNoWorkers)
}

func (s *ForwardedDispatchSuite) TestAToolHostedByAWorkerOnAnotherNodeIsOfferedAndRunHere() {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsOnTheOtherNode(agent)
	s.Require().Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 1 && offered[0].Name == "investigate_sdk"
	}, settleFor, 10*time.Millisecond, "the tool was never offered here")

	type answer struct {
		output []llm.ContentPart
		err    error
	}
	answered := make(chan answer, 1)
	go func() {
		output, err := s.elsewhere.RunHosted(context.Background(), s.customerID(), agent,
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk", Arguments: `{"sdk":"android"}`})
		answered <- answer{output, err}
	}()

	call := s.await(worker, "tool_call")
	s.Equal(`{"sdk":"android"}`, call["arguments"])
	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_result", "id": call["id"], "output": "targetSdkVersion 35"}))

	select {
	case got := <-answered:
		s.Require().NoError(got.err)
		s.Equal("targetSdkVersion 35", llm.TextOf(got.output))
	case <-time.After(settleFor):
		s.Fail("the answer never came back from the other node")
	}
}

func (s *ForwardedDispatchSuite) TestAPictureAHostedToolAnswersWithComesBackFromAnotherNode() {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsOnTheOtherNode(agent)
	s.Require().Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 1
	}, settleFor, 10*time.Millisecond, "the tool was never offered here")
	answered := make(chan []llm.ContentPart, 1)
	go func() {
		parts, err := s.elsewhere.RunHosted(context.Background(), s.customerID(), agent,
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk"})
		s.NoError(err)
		answered <- parts
	}()
	call := s.await(worker, "tool_call")

	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_result", "tool_call_id": call["id"], "output": []frame{
		{"type": "image_url", "image_url": frame{"url": roses}},
	}}))

	select {
	case parts := <-answered:
		s.Require().Len(parts, 1)
		s.Require().NotNil(parts[0].Image, "the picture was lost between nodes")
		s.NotEmpty(parts[0].Image.Data)
	case <-time.After(settleFor):
		s.Fail("the answer never came back from the other node")
	}
}

func (s *ForwardedDispatchSuite) TestACallOnAnotherNodeTheSessionGivesUpOnIsCancelledOnTheWorker() {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsOnTheOtherNode(agent)
	s.Require().Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 1
	}, settleFor, 10*time.Millisecond, "the tool was never offered here")
	ctx, giveUp := context.WithCancel(context.Background())
	defer giveUp()
	go func() {
		_, _ = s.elsewhere.RunHosted(ctx, s.customerID(), agent,
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk"})
	}()
	s.await(worker, "tool_call")

	giveUp()

	cancel := s.await(worker, "tool_cancel")
	s.Equal("call-1", cancel["id"])
}

func (s *ForwardedDispatchSuite) TestAToolIsNoLongerOfferedOnceTheWorkerHostingItLeaves() {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsOnTheOtherNode(agent)
	s.Require().Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 1
	}, settleFor, 10*time.Millisecond)

	s.Require().NoError(worker.Close())

	s.Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 0
	}, settleFor, 10*time.Millisecond, "a tool nobody runs any more is still offered")
}

func (s *ForwardedDispatchSuite) TestAToolHostedForAnotherCustomerIsNotOffered() {
	agent := "agent-" + s.utils.uuid()
	worker := s.waitsOnTheOtherNodeAs(s.data.backendOfAnotherApp())
	s.host(worker, agent)

	offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
	s.Empty(offered)
}

func (s *ForwardedDispatchSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ForwardedDispatchSuite) TestACallArrivingWhereNoWorkerWaitsReachesTheWorkerOnAnotherNode() {
	callID := s.holdsANumber()
	worker := s.waitsOnTheOtherNode()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCall("default:"+callID)))

	call := s.await(worker, "call")
	s.Equal(callID, call["call_id"])
}

func (s *ForwardedDispatchSuite) TestACallIsAnsweredByAWorkerOnTheNodeItArrivedAtWhenThereIsOne() {
	callID := s.holdsANumber()
	elsewhere := s.waitsOnTheOtherNode()
	here, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCall("default:"+callID)))

	select {
	case call := <-here.Calls():
		s.Equal(callID, call.CallID)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker on the node it arrived at")
	}
	s.sentNothing(elsewhere, "the call was carried to another node as well")
}

func (s *ForwardedDispatchSuite) TestACallIsNotCarriedToAnotherCustomersWorker() {
	callID := s.holdsANumber()
	somebodyElse := s.waitsOnTheOtherNodeAs(s.data.backendOfAnotherApp())

	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCall("default:"+callID)))

	s.sentNothing(somebodyElse, "a call was carried to somebody else's worker")
}

func (s *ForwardedDispatchSuite) TestAMessageArrivingWhereNoWorkerWaitsReachesTheWorkerOnAnotherNode() {
	channel := "chat-" + s.utils.uuid()
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: "text"}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	worker := s.waitsOnTheOtherNode()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", fmt.Sprintf(`{
  "type": "message.new",
  "channel_id": %q,
  "channel_type": "agent",
  "channel_custom": {%q: %q},
  "message": {"id": "message-1", "text": "is anybody there?", "user": {"id": "sam", "name": "Sam"}}
}`, channel, ConfigField, config.ID)))

	message := s.await(worker, "message")
	s.Equal(channel, message["channel_id"])
	s.Equal(config.ID, message["config_id"])
}

// holdsANumber gives the suite's app a number attached to a call, and returns the call.
func (s *ForwardedDispatchSuite) holdsANumber() string {
	ctx := context.Background()
	e164 := s.utils.number()
	callID := "phone-" + e164
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: e164, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), e164,
		store.NumberAttachment{TrunkID: "trunk-" + s.utils.uuid(), CallType: "default", CallID: callID}))
	return callID
}

// waitsOnTheOtherNode connects a worker for the suite's app to the other node.
func (s *ForwardedDispatchSuite) waitsOnTheOtherNode() *websocket.Conn {
	return s.waitsOnTheOtherNodeAs(s.serverClient)
}

func (s *ForwardedDispatchSuite) waitsOnTheOtherNodeAs(client *testClient) *websocket.Conn {
	connection := client.on(s.other).opens("/v1/dispatch")
	s.await(connection, "ready")
	return connection
}

// hostsOnTheOtherNode connects a worker for the suite's app to the other node, hosting one
// tool for an agent.
func (s *ForwardedDispatchSuite) hostsOnTheOtherNode(agent string) *websocket.Conn {
	worker := s.waitsOnTheOtherNode()
	s.host(worker, agent)
	return worker
}

func (s *ForwardedDispatchSuite) host(worker *websocket.Conn, agent string) {
	s.Require().NoError(worker.WriteJSON(frame{
		"type": "host_tools", "agent_id": agent,
		"tools": []frame{{"name": "investigate_sdk", "description": "Read SDK source"}},
	}))
	s.await(worker, "hosting")
}

// sentNothing fails when a worker is sent anything but pongs before the hook could have
// reached it.
func (s *ForwardedDispatchSuite) sentNothing(connection *websocket.Conn, complaint string) {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(dropped)))
	for {
		var received frame
		err := connection.ReadJSON(&received)
		var timeout net.Error
		if errors.As(err, &timeout) && timeout.Timeout() {
			return
		}
		s.Require().NoError(err)
		if received["type"] != "pong" {
			s.Failf(complaint, "the worker was sent %v", received["type"])
			return
		}
	}
}
