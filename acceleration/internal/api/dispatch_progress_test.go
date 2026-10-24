//go:build integration

package api

import (
	"context"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// conversationHearing is the conversation a hosted call was made in, recording what the
// worker running it said. holds is whether the call is still running to be held.
type conversationHearing struct {
	holds bool

	mu      sync.Mutex
	held    []string
	updates []string
}

func (c *conversationHearing) HoldTool(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.held = append(c.held, id)
	return c.holds
}

func (c *conversationHearing) ToolUpdate(id, text string, verbatim bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if verbatim {
		text = "said: " + text
	}
	c.updates = append(c.updates, id+" "+text)
}

func (c *conversationHearing) heard() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.updates...)
}

// runsOnTheWorker hosts investigate_sdk on the suite's worker and makes one call to it from
// the conversation.
func (s *DispatchSuite) runsOnTheWorker(conversation dispatch.Progress) *websocket.Conn {
	worker := s.worker()
	s.Require().NoError(s.dispatch.Host(s.dispatch.Workers(s.customerID())[0], "stream-support",
		[]dispatch.Tool{{Name: "investigate_sdk", Description: "Read SDK source"}}, time.Minute))
	go func() {
		_, _ = s.dispatch.RunHosted(s.T().Context(), s.customerID(), "stream-support",
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk", Progress: conversation})
	}()
	s.Equal("tool_call", s.next(worker)["type"])
	return worker
}

func (s *DispatchSuite) TestAWorkerTellsTheConversationHowAHostedCallIsGoing() {
	conversation := &conversationHearing{}
	worker := s.runsOnTheWorker(conversation)

	s.Require().NoError(worker.WriteJSON(frame{
		"type": "tool_update", "tool_call_id": "call-1", "text": "reading the gradle files"}))
	s.Require().NoError(worker.WriteJSON(frame{
		"type": "tool_update", "tool_call_id": "call-1", "text": "One moment.", "verbatim": true}))

	s.Eventually(func() bool { return len(conversation.heard()) == 2 }, settleFor, 10*time.Millisecond)
	s.Equal([]string{"call-1 reading the gradle files", "call-1 said: One moment."}, conversation.heard())
}

func (s *DispatchSuite) TestAWorkerHoldsAHostedCallTheCallerMayTalkOver() {
	worker := s.runsOnTheWorker(&conversationHearing{holds: true})

	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_hold", "tool_call_id": "call-1"}))

	held := s.next(worker)
	s.Equal("tool_held", held["type"])
	s.Equal("call-1", held["id"])
}

func (s *DispatchSuite) TestAHostedCallItIsTooLateToHoldIsCancelled() {
	worker := s.runsOnTheWorker(&conversationHearing{})

	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_hold", "tool_call_id": "call-1"}))

	cancel := s.next(worker)
	s.Equal("tool_cancel", cancel["type"])
	s.Equal("call-1", cancel["id"])
}

// runsOnTheOtherNode hosts investigate_sdk on a worker on the other node and makes one call
// to it from a conversation here.
func (s *ForwardedDispatchSuite) runsOnTheOtherNode(conversation dispatch.Progress) *websocket.Conn {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsOnTheOtherNode(agent)
	s.Require().Eventually(func() bool {
		offered, _ := s.elsewhere.HostedTools(s.customerID(), agent)
		return len(offered) == 1
	}, settleFor, 10*time.Millisecond, "the tool was never offered here")
	ctx, giveUp := context.WithCancel(context.Background())
	s.T().Cleanup(giveUp)
	go func() {
		_, _ = s.elsewhere.RunHosted(ctx, s.customerID(), agent,
			dispatch.ToolCall{ID: "call-1", SessionID: "session-1", Name: "investigate_sdk", Progress: conversation})
	}()
	s.await(worker, "tool_call")
	return worker
}

func (s *ForwardedDispatchSuite) TestAWorkerOnAnotherNodeTellsTheConversationHowItsCallIsGoing() {
	conversation := &conversationHearing{}
	worker := s.runsOnTheOtherNode(conversation)

	s.Require().NoError(worker.WriteJSON(frame{
		"type": "tool_update", "tool_call_id": "call-1", "text": "reading the gradle files"}))

	s.Eventually(func() bool { return len(conversation.heard()) == 1 }, settleFor, 10*time.Millisecond,
		"the update never came back from the other node")
	s.Equal([]string{"call-1 reading the gradle files"}, conversation.heard())
}

func (s *ForwardedDispatchSuite) TestAWorkerOnAnotherNodeHoldsItsCallThroughTheConversation() {
	worker := s.runsOnTheOtherNode(&conversationHearing{holds: true})

	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_hold", "tool_call_id": "call-1"}))

	s.Equal("call-1", s.await(worker, "tool_held")["id"])
}

func (s *ForwardedDispatchSuite) TestAWorkerOnAnotherNodeIsToldWhenItsCallCannotBeHeld() {
	worker := s.runsOnTheOtherNode(&conversationHearing{})

	s.Require().NoError(worker.WriteJSON(frame{"type": "tool_hold", "tool_call_id": "call-1"}))

	s.Equal("call-1", s.await(worker, "tool_cancel")["id"])
}
