//go:build integration

package store

import "time"

func (s *StoreSuite) TestAnAgentIdSaysWhoseChannelItIsAndWhatRanOnIt() {
	// A message arriving on a channel names the agent and nothing else. Without this row
	// there is no way to tell whose message it is, and a message that cannot be billed to
	// anybody cannot be answered.
	//
	// The agent id belongs to this test rather than being shared: a channel is looked up
	// by it alone, so two tests naming the same one are asking about each other's calls.
	agentID := "agent-whose-channel"
	call := Call{
		ID: "session-whose-channel", CustomerID: "acme",
		CallID: "call-1", AgentID: agentID, ConfigID: "chat_support", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	found, err := s.store.CallByAgentInApp(s.ctx, AppScope{Unpinned: true}, agentID)

	s.Require().NoError(err)
	s.Equal("acme", found.CustomerID)
	s.Equal("chat_support", found.ConfigID, "which agent is being written to")
}

func (s *StoreSuite) TestWritingToAChannelReachesTheLastConversationOnIt() {
	// An agent id outlives the call that made it, so the same channel can hold several.
	// The one somebody writing there is continuing is the last one.
	agentID := "agent-last-conversation"
	older := Call{
		ID: "older-last-conversation", CustomerID: "acme",
		CallID: "call-1", AgentID: agentID, ConfigID: "old_config",
		StartedAt: s.base.Add(-time.Hour),
	}
	newer := Call{
		ID: "newer-last-conversation", CustomerID: "acme",
		CallID: "call-2", AgentID: agentID, ConfigID: "chat_support",
		StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &older))
	s.Require().NoError(s.store.StartCall(s.ctx, &newer))

	found, err := s.store.CallByAgentInApp(s.ctx, AppScope{Unpinned: true}, agentID)

	s.Require().NoError(err)
	s.Equal(newer.ID, found.ID)
	s.Equal("chat_support", found.ConfigID)
}

func (s *StoreSuite) TestNoConversationIsFoundForAChannelNoAgentHasBeenOn() {
	// This is a message in a channel that merely looks like an agent's. There is nobody to
	// bill it to and nothing to start, and saying so is the whole answer.
	_, err := s.store.CallByAgentInApp(s.ctx, AppScope{Unpinned: true}, "a-channel-nobody-ran-on")

	s.Require().Error(err)
}

func (s *StoreSuite) TestACallKeepsTheTimeItFirstEnded() {
	// An agent leaves once. A second close is the same leaving reported again, and must
	// not stretch the call to cover it.
	call := Call{
		ID: "session-ended-once", CustomerID: "acme",
		CallID: "call-1", AgentID: "agent-ended-once", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Minute)))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Hour)))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Require().NotNil(read.EndedAt)
	s.WithinDuration(s.base.Add(time.Minute), *read.EndedAt, time.Second)
}
