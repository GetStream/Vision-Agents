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

func (s *StoreSuite) TestACallStartedAgainLearnsTheConversationItWasBoundTo() {
	// A call row from before the column has none. The session carrying on under the same id
	// names its conversation, and the row keeps it from then on.
	call := Call{
		ID: "session-learns-conversation", CustomerID: "acme",
		CallID: "call-1", AgentID: "agent-learns-conversation", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	before, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Require().Empty(before.ConversationID)

	again := call
	again.ConversationID = "agent:support-1"
	s.Require().NoError(s.store.StartCall(s.ctx, &again))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Equal("agent:support-1", read.ConversationID)
}

func (s *StoreSuite) TestACallStartedAgainKeepsTheConversationItWasBoundTo() {
	call := Call{
		ID: "session-keeps-conversation", CustomerID: "acme",
		CallID: "call-1", AgentID: "agent-keeps-conversation", StartedAt: s.base,
		ConversationID: "agent:first",
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	again := call
	again.ConversationID = "agent:second"
	s.Require().NoError(s.store.StartCall(s.ctx, &again))
	third := call
	third.ConversationID = ""
	s.Require().NoError(s.store.StartCall(s.ctx, &third))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Equal("agent:first", read.ConversationID)
}

func (s *StoreSuite) TestACallTheRouterPlacedIsStoredAsOutboundBetweenItsNumbers() {
	// A client opening the session of a call it placed may name neither number.
	s.placedPhoneCall("placed", func(call *PhoneCall) {
		call.CustomerID, call.CallType, call.CallID = "acme", "agent", "session-placed"
	})
	call := Call{ID: "session-placed", CustomerID: "acme", CallID: "session-placed", AgentID: "agent-placed", StartedAt: s.base}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Equal(Outbound, read.Direction)
	s.Equal("+15125550100", read.FromNumber)
	s.Equal("+15125550199", read.ToNumber)
}

func (s *StoreSuite) TestAnArrivingCallIsStoredAsInboundEvenWhereAnotherCustomerPlacedOne() {
	s.placedPhoneCall("theirs", func(call *PhoneCall) {
		call.CustomerID, call.CallType, call.CallID = "someone-else", "agent", "session-arrived"
	})
	call := Call{ID: "session-arrived", CustomerID: "acme", CallID: "session-arrived", AgentID: "agent-arrived",
		FromNumber: "+15125550142", StartedAt: s.base}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Equal(Inbound, read.Direction)
	s.Equal("+15125550142", read.FromNumber)
	s.Empty(read.ToNumber)
}

func (s *ShardedStoreSuite) TestACallAnotherShardPlacedIsStoredAsOutbound() {
	placed := PhoneCall{Token: "there", CustomerID: "acme", CallType: "agent", CallID: "session-there", Vendor: "telnyx",
		LocalNumber: "+15125550100", RemoteNumber: "+15125550199", PlacedAt: s.base}
	s.Require().NoError(s.other.InsertPhoneCall(s.ctx, &placed))
	call := Call{ID: "session-there", CustomerID: "acme", CallID: "session-there", AgentID: "agent-there", StartedAt: s.base}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	read, err := s.store.Call(s.ctx, "acme", call.ID)
	s.Require().NoError(err)
	s.Equal(Outbound, read.Direction)
	s.Equal("+15125550199", read.ToNumber)
}
