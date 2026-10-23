//go:build integration

package store

import (
	"time"
)

// replied records one response with the items given, numbered in order.
func (s *StoreSuite) replied(sessionID, responseID, said string, at time.Time, items ...AgentResponseItem) {
	s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
		ID: responseID, SessionID: sessionID, CustomerID: "app", Said: said, CreatedAt: at,
	}))
	for i := range items {
		items[i].ResponseID, items[i].Ordinal, items[i].SessionID = responseID, i, sessionID
		items[i].At = at.Add(time.Duration(i) * time.Second)
	}
	s.Require().NoError(s.store.AppendResponseItems(s.ctx, items))
	s.Require().NoError(s.store.FinishResponse(s.ctx, responseID, ResponseCompleted, "", at.Add(time.Minute)))
}

func answer(turn, text string) AgentResponseItem {
	return AgentResponseItem{Kind: ItemAnswer, Text: text, Payload: map[string]any{"turn_id": turn}}
}

func result(callID, text string) AgentResponseItem {
	return AgentResponseItem{Kind: ItemToolResult, ToolName: "lookup", Text: text, Payload: map[string]any{"call_id": callID}}
}

func (s *StoreSuite) TestAReplyIsFoundByTheTurnItsTranscriptWasWrittenUnder() {
	s.opened("one", "app", s.base, func(session *AgentSession) { session.ConversationID = "agent:chat" })
	s.replied("one", "first", "look it up", s.base,
		AgentResponseItem{Kind: ItemSaid, Text: "look it up"},
		answer("turn-1", ""),
		AgentResponseItem{Kind: ItemToolCall, ToolName: "lookup", Payload: map[string]any{"call_id": "call-1"}},
		result("call-1", "42"),
		answer("turn-2", "It is 42."),
	)

	replies, err := s.store.ReplyItems(s.ctx, "app", "agent:chat", []string{"turn-2", "turn-unknown"})

	s.Require().NoError(err)
	s.Require().Len(replies, 1)
	kinds := []string{}
	for _, item := range replies["turn-2"] {
		kinds = append(kinds, item.Kind)
	}
	s.Equal([]string{ItemAnswer, ItemToolResult, ItemAnswer}, kinds)
}

func (s *StoreSuite) TestAnotherConversationsTurnFindsNothing() {
	s.opened("one", "app", s.base, func(session *AgentSession) { session.ConversationID = "agent:theirs" })
	s.replied("one", "first", "secret", s.base, result("call-1", "private"), answer("turn-1", "done"))
	s.opened("two", "app", s.base, func(session *AgentSession) { session.ConversationID = "agent:mine" })

	replies, err := s.store.ReplyItems(s.ctx, "app", "agent:mine", []string{"turn-1"})

	s.Require().NoError(err)
	s.Empty(replies)
}

func (s *StoreSuite) TestARewoundReplyFindsNothing() {
	s.opened("one", "app", s.base, func(session *AgentSession) { session.ConversationID = "agent:chat" })
	s.replied("one", "first", "a", s.base, answer("turn-1", "one"))
	s.replied("one", "second", "b", s.base.Add(time.Hour), result("call-1", "x"), answer("turn-2", "two"))
	s.Require().NoError(s.store.RewindResponses(s.ctx, "app", "one", "first", s.base.Add(2*time.Hour)))

	replies, err := s.store.ReplyItems(s.ctx, "app", "agent:chat", []string{"turn-1", "turn-2"})

	s.Require().NoError(err)
	s.Len(replies, 1)
	s.Contains(replies, "turn-1")
}

func (s *StoreSuite) TestTheOpeningIsTheFirstThingSaidAcrossTheConversationsSessions() {
	s.opened("later", "app", s.base.Add(time.Hour), func(session *AgentSession) { session.ConversationID = "agent:chat" })
	s.opened("first", "app", s.base, func(session *AgentSession) { session.ConversationID = "agent:chat" })
	s.replied("later", "r2", "and then", s.base.Add(time.Hour), answer("turn-2", "ok"))
	s.replied("first", "r1", "the task", s.base, answer("turn-1", "ok"))

	said, err := s.store.Opening(s.ctx, "app", "agent:chat")

	s.Require().NoError(err)
	s.Equal("the task", said)
}

func (s *StoreSuite) TestAConfigKeepsItsResumeHistory() {
	config := &AgentConfig{CustomerID: "app", Name: "resume", Mode: AgentModeText,
		ResumeHistory: ResumeHistory{IncludeTools: true, MaxCharacters: 120000}}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	config.ResumeHistory.MaxMessages = 50
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, config))

	stored, err := s.store.AgentConfig(s.ctx, "app", config.ID)

	s.Require().NoError(err)
	s.Equal(ResumeHistory{IncludeTools: true, MaxMessages: 50, MaxCharacters: 120000}, stored.ResumeHistory)
}
