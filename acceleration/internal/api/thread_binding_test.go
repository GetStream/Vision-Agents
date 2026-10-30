//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A thread opened under one agent config is answered by the config the line is bound to now:
// the binding moved, and the thread's earlier turns, under the old config, do not keep it.
func (s *TelnyxChannelSuite) TestAThreadIsAnsweredByTheConfigTheLineIsBoundToNow() {
	person := "+13125550001"
	s.smsTurn(person, "Hi, is my order ready?")
	s.Equal("Noted.", s.took(1)[0].text)

	s.rebind()
	s.smsTurn(person, "And the invoice?")

	s.Equal("Counted.", s.took(2)[1].text)
}

// A thread whose earlier turns ran on a config deleted since is answered by the config the
// line is bound to now.
func (s *TelnyxChannelSuite) TestAThreadWhoseConfigWasDeletedIsAnsweredByTheConfigTheLineIsBoundToNow() {
	person := "+13125550001"
	s.smsTurn(person, "Hi, is my order ready?")

	old := s.config.ID
	s.rebind()
	s.Require().NoError(s.store.DeleteAgentConfig(context.Background(), s.customerID(), old))
	s.smsTurn(person, "And the invoice?")

	s.Equal("Counted.", s.took(2)[1].text)
}

// Another customer's config that names the line's connection, older than the config the line's
// customer binds it with, never answers the customer's thread. A save refuses such a binding
// (refuseUnbindable), so the row is written as a bad migration or a bug would leave it.
func (s *TelnyxChannelSuite) TestAnotherCustomersConfigNamingTheLineNeverAnswersItsThread() {
	ctx := context.Background()
	person := "+13125550001"
	s.smsTurn(person, "Hi, is my order ready?")
	forged := store.AgentConfig{
		CustomerID: s.data.createApp().app.ID, Name: "forged-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "echo/echo-model",
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &forged))
	_, err := s.store.DB().ExecContext(ctx,
		"UPDATE agent_configs SET connectors = (SELECT connectors FROM agent_configs WHERE id = ?) WHERE id = ?", s.config.ID, forged.ID)
	s.Require().NoError(err)

	s.rebind()
	s.smsTurn(person, "And the invoice?")

	s.Equal("Counted.", s.took(2)[1].text)
}

// A line two configs of the customer bind, as rows saved before a save refused the second
// (refuseSecondChannelAgent) are, is answered by the oldest, the one the bridge picks.
func (s *TelnyxChannelSuite) TestALineTwoConfigsBindIsAnsweredByTheOldest() {
	ctx := context.Background()
	newer := store.AgentConfig{
		CustomerID: s.customerID(), Name: "telnyx-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "counted/counted-model",
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &newer))
	_, err := s.store.DB().ExecContext(ctx,
		"UPDATE agent_configs SET connectors = (SELECT connectors FROM agent_configs WHERE id = ?) WHERE id = ?", s.config.ID, newer.ID)
	s.Require().NoError(err)

	s.smsTurn("+13125550001", "Hi, is my order ready?")

	s.Equal("Noted.", s.took(1)[0].text)
}

// A text written into a thread whose line no config binds any more by the time the agent is
// handed it is answered by nobody.
func (s *TelnyxChannelSuite) TestAThreadWhoseLineNoConfigBindsAnyMoreIsNotAnswered() {
	handOver := s.holdHandOffs()
	person := "+13125550001"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "Hi, is my order ready?"), time.Now()))
	channel := s.threadChannel(person)
	s.written(channel, 1)
	s.config.Connectors = nil
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))

	handOver()

	s.Never(func() bool { return s.repliesIn(channel) > 0 }, dropped, 20*time.Millisecond)
	s.Eventually(func() bool { return strings.Contains(s.logged.String(), "no agent config binds its line") },
		time.Second, 10*time.Millisecond)
	for line := range strings.Lines(s.logged.String()) {
		if strings.Contains(line, "no agent config binds its line") {
			s.Contains(line, "level=WARN", "a line nobody binds is the customer's to fix, not a failure")
		}
	}
}

// rebind moves the line's binding off the suite's config onto a new one of the same customer,
// whose model answers "Counted.", and makes it the suite's config.
func (s *TelnyxChannelSuite) rebind() {
	ctx := context.Background()
	bindings := s.config.Connectors
	s.config.Connectors = nil
	s.Require().NoError(s.store.UpdateAgentConfig(ctx, &s.config))
	s.config = store.AgentConfig{
		CustomerID: s.customerID(), Name: "telnyx-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "counted/counted-model",
		Connectors: bindings,
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &s.config))
}
