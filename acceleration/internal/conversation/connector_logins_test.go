package conversation

import (
	"encoding/json"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// login is a consent a session would ask employee's conversation to show.
func login(name, attempt string) ConnectorAuthorization {
	return ConnectorAuthorization{
		Name: name, ConnectorID: "slack", ConnectionID: "connection-1", AuthorizationID: attempt,
		Title: "Connect Slack", LaunchURL: "https://router.example/v1/agents/connectors/oauth/launch/" + attempt,
		HandoffToken: "handoff-" + attempt, ExpiresAt: time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC),
	}
}

func (s *DisplaySuite) TestALoginAConnectorAsksForIsAttachedToTheReplyAndRestored() {
	c := s.open("on_call")
	receipt, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)

	s.True(c.AskToConnect("employee", login("slack", "attempt-1")))
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	var reply struct {
		Attachments []map[string]any `json:"attachments"`
	}
	s.Require().NoError(json.Unmarshal([]byte(s.raw(receipt.AssistantMessageID)), &reply))
	s.Require().NotEmpty(reply.Attachments)
	button := reply.Attachments[len(reply.Attachments)-1]
	s.Equal(ConnectorAuthorizationType, button["type"])
	s.Equal("Connect Slack", button["title"])
	c.Release()
	page, err := s.service.HistoryForCaller(s.T().Context(), "customer", "on_call", c.CID(), "", "employee")
	s.Require().NoError(err)
	s.Require().Len(page.Messages, 2)
	asked := login("slack", "attempt-1")
	asked.Type = ConnectorAuthorizationType
	s.Equal([]ConnectorAuthorization{asked}, page.Messages[1].ConnectorAuthorizations)
	// A session's client reads it where it reads a plugin's: among the attachments.
	encoded, err := json.Marshal(page.Messages[1])
	s.Require().NoError(err)
	var wire struct {
		Attachments []map[string]any `json:"attachments"`
	}
	s.Require().NoError(json.Unmarshal(encoded, &wire))
	shown := wire.Attachments[len(wire.Attachments)-1]
	s.Equal(ConnectorAuthorizationType, shown["type"])
	s.Equal(asked.LaunchURL, shown["launch_url"])
	s.Equal(asked.HandoffToken, shown["handoff_token"])
}

// TestALoginIsShownOnlyInItsOwnersConversation: whoever reads the attachment first can hand
// the consent off, so a conversation that is not the owner's own is never asked to show it.
func (s *DisplaySuite) TestALoginIsShownOnlyInItsOwnersConversation() {
	c := s.open("on_call")
	receipt, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)

	s.False(c.AskToConnect("someone-else", login("slack", "attempt-1")))
	s.False(c.AskToConnect("", login("slack", "attempt-1")))
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	s.NotContains(s.raw(receipt.AssistantMessageID), ConnectorAuthorizationType)
	s.NotContains(s.raw(receipt.AssistantMessageID), "handoff-attempt-1")
}

func (s *DisplaySuite) TestALoginIsShownOnlyOnAReplyBeingWritten() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	s.False(c.AskToConnect("employee", login("slack", "attempt-1")), "the reply is finished")
}

// TestAFinishedConnectorLoginMarksTheReplyAndDropsItsHandoffToken: the person has moved on by
// the time the provider hands them back; the reply that asked still says the login is done.
func (s *DisplaySuite) TestAFinishedConnectorLoginMarksTheReplyAndDropsItsHandoffToken() {
	c := s.open("on_call")
	asked, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	s.Require().True(c.AskToConnect("employee", login("slack", "attempt-1")))
	c.Observe(agent.Responded{})
	saved(s.T(), c)
	_, err = c.BeginCommand("command-b", "And on the crm?", "")
	s.Require().NoError(err)

	s.False(c.ConnectorConnected("attempt-2"), "another attempt")
	s.True(c.ConnectorConnected("attempt-1"))
	s.False(c.ConnectorConnected("attempt-1"), "finished once")
	saved(s.T(), c)

	raw := s.raw(asked.AssistantMessageID)
	s.Contains(raw, `"status":"connected"`)
	s.NotContains(raw, "handoff-attempt-1")
}

// TestAMessageWithNoConnectorLoginWritesWhatItDidBefore: no connector login, no new
// attachment, on the wire or in Chat.
func (s *DisplaySuite) TestAMessageWithNoConnectorLoginWritesWhatItDidBefore() {
	m := Message{ID: "m", Role: "assistant", Tools: []Tool{}}

	encoded, err := json.Marshal(m)

	s.Require().NoError(err)
	s.Contains(string(encoded), `"attachments":[]`)
	s.NotContains(string(encoded), ConnectorAuthorizationType)
	s.Empty(connectorAuthorizationAttachments(nil))
}

// TestAnAttachmentThatIsNotALaunchPageIsNotReadBack: what Chat hands back is read as a login
// only when its launch URL is the launch page of the attempt it names.
func (s *DisplaySuite) TestAnAttachmentThatIsNotALaunchPageIsNotReadBack() {
	elsewhere := login("slack", "attempt-1")
	elsewhere.LaunchURL = "https://evil.example/authorize"
	otherAttempt := login("slack", "attempt-1")
	otherAttempt.AuthorizationID = "attempt-2"
	plain := login("slack", "attempt-1")
	plain.LaunchURL = "http://router.example/v1/agents/connectors/oauth/launch/attempt-1"

	s.True(validConnectorAuthorization(login("slack", "attempt-1")))
	s.False(validConnectorAuthorization(elsewhere))
	s.False(validConnectorAuthorization(otherAttempt))
	s.False(validConnectorAuthorization(plain), "https only")
}

// TestTheReplyBeingWrittenSaysWhichLoginItShows: a second call in the same reply reuses the
// consent it shows; a finished reply, or a finished login, shows none.
func (s *DisplaySuite) TestTheReplyBeingWrittenSaysWhichLoginItShows() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	s.Require().True(c.AskToConnect("employee", login("slack", "attempt-1")))

	s.True(c.ShowsLogin("attempt-1"))
	s.False(c.ShowsLogin("attempt-2"))
	c.Observe(agent.Responded{})
	s.False(c.ShowsLogin("attempt-1"), "the reply is finished")
}
