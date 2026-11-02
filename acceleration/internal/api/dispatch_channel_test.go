//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// AI-2505: the session a dispatch worker opens to answer a message the router handed it
// answers in the channel the message was written in. On base it answered in agent:<session
// id>, a channel of its own that nobody reads.

// TestAWorkersSessionAnswersInTheChannelItWasHandedAMessageFrom fails on base: the session's
// conversation is agent:<session id>, and the reply is written there.
func (s *MessageHooksSuite) TestAWorkersSessionAnswersInTheChannelItWasHandedAMessageFrom() {
	handed := s.handedOut()

	opened := s.workersSession(handed)

	s.Equal(chatlog.ChannelType+":"+s.channelID, value(opened.ConversationId))
	s.asked(opened.Id, handed.Text)
	s.answered(handed.Text, finishedReply)
	_, kept := s.chat.Channel(opened.Id)
	s.False(kept, "the session made a channel of its own")
}

// TestAMessageWrittenAgainIsAnsweredByTheSessionAlreadyAnsweringTheChannel: no second
// dispatch, and one reply, in the channel. On base the running session took it and nothing
// answered it.
func (s *MessageHooksSuite) TestAMessageWrittenAgainIsAnsweredByTheSessionAlreadyAnsweringTheChannel() {
	handed := s.handedOut()
	opened := s.workersSession(handed)
	s.asked(opened.Id, handed.Text)
	s.answered(handed.Text, finishedReply)

	s.inChannel("and what about Spain?")
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.again("and what about Spain?")))

	s.answered(handed.Text, finishedReply, "and what about Spain?", finishedReply)
	s.nothingReaches(s.worker, "a message to a running session was dispatched again")
}

// TestOnlyTheBackendOfTheChannelsCustomerOpensItsChannel: a session another customer's
// backend, the customer's own end user, or its backend acting for one of its users opens
// under the channel's id keeps a channel of its own, as on base.
func (s *MessageHooksSuite) TestOnlyTheBackendOfTheChannelsCustomerOpensItsChannel() {
	handed := s.handedOut()
	owner := s.app
	asked := CreateSessionRequest{AgentId: &handed.AgentID, Llm: pointerTo("en-low-latency")}

	for name, opener := range map[string]*testClient{
		"end user": s.client,
		"backend acting for a user": s.serverClient.actingFor(s.client),
	} {
		opened := opener.createSession(asked)
		s.Equal(chatlog.ChannelType+":"+opened.Id, value(opened.ConversationId), name)
	}
	s.useApp(s.data.createApp())
	opened := s.serverClient.createSession(asked)
	s.Equal(chatlog.ChannelType+":"+opened.Id, value(opened.ConversationId), "another customer")
	s.useApp(owner)
}

// TestASessionUnderAChannelNothingWasHandedOutFromKeepsAChannelOfItsOwn is the control: a
// backend's text session under any agent id has a conversation of its own, as on base
// (agent:<session id>; probe in the PR's author log).
func (s *MessageHooksSuite) TestASessionUnderAChannelNothingWasHandedOutFromKeepsAChannelOfItsOwn() {
	s.inChannel("is it raining in Rome?")

	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, Llm: pointerTo("en-low-latency")})

	s.Equal(chatlog.ChannelType+":"+opened.Id, value(opened.ConversationId))
}

// handedOut writes a message in the suite's channel, as a person in the customer's app does,
// and returns what the customer's worker is handed for it.
func (s *MessageHooksSuite) handedOut() dispatch.Message {
	configID := s.holds()
	s.worker = s.workerOf(s.customerID())
	custom := fmt.Sprintf(`{%q: %q}`, ConfigField, configID)
	s.inChannel("is it raining in Rome?")
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.writtenAs("message-1", "is it raining in Rome?", custom)))
	return s.handed(s.worker)
}

// workersSession opens the session a worker does for a message: the Go SDK's
// Dispatch.Conversation names only the agent id it was handed.
func (s *MessageHooksSuite) workersSession(handed dispatch.Message) Session {
	return s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("en-low-latency")})
}

// asked hands the session the message's text, as the worker does, and waits for its answer.
func (s *MessageHooksSuite) asked(session, text string) {
	watching := s.serverClient.opens("/v1/agents/sessions/" + session + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+session+"/responses",
		CreateResponseRequest{Text: text}, nil))
	s.await(watching, "responded")
}

// finishedReply stands for an agent's finished reply in what answered reads: a session past the
// suite's first is handed a model that answers nothing, so the reply is told by its metadata.
const finishedReply = "(finished reply)"

// answered waits for the suite's channel to hold just the messages given, in order: each
// person's message once and each reply once it is finished.
func (s *MessageHooksSuite) answered(messages ...string) {
	s.Eventually(func() bool {
		return slices.Equal(messages, s.inTheChannel()) && !s.conversations.Writing()
	}, settleFor, 20*time.Millisecond)
	s.Equal(messages, s.inTheChannel())
}

// inTheChannel is the suite's channel as answered reads it: a person's text, or finishedReply.
func (s *MessageHooksSuite) inTheChannel() []string {
	var read []string
	for _, stored := range s.chat.Stored(s.channelID) {
		custom, _ := stored["custom"].(map[string]any)
		written, _ := custom["support_message"].(map[string]any)
		switch {
		case written == nil:
			text, _ := stored["text"].(string)
			read = append(read, text)
		case written["role"] == "assistant" && written["state"] == "completed":
			read = append(read, finishedReply)
		default:
			read = append(read, fmt.Sprintf("%v %v", written["role"], written["state"]))
		}
	}
	return read
}

// inChannel writes a person's message in the suite's channel, created by the customer's app
// with none of the router's stamps.
func (s *MessageHooksSuite) inChannel(text string) {
	ctx := context.Background()
	person := "sam"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, s.channelID,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &person}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, s.channelID,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &person}})
	s.Require().NoError(err)
}

// again is the message.new for a second message in the suite's channel.
func (s *MessageHooksSuite) again(text string) string {
	return s.writtenAs("message-2", text, "{}")
}

func (s *MessageHooksSuite) writtenAs(id, text, custom string) string {
	return fmt.Sprintf(`{
  "type": "message.new",
  "cid": "agent:%s",
  "channel_id": %q,
  "channel_type": "agent",
  "channel_custom": %s,
  "message": {"id": %q, "text": %q, "user": {"id": "sam", "name": "Sam"}}
}`, s.channelID, s.channelID, custom, id, text)
}
