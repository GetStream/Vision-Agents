//go:build integration

package api

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// Who may name a thread channel. Only the customer's backend, or the Router's own message
// hook (threadhooks.go), holds a session on one. An end user's device naming one, by its
// agent id or by any other way into a session, is answered exactly as for an id no thread
// channel has, so nothing it is told says the thread exists. The tests below compare the
// two answers; SlackChannelSuite's own tests are the backend and the hook still working.

// TestADeviceNamingAThreadChannelsAgentIDKeepsAConversationOfItsOwn: a device's text session
// under a thread channel's agent id keeps a support channel of its own, as one under an agent
// id no thread channel has does.
func (s *SlackChannelSuite) TestADeviceNamingAThreadChannelsAgentIDKeepsAConversationOfItsOwn() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.client.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})
	guessed := s.client.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &unknown, Text: pointerTo(true)})

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel))
	s.Equal(channel, named.AgentId)
	s.True(strings.HasPrefix(value(named.ConversationId), "agent:support-"),
		"the conversation is the device's own, not the thread channel: %s", value(named.ConversationId))
}

// TestTheThreadIsAnsweredByTheRouterNotByADevicesSessionNamingIt: the person's message in the
// thread is answered by the Router's own session, into the Slack thread, and never reaches
// the session a device opened under the thread channel's agent id.
func (s *SlackChannelSuite) TestTheThreadIsAnsweredByTheRouterNotByADevicesSessionNamingIt() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	mine := s.client.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Equal("Noted.", s.posted(1)[0].Text, "the reply leaves into the Slack thread")
	own := strings.TrimPrefix(value(mine.ConversationId), "agent:")
	s.Empty(s.chat.Stored(own), "nothing of the thread reaches the device's conversation")
}

// TestTheBackendsSessionOnAThreadChannelAnswersTheThread: the hook finds the session the
// backend opened on the thread channel by its conversation, and it answers there; the
// Router opens none of its own, which would take the backend's over.
func (s *SlackChannelSuite) TestTheBackendsSessionOnAThreadChannelAnswersTheThread() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Equal("Noted.", s.posted(1)[0].Text)
	s.Nil(s.serverClient.getSession(held.Id).ClosedAt, "the backend's session answered and is still running")
}

// TestADevicesCallNamingAThreadChannelWritesNoTranscriptThere: a voice transcript is written
// into the channel the agent id names (chatlog.Options.Channel). A device's call under a
// thread channel's agent id writes none there; under an unknown one it writes its own.
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.client.createSession(CreateSessionRequest{AgentId: &channel, CallId: pointerTo("call-" + s.utils.uuid())})
	guessed := s.client.createSession(CreateSessionRequest{AgentId: &unknown, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown agent id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's call")
}

// TestABackendsCallNamingAThreadChannelWritesItsTranscriptThereAsBefore: the backend is the
// customer's own; its call under a thread channel's agent id is as it was.
func (s *SlackChannelSuite) TestABackendsCallNamingAThreadChannelWritesItsTranscriptThereAsBefore() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	s.serverClient.createSession(CreateSessionRequest{AgentId: &channel, CallId: pointerTo("call-" + s.utils.uuid())})

	s.True(s.transcribed.opened(chatlog.ChannelType + ":" + channel))
}

// TestADevicesSocketNamingAThreadChannelWritesNoTranscriptThere: the voice socket is a way
// into a session of its own (socketws.go), answered as POST /v1/agents/sessions is.
func (s *SlackChannelSuite) TestADevicesSocketNamingAThreadChannelWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.socketSession(channel)
	guessed := s.socketSession(unknown)

	s.Equal(channel, named["agent_id"])
	s.Equal(unknown, guessed["agent_id"])
	s.Equal(guessed["state"], named["state"])
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown agent id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's socket")
}

// TestADeviceReadingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne: the session the
// backend holds on a thread channel is, to a device, an id that does not exist.
func (s *SlackChannelSuite) TestADeviceReadingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne() {
	held := s.heldOnThread()

	s.answeredAsUnknown(http.MethodGet, "/v1/agents/sessions/%s", held, nil)
}

// TestADeviceForkingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne.
func (s *SlackChannelSuite) TestADeviceForkingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne() {
	held := s.heldOnThread()

	s.answeredAsUnknown(http.MethodPost, "/v1/agents/sessions/%s/fork", held, ForkSessionRequest{})
}

// TestADeviceAnsweringInAThreadChannelsSessionIsAnsweredAsForAnUnknownOne: a response is
// how a chat that ended is reopened (sessionToAnswer), and how a running one is written in.
func (s *SlackChannelSuite) TestADeviceAnsweringInAThreadChannelsSessionIsAnsweredAsForAnUnknownOne() {
	held := s.heldOnThread()

	s.answeredAsUnknown(http.MethodPost, "/v1/agents/sessions/%s/responses", held, CreateResponseRequest{Text: "what did they say?"})
}

// TestADeviceListingAThreadChannelsResponsesIsAnsweredAsForAnUnknownSession.
func (s *SlackChannelSuite) TestADeviceListingAThreadChannelsResponsesIsAnsweredAsForAnUnknownSession() {
	held := s.heldOnThread()

	s.answeredAsUnknown(http.MethodGet, "/v1/agents/sessions/%s/responses", held, nil)
}

// TestADeviceWatchingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne: the events socket.
func (s *SlackChannelSuite) TestADeviceWatchingAThreadChannelsSessionIsAnsweredAsForAnUnknownOne() {
	held := s.heldOnThread()

	_, watched := s.client.watch("/v1/agents/sessions/" + held + "/events")
	_, guessed := s.client.watch("/v1/agents/sessions/" + s.utils.uuid() + "/events")

	s.Equal(guessed, watched)
	s.Equal(http.StatusNotFound, watched)
}

// TestReadingAThreadChannelsMessagesIsAnsweredAsForAnUnknownChannel: no request reads a
// thread channel's messages, as at baseline 646c7ad4, which answered any agent:thread- id
// as it answers one nobody holds. A device is refused this server-side operation first.
func (s *SlackChannelSuite) TestReadingAThreadChannelsMessagesIsAnsweredAsForAnUnknownChannel() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	s.Equal(s.readConversation(s.client, unknown, "/messages"), s.readConversation(s.client, channel, "/messages"))
	s.Equal(http.StatusForbidden, s.readConversation(s.client, channel, "/messages").status)
	s.Equal(s.readConversation(s.serverClient, unknown, "/messages"), s.readConversation(s.serverClient, channel, "/messages"))
}

// TestReadingACommandInAThreadChannelIsAnsweredAsForAnUnknownChannel.
func (s *SlackChannelSuite) TestReadingACommandInAThreadChannelIsAnsweredAsForAnUnknownChannel() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	s.Equal(s.readConversation(s.client, unknown, "/commands/command-1"), s.readConversation(s.client, channel, "/commands/command-1"))
	s.Equal(http.StatusForbidden, s.readConversation(s.client, channel, "/commands/command-1").status)
	s.Equal(s.readConversation(s.serverClient, unknown, "/commands/command-1"), s.readConversation(s.serverClient, channel, "/commands/command-1"))
}

// TestABackendDoesNotForkAThreadChannelsRunningSession: a thread is everyone's in it
// (Spec.Shared), and a fork would carry their words into a conversation one caller owns,
// outside the thread and its connector rule. None is made, with its history or without.
func (s *SlackChannelSuite) TestABackendDoesNotForkAThreadChannelsRunningSession() {
	held := s.heldOnThread()

	carried, carriedFailure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+held+"/fork", ForkSessionRequest{})
	fresh, freshFailure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+held+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)})

	s.Equal(http.StatusBadRequest, carried)
	s.Equal("a thread channel's conversation is not forked", carriedFailure)
	s.Equal(http.StatusBadRequest, fresh)
	s.Equal("a thread channel's conversation is not forked", freshFailure)
}

// TestABackendDoesNotForkAThreadChannelsSessionThatEnded: from its stored row either. The
// backend carries on in the thread by naming its agent id again.
func (s *SlackChannelSuite) TestABackendDoesNotForkAThreadChannelsSessionThatEnded() {
	held := s.heldOnThread()
	s.serverClient.stopSession(held)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+held+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("a thread channel's conversation is not forked", failure)
}

// TestABackendDoesNotReopenAThreadChannelsSessionThatEnded: a chat that ended is carried on
// under its id (sessionToAnswer), from the spec it ran on. A thread channel's is not, as at
// base: only its agent id opens one, after reading the thread's link again.
func (s *SlackChannelSuite) TestABackendDoesNotReopenAThreadChannelsSessionThatEnded() {
	held := s.heldOnThread()
	s.serverClient.stopSession(held)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+held+"/responses",
		CreateResponseRequest{Text: "and the deploy?"})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid conversation channel", failure)
}

// TestABackendOpensAThreadChannelAgainByItsAgentID: the one way in, after a session on it
// ended.
func (s *SlackChannelSuite) TestABackendOpensAThreadChannelAgainByItsAgentID() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})
	s.serverClient.stopSession(held.Id)

	again := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})

	s.Equal("agent:"+channel, value(again.ConversationId))
}

// TestWithNoThreadLinkedADevicesAgentIDIsAnsweredAsBefore: with no channel_threads row, as
// staging runs with connectors off, an agent id in the thread- namespace is any agent id: a
// conversation of its own, and a call's transcript in the channel it names.
func (s *SlackChannelSuite) TestWithNoThreadLinkedADevicesAgentIDIsAnsweredAsBefore() {
	texted, called := conversation.ThreadChannelPrefix+s.utils.uuid(), conversation.ThreadChannelPrefix+s.utils.uuid()

	text := s.client.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &texted, Text: pointerTo(true)})
	s.client.createSession(CreateSessionRequest{AgentId: &called, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Equal(texted, text.AgentId)
	s.True(strings.HasPrefix(value(text.ConversationId), "agent:support-"))
	s.True(s.transcribed.opened(chatlog.ChannelType + ":" + called))
	s.Zero(s.threadChannels())
}

// TestWithNoThreadLinkedABackendsAgentIDIsAnsweredAsBefore.
func (s *SlackChannelSuite) TestWithNoThreadLinkedABackendsAgentIDIsAnsweredAsBefore() {
	texted, called := conversation.ThreadChannelPrefix+s.utils.uuid(), conversation.ThreadChannelPrefix+s.utils.uuid()

	text := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &texted, Text: pointerTo(true)})
	s.serverClient.createSession(CreateSessionRequest{AgentId: &called, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Equal(texted, text.AgentId)
	s.True(strings.HasPrefix(value(text.ConversationId), "agent:support-"))
	s.True(s.transcribed.opened(chatlog.ChannelType + ":" + called))
	s.Zero(s.threadChannels())
}

// heldOnThread is the id of a session the backend opened on a thread channel, once its row
// is stored, as a worker opens one (Dispatch.Conversation).
func (s *SlackChannelSuite) heldOnThread() string {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, Text: pointerTo(true)})
	s.Require().Equal("agent:"+channel, value(held.ConversationId))
	s.Require().Eventually(func() bool {
		_, err := s.store.StoredSession(context.Background(), s.customerID(), held.Id)
		return err == nil
	}, settleFor, 20*time.Millisecond, "the session's row was never stored")
	return held.Id
}

// answeredAsUnknown asserts a device is answered for the session held as for an id no
// session has: not found, with the same error.
func (s *SlackChannelSuite) answeredAsUnknown(method, path, held string, body any) {
	thread := s.answerTo(s.client, method, strings.Replace(path, "%s", held, 1), body)
	guessed := s.answerTo(s.client, method, strings.Replace(path, "%s", s.utils.uuid(), 1), body)

	s.Equal(guessed, thread)
	s.Equal(http.StatusNotFound, thread.status)
}

// readConversation is what client is answered reading a conversation of the agent id.
func (s *SlackChannelSuite) readConversation(client *testClient, agentID, what string) reply {
	return s.answerTo(client, http.MethodGet,
		"/v1/agents/conversations/"+url.PathEscape("agent:"+agentID)+what+"?agent_id="+agentID, nil)
}

// reply is a request's status and the error it was answered with, if any.
type reply struct {
	status int
	error  ErrorDetail
}

// answerTo is what client is answered for one request.
func (s *SlackChannelSuite) answerTo(client *testClient, method, path string, body any) reply {
	status, payload := client.call(method, path, body)
	answered := reply{status: status}
	if status >= http.StatusBadRequest {
		var failure ErrorResponse
		s.Require().NoError(json.Unmarshal(payload, &failure), string(payload))
		answered.error = failure.Error
	}
	return answered
}

// answeredFor is what a created session says of itself that does not differ by its ids: the
// agent id it was asked for, the sort of conversation it keeps, its state and modality.
func answeredFor(opened Session, agentID string) []any {
	return []any{opened.AgentId == agentID, strings.HasPrefix(value(opened.ConversationId), "agent:support-"),
		opened.State, opened.Modality}
}

// socketSession opens a device's voice socket session under agentID and returns the session
// it was answered with.
func (s *SlackChannelSuite) socketSession(agentID string) map[string]any {
	connection := s.client.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "sample_rate": 24_000, "session": map[string]any{"agent_id": agentID},
	}))
	var answered map[string]any
	s.Require().NoError(connection.ReadJSON(&answered))
	s.Require().Equal("session", answered["type"], "the socket answered %v", answered)
	return answered["session"].(map[string]any)
}

// openedTranscripts keeps the channel each voice session's transcript was opened for, named
// as cmd/router's transcriptFor names it: the conversation's, else the agent id's.
type openedTranscripts struct {
	mu       sync.Mutex
	channels []string
}

func (o *openedTranscripts) open(_ context.Context, spec session.Spec, _ streamapp.Bound, _ *slog.Logger) (session.Transcript, error) {
	channel := spec.ConversationID
	if channel == "" {
		channel = chatlog.ChannelType + ":" + spec.AgentID
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.channels = append(o.channels, channel)
	return keptNowhere{}, nil
}

func (o *openedTranscripts) opened(cid string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Contains(o.channels, cid)
}

// keptNowhere is a transcript that keeps nothing: the tests read which channel it was opened
// for, not what it was told.
type keptNowhere struct{}

func (keptNowhere) Start(context.Context) error { return nil }
func (keptNowhere) Record(agent.Event)          {}
func (keptNowhere) Reply(string)                {}
func (keptNowhere) Close()                      {}
