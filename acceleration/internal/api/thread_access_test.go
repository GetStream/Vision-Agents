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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

// TestADevicesCallKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere: a voice
// session that names no agent id is keyed under its call id (Spec.Normalize), so a call id
// naming a thread channel is the same as an agent id naming it.
func (s *SlackChannelSuite) TestADevicesCallKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.client.createSession(CreateSessionRequest{CallId: &channel})
	guessed := s.client.createSession(CreateSessionRequest{CallId: &unknown})

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's call")
}

// TestADevicesCallKeyedUnderAThreadChannelByACallIDWithALeadingSpaceWritesNoTranscriptThere:
// Normalize trims a call id before it keys the session under it, so the check reads it
// trimmed too (Spec.KeyedAgentID).
func (s *SlackChannelSuite) TestADevicesCallKeyedUnderAThreadChannelByACallIDWithALeadingSpaceWritesNoTranscriptThere() {
	s.callAsForAnUnknownID(func(id string) string { return " " + id })
}

// TestADevicesCallKeyedUnderAThreadChannelByACallIDWithATrailingSpaceWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesCallKeyedUnderAThreadChannelByACallIDWithATrailingSpaceWritesNoTranscriptThere() {
	s.callAsForAnUnknownID(func(id string) string { return id + " " })
}

// TestADevicesCallKeyedUnderAThreadChannelByACallIDWithANewlineWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesCallKeyedUnderAThreadChannelByACallIDWithANewlineWritesNoTranscriptThere() {
	s.callAsForAnUnknownID(func(id string) string { return id + "\n" })
}

// TestADevicesCallKeyedUnderAThreadChannelsIDInUpperCaseWritesNoTranscriptThere: whether
// Stream Chat takes THREAD-<UUID> for thread-<uuid> is unverified, so it is kept out as if
// it did.
func (s *SlackChannelSuite) TestADevicesCallKeyedUnderAThreadChannelsIDInUpperCaseWritesNoTranscriptThere() {
	s.callAsForAnUnknownID(strings.ToUpper)
}

// TestADevicesCallNamingAThreadChannelsAgentIDInUpperCaseWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelsAgentIDInUpperCaseWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	upper, unknown := strings.ToUpper(channel), strings.ToUpper(conversation.ThreadChannelPrefix+s.utils.uuid())

	named := s.client.createSession(CreateSessionRequest{AgentId: &upper, CallId: pointerTo("call-" + s.utils.uuid())})
	guessed := s.client.createSession(CreateSessionRequest{AgentId: &unknown, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, upper))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown agent id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+upper), "the thread channel's id in upper case holds no transcript")
}

// TestADevicesCallNamingAThreadChannelsAgentIDWithALeadingSpaceWritesNoTranscriptThere:
// Normalize keeps an agent id as given, and whether Stream Chat takes " thread-<uuid>" for
// the thread channel is unverified, so it is kept out as if it did.
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelsAgentIDWithALeadingSpaceWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	padded, unknown := " "+channel, " "+conversation.ThreadChannelPrefix+s.utils.uuid()

	named := s.client.createSession(CreateSessionRequest{AgentId: &padded, CallId: pointerTo("call-" + s.utils.uuid())})
	guessed := s.client.createSession(CreateSessionRequest{AgentId: &unknown, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, padded))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown agent id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+padded), "the padded thread channel id holds no transcript")
}

// TestABackendNamingAThreadChannelInUpperCaseIsAnsweredAsBefore: the backend opens a thread
// channel by its exact id, as at base; any other spelling is any other agent id.
func (s *SlackChannelSuite) TestABackendNamingAThreadChannelInUpperCaseIsAnsweredAsBefore() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	upper := strings.ToUpper(channel)

	opened := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &upper, Text: pointerTo(true)})

	s.Equal(upper, opened.AgentId)
	s.True(strings.HasPrefix(value(opened.ConversationId), "agent:support-"), value(opened.ConversationId))
}

// TestADevicesSocketKeyedUnderAThreadChannelByAPaddedCallIDWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesSocketKeyedUnderAThreadChannelByAPaddedCallIDWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.socketSessionWith(map[string]any{"call_id": " " + channel})
	guessed := s.socketSessionWith(map[string]any{"call_id": " " + unknown})

	s.Equal(channel, named["agent_id"])
	s.Equal(unknown, guessed["agent_id"])
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's socket")
}

// TestADevicesForkKeyedUnderAThreadChannelByAPaddedCallIDWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesForkKeyedUnderAThreadChannelByAPaddedCallIDWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
	parent := s.client.createSession(CreateSessionRequest{CallId: pointerTo("call-" + s.utils.uuid())})

	var named, guessed Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork",
		ForkSessionRequest{CallId: pointerTo(channel + " ")}, &named))
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork",
		ForkSessionRequest{CallId: pointerTo(unknown + " ")}, &guessed))

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's fork")
}

// TestADevicesSocketKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesSocketKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()

	named := s.socketSessionWith(map[string]any{"call_id": channel})
	guessed := s.socketSessionWith(map[string]any{"call_id": unknown})

	s.Equal(channel, named["agent_id"])
	s.Equal(unknown, guessed["agent_id"])
	s.Equal(guessed["state"], named["state"])
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's socket")
}

// TestADevicesForkKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere: a voice fork
// joins the call the request names and is keyed under it (forkSpec).
func (s *SlackChannelSuite) TestADevicesForkKeyedUnderAThreadChannelByItsCallIDWritesNoTranscriptThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
	parent := s.client.createSession(CreateSessionRequest{CallId: pointerTo("call-" + s.utils.uuid())})

	var named, guessed Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork",
		ForkSessionRequest{CallId: &channel}, &named))
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork",
		ForkSessionRequest{CallId: &unknown}, &guessed))

	s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's fork")
}

// TestADevicesCallIsBarredFromAChannelWhoseThreadLinkCannotBeRead: when channel_threads
// cannot be read, whether the agent id is a thread channel is not known, and a device is kept
// out of it as if it were one.
func (s *SlackChannelSuite) TestADevicesCallIsBarredFromAChannelWhoseThreadLinkCannotBeRead() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.unreadableLink(channel)

	s.client.createSession(CreateSessionRequest{AgentId: &channel, CallId: pointerTo("call-" + s.utils.uuid())})

	s.False(s.transcribed.opened(chatlog.ChannelType + ":" + channel))
}

// TestAMessageInAChannelWhoseThreadLinkCannotBeReadIsAnsweredNowhere: the hook does not hand
// the thread's message to a session under the channel's agent id when it cannot tell whether
// the channel holds a thread.
func (s *SlackChannelSuite) TestAMessageInAChannelWhoseThreadLinkCannotBeReadIsAnsweredNowhere() {
	said := "is the build green? " + s.utils.uuid()
	channel := s.messaged("U0000ALICE", said, "1759740000.000100", "")
	s.unreadableLink(channel)
	s.client.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel, CallId: pointerTo("call-" + s.utils.uuid())})

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Never(func() bool { return s.told(said) }, dropped, 20*time.Millisecond, "no session was told the thread's message")
	s.Empty(s.slack.Posts())
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

// A conversation_id is a second way to name where a session writes. Each call shape a
// device has is below (AI-936). Only an agent: channel Openable allows is a conversation:
// any other conversation_id is refused before anything is written (Manager.Create reads its
// history first, conversation.historyIn), so a session never falls back to writing under a
// thread channel's agent id. The transcripts these tests read are named as cmd/router's
// transcriptFor names them (openedTranscripts), so a session that did fall back would show.

// TestADevicesCallNamingAThreadChannelOnAMessagingConversationWritesNoTranscriptThere: a
// conversation_id of another channel type would leave the transcript under the agent id
// (Spec.TranscriptChannel).
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelOnAMessagingConversationWritesNoTranscriptThere() {
	s.conversationAsForAnUnknownID(false, func(string) string { return "messaging:" + s.utils.uuid() })
}

// TestADevicesCallNamingAThreadChannelOnAConversationWithNoTypeWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelOnAConversationWithNoTypeWritesNoTranscriptThere() {
	s.conversationAsForAnUnknownID(false, func(string) string { return s.utils.uuid() })
}

// TestADevicesCallNamingAThreadChannelOnAMessagingChannelOfTheSameIDWritesNoTranscriptThere.
func (s *SlackChannelSuite) TestADevicesCallNamingAThreadChannelOnAMessagingChannelOfTheSameIDWritesNoTranscriptThere() {
	s.conversationAsForAnUnknownID(false, func(id string) string { return "messaging:" + id })
}

// TestADevicesChatNamingAThreadChannelOnAMessagingConversationWritesNothingThere.
func (s *SlackChannelSuite) TestADevicesChatNamingAThreadChannelOnAMessagingConversationWritesNothingThere() {
	s.conversationAsForAnUnknownID(true, func(string) string { return "messaging:" + s.utils.uuid() })
}

// TestADevicesCallOnAThreadChannelsConversationWritesNoTranscriptThere: conversation_id
// naming the thread channel itself is not Openable for a device.
func (s *SlackChannelSuite) TestADevicesCallOnAThreadChannelsConversationWritesNoTranscriptThere() {
	s.threadConversationAsForAnUnknownOne(false)
}

// TestADevicesChatOnAThreadChannelsConversationWritesNothingThere.
func (s *SlackChannelSuite) TestADevicesChatOnAThreadChannelsConversationWritesNothingThere() {
	s.threadConversationAsForAnUnknownOne(true)
}

// TestADevicesIncognitoCallNamingAThreadChannelWritesNoTranscriptThere: incognito clears the
// conversation_id (Spec.Normalize) and keeps no transcript, whatever conversation was named.
func (s *SlackChannelSuite) TestADevicesIncognitoCallNamingAThreadChannelWritesNoTranscriptThere() {
	for _, conversationID := range []string{"", "messaging:" + s.utils.uuid()} {
		channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
		unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
		asked := func(agentID string) CreateSessionRequest {
			request := CreateSessionRequest{AgentId: &agentID, CallId: pointerTo("call-" + s.utils.uuid()), Incognito: pointerTo(true)}
			if conversationID != "" {
				request.ConversationId = &conversationID
			}
			return request
		}

		named := s.client.createSession(asked(channel))
		guessed := s.client.createSession(asked(unknown))

		s.Equal(answeredFor(guessed, unknown), answeredFor(named, channel), conversationID)
		s.False(s.transcribed.opened(chatlog.ChannelType+":"+unknown), "an incognito call keeps no transcript: %q", conversationID)
		s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript: %q", conversationID)
	}
}

// TestWithNoThreadLinkedADevicesConversationOfAnotherTypeIsAnsweredAsBefore: with no
// channel_threads row, as staging runs with connectors off, a conversation_id of another
// type is refused as it always was.
func (s *SlackChannelSuite) TestWithNoThreadLinkedADevicesConversationOfAnotherTypeIsAnsweredAsBefore() {
	agentID := conversation.ThreadChannelPrefix + s.utils.uuid()

	answered := s.answerTo(s.client, http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{
		AgentId: &agentID, CallId: pointerTo("call-" + s.utils.uuid()), ConversationId: pointerTo("messaging:" + s.utils.uuid()),
	})

	s.Equal(http.StatusBadRequest, answered.status)
	s.Equal("invalid conversation channel", answered.error.Message)
	s.False(s.transcribed.opened(chatlog.ChannelType + ":" + agentID))
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

// callAsForAnUnknownID asserts a device's call under form(a thread channel's id) is answered
// as one under form(an unknown id), and writes no transcript where it is keyed.
func (s *SlackChannelSuite) callAsForAnUnknownID(form func(id string) string) {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
	asked, guessedAsked := form(channel), form(unknown)
	keyed, guessedKeyed := strings.TrimSpace(asked), strings.TrimSpace(guessedAsked)

	named := s.client.createSession(CreateSessionRequest{CallId: &asked})
	guessed := s.client.createSession(CreateSessionRequest{CallId: &guessedAsked})

	s.Equal(answeredFor(guessed, guessedKeyed), answeredFor(named, keyed))
	s.True(s.transcribed.opened(chatlog.ChannelType+":"+guessedKeyed), "an unknown call id's channel holds the transcript")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+keyed), "no transcript where the device's call is keyed")
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's call")
}

// conversationAsForAnUnknownID asserts a device's session under a thread channel's agent id,
// on the conversation_id conversationOf(the agent id) names, is answered as one under an
// unknown id, and writes nothing into the thread channel.
func (s *SlackChannelSuite) conversationAsForAnUnknownID(text bool, conversationOf func(id string) string) {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
	before := len(s.chat.Stored(channel))

	named := s.answerTo(s.client, http.MethodPost, "/v1/agents/sessions", s.deviceSession(text, &channel, conversationOf(channel)))
	guessed := s.answerTo(s.client, http.MethodPost, "/v1/agents/sessions", s.deviceSession(text, &unknown, conversationOf(unknown)))

	s.Equal(guessed, named)
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's session")
	s.Len(s.chat.Stored(channel), before, "nothing of the device's session is written into the thread channel")
}

// threadConversationAsForAnUnknownOne asserts a device's session with conversation_id naming
// a thread channel is answered as one naming an unknown thread channel, and writes nothing
// into it.
func (s *SlackChannelSuite) threadConversationAsForAnUnknownOne(text bool) {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	unknown := conversation.ThreadChannelPrefix + s.utils.uuid()
	before := len(s.chat.Stored(channel))

	named := s.answerTo(s.client, http.MethodPost, "/v1/agents/sessions", s.deviceSession(text, nil, chatlog.ChannelType+":"+channel))
	guessed := s.answerTo(s.client, http.MethodPost, "/v1/agents/sessions", s.deviceSession(text, nil, chatlog.ChannelType+":"+unknown))

	s.Equal(guessed, named)
	s.False(s.transcribed.opened(chatlog.ChannelType+":"+channel), "the thread channel holds no transcript of the device's session")
	s.Len(s.chat.Stored(channel), before, "nothing of the device's session is written into the thread channel")
}

// deviceSession is a device's request for a session under agentID, if any, on
// conversationID: a text one with the suite's agent config, else a call.
func (s *SlackChannelSuite) deviceSession(text bool, agentID *string, conversationID string) CreateSessionRequest {
	request := CreateSessionRequest{AgentId: agentID, ConversationId: &conversationID}
	if text {
		request.Agent, request.Text = &s.config.Name, pointerTo(true)
	} else {
		request.CallId = pointerTo("call-" + s.utils.uuid())
	}
	return request
}

// unreadableLink leaves a thread channel's channel_threads row one the store cannot read:
// thread_parts holds a number where a string goes, so ChannelThread fails to scan it.
func (s *SlackChannelSuite) unreadableLink(channel string) {
	_, err := s.store.DB().ExecContext(context.Background(),
		`UPDATE channel_threads SET thread_parts = '{"ts": 1}'::jsonb WHERE channel_id = ?`, channel)
	s.Require().NoError(err)
	_, err = s.store.ChannelThread(context.Background(), channel)
	s.Require().Error(err)
	s.Require().NotErrorIs(err, store.ErrNoChannelThread)
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
	return s.socketSessionWith(map[string]any{"agent_id": agentID})
}

// socketSessionWith opens a device's voice socket session asked for as asked.
func (s *SlackChannelSuite) socketSessionWith(asked map[string]any) map[string]any {
	connection := s.client.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "sample_rate": 24_000, "session": asked,
	}))
	var answered map[string]any
	s.Require().NoError(connection.ReadJSON(&answered))
	s.Require().Equal("session", answered["type"], "the socket answered %v", answered)
	return answered["session"].(map[string]any)
}

// openedTranscripts keeps the channel each voice session's transcript was opened for, named
// as cmd/router's transcriptFor names it (Spec.TranscriptChannel): the conversation's agent
// channel, else the agent id's, whatever other type of conversation_id was named.
type openedTranscripts struct {
	mu       sync.Mutex
	channels []string
}

func (o *openedTranscripts) open(_ context.Context, spec session.Spec, _ streamapp.Bound, _ *slog.Logger) (session.Transcript, error) {
	channel := spec.TranscriptChannel()
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
