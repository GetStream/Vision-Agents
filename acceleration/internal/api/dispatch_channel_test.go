//go:build integration

package api

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

// TestAWorkersSessionOpenedOnAnotherNodeAnswersInTheChannel: the worker opens its session
// with a request of its own, which the load balancer may hand any node, not the one that
// handed the message out. In memory alone, that node had no word of it: agent:<session id>.
func (s *MessageHooksSuite) TestAWorkersSessionOpenedOnAnotherNodeAnswersInTheChannel() {
	handed := s.handedOut()

	opened := s.serverClient.on(s.otherNode()).createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("en-low-latency")})

	s.Equal(chatlog.ChannelType+":"+s.channelID, value(opened.ConversationId))
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
		"end user":                  s.client,
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

// AI-2551: what a worker hands on from the channel is the person's message, already there.
// Asked as a command (a request id), the session wrote it a second time, as support-operator.

// TestAWorkersQuestionWithARequestIDIsNotWrittenIntoTheChannelAgain fails on base: the
// channel holds the person's message, a copy of it ("user completed"), then the reply.
func (s *MessageHooksSuite) TestAWorkersQuestionWithARequestIDIsNotWrittenIntoTheChannelAgain() {
	handed := s.handedOut()
	opened := s.workersSession(handed)

	s.asked(opened.Id, handed.Text)

	s.answered(handed.Text, finishedReply)
}

// TestAWorkersQuestionWithoutARequestIDIsNotWrittenIntoTheChannelAgain: a worker that sends
// no request id was answered with the reply alone on base too (Respond). The cap it asks for
// still reaches the model.
func (s *MessageHooksSuite) TestAWorkersQuestionWithoutARequestIDIsNotWrittenIntoTheChannelAgain() {
	handed := s.handedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	seen := len(s.vision.requests())

	s.askedAs(opened.Id, CreateResponseRequest{Text: handed.Text, MaxOutputTokens: pointerTo(77)})

	s.answered(handed.Text, finishedReply)
	s.True(slices.ContainsFunc(s.vision.requests()[seen:], func(asked llm.ResponseParams) bool { return asked.MaxOutputTokens == 77 }),
		"the worker's cap never reached the model")
}

// TestAWorkersRespondCommandIsNotWrittenIntoTheChannelAgain fails on base, as the response
// with a request id does: /respond with one is the same command. Its receipt is still
// announced as command_accepted, before the reply.
func (s *MessageHooksSuite) TestAWorkersRespondCommandIsNotWrittenIntoTheChannelAgain() {
	handed := s.handedOut()
	opened := s.workersSession(handed)
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	id := s.utils.uuid()

	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		map[string]any{"text": handed.Text, "request_id": id}, &receipt))
	accepted, _ := s.await(watching, "command_accepted")["command"].(map[string]any)
	s.await(watching, "responded")

	s.Equal(id, accepted["request_id"])
	s.Equal(receipt.AssistantMessageId, accepted["assistant_message_id"])
	s.Equal(id, receipt.RequestId)
	s.Empty(receipt.UserMessageId, "the person's message is not the session's")
	s.NotEmpty(receipt.AssistantMessageId)
	s.answered(handed.Text, finishedReply)
}

// TestAWorkersCommandWithAPhotoUploadsNothingIntoTheChannel fails on base: the photo is
// already on the person's message, and was uploaded again for the copy. The model is still
// shown it, under the cap the worker asked for.
func (s *MessageHooksSuite) TestAWorkersCommandWithAPhotoUploadsNothingIntoTheChannel() {
	handed := s.handedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"+s.utils.uuid()))
	uploaded, seen := len(s.chat.Uploads()), len(s.vision.requests())

	s.askedAs(opened.Id, CreateResponseRequest{Text: handed.Text, RequestId: pointerTo(s.utils.uuid()), Images: &[]ImageSource{{Url: photo}}, MaxOutputTokens: pointerTo(77)})

	s.answered(handed.Text, finishedReply)
	s.Empty(s.chat.Uploads()[uploaded:])
	s.True(slices.ContainsFunc(s.vision.requests()[seen:], llm.ResponseParams.HasImage), "the model was never shown the photo")
	s.True(slices.ContainsFunc(s.vision.requests()[seen:], func(asked llm.ResponseParams) bool { return asked.MaxOutputTokens == 77 }),
		"the worker's cap never reached the model")
}

// TestAWorkersQuestionAskedAgainIsAnsweredOnce: the Go SDK asks after its request id once
// its socket drops; the same id again starts no second reply and writes nothing.
func (s *MessageHooksSuite) TestAWorkersQuestionAskedAgainIsAnsweredOnce() {
	handed := s.handedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	seen := len(s.vision.requests())
	asked := CreateResponseRequest{Text: handed.Text, RequestId: pointerTo(s.utils.uuid())}
	s.askedAs(opened.Id, asked)
	s.answered(handed.Text, finishedReply)

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses", asked, nil))

	s.Never(func() bool { return len(s.vision.requests())-seen > 1 }, dropped, 20*time.Millisecond, "asked again, the model was asked again")
	s.Len(s.vision.requests()[seen:], 1)
	s.answered(handed.Text, finishedReply)
}

// TestACommandToASessionOfItsOwnStillWritesThePersonsMessage is the control: a session no
// message was handed out for writes the question asked with a request id into its own
// channel, under that id, by its owner or support-operator, then the reply, as on base
// (probe in the PR's author log).
func (s *MessageHooksSuite) TestACommandToASessionOfItsOwnStillWritesThePersonsMessage() {
	for name, opener := range map[string]struct {
		client *testClient
		author string
	}{
		"a backend's session":   {s.serverClient, "support-operator"},
		"an end user's session": {s.client, s.client.userID},
	} {
		opened := opener.client.createSession(CreateSessionRequest{AgentId: pointerTo("agent-" + s.utils.uuid()), Llm: pointerTo("en-low-latency")})
		s.Require().Equal(chatlog.ChannelType+":"+opened.Id, value(opened.ConversationId), name)
		watching := opener.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
		id := s.utils.uuid()

		s.Require().Equal(http.StatusAccepted, opener.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
			CreateResponseRequest{Text: "is it raining in Rome?", RequestId: &id}, nil), name)
		s.await(watching, "responded")

		var written []string
		s.Eventually(func() bool {
			written = written[:0]
			for _, stored := range s.chat.Stored(opened.Id) {
				custom, _ := stored["custom"].(map[string]any)
				metadata, _ := custom["support_message"].(map[string]any)
				written = append(written, fmt.Sprintf("%v %v %v %v", stored["id"], stored["user_id"], metadata["role"], stored["text"]))
			}
			return len(written) == 2 && !s.conversations.Writing()
		}, settleFor, 20*time.Millisecond, name)
		s.Equal(fmt.Sprintf("%s %s user is it raining in Rome?", id, opener.author), written[0], name)
		s.Contains(written[1], " assistant ", name)
	}
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
// The Go SDK's Responses.Create gives a question with no photos a request id of its own.
func (s *MessageHooksSuite) asked(session, text string) {
	s.askedAs(session, CreateResponseRequest{Text: text, RequestId: pointerTo(s.utils.uuid())})
}

// askedAs creates the response asked on session and waits for its answer. The response names
// the reply, which is how a worker follows it: its id is one of the session's responses.
func (s *MessageHooksSuite) askedAs(session string, asked CreateResponseRequest) {
	watching := s.serverClient.opens("/v1/agents/sessions/" + session + "/events")
	var created AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+session+"/responses", asked, &created))
	s.await(watching, "responded")
	s.Require().NotEmpty(created.Id, "the response names no reply to follow")
	s.Eventually(func() bool {
		var page AgentResponsePage
		return s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+session+"/responses", nil, &page) == http.StatusOK &&
			slices.ContainsFunc(page.Items, func(listed AgentResponse) bool { return listed.Id == created.Id })
	}, settleFor, 20*time.Millisecond, "the reply is not among the session's responses")
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
	s.writesIn(s.channelID, text)
}

// writesIn writes a person's message in channel, created by the customer's app with none of
// the router's stamps.
func (s *RouterSuite) writesIn(channel, text string) {
	ctx := context.Background()
	person := "sam"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &person}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channel,
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

// TestQueuedMessagesToAWorkersSessionAreEachAnsweredOnceInTurn: messages queued for the channel
// a worker's session answers are answered there, each once: one written while a reply is still
// being written once that reply is done, every later one with a follow-up of its own, and a
// second try of one not at all.
func (s *ChatInboundQueueSuite) TestQueuedMessagesToAWorkersSessionAreEachAnsweredOnceInTurn() {
	ctx := context.Background()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	config := deliveredConfig(&s.RouterSuite, s.customerID())
	first := "is it raining in " + s.utils.uuid() + "?"
	s.Require().NoError(s.router.AnswerChatInbound(ctx, s.writtenAndQueued(config, first)))
	handed := s.handed(worker.Messages())
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("holding/holding-model")})
	s.Require().Equal(chatlog.ChannelType+":"+s.channelID, value(opened.ConversationId))
	s.holding.holds()
	s.T().Cleanup(s.holding.answers)
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: first}, nil))
	s.Require().Eventually(func() bool { return s.told(first) == 1 }, settleFor, 20*time.Millisecond)

	second := "and in " + s.utils.uuid() + "?"
	queued := s.writtenAndQueued(config, second)
	ended := make(chan error, 1)
	go func() { ended <- s.router.AnswerChatInbound(ctx, queued) }()
	// Its row is settled just before the session is handed the message.
	s.Require().Eventually(func() bool { return s.row(queued.MessageID).State == store.ChatInboundDone }, settleFor, 5*time.Millisecond)
	s.Never(func() bool { return len(ended) > 0 }, dropped, 20*time.Millisecond, "gave up on a message while the reply before it was written")
	s.holding.answers()
	select {
	case err := <-ended:
		s.Require().NoError(err)
	case <-time.After(settleFor):
		s.FailNow("the message was never answered")
	}
	s.Eventually(func() bool { return s.told(second) == 1 }, settleFor, 20*time.Millisecond, "the message written during a reply")

	third := "and tomorrow in " + s.utils.uuid() + "?"
	queued = s.writtenAndQueued(config, third)
	s.Require().NoError(s.router.AnswerChatInbound(ctx, queued))
	s.Eventually(func() bool { return s.told(third) == 1 }, settleFor, 20*time.Millisecond, "a later message")
	s.Require().NoError(s.router.AnswerChatInbound(ctx, queued))
	s.Never(func() bool { return s.told(third) > 1 }, dropped, 20*time.Millisecond, "one queued delivery is one answer")
	s.False(s.reachesWorker(worker.Messages()), "a message to a running session was dispatched again")
}

// writtenAndQueued writes a person's message in the suite's channel and queues its delivery,
// naming config, as the endpoint would.
func (s *ChatInboundQueueSuite) writtenAndQueued(config, text string) store.ChatInbound {
	s.writesIn(s.channelID, text)
	return s.row(s.queue(strings.Replace(s.message(suiteStreamApp, config, ""), "is my order on its way?", text, 1)))
}

// told is how many replies the holding model was asked with text the last thing said to it.
func (s *ChatInboundQueueSuite) told(text string) int {
	count := 0
	for _, asked := range s.holding.replies() {
		said := harness.WithoutNotes(asked.Input)
		if len(said) > 0 && said[len(said)-1].Content == text {
			count++
		}
	}
	return count
}
