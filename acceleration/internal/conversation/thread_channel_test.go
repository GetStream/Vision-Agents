package conversation_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ThreadChannelSuite is a persistent conversation held on a thread channel: the agent channel
// the channel bridge writes one external thread into, stamped as internal/channelbridge
// stamps it, with the people's messages written without a source. The suite opens it as the
// Router's thread path does (conversation.RouterOpensThread); a request naming it is refused.
type ThreadChannelSuite struct {
	suite.Suite
	chat    *chattest.Server
	service *conversation.Service
	// channel is the thread channel's id, and cid its agent cid.
	channel, cid string
	// finished are the replies the service handed over, in order.
	mu       sync.Mutex
	finished []conversation.FinishedReply
}

func TestThreadChannelSuite(t *testing.T) {
	suite.Run(t, new(ThreadChannelSuite))
}

func (s *ThreadChannelSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.service = conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.finished = nil
	s.service.OnFinishedReply(func(reply conversation.FinishedReply) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.finished = append(s.finished, reply)
	})
	s.channel = conversation.ThreadChannelPrefix + uuid.NewString()
	s.cid = "agent:" + s.channel
	creator := "slack_bot-person"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), "agent", s.channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &creator, Custom: map[string]any{
			conversation.CustomerField: "customer", "support_agent_id": s.channel, "agent_config_id": "config",
		}},
	})
	s.Require().NoError(err)
}

func (s *ThreadChannelSuite) TestAThreadChannelIsNoSessionCommandChannel() {
	s.False(conversation.SessionCommandChannel("agent", s.channel), "the message hook must still see a person's message in it")
}

// The person's message the session is about to be told is not history yet, and once it is
// answered it is.
func (s *ThreadChannelSuite) TestThePeoplesMessagesAreHistoryOnceAnswered() {
	s.person("U1", "is the build green?")
	c, history, _, err := s.service.Open(s.routerOpens(), "customer", "", s.cid)
	s.Require().NoError(err)
	s.Empty(history, "the unanswered message is the turn about to be told")
	s.Equal(s.channel, c.Agent(), "the agent is the one the channel names")
	s.answer(c, "is the build green?", "It is.")
	s.person("U2", "and the deploy?")
	c.Release()

	_, history, _, err = s.service.Open(s.routerOpens(), "customer", "", s.cid)

	s.Require().NoError(err)
	s.Equal([]llm.Message{
		{Role: llm.User, Content: "is the build green?"},
		{Role: llm.Assistant, Content: "It is."},
	}, history)
}

func (s *ThreadChannelSuite) TestAFinishedReplyIsHandedOverWithItsStoredText() {
	s.person("U1", "is the build green?")
	c, _, _, err := s.service.Open(s.routerOpens(), "customer", "", s.cid)
	s.Require().NoError(err)

	s.answer(c, "is the build green?", "It is.")

	s.Require().Eventually(func() bool { return len(s.handedOver()) == 1 }, 5*time.Second, 10*time.Millisecond)
	reply := s.handedOver()[0]
	s.Equal(conversation.FinishedReply{Customer: "customer", CID: s.cid, MessageID: reply.MessageID, Text: "It is."}, reply)
	s.Contains(s.chat.Messages(s.channel), "It is.", "the reply is written into the thread channel")
	s.Never(func() bool { return len(s.handedOver()) > 1 }, 200*time.Millisecond, 20*time.Millisecond,
		"the reply being written is not handed over, only the stored final text")
}

// A conversation in a session command channel has no external thread to hand a reply to.
func (s *ThreadChannelSuite) TestAReplyInASupportChannelIsNotHandedOver() {
	c, _, _, err := s.service.Open(context.Background(), "customer", "agent", "")
	s.Require().NoError(err)

	s.Require().NoError(c.Begin("question"))
	c.Observe(agent.ResponseDelta{Text: "An answer."})
	c.Observe(agent.Responded{})

	s.Require().Eventually(func() bool { return len(s.chat.Messages(c.CID()[len("agent:"):])) == 2 }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return len(s.handedOver()) > 0 }, 200*time.Millisecond, 20*time.Millisecond)
}

// A conversation id a request names opens no thread channel, as at baseline 646c7ad4, where
// only support- channels opened: the same "invalid conversation channel".
func (s *ThreadChannelSuite) TestAThreadChannelARequestNamesDoesNotOpen() {
	s.person("U1", "is the build green?")

	_, _, _, err := s.service.Open(context.Background(), "customer", "", s.cid)

	s.Require().Error(err)
	s.Contains(err.Error(), "invalid conversation channel")
}

// The Router's word is for the one channel it found the channel_threads row of.
func (s *ThreadChannelSuite) TestTheRoutersWordForAnotherThreadChannelDoesNotOpenThisOne() {
	other := conversation.RouterOpensThread(context.Background(), "agent:"+conversation.ThreadChannelPrefix+uuid.NewString())

	_, _, _, err := s.service.Open(other, "customer", "", s.cid)

	s.Require().Error(err)
	s.Contains(err.Error(), "invalid conversation channel")
}

// Reading a thread channel's history by a conversation id a request names is refused as
// opening it is: the history endpoint, a fork's recall and a voice session's context.
func (s *ThreadChannelSuite) TestAThreadChannelARequestNamesHasNoHistoryToRead() {
	s.person("U1", "is the build green?")

	_, historyErr := s.service.HistoryForCaller(context.Background(), "customer", "", s.cid, "", "")
	_, _, contextErr := s.service.ContextForCaller(context.Background(), "customer", "", s.cid, "")

	s.Require().Error(historyErr)
	s.Contains(historyErr.Error(), "invalid conversation channel")
	s.Require().Error(contextErr)
	s.Contains(contextErr.Error(), "invalid conversation channel")
}

// A command in a thread channel is no command a request can look up, as at baseline.
func (s *ThreadChannelSuite) TestACommandInAThreadChannelIsNotFoundByARequest() {
	c, _, _, err := s.service.Open(s.routerOpens(), "customer", "", s.cid)
	s.Require().NoError(err)
	_, err = c.BeginCommand("asked", "is the build green?", "")
	s.Require().NoError(err)

	_, err = s.service.CommandForCaller(context.Background(), "customer", "", s.cid, "", "asked")

	s.ErrorIs(err, conversation.ErrCommandNotFound)
}

// While one conversation opens, which holds the service across its Stream Chat calls for up
// to OpenInApp's 20 s budget, every other conversation's writes go on. The other channel's
// read is held open by the test until the writes are in.
func (s *ThreadChannelSuite) TestAWriteDoesNotWaitForAnotherConversationOpening() {
	c, _, _, err := s.service.Open(context.Background(), "customer", "agent", "")
	s.Require().NoError(err)
	release := s.openingHeld()
	defer release()

	s.Require().NoError(c.Begin("question"))
	c.Observe(agent.ResponseDelta{Text: "An answer."})
	c.Observe(agent.Responded{})

	s.Require().Eventually(func() bool {
		return slices.Contains(s.chat.Messages(c.CID()[len("agent:"):]), "An answer.")
	}, writeBound, 10*time.Millisecond, "the reply's final text is written while the other conversation opens")
}

// A reply finished in a thread channel is handed over while another conversation opens, too.
func (s *ThreadChannelSuite) TestAFinishedReplyDoesNotWaitForAnotherConversationOpening() {
	s.person("U1", "is the build green?")
	c, _, _, err := s.service.Open(s.routerOpens(), "customer", "", s.cid)
	s.Require().NoError(err)
	release := s.openingHeld()
	defer release()

	s.answer(c, "is the build green?", "It is.")

	s.Require().Eventually(func() bool { return len(s.handedOver()) == 1 }, writeBound, 10*time.Millisecond,
		"the reply is handed over while the other conversation opens")
}

// writeBound is how long a write is given while another conversation opens: far over the
// milliseconds a write to chattest takes, and far under OpenInApp's 20 s budget, which a
// write waiting on the service's lock would sit through. The other open is held until after
// the wait, so a write that waits on it never lands within any bound.
const writeBound = 5 * time.Second

// openingHeld has another conversation open on a support channel of its own, its history
// read held open in Stream Chat, and returns once it is held. release lets it finish and
// waits for it.
func (s *ThreadChannelSuite) openingHeld() (release func()) {
	channel := "support-" + uuid.NewString()
	// The creator and owner OpenInApp stamps on a channel no end user owns.
	operator, owner := "support-operator", ""
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), "agent", channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &operator, Custom: map[string]any{
			conversation.CustomerField: "customer", "support_agent_id": "agent", "support_owner_id": owner,
			conversation.TriggerField: conversation.SessionCommandTrigger,
		}},
	})
	s.Require().NoError(err)
	// The channel query OpenInApp reads history with: POST /api/v2/chat/channels/{type}/{id}/query
	// (getstream-go GetOrCreateChannel).
	waiting, let := s.chat.Hold("/agent/" + channel + "/query")
	opened := make(chan error, 1)
	go func() {
		_, _, _, err := s.service.Open(context.Background(), "customer", "agent", "agent:"+channel)
		opened <- err
	}()
	select {
	case <-waiting:
	case err := <-opened:
		s.FailNow("the other conversation opened without reading its history", "%v", err)
	}
	return func() {
		let()
		s.NoError(<-opened)
	}
}

// routerOpens is the Router's word that it opens the suite's thread channel.
func (s *ThreadChannelSuite) routerOpens() context.Context {
	return conversation.RouterOpensThread(context.Background(), s.cid)
}

// person writes a message into the thread channel as the bridge does: no source.
func (s *ThreadChannelSuite) person(user, text string) {
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "agent", s.channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &user},
	})
	s.Require().NoError(err)
}

// answer has the conversation answer told with reply, as Session.FollowUp leads it to.
func (s *ThreadChannelSuite) answer(c *conversation.Conversation, told, reply string) {
	_, err := c.BeginFollowUp(told)
	s.Require().NoError(err)
	c.Observe(agent.ResponseDelta{Text: reply})
	c.Observe(agent.Responded{})
	s.Require().Eventually(func() bool {
		for _, text := range s.chat.Messages(s.channel) {
			if text == reply {
				return true
			}
		}
		return false
	}, 5*time.Second, 10*time.Millisecond)
}

func (s *ThreadChannelSuite) handedOver() []conversation.FinishedReply {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]conversation.FinishedReply(nil), s.finished...)
}
