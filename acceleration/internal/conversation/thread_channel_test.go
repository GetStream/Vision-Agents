package conversation_test

import (
	"context"
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
// stamps it, with the people's messages written without a source.
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
	c, history, _, err := s.service.Open(context.Background(), "customer", "", s.cid)
	s.Require().NoError(err)
	s.Empty(history, "the unanswered message is the turn about to be told")
	s.Equal(s.channel, c.Agent(), "the agent is the one the channel names")
	s.answer(c, "is the build green?", "It is.")
	s.person("U2", "and the deploy?")
	c.Release()

	_, history, _, err = s.service.Open(context.Background(), "customer", "", s.cid)

	s.Require().NoError(err)
	s.Equal([]llm.Message{
		{Role: llm.User, Content: "is the build green?"},
		{Role: llm.Assistant, Content: "It is."},
	}, history)
}

func (s *ThreadChannelSuite) TestAFinishedReplyIsHandedOverWithItsStoredText() {
	s.person("U1", "is the build green?")
	c, _, _, err := s.service.Open(context.Background(), "customer", "", s.cid)
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
