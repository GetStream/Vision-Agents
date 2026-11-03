package conversation_test

import (
	"context"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// DispatchedChannelSuite is a conversation held on the channel a message the Router handed to
// a worker was written in: the customer's own channel, with none of the router's stamps. Only
// the Router's word for it opens it (conversation.RouterOpensDispatched).
type DispatchedChannelSuite struct {
	suite.Suite
	chat    *chattest.Server
	service *conversation.Service
	// channel is the channel's id, and cid its agent cid.
	channel, cid string
}

func TestDispatchedChannelSuite(t *testing.T) {
	suite.Run(t, new(DispatchedChannelSuite))
}

func (s *DispatchedChannelSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.service = conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.channel = "chat-" + uuid.NewString()
	s.cid = "agent:" + s.channel
}

func (s *DispatchedChannelSuite) TestTheRoutersWordOpensTheChannelSomebodyWroteIn() {
	s.created(map[string]any{"agent_config_id": "config"})
	s.wrote()

	c, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid)

	s.Require().NoError(err)
	s.Equal(s.cid, c.CID())
}

// On base, and still: a request naming the channel is refused, the router never stamped it.
func (s *DispatchedChannelSuite) TestWithoutTheRoutersWordTheChannelIsRefused() {
	s.created(nil)
	s.wrote()

	_, _, _, err := s.service.Open(context.Background(), "customer", s.channel, s.cid)

	s.Require().Error(err)
	s.Contains(err.Error(), "belongs to another customer or agent")
}

func (s *DispatchedChannelSuite) TestTheRoutersWordForAnotherChannelDoesNotOpenThisOne() {
	s.created(nil)
	s.wrote()

	_, _, _, err := s.service.Open(s.routerOpens("agent:chat-"+uuid.NewString()), "customer", s.channel, s.cid)

	s.Require().Error(err)
}

// A channel the router stamped is placed by its stamps alone: another customer's, a session's
// own, or one shared with members is not opened on the Router's word.
func (s *DispatchedChannelSuite) TestAStampedChannelIsPlacedByItsStampsAlone() {
	for stamp, value := range map[string]any{
		conversation.CustomerField: "somebody-else",
		"support_agent_id":         "another-agent",
		"support_owner_id":         "sam",
		"support_access":           "members",
		conversation.TriggerField:  conversation.SessionCommandTrigger,
	} {
		s.SetupTest()
		s.created(map[string]any{stamp: value})
		s.wrote()

		_, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid)

		s.Error(err, stamp)
	}
}

// A channel with nothing written in it is not the one a message came from, such as one of
// that name in another app than the message was written in.
func (s *DispatchedChannelSuite) TestAChannelNothingWasWrittenInIsRefused() {
	s.created(nil)

	_, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid)

	s.Require().Error(err)
}

// The router never stamped the channel with a memory scope, so the session's own is kept: a
// worker remembering per user opens it as one that does not, on first open and on a later one.
func (s *DispatchedChannelSuite) TestTheRoutersWordOpensTheChannelUnderTheSessionsMemoryScope() {
	s.created(nil)
	s.wrote()
	scope := memory.Scope{UserID: "sam"}

	c, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid, scope)
	s.Require().NoError(err)
	c.Release()
	_, _, _, err = s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid, scope)

	s.Require().NoError(err, "opened again while this service still holds it")
}

// A session opened on the channel later is handed what the person asked as well as what was
// answered. The person's message nothing answered yet is the one the session is about to be
// told, so it is not history.
func (s *DispatchedChannelSuite) TestTheChannelIsReadBackWithWhatThePersonWrote() {
	s.created(nil)
	s.wrote()
	c, history, _, err := s.service.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid)
	s.Require().NoError(err)
	s.Empty(history)
	_, err = c.BeginFollowUp("message-1", "is it raining in Rome?")
	s.Require().NoError(err)
	c.Observe(agent.Responding{TurnID: "turn-1"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-1", Text: "It is."})
	c.Observe(agent.Responded{TurnID: "turn-1", Text: "It is."})
	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.channel)) == 2 && !s.service.Writing() }, 5*time.Second, 10*time.Millisecond)
	c.Release()

	later := conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(later.Close)
	_, history, _, err = later.Open(s.routerOpens(s.cid), "customer", s.channel, s.cid)

	s.Require().NoError(err)
	s.Equal([]llm.Message{{Role: llm.User, Content: "is it raining in Rome?", Author: "sam"}, {Role: llm.Assistant, Content: "It is."}}, history)
}

func (s *DispatchedChannelSuite) routerOpens(cid string) context.Context {
	return conversation.RouterOpensDispatched(context.Background(), cid)
}

// created makes the channel as the customer's app does, with the custom data given.
func (s *DispatchedChannelSuite) created(custom map[string]any) {
	creator := "sam"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), "agent", s.channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &creator, Custom: custom},
	})
	s.Require().NoError(err)
}

// wrote writes the person's message in the channel.
func (s *DispatchedChannelSuite) wrote() {
	person, text := "sam", "is it raining in Rome?"
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "agent", s.channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &person},
	})
	s.Require().NoError(err)
}
