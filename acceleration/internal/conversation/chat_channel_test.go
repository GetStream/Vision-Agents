package conversation_test

import (
	"context"
	"slices"
	"strings"
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

// ChatChannelSuite is a conversation the Router holds on a channel of a type other than agent,
// one an agent config answers in: a customer's group channel callouts:<id>, which opens on the
// Router's word alone (conversation.RouterOpensDispatched). Every request about the channel is
// made with its own type: the fake holds channels by cid, so one made under agent finds nothing.
type ChatChannelSuite struct {
	suite.Suite
	chat    *chattest.Server
	service *conversation.Service
	// channel is the channel's id, and cid its cid under the callouts type.
	channel, cid string
}

func TestChatChannelSuite(t *testing.T) {
	suite.Run(t, new(ChatChannelSuite))
}

func (s *ChatChannelSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.service = conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.channel = "team-" + uuid.NewString()
	s.cid = "callouts:" + s.channel
}

// The query that opens and reads the channel names its type.
func (s *ChatChannelSuite) TestTheRoutersWordOpensAChannelOfAnotherType() {
	s.created(nil)
	s.wrote()

	c, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", "config-1", s.cid)

	s.Require().NoError(err)
	s.Equal(s.cid, c.CID())
}

// A request naming the channel, as a caller's conversation_id does, holds no word for it.
func (s *ChatChannelSuite) TestWithoutTheRoutersWordAChannelOfAnotherTypeIsRefused() {
	s.created(nil)
	s.wrote()

	_, _, _, err := s.service.Open(context.Background(), "customer", "config-1", s.cid)

	s.Require().Error(err)
	s.Contains(err.Error(), "invalid conversation channel")
}

func (s *ChatChannelSuite) TestTheRoutersWordForAnotherChannelOfTheTypeDoesNotOpenThisOne() {
	s.created(nil)
	s.wrote()

	_, _, _, err := s.service.Open(s.routerOpens("callouts:team-"+uuid.NewString()), "customer", "config-1", s.cid)

	s.Require().Error(err)
	s.Contains(err.Error(), "invalid conversation channel")
}

// Whoever made the channel wrote its custom data: stamps naming another customer, a session's
// own channel or an owner, or this customer and agent, are not read, and the person's message
// is read back as theirs.
func (s *ChatChannelSuite) TestTheStampsOnAChannelOfAnotherTypeAreNotRead() {
	for _, custom := range []map[string]any{
		{conversation.CustomerField: "somebody-else", "support_agent_id": "another-agent", "support_owner_id": "ben",
			conversation.TriggerField: conversation.SessionCommandTrigger},
		{conversation.CustomerField: "customer", "support_agent_id": "config-1"},
	} {
		s.SetupTest()
		s.created(custom)
		s.wrote()
		s.answered(s.routerOpens(s.cid))
		later := conversation.NewForChat(s.chat.Client)
		s.T().Cleanup(later.Close)

		_, history, _, err := later.Open(s.routerOpens(s.cid), "customer", "config-1", s.cid)

		s.Require().NoError(err)
		s.Equal([]llm.Message{{Role: llm.User, Content: "I'm sick, I can't come at 6", Author: "anya"}, {Role: llm.Assistant, Content: "Feel better."}}, history, custom)
	}
}

// A channel id a session's own channel could not have is refused under any type, the word or
// none: a space, a colon, more than 64 characters.
func (s *ChatChannelSuite) TestAChannelIDNoConversationCanHaveIsRefusedUnderAnyType() {
	for _, id := range []string{"team 1", "team:1", strings.Repeat("t", 65)} {
		for _, cid := range []string{"agent:" + id, "callouts:" + id} {
			s.False(conversation.Openable(s.routerOpens(cid), cid), cid)
		}
	}
}

// The word names a cid, and one with no type before its colon is no channel at all.
func (s *ChatChannelSuite) TestTheRoutersWordForACIDWithNoTypeOpensNothing() {
	_, _, _, err := s.service.Open(s.routerOpens(":"+s.channel), "customer", "config-1", ":"+s.channel)

	s.Require().Error(err)
	s.Contains(err.Error(), "invalid conversation channel")
}

// The reply is written into the channel, and its typing indicator sent there while it runs.
func (s *ChatChannelSuite) TestTheReplyAndItsIndicatorGoToTheChannelOfItsType() {
	s.created(nil)
	s.wrote()
	c, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", "config-1", s.cid)
	s.Require().NoError(err)
	_, err = c.BeginFollowUp("message-1", "I'm sick, I can't come at 6")
	s.Require().NoError(err)

	c.Observe(agent.Responding{TurnID: "turn-1"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-1", Text: "Feel"})
	s.Require().Eventually(func() bool { return len(s.chat.Events(s.cid)) > 0 }, 5*time.Second, 10*time.Millisecond)
	c.Observe(agent.Responded{TurnID: "turn-1", Text: "Feel better."})

	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.cid)) == 2 && !s.service.Writing() }, 5*time.Second, 10*time.Millisecond)
	stored := s.chat.Stored(s.cid)
	s.Equal("config-1", stored[1]["user_id"])
}

func (s *ChatChannelSuite) TestAPhotoIsUploadedToTheChannelOfItsType() {
	s.created(nil)
	s.wrote()
	c, _, _, err := s.service.Open(s.routerOpens(s.cid), "customer", "config-1", s.cid)
	s.Require().NoError(err)

	_, err = c.PublishPhotos(context.Background(), []llm.ImagePart{{MIME: "image/jpeg", Data: []byte("jpeg")}})

	s.Require().NoError(err)
	uploads := s.chat.Uploads()
	s.Require().Len(uploads, 1)
	s.Equal("callouts", uploads[0].Type)
	s.Equal(s.channel, uploads[0].Channel)
}

// Describe renames the channel of its type only for the Router's word, which no session
// renaming a conversation holds: a customer's channel keeps the name the customer gave it.
func (s *ChatChannelSuite) TestAChannelOfAnotherTypeIsRenamedOnlyOnTheRoutersWord() {
	s.created(nil)

	s.Require().Error(s.service.Describe(context.Background(), "customer", s.cid, "Sick days", ""))
	s.Require().NoError(s.service.Describe(s.routerOpens(s.cid), "customer", s.cid, "Sick days", ""))

	data, _ := s.chat.Channel(s.cid)
	s.Equal("Sick days", data["name"])
}

// A command is looked up in the channel of its type, on the Router's word: here a channel the
// customer stamped as the customer's own, since the lookup still asks the stamps whose it is.
func (s *ChatChannelSuite) TestACommandIsLookedUpInTheChannelOfItsTypeOnlyOnTheRoutersWord() {
	s.created(map[string]any{conversation.CustomerField: "customer", "support_agent_id": "config-1"})
	s.wrote()
	s.answered(s.routerOpens(s.cid))

	_, err := s.service.CommandForCaller(context.Background(), "customer", "config-1", s.cid, "", "message-1")
	s.Require().ErrorIs(err, conversation.ErrCommandNotFound)
	receipt, err := s.service.CommandForCaller(s.routerOpens(s.cid), "customer", "config-1", s.cid, "", "message-1")

	s.Require().NoError(err)
	s.Equal("message-1", receipt.CommandID)
}

func (s *ChatChannelSuite) routerOpens(cid string) context.Context {
	return conversation.RouterOpensDispatched(context.Background(), cid)
}

// answered opens the conversation on ctx and has the agent answer the person's message with a
// follow-up, as the Router's session does, then lets the conversation go.
func (s *ChatChannelSuite) answered(ctx context.Context) {
	c, _, _, err := s.service.Open(ctx, "customer", "config-1", s.cid)
	s.Require().NoError(err)
	_, err = c.BeginFollowUp("message-1", "I'm sick, I can't come at 6")
	s.Require().NoError(err)
	c.Observe(agent.Responding{TurnID: "turn-1"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-1", Text: "Feel better."})
	c.Observe(agent.Responded{TurnID: "turn-1", Text: "Feel better."})
	s.Require().Eventually(func() bool {
		return len(s.chat.Stored(s.cid)) == 2 && !s.service.Writing()
	}, 5*time.Second, 10*time.Millisecond)
	c.Release()
}

// created makes the channel as the customer's app does, under the callouts type, with the
// custom data given.
func (s *ChatChannelSuite) created(custom map[string]any) {
	creator := "anya"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), "callouts", s.channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &creator, Custom: custom},
	})
	s.Require().NoError(err)
}

// wrote writes the person's message in the channel.
func (s *ChatChannelSuite) wrote() {
	person, text := "anya", "I'm sick, I can't come at 6"
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "callouts", s.channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &person},
	})
	s.Require().NoError(err)
	s.Require().True(slices.ContainsFunc(s.chat.Stored(s.cid), func(m map[string]any) bool { return m["text"] == text }))
}
