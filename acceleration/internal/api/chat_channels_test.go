//go:build integration

package api

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A message in a channel of a type the customer's config names, callouts here, is answered by
// the Router's own session there, as the config's Stream user (chat_channels).

// memberWrites are a group channel type's grants as a customer sets one up for the agent:
// members read and write, and only the app's backend makes, changes or joins a channel.
var memberWrites = map[string][]string{"channel_member": {"read-channel", "read-channel-members", "create-message"}}

func (s *AppHooksSuite) TestAMessageInAChannelOfATypeTheConfigNamesIsAnsweredThere() {
	s.holdsType("callouts", memberWrites)
	config := s.answeringIn("callouts")
	channel := s.groupChannel("callouts", nil)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, nil)))

	reply := s.replyIn("callouts:" + channel)
	s.Require().NotNil(reply, "nobody answered in the channel")
	s.Equal(config.ID, reply["user_id"])
	s.Equal("Noted.", reply["text"])
	user, made := s.chat.User(config.ID)
	s.Require().True(made, "the agent's Stream user was not made")
	s.Equal(config.Name, user["name"])
}

// The agent's own reply carries a source, as everything it stores does: answering it would be
// the agent talking to itself.
func (s *AppHooksSuite) TestTheAgentsOwnReplyInAChannelOfTheTypeIsNotAnswered() {
	s.holdsType("callouts", memberWrites)
	config := s.answeringIn("callouts")
	channel := s.groupChannel("callouts", nil)
	own := strings.Replace(s.groupMessage("callouts", channel, nil), `"user": {"id": "anya", "name": "Anya"}`,
		fmt.Sprintf(`"user": {"id": %q, "name": "Rota"}, %q: "agent"`, config.ID, chatlog.SourceField), 1)
	s.Require().Contains(own, chatlog.SourceField)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), own))

	s.nobodyAnswered("callouts:" + channel)
}

// A second customer's config naming the type answers in its own app alone: the type in this
// app is answered by this app's customer or nobody.
func (s *AppHooksSuite) TestAnotherCustomersConfigNamingTheTypeDoesNotAnswerInThisApp() {
	s.holdsType("callouts", memberWrites)
	other := s.numberedApp(streamAppID())
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &store.AgentConfig{
		CustomerID: other.app.ID, Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: "callouts"}},
	}))
	channel := s.groupChannel("callouts", nil)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, nil)))

	s.nobodyAnswered("callouts:" + channel)
}

// The channel's custom data, which whoever made it wrote, names another customer, their config
// and a session's own channel: the config naming the type answers all the same, as its own
// user.
func (s *AppHooksSuite) TestTheChannelsCustomDataNamingAnotherOwnerIsNotRead() {
	s.holdsType("callouts", memberWrites)
	config := s.answeringIn("callouts")
	other := s.numberedApp(streamAppID())
	theirs := store.AgentConfig{CustomerID: other.app.ID, Name: "theirs-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model"}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &theirs))
	custom := map[string]any{
		ConfigField: theirs.ID, conversation.CustomerField: other.app.ID, "support_agent_id": theirs.ID,
		"support_owner_id": "ben", conversation.TriggerField: conversation.SessionCommandTrigger,
	}
	channel := s.groupChannel("callouts", custom)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, custom)))

	reply := s.replyIn("callouts:" + channel)
	s.Require().NotNil(reply, "nobody answered in the channel")
	s.Equal(config.ID, reply["user_id"])
}

// The grants check an agent channel's type is held to holds for the type a config names: one
// that lets a user join any channel of it would let anybody reach the agent there.
func (s *AppHooksSuite) TestAChannelTypeWhoseGrantsLetAUserJoinIsNotAnswered() {
	s.holdsType("callouts", map[string][]string{
		"channel_member": {"read-channel", "create-message"}, "user": {"add-own-channel-membership"},
	})
	s.answeringIn("callouts")
	channel := s.groupChannel("callouts", nil)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, nil)))

	s.nobodyAnswered("callouts:" + channel)
}

func (s *AppHooksSuite) TestAChannelTypeTheAppDoesNotHaveIsNotAnswered() {
	s.answeringIn("callouts")
	channel := s.groupChannel("callouts", nil)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, nil)))

	s.nobodyAnswered("callouts:" + channel)
}

// A config naming one type answers in no other, and the agent channel of the same id is not
// the group channel: its sessions and workers are told nothing.
func (s *AppHooksSuite) TestAConfigNamingOneTypeAnswersInNoOther() {
	s.holdsType("shifts", memberWrites)
	s.answeringIn("callouts")
	channel := s.groupChannel("shifts", nil)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("shifts", channel, nil)))

	s.nobodyAnswered("shifts:" + channel)
}

// The shared app every customer writes in has no customer of its own, so no customer's
// config answers in a type there.
func (s *MessageHooksSuite) TestTheSharedAppAnswersNoChannelTypeAConfigNames() {
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &store.AgentConfig{
		CustomerID: s.customerID(), Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: "callouts"}},
	}))
	s.chat.SetApp(chattest.App{ID: 1, ChannelTypes: map[string]map[string][]string{"agent": memberWrites, "callouts": memberWrites}, CallTypes: []string{"agent"}})
	channel := "team-" + s.utils.uuid()
	s.writesInType("callouts", channel, nil)

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.groupMessage("callouts", channel, nil)))

	time.Sleep(dropped)
	s.Len(s.chat.Stored("callouts:"+channel), 1, "something answered in the channel")
}

// holdsType has the app's Stream say it holds the channel type named with the grants given,
// beside the agent types the router needs.
func (s *AppHooksSuite) holdsType(name string, grants map[string][]string) {
	s.chat.SetAppFor(s.apiKey, chattest.App{
		ID:           s.appID(),
		ChannelTypes: map[string]map[string][]string{"agent": memberWrites, name: grants},
		CallTypes:    []string{"agent"},
	})
}

// answeringIn is a text config of the app's customer naming the channel types given.
func (s *AppHooksSuite) answeringIn(types ...string) store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model"}
	for _, named := range types {
		config.ChatChannels = append(config.ChatChannels, store.ChatChannel{Type: named})
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config
}

// groupChannel is a new channel of the type named, made by the customer's app with the custom
// data given and Anya's message in it.
func (s *AppHooksSuite) groupChannel(channelType string, custom map[string]any) string {
	channel := "team-" + s.utils.uuid()
	s.writesInType(channelType, channel, custom)
	return channel
}

// writesInType writes Anya's message in the channel channelType:channel, made with the custom
// data given when it is not there yet.
func (s *RouterSuite) writesInType(channelType, channel string, custom map[string]any) {
	ctx, person, text := context.Background(), "anya", "I'm sick, I can't come at 6"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, channelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &person, Custom: custom}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(ctx, channelType, channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &person}})
	s.Require().NoError(err)
}

// groupMessage is the message.new for Anya's message in channelType:channel.
func (s *RouterSuite) groupMessage(channelType, channel string, custom map[string]any) string {
	if custom == nil {
		custom = map[string]any{}
	}
	encoded, err := json.Marshal(custom)
	s.Require().NoError(err)
	return fmt.Sprintf(`{
  "type": "message.new",
  "cid": "%s:%s",
  "channel_id": %q,
  "channel_type": %q,
  "created_at": %q,
  "channel_custom": %s,
  "message": {"id": %q, "text": "I'm sick, I can't come at 6", "user": {"id": "anya", "name": "Anya"}}
}`, channelType, channel, channel, channelType, time.Now().UTC().Format(time.RFC3339Nano), encoded, "message-"+s.utils.uuid())
}

// replyIn waits for the agent's finished reply in the channel cid, nil when none comes.
func (s *RouterSuite) replyIn(cid string) map[string]any {
	var reply map[string]any
	s.Eventually(func() bool {
		for _, stored := range s.chat.Stored(cid) {
			custom, _ := stored["custom"].(map[string]any)
			written, _ := custom["support_message"].(map[string]any)
			if written["role"] == "assistant" && written["state"] == "completed" {
				reply = stored
				return true
			}
		}
		return false
	}, settleFor, 20*time.Millisecond)
	return reply
}

// nobodyAnswered checks that the channel cid still holds Anya's message alone a while after it was
// delivered, and that no agent channel of its id was written in.
func (s *RouterSuite) nobodyAnswered(cid string) {
	time.Sleep(dropped)
	s.Len(s.chat.Stored(cid), 1, "something answered in the channel")
	_, channel, _ := strings.Cut(cid, ":")
	s.Empty(s.chat.Stored(chatlog.ChannelType+":"+channel), "something wrote in the agent channel of the same id")
}

// A delivery Stream Chat makes of a message in a channel of a type the app's own customer names
// is queued and answered there by the Router's own session; one in a type nobody names is not
// queued.
func (s *ChatInboundQueueSuite) TestADeliveryInAChannelOfATypeAConfigNamesIsQueuedAndAnsweredThere() {
	customer := s.inAppMode()
	// Types of the test's own: the deployment app's customer is the same in every test.
	named, unnamed := "callouts-"+strings.ToLower(rand.Text()), "shifts-"+strings.ToLower(rand.Text())
	config := store.AgentConfig{CustomerID: customer, Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: named}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.channelID = "team-" + s.utils.uuid()
	s.writesInType(named, s.channelID, nil)

	s.Equal(http.StatusOK, s.chatInbounds.delivers(&s.RouterSuite, s.ofType(named), s.current))
	s.answering()

	reply := s.replyIn(named + ":" + s.channelID)
	s.Require().NotNil(reply, "nobody answered in the channel")
	s.Equal(config.ID, reply["user_id"])

	s.channelID = "team-" + s.utils.uuid()
	s.writesInType(unnamed, s.channelID, nil)
	s.Equal(http.StatusOK, s.chatInbounds.delivers(&s.RouterSuite, s.ofType(unnamed), s.current))
	s.Zero(s.chatInbounds.queued(&s.RouterSuite), "a type nobody names was queued")
}

// A customer whose work in the deployment's app is only to be read any more has no channel
// there answered, as no thread there is (mayWrite).
func (s *ChatInboundQueueSuite) TestADeliveryInAChannelOfATypeForACustomerNoLongerWritingThereIsNotAnswered() {
	customer := s.inAppMode()
	named := "callouts-" + strings.ToLower(rand.Text())
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &store.AgentConfig{CustomerID: customer, Name: "rota-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "noted/noted-model", ChatChannels: []store.ChatChannel{{Type: named}}}))
	s.setApps(customer, func(apps *suiteApps) { apps.readOnly[customer] = true })
	s.channelID = "team-" + s.utils.uuid()
	s.writesInType(named, s.channelID, nil)

	s.Equal(http.StatusOK, s.chatInbounds.delivers(&s.RouterSuite, s.ofType(named), s.current))
	s.answering()

	s.nobodyAnswered(named + ":" + s.channelID)
}

// ofType is a delivery of a message in the suite's channel under the channel type named.
func (s *ChatInboundQueueSuite) ofType(channelType string) string {
	body := strings.Replace(s.message(suiteStreamApp, "", ""), `"channel_type": "agent"`, fmt.Sprintf(`"channel_type": %q`, channelType), 1)
	s.Require().Contains(body, channelType)
	return body
}
