//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A channel type a config answers only its mentions in (answer: mentioned): Anya's message
// mentioning the agent is answered, Ben's "feel better!" is not, and stays in the channel for
// the next answer to read.

// The agent picked from the mention list is a mention, though the text names no @ at all.
func (s *AppHooksSuite) TestAMentionByTheListIsAnswered() {
	s.holdsType("callouts", memberWrites)
	config := s.mentionedIn("callouts")
	channel := s.emptyChannel("callouts")

	s.says("callouts", channel, "anya", "can you cover my 6pm?", fmt.Sprintf(`[{"id": %q, "name": %q}]`, config.ID, config.Name))

	reply := s.replyIn("callouts:" + channel)
	s.Require().NotNil(reply, "a mention by the list was not answered")
	s.Equal(config.ID, reply["user_id"])
}

// @ and the agent's id or name, as Stream's clients write it or a person types it, is a
// mention with no mentioned_users at all.
func (s *AppHooksSuite) TestAMentionInTheTextIsAnswered() {
	s.holdsType("callouts", memberWrites)
	config := s.mentionedIn("callouts")
	for _, text := range []string{
		"@" + config.Name + " I'm sick, I can't come at 6",
		"I'm sick, can you cover my 6pm, @" + strings.ToUpper(config.Name) + "?",
		"@" + config.ID,
	} {
		channel := s.emptyChannel("callouts")

		s.says("callouts", channel, "anya", text, "")

		s.NotNil(s.replyIn("callouts:"+channel), "%q was not answered", text)
	}
}

// A message that does not mention the agent gets no reply, no session and no model, whether it
// mentions nobody, mentions Ben, or names the agent without making a mention of it.
func (s *AppHooksSuite) TestAMessageNotMentioningTheAgentIsNotAnswered() {
	s.holdsType("callouts", memberWrites)
	config := s.mentionedIn("callouts")
	// Each message ends with words of its own, so a model handed any of them is found among
	// the models other tests opened.
	marker := " " + s.utils.uuid()
	for _, c := range []struct{ text, users string }{
		{"feel better!", ""},
		{"@Ben feel better!", `[{"id": "ben", "name": "Ben"}]`},
		{config.Name + " can you cover?", ""},
		{"write to anya@" + config.ID, ""},
		{"@" + config.ID + "-2 can you cover?", ""},
	} {
		channel := s.emptyChannel("callouts")

		s.says("callouts", channel, "ben", c.text+marker, c.users)

		time.Sleep(dropped)
		s.Len(s.chat.Stored("callouts:"+channel), 1, "%q was answered", c.text)
		_, running := s.manager.ByConversationWhere("callouts:"+channel, func(string, int64) bool { return true })
		s.False(running, "%q opened a session", c.text)
	}
	s.NotContains(s.notedInputs(), marker, "a model was handed a message that does not mention the agent")
}

// Ben's message, which mentions nobody, is not answered; Anya's mention next is, and the model
// answering her is handed his message as what was said before hers, and hers once.
func (s *AppHooksSuite) TestAMessageNobodyAnsweredIsReadWithTheNextMention() {
	s.forgetNoted()
	s.holdsType("callouts", memberWrites)
	config := s.mentionedIn("callouts")
	channel := s.emptyChannel("callouts")
	s.says("callouts", channel, "ben", "the 6pm shift is short one person", "")
	time.Sleep(dropped)
	asked := "@" + config.Name + " I'm sick, I can't come at 6"

	s.says("callouts", channel, "anya", asked, "")

	s.Require().NotNil(s.replyIn("callouts:"+channel), "the mention was not answered")
	request, _ := s.askedWith(asked)
	input := request.Input
	ben := slices.IndexFunc(input, func(m llm.Message) bool { return strings.Contains(m.Content, "the 6pm shift is short one person") })
	anya := slices.IndexFunc(input, func(m llm.Message) bool { return strings.HasSuffix(m.Content, asked) })
	s.Require().NotEqual(-1, ben, "the message nobody answered did not reach the model: %+v", input)
	s.Less(ben, anya, "the message nobody answered came after the one answered")
	s.Equal(1, s.count(input, asked), "the message answered was handed twice: %+v", input)
}

// A message written after Anya's mention and before its session reads the channel, as when
// the session waits for the channel's turn, is no message nobody answered: Anya's mention is
// handed once, as the message answered, and Ben's later words not at all.
func (s *AppHooksSuite) TestAMessageWrittenAfterTheMentionIsNotReadBeforeIt() {
	s.forgetNoted()
	s.holdsType("callouts", memberWrites)
	config := s.mentionedIn("callouts")
	channel := s.emptyChannel("callouts")
	asked := "@" + config.Name + " I'm sick, I can't come at 6"
	mention := s.writes("callouts", channel, "anya", asked)
	later := "ok noted " + s.utils.uuid()
	s.writes("callouts", channel, "ben", later)

	s.delivers("callouts", channel, mention, "anya", asked, "")

	s.Require().NotNil(s.replyIn("callouts:"+channel), "the mention was not answered")
	request, _ := s.askedWith(asked)
	s.Equal(1, s.count(request.Input, asked), "the message answered was handed twice: %+v", request.Input)
	s.Zero(s.count(request.Input, later), "a message written after the mention was read before it: %+v", request.Input)
}

// A config answering only its mentions in one type answers every message in another it names
// with no answer, as before.
func (s *AppHooksSuite) TestATypeWithNoAnswerAnswersEveryMessageBesideAMentionsOnlyOne() {
	s.holdsType("shifts", memberWrites)
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: "callouts", Answer: store.ChatChannelAnswerMentioned}, {Type: "shifts"}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	channel := s.emptyChannel("shifts")

	s.says("shifts", channel, "ben", "feel better!", "")

	s.NotNil(s.replyIn("shifts:"+channel), "a type answering every message waited for a mention")
}

// mentionedIn is a text config of the app's customer answering only its mentions in the channel
// type named.
func (s *AppHooksSuite) mentionedIn(channelType string) store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: channelType, Answer: store.ChatChannelAnswerMentioned}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config
}

// emptyChannel is a new channel of the type named, made by the customer's app, nothing in it.
func (s *AppHooksSuite) emptyChannel(channelType string) string {
	channel, person := "team-"+s.utils.uuid(), "anya"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), channelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &person}})
	s.Require().NoError(err)
	return channel
}

// says writes text in channelType:channel as the user given and delivers its message.new, with
// mentioned_users as given (raw JSON, none when empty), under the id Stream stored it by.
func (s *AppHooksSuite) says(channelType, channel, user, text, mentioned string) {
	s.delivers(channelType, channel, s.writes(channelType, channel, user, text), user, text, mentioned)
}

// writes stores text in channelType:channel as the user given, delivering no message.new, and
// is the id Stream stored it by.
func (s *AppHooksSuite) writes(channelType, channel, user, text string) string {
	id := "message-" + s.utils.uuid()
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), channelType, channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &user}})
	s.Require().NoError(err)
	return id
}

// delivers the message.new of the message id the user wrote in channelType:channel, with
// mentioned_users as given (raw JSON, none when empty).
func (s *AppHooksSuite) delivers(channelType, channel, id, user, text, mentioned string) {
	listed := ""
	if mentioned != "" {
		listed = `, "mentioned_users": ` + mentioned
	}
	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), fmt.Sprintf(`{
  "type": "message.new", "cid": "%s:%s", "channel_id": %q, "channel_type": %q, "created_at": %q, "channel_custom": {},
  "message": {"id": %q, "text": %q, "user": {"id": %q, "name": %q}%s}
}`, channelType, channel, channel, channelType, time.Now().UTC().Format(time.RFC3339Nano), id, text, user, user, listed)))
}

// notedInputs is every message any "noted" model opened so far was handed, as one string.
func (s *RouterSuite) notedInputs() string {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	var inputs strings.Builder
	for _, model := range s.noted {
		for _, request := range model.requests() {
			for _, message := range request.Input {
				inputs.WriteString(message.Content + "\n")
			}
		}
	}
	return inputs.String()
}

// count is how many messages of input say text.
func (s *RouterSuite) count(input []llm.Message, text string) int {
	return len(slices.DeleteFunc(slices.Clone(input), func(m llm.Message) bool { return !strings.Contains(m.Content, text) }))
}

// A delivery Stream Chat makes of a message not mentioning the agent, in a type it answers only
// its mentions in, is not queued; one mentioning it is queued and answered.
func (s *ChatInboundQueueSuite) TestADeliveryNotMentioningTheAgentIsNotQueued() {
	customer := s.inAppMode()
	named := "callouts-" + strings.ToLower(s.utils.uuid())
	config := store.AgentConfig{CustomerID: customer, Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		ChatChannels: []store.ChatChannel{{Type: named, Answer: store.ChatChannelAnswerMentioned}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.channelID = "team-" + s.utils.uuid()
	s.writesInType(named, s.channelID, nil)

	s.Equal(http.StatusOK, s.chatInbounds.delivers(&s.RouterSuite, s.ofType(named), s.current))
	s.Zero(s.chatInbounds.queued(&s.RouterSuite), "a message not mentioning the agent was queued")
	s.nobodyAnswered(named + ":" + s.channelID)

	s.Equal(http.StatusOK, s.chatInbounds.delivers(&s.RouterSuite, s.mentioning(named, config.Name), s.current))
	s.answering()
	reply := s.replyIn(named + ":" + s.channelID)
	s.Require().NotNil(reply, "a mention delivered by Stream Chat was not answered")
	s.Equal(config.ID, reply["user_id"])
}

// A delivery queued while the config still answered every message there, tried once it answers
// only its mentions, is not answered.
func (s *ChatInboundQueueSuite) TestAQueuedDeliveryNotMentioningTheAgentIsNotAnswered() {
	customer := s.inAppMode()
	named := "callouts-" + strings.ToLower(s.utils.uuid())
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &store.AgentConfig{CustomerID: customer, Name: "rota-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "noted/noted-model", ChatChannels: []store.ChatChannel{{Type: named, Answer: store.ChatChannelAnswerMentioned}}}))
	s.channelID = "team-" + s.utils.uuid()
	s.writesInType(named, s.channelID, nil)
	queued := s.queue(s.ofType(named))

	s.Require().NoError(s.router.AnswerChatInbound(context.Background(), s.row(queued)))

	s.nobodyAnswered(named + ":" + s.channelID)
}

// mentioning is ofType with the text mentioning the user named.
func (s *ChatInboundQueueSuite) mentioning(channelType, name string) string {
	body := strings.Replace(s.ofType(channelType), `"text": "is my order on its way?"`, `"text": "@`+name+` is my order on its way?"`, 1)
	s.Require().Contains(body, "@"+name)
	return body
}

// A type named with no answer is stored and served with none, which answers every message; one
// named with an answer keeps it through create, update, patch and sync.
func (s *ConfigsSuite) TestAChatChannelKeepsTheAnswerItWasNamedWith() {
	unsaid := s.createConfig(map[string]any{"name": "rota", "chat_channels": []map[string]any{{"type": "callouts"}}})
	s.Equal(&[]AgentChatChannel{{Type: "callouts"}}, unsaid.ChatChannels)
	s.storedChatChannels(unsaid.Id, []store.ChatChannel{{Type: "callouts"}})

	mentioned, always := ChatChannelAnswer(store.ChatChannelAnswerMentioned), ChatChannelAnswer(store.ChatChannelAnswerAlways)
	created := s.createConfig(map[string]any{"name": "rota-2", "chat_channels": []map[string]any{{"type": "callouts", "answer": "mentioned"}}})
	s.Equal(&[]AgentChatChannel{{Type: "callouts", Answer: &mentioned}}, created.ChatChannels)
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts", Answer: store.ChatChannelAnswerMentioned}})

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "rota-2", "chat_channels": []map[string]any{{"type": "callouts", "answer": "always"}, {"type": "shifts", "answer": "mentioned"}}}, &saved))
	s.Equal(&[]AgentChatChannel{{Type: "callouts", Answer: &always}, {Type: "shifts", Answer: &mentioned}}, saved.ChatChannels)
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts", Answer: store.ChatChannelAnswerAlways}, {Type: "shifts", Answer: store.ChatChannelAnswerMentioned}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"chat_channels": []map[string]any{{"type": "shifts"}}}, &patched))
	s.Equal(&[]AgentChatChannel{{Type: "shifts"}}, patched.ChatChannels)
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "shifts"}})
}

// answer is always or mentioned, on create, update and patch; nothing else is stored.
func (s *ConfigsSuite) TestAChatChannelAnswerIsAlwaysOrMentioned() {
	created := s.createConfig(map[string]any{"name": "rota", "chat_channels": []map[string]any{{"type": "callouts", "answer": "mentioned"}}})
	for _, answer := range []any{"sometimes", "Mentioned", "", true, []string{"mentioned"}} {
		for _, write := range []struct{ method, path string }{
			{http.MethodPost, "/v1/agents/configs"},
			{http.MethodPut, "/v1/agents/configs/" + created.Id},
			{http.MethodPatch, "/v1/agents/configs/" + created.Id},
		} {
			status, body := s.serverClient.call(write.method, write.path,
				map[string]any{"name": "rota-" + s.utils.uuid(), "chat_channels": []map[string]any{{"type": "shifts", "answer": answer}}})
			s.Equal(http.StatusBadRequest, status, "%s %v: %s", write.method, answer, body)
		}
	}
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts", Answer: store.ChatChannelAnswerMentioned}})
}

// agent.yaml's chat_channels carry their answer to the stored config.
func (s *SyncSuite) TestSyncingAnAgentStoresHowItAnswersInEachChannelType() {
	result := s.sync(map[string]any{"name": "rota", "hash": "v1", "mode": "text",
		"chat_channels": []map[string]string{{"type": "callouts", "answer": "mentioned"}, {"type": "shifts"}}})
	mentioned := ChatChannelAnswer(store.ChatChannelAnswerMentioned)
	s.Equal([]AgentChatChannel{{Type: "callouts", Answer: &mentioned}, {Type: "shifts"}}, value(result.Config.ChatChannels))
	s.Equal(result.Config.ChatChannels, s.configsNamed("rota")[0].ChatChannels)
}
