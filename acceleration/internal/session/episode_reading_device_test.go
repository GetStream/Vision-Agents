//go:build integration

package session

import (
	"maps"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// The person's own device reading their app chat's cards (AI-2709): a session the app opens
// with the person's own token, as api.createSession fills Caller and CallerKind in from the
// credential (CallerFrom, KindFrom).

// opener is who opens a session: the caller and its kind, as the credential makes them.
type opener struct {
	caller routing.Caller
	kind   auth.Kind
}

// signedIn is a user's own device, holding a token signed for them (server.go withCustomer:
// Verified for a kind that is).
func signedIn(user string) opener {
	return opener{routing.Caller{UserID: user, Verified: true}, auth.KindAuthenticated}
}

// anyasCard is Anya's app chat episode, summarized, an hour before the test's now, and its
// channel.
func (s *EpisodeReadingSuite) anyasCard() (string, omnichannel.Opened) {
	s.at(-3600)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "my appointment is Friday at 3", nil)
	card := s.chatEpisode("anya_123", chat, 0)
	s.summarize(card, "Her appointment is Friday at 3.")
	s.at(0)
	return chat, card
}

// deviceSession opens a text session as by, on conversation (empty for a new one), and has it
// answer text, as a device's command does.
func (s *EpisodeReadingSuite) deviceSession(by opener, conversation string, cards bool, text string) (*Session, llm.ResponseParams) {
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards, ConversationID: conversation,
		Text: true, PersistConversation: true, Caller: by.caller, CallerKind: by.kind,
		LLMTarget: "en-low-latency", Instructions: "be brief",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	s.Require().False(created.Spec().Dispatched)
	_, _, err = created.RespondCommand(s.ctx, "command-"+uuid.NewString(), text, "", nil, options.LLM{})
	s.Require().NoError(err)
	return created, s.replyTo(text)
}

// deviceCall opens a voice session as by on a call the person's app joined, with no phone on
// it unless on names who else is, and has it answer text.
func (s *EpisodeReadingSuite) deviceCall(by opener, text string, on ...string) (*Session, llm.ResponseParams) {
	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, append([]string{by.caller.UserID}, on...)...)
	return s.deviceCallOn(by, call, text)
}

// deviceCallOn is deviceCall on a call the test put.
func (s *EpisodeReadingSuite) deviceCallOn(by opener, call, text string) (*Session, llm.ResponseParams) {
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, CallID: call,
		Caller: by.caller, CallerKind: by.kind,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	_, err = created.Respond(s.ctx, text, nil, options.LLM{})
	s.Require().NoError(err)
	return created, s.replyTo(text)
}

// Defect: a session Anya's own device opens with her token reads none of her app chat's
// cards. Her text session and her call from the app start with the card's summary, and each
// names her as the one person it answers (CardsFor).
func (s *EpisodeReadingSuite) TestASessionThePersonsOwnDeviceOpensReadsTheirChatCard() {
	s.anyasCard()

	text, params := s.deviceSession(signedIn("anya_123"), "", true, "when is my appointment?")

	handed := s.handed(params)
	s.Require().Len(handed, 1, "her text session")
	s.Equal(store.EpisodeChat, handed[0]["source"])
	s.Equal("Her appointment is Friday at 3.", handed[0]["summary"])
	s.Equal("anya_123", text.CardsFor())

	call, params := s.deviceCall(signedIn("anya_123"), "when is it again? "+uuid.NewString())

	handed = s.handed(params)
	s.Require().Len(handed, 1, "her call from the app")
	s.Equal("Her appointment is Friday at 3.", handed[0]["summary"])
	s.Equal("anya_123", call.CardsFor())
}

// Defect: a call with a phone on it is taken for the device's that placed it. Anya's app
// ringing a number reads that number's cards, as any call with a phone on it does, not hers.
func (s *EpisodeReadingSuite) TestACallWithAPhoneOnItReadsTheNumbersCardsWhoeverOpenedIt() {
	s.anyasCard()
	s.at(-30)
	s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	s.at(0)

	call, params := s.deviceCall(signedIn("anya_123"), "hello", "sip-+15550100100")

	handed := s.handed(params)
	s.Require().Len(handed, 1)
	s.Equal("sms", handed[0]["source"])
	s.Empty(call.CardsFor())
}

// Defect: a session reads Anya's cards for a caller whose name nobody vouched for as hers.
// Bob's token, a backend with no user, a backend or a scheduled task naming her
// (ServerSide), an anonymous caller or a guest going by her name, and a token naming nobody
// read none, by text or by call; nor does a session of another agent config of hers.
func (s *EpisodeReadingSuite) TestOnlyThePersonsOwnVerifiedTokenReadsTheirCards() {
	s.anyasCard()
	for name, by := range map[string]opener{
		"Bob's token":                {routing.Caller{UserID: "bob_456", Verified: true}, auth.KindAuthenticated},
		"a backend with no user":     {routing.Caller{ServerSide: true}, auth.KindServer},
		"a backend naming her":       {routing.Caller{UserID: "anya_123", ServerSide: true, Verified: true}, auth.KindServer},
		"a scheduled task of hers":   {routing.Caller{UserID: "anya_123", ServerSide: true}, auth.KindAuthenticated},
		"an anonymous caller as her": {routing.Caller{UserID: "anya_123"}, auth.KindAnonymous},
		"a guest as her":             {routing.Caller{UserID: "anya_123", Verified: true}, auth.KindGuest},
		"a token naming nobody":      {routing.Caller{Verified: true}, auth.KindAuthenticated},
	} {
		text, params := s.deviceSession(by, "", true, "when is my appointment? "+uuid.NewString())
		s.Nil(s.handed(params), name+", by text")
		s.Empty(text.CardsFor(), name+", by text")
		if by.caller.UserID == "" {
			continue
		}
		call, params := s.deviceCall(by, "when is it? "+uuid.NewString())
		s.Nil(s.handed(params), name+", by call")
		s.Empty(call.CardsFor(), name+", by call")
	}

	s.configID = "config-" + uuid.NewString()
	_, params := s.deviceSession(signedIn("anya_123"), "", true, "and under another agent? "+uuid.NewString())
	s.Nil(s.handed(params), "another agent config of hers")
}

// Defect: a Stream user is taken at their word in an app Stream says checks no tokens, where a
// client may sign in, or write in the app chat, as anybody. Neither Anya's device session nor
// the worker's session answering "her" app chat reads her card there, nor where Stream cannot
// be asked about the app; once the app checks tokens again, both do.
func (s *EpisodeReadingSuite) TestNoCardOfAStreamUserIsReadInAnAppThatChecksNoTokens() {
	s.anyasCard()
	for name, app := range map[string]chattest.App{
		"auth checks off":         {ID: 1, DisableAuthChecks: true},
		"Stream refuses to tell":  {ID: 1, Refuses: true},
		"checks tokens (control)": {ID: 1},
	} {
		// Clients of their own, as Readiness keeps what Stream said for a minute.
		s.chat.SetApp(app)
		s.apps = streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
			APIKey: streamKey, Secret: "deploy-secret", BaseURL: s.chat.URL, App: suiteDeploymentApp,
		}), streamapp.ClientsOptions{})
		var err error
		s.cards, err = omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.apps})
		s.Require().NoError(err)
		s.manager = s.managerOver()
		reads := name == "checks tokens (control)"

		_, params := s.deviceSession(signedIn("anya_123"), "", true, "when is my appointment? "+uuid.NewString())
		s.Equal(reads, s.handed(params) != nil, name+", her device")

		chat := s.chatChannel("anya_123")
		said := "and the time? " + uuid.NewString()
		s.write(chat, "anya_123", said, nil)
		s.Equal(reads, s.handed(s.deploymentChat(chat, "anya_123", true, said)) != nil, name+", the app chat")
	}
}

// Defect: a device session that carries on a conversation is handed that conversation's own
// card, which it reads word for word. Anya's resumed conversation, and her call on it, read her
// other chat's card and not the one of the conversation they carry on.
func (s *EpisodeReadingSuite) TestAResumedDeviceSessionLeavesItsOwnConversationsCardOut() {
	s.anyasCard()
	first, _ := s.deviceSession(signedIn("anya_123"), "", false, "hello")
	kept := first.Spec().ConversationID
	s.Require().Eventually(func() bool {
		for _, message := range s.chat.Stored(strings.TrimPrefix(kept, "agent:")) {
			if user, _ := message["user"].(map[string]any); user["id"] == "anya_123" && message["text"] == "hello" {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "her turn is not in the conversation's channel")
	s.summarize(s.chatEpisode("anya_123", kept, 0), "Said hello in this conversation.")
	_, err := s.manager.Close(first.ID(), OwnerOf(first.Spec()))
	s.Require().NoError(err)

	_, params := s.deviceSession(signedIn("anya_123"), kept, true, "when is my appointment?")

	handed := s.handed(params)
	s.Require().Len(handed, 1, "her other chat's card alone")
	s.Equal("Her appointment is Friday at 3.", handed[0]["summary"])

	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "anya_123")
	onCall, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, CallID: call, ConversationID: kept, AgentID: first.Spec().AgentID,
		Caller: signedIn("anya_123").caller, CallerKind: auth.KindAuthenticated,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(onCall.ID(), OwnerOf(onCall.Spec())) })
	said := "and on a call? " + uuid.NewString()
	_, err = onCall.Respond(s.ctx, said, nil, options.LLM{})
	s.Require().NoError(err)
	handed = s.handed(s.replyTo(said))
	s.Require().Len(handed, 1, "her call on the conversation: her other chat's card alone")
	s.Equal("Her appointment is Friday at 3.", handed[0]["summary"])
}

// anyasConversation is a conversation Anya's device opened and closed, then opened to its
// members when shared, with the members others added, and its agent's user id.
func (s *EpisodeReadingSuite) anyasConversation(shared bool, others ...string) (string, string) {
	first, _ := s.deviceSession(signedIn("anya_123"), "", false, "hello "+uuid.NewString())
	kept, agent := first.Spec().ConversationID, first.Spec().AgentID
	_, err := s.manager.Close(first.ID(), OwnerOf(first.Spec()))
	s.Require().NoError(err)
	id := strings.TrimPrefix(kept, "agent:")
	s.Require().ElementsMatch([]string{agent, "anya_123"}, s.chat.Members(id))
	if shared {
		data, _ := s.chat.Channel(id)
		custom := maps.Clone(data["custom"].(map[string]any))
		custom["support_access"] = "members"
		_, err := s.chat.Client.Chat().UpdateChannelPartial(s.ctx, "agent", id, &getstream.UpdateChannelPartialRequest{Set: map[string]any{"custom": custom}})
		s.Require().NoError(err)
	}
	for _, other := range others {
		_, err := s.chat.Client.Chat().UpdateChannel(s.ctx, "agent", id, &getstream.UpdateChannelRequest{AddMembers: []getstream.ChannelMemberRequest{{UserID: other}}})
		s.Require().NoError(err)
	}
	return kept, agent
}

// Defect: Anya's device session on a conversation somebody else reads the answers in is handed
// her cards, which its answers then carry to them. A conversation open to its members that
// Carl is a member of, one Carl was added to, and one open to its members that only she and
// the agent are in yet read none, by text or by a call on it; her own conversation does.
func (s *EpisodeReadingSuite) TestADeviceSessionOnAConversationSomebodyElseReadsReadsNoCards() {
	s.anyasCard()
	for name, shape := range map[string]struct {
		shared bool
		others []string
		reads  bool
	}{
		"shared, Carl a member":   {shared: true, others: []string{"carl_789"}},
		"Carl a member":           {others: []string{"carl_789"}},
		"shared, no other member": {shared: true},
		"hers alone (control)":    {reads: true},
	} {
		kept, agent := s.anyasConversation(shape.shared, shape.others...)

		text, params := s.deviceSession(signedIn("anya_123"), kept, true, "when is my appointment? "+uuid.NewString())
		s.Equal(shape.reads, s.handed(params) != nil, name+", by text")
		s.Equal(shape.reads, text.CardsFor() == "anya_123", name+", by text")
		_, err := s.manager.Close(text.ID(), OwnerOf(text.Spec()))
		s.Require().NoError(err)

		call := "call-" + uuid.NewString()
		s.chat.PutCall("agent", call, "anya_123")
		onCall, err := s.manager.Create(s.ctx, Spec{
			CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, CallID: call, ConversationID: kept, AgentID: agent,
			Caller: signedIn("anya_123").caller, CallerKind: auth.KindAuthenticated,
			LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency", Instructions: "be brief",
		})
		s.Require().NoError(err)
		said := "and on a call? " + uuid.NewString()
		_, err = onCall.Respond(s.ctx, said, nil, options.LLM{})
		s.Require().NoError(err)
		s.Equal(shape.reads, s.handed(s.replyTo(said)) != nil, name+", by a call on it")
		_, err = s.manager.Close(onCall.ID(), OwnerOf(onCall.Spec()))
		s.Require().NoError(err)
	}
}

// Defect: Anya's call from her app is handed her cards while somebody else is on it to hear
// the answers. Bob in the call's session, or a member of the call, reads none; the agent on
// it, as the user it joins as, is no somebody else.
func (s *EpisodeReadingSuite) TestADeviceCallSomebodyElseIsOnReadsNoCards() {
	s.anyasCard()

	_, params := s.deviceCall(signedIn("anya_123"), "when is it again? "+uuid.NewString(), "bob_456")
	s.Nil(s.handed(params), "Bob in the call")

	call := "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "anya_123")
	s.chat.PutCallMembers("agent", call, "anya_123", "bob_456")
	_, params = s.deviceCallOn(signedIn("anya_123"), call, "when is it again? "+uuid.NewString())
	s.Nil(s.handed(params), "Bob a member of the call")

	call = "call-" + uuid.NewString()
	s.chat.PutCall("agent", call, "anya_123", defaultUserID)
	s.chat.PutCallMembers("agent", call, "anya_123", defaultUserID)
	created, params := s.deviceCallOn(signedIn("anya_123"), call, "when is it again? "+uuid.NewString())
	s.Len(s.handed(params), 1, "the agent and her")
	s.Equal("anya_123", created.CardsFor())
}

// Defect: a device session whose app has no Stream client, as Manager.stream binds one for a
// customer with no identity (nowhere), asks it whether the app checks tokens and panics. It
// reads no cards and names nobody.
func (s *EpisodeReadingSuite) TestADeviceSessionWithNoStreamClientReadsNoCards() {
	s.anyasCard()
	spec := Spec{
		ID: "session-" + uuid.NewString(), CustomerID: s.customerID, ConfigID: s.configID, EpisodeCards: true,
		Text: true, PersistConversation: true, Caller: signedIn("anya_123").caller, CallerKind: auth.KindAuthenticated,
	}

	cards, person := s.manager.cards.read(s.ctx, spec, streamapp.Bound{})

	s.Nil(cards)
	s.Empty(person)
}

// Control, from base 3b7a53f97 (probe: this test on the base's code, base-probe.log): with
// episode_cards off Anya's device session is handed what it was, her message and the system's
// note of the date, and asks Stream nothing of her card, her omni-channel or the app.
func (s *EpisodeReadingSuite) TestWithTheCardsOffADeviceSessionIsAsBefore() {
	chat, card := s.anyasCard()
	asked := len(s.requests())

	created, off := s.deviceSession(signedIn("anya_123"), "", false, "when is my appointment?")

	s.Nil(s.handed(off))
	s.Empty(created.CardsFor())
	s.Require().Len(off.Input, 2)
	s.Equal(llm.Message{Role: llm.User, Content: "when is my appointment?"}, off.Input[0])
	s.Equal(llm.System, off.Input[1].Role)
	s.True(strings.HasPrefix(off.Input[1].Content, "[Note from the system running this call]\nToday is "), off.Input[1].Content)
	s.Empty(namingAny(s.requests()[asked:], chat, card.Contact.ConversationID, card.Episode.CardMessageID))
	for _, request := range s.requests()[asked:] {
		s.NotEqual("/api/v2/app", request.Path, "the app's settings are read")
	}
}
