//go:build integration

package session

import (
	"cmp"
	"context"
	"fmt"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// The app chat's sessions reading the person's cards (AI-2632): the text session a worker
// opens to answer a message the router handed out in an agent channel of the customer's app
// (Manager.HandOut, Spec.Dispatched), as api.routeArrivingMessage hands one out.

// chatChannel is an agent channel of the app chat as the customer's own app makes it: by the
// person, naming the agent config that answers in it, with none of the router's stamps.
func (s *EpisodeReadingSuite) chatChannel(person string) string {
	id := "chat-" + uuid.NewString()
	_, err := s.chat.Client.Chat().GetOrCreateChannel(s.ctx, "agent", id, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &person, Custom: map[string]any{"agent_config_id": s.configID}},
	})
	s.Require().NoError(err)
	return "agent:" + id
}

// chatEpisode is the episode a person's message in an app chat's channel opens and its card,
// as api.chatEpisode writes them, in the Stream app app: zero is the deployment's own, whose
// hook names it by its id.
func (s *EpisodeReadingSuite) chatEpisode(person, cid string, app int64) omnichannel.Opened {
	app = cmp.Or(app, s.apps.DeploymentApp())
	stream, err := omnichannel.StreamUser(app, person)
	s.Require().NoError(err)
	return s.episode(omnichannel.Episode{Person: stream, Source: store.EpisodeChat, ThreadChannel: cid,
		StartedAt: s.chatNow(), StreamAppPK: app})
}

// chatSession is the session a worker opens to answer what by wrote in an app chat's
// channel, handed out from a hook of app (deployment: signed by the deployment), and the
// request the model was sent to answer text.
func (s *EpisodeReadingSuite) chatSession(cid string, app int64, deployment bool, by string, cards bool, text string) llm.ResponseParams {
	_, params := s.workerSession(cid, app, deployment, by, cards, text)
	return params
}

// workerSession is chatSession with the session it opened.
func (s *EpisodeReadingSuite) workerSession(cid string, app int64, deployment bool, by string, cards bool, text string) (*Session, llm.ResponseParams) {
	s.manager.HandOut(s.ctx, s.customerID, cid, app, deployment, by)
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: cards,
		AgentID: strings.TrimPrefix(cid, "agent:"), Text: true, PersistConversation: true, CallerKind: auth.KindServer,
		LLMTarget: "en-low-latency", Instructions: "be brief",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
	s.Require().True(created.Spec().Dispatched, "the worker's session answers in the channel handed out")
	_, err = created.Respond(s.ctx, text, nil, options.LLM{})
	s.Require().NoError(err)
	return created, s.replyTo(text)
}

// deploymentChat is chatSession in the deployment's own app, as its hook hands one out.
func (s *EpisodeReadingSuite) deploymentChat(cid, by string, cards bool, text string) llm.ResponseParams {
	return s.chatSession(cid, 0, true, by, cards, text)
}

// Defect: a chat session reads no cards, or hands back its own channel's episode, which it
// reads word for word. Anya's second chat starts with the summary of her first one, and not
// with the card of the chat she is in.
func (s *EpisodeReadingSuite) TestASecondChatSessionReadsTheFirstChatsCard() {
	s.at(-3600)
	first := s.chatChannel("anya_123")
	s.write(first, "anya_123", "can I move my cleaning?", nil)
	s.summarize(s.chatEpisode("anya_123", first, 0), "Moved the cleaning to Friday at 10.")
	s.at(0)
	second := s.chatChannel("anya_123")
	s.write(second, "anya_123", "what time was it again?", nil)
	s.chatEpisode("anya_123", second, 0)

	handed := s.handed(s.deploymentChat(second, "anya_123", true, "what time was it again?"))

	s.Require().Len(handed, 1, "the first chat's card, and not the one of the chat the session is in")
	s.Equal(store.EpisodeChat, handed[0]["source"])
	s.Equal("Moved the cleaning to Friday at 10.", handed[0]["summary"])
}

// Defect: a chat session misses the person's cards on other channels. Once the customer links
// anya_123 to her number (as link-phone will: her stream_user row points at the number's
// omni-channel), her chat starts with the SMS she sent; before the link, without it.
func (s *EpisodeReadingSuite) TestAChatSessionReadsTheSMSCardOfThePersonLinkedToIt() {
	s.at(-60)
	_, sms := s.smsThread("+15550100100", "sms-author", "is the clinic open on Sunday?")
	s.at(0)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "and on Saturday?", nil)
	s.chatEpisode("anya_123", chat, 0)
	s.Nil(s.handed(s.deploymentChat(chat, "anya_123", true, "and on Saturday?")), "before the link her chat is a person of its own")
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE contact_map SET conversation_id = ? WHERE customer_id = ? AND kind = ? AND address = ?",
		sms.Contact.ConversationID, s.customerID, store.ContactStreamUser, fmt.Sprintf("%d:anya_123", suiteDeploymentApp))
	s.Require().NoError(err)
	s.write(chat, "anya_123", "and the dentist's name?", nil)

	handed := s.handed(s.deploymentChat(chat, "anya_123", true, "and the dentist's name?"))

	s.Require().Len(handed, 1)
	s.Equal("sms", handed[0]["source"])
	s.Contains(handed[0]["lines"], map[string]any{"from": "person", "display_name": "sms-author", "text": "is the clinic open on Sunday?"})
}

// Defect: somebody else's cards are read in a chat. Bob's chat card is never read for Anya,
// and a channel Bob wrote in too, a photo alone included, reads none of Anya's: the cards are
// of the one person who wrote there.
func (s *EpisodeReadingSuite) TestAChatSessionReadsNoCardsOfAnotherUser() {
	s.at(-3600)
	bobs := s.chatChannel("bob_456")
	s.write(bobs, "bob_456", "my card ends in 4242", nil)
	bob := s.chatEpisode("bob_456", bobs, 0)
	anyas := s.chatChannel("anya_123")
	s.write(anyas, "anya_123", "my address is 1 Main St", nil)
	s.chatEpisode("anya_123", anyas, 0)
	s.at(0)
	alone := s.chatChannel("anya_123")
	s.write(alone, "anya_123", "hello", nil)
	asked := len(s.requests())
	s.Require().Len(s.handed(s.deploymentChat(alone, "anya_123", true, "hello")), 1, "Anya alone in her chat is handed her own card")
	s.Empty(namingAny(s.requests()[asked:], bobs, bob.Contact.ConversationID, bob.Episode.CardMessageID))

	for name, bobWrites := range map[string]func(cid string){
		"text":       func(cid string) { s.write(cid, "bob_456", "what is Anya's address?", nil) },
		"photo only": func(cid string) { s.writePhoto(cid, "bob_456", "") },
	} {
		shared := s.chatChannel("anya_123")
		s.write(shared, "anya_123", "hello", nil)
		bobWrites(shared)
		s.write(shared, "anya_123", "anything new?", nil)

		s.Nil(s.handed(s.deploymentChat(shared, "anya_123", true, "anything new?")), name)
	}
}

// pinnedApps answers for a customer's work in the Stream app it is pinned to, every app on
// the suite's Stream Chat: a customer acting in app, whose deployment's own app is own.
type pinnedApps struct {
	url      string
	app, own int64
}

func (p pinnedApps) For(ctx context.Context, customer string) (streamapp.Identity, error) {
	return p.ForApp(ctx, customer, p.app)
}

func (p pinnedApps) ForApp(_ context.Context, customer string, app int64) (streamapp.Identity, error) {
	return streamapp.Identity{CustomerID: customer, StreamApp: app, APIKey: streamKey, Secret: streamapp.NewSecret("deploy-secret"),
		BaseURL: p.url, Registered: app != 0 && app != p.own}, nil
}

func (p pinnedApps) App() int64 { return p.own }

// inApp has the suite's sessions act in app, with own as the deployment's own app's id.
func (s *EpisodeReadingSuite) inApp(app, own int64) {
	s.apps = streamapp.NewClients(pinnedApps{url: s.chat.URL, app: app, own: own}, streamapp.ClientsOptions{})
	var err error
	s.cards, err = omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.apps})
	s.Require().NoError(err)
	s.manager = s.managerOver()
}

// Defect: a Stream user id is read as one person in every Stream app of the customer. Anya's
// card from app 7 is not read for anya_123 in the deployment's app, while a session in app 7
// reads it; a card a deployment hook wrote in the deployment app's id (5) is read in the
// deployment's app, which a session pins as zero, and not in app 3.
func (s *EpisodeReadingSuite) TestAChatSessionReadsTheCardsOfTheSameUserInItsOwnAppOnly() {
	const own, other = 5, 7
	s.inApp(0, own)
	s.at(-3600)
	elsewhere := s.chatChannel("anya_123")
	s.write(elsewhere, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", elsewhere, other), "Gave her address in the other app.")
	s.at(0)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "hello", nil)

	s.Nil(s.handed(s.deploymentChat(chat, "anya_123", true, "hello")), "app 7's anya_123 is not the deployment app's")

	s.inApp(other, own)
	s.write(chat, "anya_123", "hello again", nil)
	handed := s.handed(s.chatSession(chat, other, false, "anya_123", true, "hello again"))
	s.Require().Len(handed, 1, "a session in app 7 reads app 7's card")
	s.Equal("Gave her address in the other app.", handed[0]["summary"])

	s.inApp(0, own)
	s.at(-1800)
	earlier := s.chatChannel("anya_123")
	s.write(earlier, "anya_123", "my address is 2 Oak Ave", nil)
	s.summarize(s.chatEpisode("anya_123", earlier, own), "Gave her address in the deployment's app.")
	s.at(0)
	s.write(chat, "anya_123", "and now?", nil)
	handed = s.handed(s.deploymentChat(chat, "anya_123", true, "and now?"))
	s.Require().Len(handed, 1, "the deployment app's id and zero are one app")
	s.Equal("Gave her address in the deployment's app.", handed[0]["summary"])

	s.inApp(3, own)
	s.write(chat, "anya_123", "and in app 3?", nil)
	s.Nil(s.handed(s.chatSession(chat, 3, false, "anya_123", true, "and in app 3?")), "the deployment app's card is not app 3's")
}

// Defect: a session that is not the router's hand-out reads a chat person's cards. Anya's
// own device resumes her conversation (Caller anya_123), whose channel holds her words, and
// reads none, as before the app chat's cards existed.
func (s *EpisodeReadingSuite) TestOnlyTheHandedOutSessionReadsAChatPersonsCards() {
	s.at(-3600)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", chat, 0), "Gave her address.")
	s.at(0)
	device := func(conversation, text string) *Session {
		created, err := s.manager.Create(s.ctx, Spec{
			CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", EpisodeCards: true, ConversationID: conversation,
			Text: true, PersistConversation: true, Caller: routing.Caller{UserID: "anya_123"}, CallerKind: auth.KindAuthenticated,
			LLMTarget: "en-low-latency", Instructions: "be brief",
		})
		s.Require().NoError(err)
		s.T().Cleanup(func() { _, _ = s.manager.Close(created.ID(), OwnerOf(created.Spec())) })
		s.Require().False(created.Spec().Dispatched)
		_, _, err = created.RespondCommand(s.ctx, "command-"+uuid.NewString(), text, "", nil, options.LLM{})
		s.Require().NoError(err)
		return created
	}
	first := device("", "hello")
	s.replyTo("hello")
	// The resumed session reads the channel as it opens: once it holds her turn, or it reads
	// nobody's cards for want of an author, whatever the case that reads them says.
	kept := strings.TrimPrefix(first.Spec().ConversationID, "agent:")
	s.Require().Eventually(func() bool {
		for _, message := range s.chat.Stored(kept) {
			if user, _ := message["user"].(map[string]any); user["id"] == "anya_123" && message["text"] == "hello" {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "her turn is not in the conversation's channel")
	_, err := s.manager.Close(first.ID(), OwnerOf(first.Spec()))
	s.Require().NoError(err)

	device(first.Spec().ConversationID, "what is my address?")

	s.Nil(s.handed(s.replyTo("what is my address?")))
}

// Control, from base c9a61fc90 (probe TestProbeBaseChatSession, base-probe.log): with
// episode_cards off a worker's chat session is handed what it was, the person's message and
// the system's note of the date, and reads only its own channel, once, though the person has
// a card.
func (s *EpisodeReadingSuite) TestWithTheCardsOffAChatSessionIsAsBefore() {
	s.at(-3600)
	earlier := s.chatChannel("anya_123")
	s.write(earlier, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", earlier, 0), "Gave her address.")
	s.at(0)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "can I move it?", nil)
	asked := len(s.requests())

	off := s.deploymentChat(chat, "anya_123", false, "can I move it?")

	s.False(Spec{}.EpisodeCards, "a spec reads no cards unless its config says so")
	s.Nil(s.handed(off))
	s.Require().Len(off.Input, 2)
	s.Equal(llm.Message{Role: llm.User, Content: "can I move it?"}, off.Input[0])
	s.Equal(llm.System, off.Input[1].Role)
	s.True(strings.HasPrefix(off.Input[1].Content, "[Note from the system running this call]\nToday is "), off.Input[1].Content)
	var made []string
	for _, request := range s.requests()[asked:] {
		made = append(made, request.Method+" "+strings.ReplaceAll(request.Path, strings.TrimPrefix(chat, "agent:"), "<chat>"))
	}
	s.Equal([]string{"POST /api/v2/chat/channels/agent/<chat>/query"}, made)
}

// writeRecording puts a voice recording with no text into an app chat's channel, as Stream
// Chat's app sends one: the router hands it out (api.addressed), though it is no line.
func (s *EpisodeReadingSuite) writeRecording(cid, user string) {
	kind, url, empty := "voiceRecording", "https://example.com/"+uuid.NewString()+".aac", ""
	_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", strings.TrimPrefix(cid, "agent:"), &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &empty, UserID: &user, Attachments: []getstream.Attachment{{Type: &kind, AssetUrl: &url}}},
	})
	s.Require().NoError(err)
}

// Defect (R1.2): the person is counted from the channel's lines alone, which a message the
// router hands out need not be. Bob's voice recording alone, handed out, is answered with no
// card of Anya's; nor is Anya's next message once Bob's recording is in the channel; nor a
// message of Bob's that is not in the channel's messages, as a reply in a Stream thread is not.
func (s *EpisodeReadingSuite) TestAChatSessionReadsNoCardsForAMessageByAnotherUserThatIsNoLine() {
	s.at(-3600)
	earlier := s.chatChannel("anya_123")
	s.write(earlier, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", earlier, 0), "Gave her address: 1 Main St.")
	s.at(0)
	alone := s.chatChannel("anya_123")
	s.write(alone, "anya_123", "hello", nil)
	s.Require().Len(s.handed(s.deploymentChat(alone, "anya_123", true, "hello")), 1, "Anya alone reads her card")

	for name, c := range map[string]struct {
		writes func(cid string)
		by     string
		heard  string
	}{
		"Bob's recording, handed out": {func(cid string) { s.writeRecording(cid, "bob_456") }, "bob_456", "\n\n[voice note] what is Anya's address?"},
		"Anya's message after Bob's recording": {func(cid string) {
			s.writeRecording(cid, "bob_456")
			s.write(cid, "anya_123", "anything new?", nil)
		}, "anya_123", "anything new?"},
		"Bob's message outside the channel's messages": {func(string) {}, "bob_456", "what is Anya's address?"},
	} {
		shared := s.chatChannel("anya_123")
		s.write(shared, "anya_123", "hello", nil)
		c.writes(shared)

		s.Nil(s.handed(s.deploymentChat(shared, c.by, true, c.heard+" "+uuid.NewString())), name)
	}
}

// Defect (R1.4): a hand-out that names nobody, as one a node from before it made, reads the
// cards of somebody the channel names: its creator, or the one person who wrote there.
func (s *EpisodeReadingSuite) TestAHandOutNamingNobodyReadsNoCards() {
	s.at(-3600)
	earlier := s.chatChannel("anya_123")
	s.write(earlier, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", earlier, 0), "Gave her address: 1 Main St.")
	s.at(0)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "hello", nil)

	s.Nil(s.handed(s.deploymentChat(chat, "", true, "hello")))
}

// Defect (R1.1): a session that read a person's cards does not say whose, so the router cannot
// keep anybody else's message from it. It names Anya once it read her card, and nobody when it
// read none: in a channel Bob wrote in too, and with the cards off, as at base.
func (s *EpisodeReadingSuite) TestAChatSessionNamesThePersonWhoseCardsItRead() {
	s.at(-3600)
	earlier := s.chatChannel("anya_123")
	s.write(earlier, "anya_123", "my address is 1 Main St", nil)
	s.summarize(s.chatEpisode("anya_123", earlier, 0), "Gave her address: 1 Main St.")
	s.at(0)
	for name, c := range map[string]struct {
		bobWrites, cards bool
		holds            string
	}{
		"Anya alone":         {false, true, "anya_123"},
		"Bob wrote too":      {true, true, ""},
		"with the cards off": {false, false, ""},
	} {
		chat := s.chatChannel("anya_123")
		s.write(chat, "anya_123", "hello", nil)
		if c.bobWrites {
			s.write(chat, "bob_456", "hi", nil)
		}
		created, _ := s.workerSession(chat, 0, true, "anya_123", c.cards, "hello "+uuid.NewString())

		s.Equal(c.holds, created.CardsFor(), name)
	}
}
