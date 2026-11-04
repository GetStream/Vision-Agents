//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// UserPhonesSuite is PUT /v1/agents/users/{user_id}: the app's backend links a Stream user of
// the app chat to a phone number, so the person's chat card and their SMS share one
// omni-channel. The SMS comes through the customer's own Telnyx line (telnyxLine), answered by
// an agent with episode_cards on whose model recites what it was handed, so its reply holds the
// cards the session read. A database of its own, as the line's inbound worker claims every
// row in it.
type UserPhonesSuite struct {
	telnyxLine

	// user is the Stream user, person their number, both unique to the test.
	user   string
	person string
}

func TestUserPhonesSuite(t *testing.T) {
	runSuite(t, new(UserPhonesSuite))
}

func (s *UserPhonesSuite) SetupSuite() {
	s.database = "user_phones"
	// The message hook names the deployment's own app by its id once it is known, and opens
	// no chat episode for a message of an app it cannot name.
	s.deploymentKnown = true
	s.telnyxLine.SetupSuite()
}

func (s *UserPhonesSuite) SetupTest() {
	s.telnyxLine.SetupTest()
	s.config.EpisodeCards, s.config.LLM = true, "recites/recites-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	s.user, s.person = "user-"+s.utils.uuid(), s.utils.number()
}

// chatSummary is the summary the test's user's chat card holds.
const chatSummary = "Anya asked to move her cleaning to Tuesday at 10."

// Catches an SMS from a linked number that gets an omni-channel of its own: it takes the
// chat's, older, and its session reads the chat's card.
func (s *UserPhonesSuite) TestAnSMSFromALinkedNumberReadsTheUsersChatCard() {
	chat := s.chatCard()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.Contains(s.took(1)[0].text, chatSummary)
	s.Equal(chat, s.omniChannel(s.person))
}

// Catches a link that leaves the person on two omni-channels or joins onto the newer one: the
// number's row is made after the link, by its first SMS, and the user's chat row made after it
// takes its omni-channel, the older.
func (s *UserPhonesSuite) TestAChatRowMadeAfterTheLinkJoinsTheOlderSMSOmniChannel() {
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "hi"), time.Now()))
	s.NotContains(s.took(1)[0].text, chatSummary)
	sms := s.omniChannel(s.person)
	s.chatCard()

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.Contains(s.took(2)[1].text, chatSummary)
	s.Equal(sms, s.omniChannel(s.person))
	s.Equal(sms, s.chatOmniChannel())
}

// Catches a link that joins the number's history to the user: the number texted before the
// link (a former holder, or the user before the customer vouched for them), and the user's app
// chat would read that SMS card once linked.
func (s *UserPhonesSuite) TestALinkedUserReadsNoneOfTheNumbersSMSFromBeforeTheLink() {
	s.textedBeforeTheLink()
	s.chatCard()

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	s.Equal([]string{store.EpisodeChat}, s.chatReads(s.user), "the user's own chat card alone")
}

// Catches a fix that leaves the number off the user for good: an SMS after the link joins the
// user's omni-channel, so the user's chat reads its card and the SMS reads the chat's.
func (s *UserPhonesSuite) TestAnSMSAfterTheLinkReachesTheUsersChat() {
	s.textedBeforeTheLink()
	s.chatCard()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.Contains(s.took(2)[1].text, chatSummary)
	s.ElementsMatch([]string{store.EpisodeSMS, store.EpisodeChat}, s.chatReads(s.user))
	s.Equal(s.chatOmniChannel(), s.omniChannel(s.person))
}

// Catches a number given to somebody else that carries its last holder's SMS: anya texted from
// it while linked, the customer unlinked her and linked bob, and bob's chat would read her SMS
// card, which the unlink left with the number.
func (s *UserPhonesSuite) TestAUserLinkedToANumberSomebodyHadBeforeReadsNoneOfTheirSMS() {
	s.chatCard()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "this is anya"), time.Now()))
	s.took(1)
	s.idle()
	s.Require().ElementsMatch([]string{store.EpisodeSMS, store.EpisodeChat}, s.chatReads(s.user), "linked, her SMS is hers")
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": null}`, nil))
	bob := "bob-" + s.utils.uuid()
	s.chatCardOf(bob, "Bob asked about his bill.")

	s.Require().Equal(http.StatusOK, s.put(bob, `{"phone": "`+s.person+`"}`, nil))

	s.Equal([]string{store.EpisodeChat}, s.chatReads(bob), "bob's own chat card alone")
}

// The control: the same link again leaves everything as it was, the cards of the SMS written
// since the first included, as the probe of this test at 3b7a53f97 (before this change)
// answered.
func (s *UserPhonesSuite) TestTheSameLinkAgainKeepsTheCardsWrittenSinceTheFirst() {
	s.chatCard()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "first"), time.Now()))
	s.took(1)
	s.idle()
	omni := s.omniChannel(s.person)

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	s.ElementsMatch([]string{store.EpisodeSMS, store.EpisodeChat}, s.chatReads(s.user))
	s.Equal(omni, s.omniChannel(s.person))
	s.Equal(omni, s.chatOmniChannel())
}

// The control: under a config without episode_cards the channel bridge still writes the SMS's
// row and episode, and the link leaves them as they were: one row under the number, one
// episode, and no cards read, as the probe of this test at 3b7a53f97 (before this change)
// answered.
func (s *UserPhonesSuite) TestAConfigWithoutCardsLinksAsBefore() {
	s.config.EpisodeCards = false
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "hi"), time.Now()))
	s.took(1)

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.NotContains(s.took(2)[1].text, "earlier episodes")
	s.Equal(1, s.episodes())
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM contact_map WHERE customer_id = ?", s.customerID()).Scan(&rows))
	s.Equal(1, rows)
	s.Equal([]string{s.person}, s.links())
}

// The control: without a link the SMS reads no chat card and keeps its own omni-channel, as
// the probe of this test at 13e088981 (the chat card's PR, before this one) answered.
func (s *UserPhonesSuite) TestWithoutALinkAnSMSReadsNoChatCard() {
	chat := s.chatCard()

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.NotContains(s.took(1)[0].text, chatSummary)
	s.NotEqual(chat, s.omniChannel(s.person))
	s.Equal(chat, s.chatOmniChannel())
}

// Catches a number kept as it was written, so the SMS's +1… never matches it, and a second
// PUT of the same number that fails or adds a row.
func (s *UserPhonesSuite) TestAPutAnswersTheNumberInE164AndIsIdempotent() {
	var user AgentUser
	formatted := s.person[:2] + " (" + s.person[2:5] + ") " + s.person[5:8] + "-" + s.person[8:]

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+formatted+`"}`, &user))
	s.Equal(AgentUser{UserID: s.user, Phone: &s.person}, user, "the router's form, not the one sent")
	s.Require().Equal([]string{s.person}, s.links(), "kept as an SMS from it names it")
	user = AgentUser{}
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, &user))

	s.Equal(AgentUser{UserID: s.user, Phone: &s.person}, user)
	s.Equal([]string{s.person}, s.links())
}

// Catches a number linked to a second user of the app, and an answer that names who holds it.
func (s *UserPhonesSuite) TestANumberAnotherUserHoldsIsAConflict() {
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/users/other-"+s.utils.uuid(), `{"phone": "`+s.person+`"}`)

	s.Equal(http.StatusConflict, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered))
	s.Equal(codePhoneTaken, answered.Error.Code)
	s.NotContains(string(payload), s.person[2:])
	s.NotContains(string(payload), s.user)
}

// Catches a null that is refused or keeps the link, and a number still held after it.
func (s *UserPhonesSuite) TestNullUnlinksAndFreesTheNumber() {
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	var user AgentUser

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": null}`, &user))

	s.Equal(AgentUser{UserID: s.user}, user)
	s.Empty(s.links())
	s.Equal(http.StatusOK, s.put("other-"+s.utils.uuid(), `{"phone": "`+s.person+`"}`, nil))
}

// Catches an unlink that leaves the number on the user's omni-channel: an SMS from it then still
// reads the user's chat card, and a second user linked to the freed number (a number given to
// somebody else) reads the first user's card and is read by them.
func (s *UserPhonesSuite) TestAnUnlinkedNumberReadsNoneOfTheUsersCards() {
	anya := s.chatCard()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "first"), time.Now()))
	s.Require().Contains(s.took(1)[0].text, chatSummary, "linked, the SMS reads the chat card")

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": null}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "after the unlink"), time.Now()))

	turns := s.took(2)
	s.NotContains(s.cardsBefore(turns[1].text, "first"), chatSummary)
	s.Equal(anya, s.chatOmniChannel(), "the user keeps their omni-channel")
	s.NotEqual(anya, s.omniChannel(s.person))
	bob := "bob-" + s.utils.uuid()
	s.chatCardOf(bob, "Bob asked about his bill.")
	s.Require().Equal(http.StatusOK, s.put(bob, `{"phone": "`+s.person+`"}`, nil))
	s.NotEqual(anya, s.chatOmniChannelOf(bob), "bob is not on anya's omni-channel")
	s.Equal(anya, s.chatOmniChannel(), "anya is not on bob's")
	// A number is given to somebody else long after its last SMS, once its episode closed.
	s.idle()
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "bob here"), time.Now()))
	turns = s.took(3)
	s.NotContains(s.cardsBefore(turns[2].text, "first"), chatSummary, "bob's SMS reads none of anya's cards")
	s.Contains(s.cardsBefore(turns[2].text, "first"), "Bob asked about his bill.")
}

// Catches a new number that leaves the old one on the user's omni-channel: the old number
// linked next to another user would join that user onto the first user's omni-channel.
func (s *UserPhonesSuite) TestAnOldNumberLinkedToAnotherUserJoinsNoneOfTheFirstUsersCards() {
	s.chatCard()
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "hi"), time.Now()))
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.utils.number()+`"}`, nil))

	bob := "bob-" + s.utils.uuid()
	s.chatCardOf(bob, "Bob asked about his bill.")
	s.Require().Equal(http.StatusOK, s.put(bob, `{"phone": "`+s.person+`"}`, nil))

	s.NotEqual(s.chatOmniChannel(), s.chatOmniChannelOf(bob))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "bob here"), time.Now()))
	s.NotContains(s.took(2)[1].text, chatSummary, "the old number reads none of anya's cards")
	s.NotEqual(s.chatOmniChannel(), s.omniChannel(s.person))
}

// Catches a number that is not E.164 stored as a link no SMS could ever match, and a body of
// the wrong shape taken as an unlink.
func (s *UserPhonesSuite) TestABodyThatIsNotANumberIsRefused() {
	for _, body := range []string{
		`{"phone": "5550100100"}`, `{"phone": "911"}`, `{"phone": ""}`, `{"phone": "+44 (020) 7946 0018"}`,
		`{"phone": "+1555abc0100"}`,
	} {
		status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/users/"+s.user, body)
		s.Equal(http.StatusBadRequest, status, body)
		s.Contains(string(payload), "E.164", body)
	}
	for _, body := range []string{`{"phone": 15550100100}`, `{}`, `[]`, `{"phone": ["+15550100100"]}`} {
		status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/users/"+s.user, body)
		s.Equal(http.StatusBadRequest, status, body)
		s.Contains(string(payload), codeValidationFailed, body)
	}
	status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/users/"+s.user, ``)
	s.Equal(http.StatusBadRequest, status, "no body")
	s.Contains(string(payload), "request body is required", "no body, refused by the schema")
	s.Empty(s.links())
}

// Catches a link scoped to the user id alone: another customer's same user and number are its
// own, and neither is a conflict with the other.
func (s *UserPhonesSuite) TestAnotherCustomerLinksItsOwnUser() {
	first := s.customerID()
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.useApp(s.data.createApp())

	s.Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))

	s.Equal([]string{s.person}, s.links())
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM user_phones WHERE customer_id = ?", first).Scan(&count))
	s.Equal(1, count)
}

// Catches a link kept for the customer rather than for the Stream app the request acts in:
// the same user id in a second app is another person, who may have a number of their own, and
// a number the first app's user holds is free in the second.
func (s *UserPhonesSuite) TestTheSameUserInASecondStreamAppIsAnotherPerson() {
	second := s.utils.number()
	s.giveApp(s.customerID(), 7101, "key-"+s.utils.uuid())
	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.giveApp(s.customerID(), 7102, "key-"+s.utils.uuid())

	s.Equal(http.StatusOK, s.put(s.user, `{"phone": "`+second+`"}`, nil))
	s.Equal(http.StatusOK, s.put("other-"+s.utils.uuid(), `{"phone": "`+s.person+`"}`, nil))

	var apps []int64
	s.Require().NoError(s.store.DB().NewSelect().Table("user_phones").Column("stream_app_pk").
		Where("customer_id = ? AND user_id = ?", s.customerID(), s.user).Order("stream_app_pk").Scan(context.Background(), &apps))
	s.Equal([]int64{7101, 7102}, apps)
}

// Catches the operation reachable from an end user's device: a user could link their own id to
// anybody's number and read that number's cards.
func (s *UserPhonesSuite) TestAnEndUserCannotLinkAUser() {
	status, _ := s.client.call(http.MethodPut, "/v1/agents/users/"+s.user, `{"phone": "`+s.person+`"}`)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.links())
}

// LinqSandboxSuite is the Linq line of a customer the sandbox holds, under an agent with
// episode_cards on: before each reply, and each scheduled delivery, the bridge asks dlc.Gate
// about the person a chat's replies reach, the contact map row of the chat's episode
// (channelbridge.replyTo), and the sandbox lists only the test's person.
type LinqSandboxSuite struct {
	linqLine

	person string
}

func TestLinqSandboxSuite(t *testing.T) {
	runSuite(t, new(LinqSandboxSuite))
}

func (s *LinqSandboxSuite) SetupSuite() {
	s.database = "linq_sandbox"
	s.sandbox = dlc.Sandbox{Enabled: true, Recipients: 2}
	s.linqLine.SetupSuite()
}

func (s *LinqSandboxSuite) SetupTest() {
	s.linqLine.SetupTest()
	s.config.EpisodeCards = true
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	s.person = s.utils.number()
	s.Require().NoError(s.store.SetSandboxRecipients(context.Background(), s.customerID(), []string{s.person}))
}

// Catches a link that leaves a chat's replies to its set-aside row's address, retired:<id>,
// which the sandbox does not list: the customer links the person's number to a user while
// their iMessage chat is going on, and the chat's next reply, and a scheduled delivery into
// it, are held back.
func (s *LinqSandboxSuite) TestAChatWhoseNumberIsLinkedMidConversationIsStillAnswered() {
	chat := s.utils.uuid()
	s.Require().Equal(http.StatusOK, s.deliver(s.received(chat, s.line, s.person, "hi"), time.Now()))
	s.took(1)
	s.Require().Eventually(func() bool { return s.rows("SELECT count(*) FROM episodes WHERE customer_id = ?") == 1 },
		settleFor, 10*time.Millisecond, "the chat's episode")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/users/user-"+s.utils.uuid(),
		`{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(1, s.rows("SELECT count(*) FROM contact_map WHERE customer_id = ? AND address LIKE 'retired:%'"),
		"the link set the chat's row aside")
	s.Require().Equal(http.StatusOK, s.deliver(s.received(chat, s.line, s.person, "still there?"), time.Now()))

	s.Equal(chat, s.took(2)[1].chat, "the reply after the link")
	s.NoError(s.reachable(s.threadChannel(chat)), "a scheduled delivery into the chat")
}

// rows is the count a query of the test's customer answers.
func (s *LinqSandboxSuite) rows(query string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), query, s.customerID()).Scan(&count))
	return count
}

// cardsBefore is what a turn's session was handed before the thread's history: the recited
// cards come first, and the history starts with the thread's first message, first.
func (s *UserPhonesSuite) cardsBefore(reply, first string) string {
	at := strings.Index(reply, "user: "+first+"\n")
	s.Require().NotEqual(-1, at, "the reply recites the thread's first message")
	return reply[:at]
}

// textedBeforeTheLink has the test's number text the agent before any link, and its episode
// close as idle.
func (s *UserPhonesSuite) textedBeforeTheLink() {
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "hi, this is carl"), time.Now()))
	s.took(1)
	s.idle()
}

// idle closes the test customer's open episodes, as the idle sweeper does an hour after their
// last message, so the next message opens an episode of its own.
func (s *UserPhonesSuite) idle() {
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE episodes SET status = ?, ended_at = now() WHERE customer_id = ? AND status = 'in_progress'",
		store.EpisodeEnded, s.customerID())
	s.Require().NoError(err)
}

// chatReads are the sources of the cards a session of the user's app chat reads, newest first:
// their stream_user row and the cards of its omni-channel (omnichannel.Cards.Context), none
// left out.
func (s *UserPhonesSuite) chatReads(user string) []string {
	person, err := omnichannel.StreamUser(s.hookApp(), user)
	s.Require().NoError(err)
	contact, err := s.store.Contact(context.Background(), s.customerID(), s.config.ID, store.ContactStreamUser, person.Address)
	s.Require().NoError(err)
	cards, err := s.store.EpisodeCards(context.Background(), store.CardsQuery{
		CustomerID: s.customerID(), AgentConfigID: s.config.ID, ConversationID: contact.ConversationID, Limit: 100,
	})
	s.Require().NoError(err)
	var sources []string
	for _, card := range cards {
		sources = append(sources, card.Source)
	}
	return sources
}

// put sends a PUT of the user with the raw body as the app's backend, and decodes the answer.
func (s *UserPhonesSuite) put(user, body string, into any) int {
	return s.serverClient.do(http.MethodPut, "/v1/agents/users/"+user, body, into)
}

// links are the numbers the test's customer has linked.
func (s *UserPhonesSuite) links() []string {
	var phones []string
	s.Require().NoError(s.store.DB().NewSelect().Table("user_phones").Column("phone").
		Where("customer_id = ?", s.customerID()).Scan(context.Background(), &phones))
	return phones
}

// chatCard writes the test's user's chat card into their omni-channel, as the app chat does
// (api.chatEpisode), summarized as the idle sweeper leaves it, and returns the omni-channel.
func (s *UserPhonesSuite) chatCard() string {
	return s.chatCardOf(s.user, chatSummary)
}

// chatCardOf is chatCard for a user of the app and the summary given.
func (s *UserPhonesSuite) chatCardOf(user, summary string) string {
	ctx := context.Background()
	cards, err := omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.stream})
	s.Require().NoError(err)
	app := s.hookApp()
	person, err := omnichannel.StreamUser(app, user)
	s.Require().NoError(err)
	opened, err := cards.Open(ctx, omnichannel.Episode{
		CustomerID: s.customerID(), AgentConfigID: s.config.ID, AgentName: s.config.Name, Person: person,
		Source: store.EpisodeChat, ThreadChannel: "agent:chat-" + s.utils.uuid(), StreamAppPK: app,
	})
	s.Require().NoError(err)
	s.Require().NoError(cards.Write(ctx, opened))
	omni := strings.TrimPrefix(opened.Contact.ConversationID, "agent:")
	bound, err := s.stream.ForApp(ctx, s.customerID(), app)
	s.Require().NoError(err)
	_, err = bound.Client.Chat().UpdateMessagePartial(ctx, opened.Episode.CardMessageID, &getstream.UpdateMessagePartialRequest{
		UserID: &omni, Set: map[string]any{"text": summary, "status": store.EpisodeSummarized},
	})
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, "UPDATE episodes SET status = ?, ended_at = now() WHERE id = ?",
		store.EpisodeSummarized, opened.Episode.ID)
	s.Require().NoError(err)
	return omni
}

// hookApp is the app the message hook names a message of the test's customer by: the pin the
// customer acts in, and the deployment's own app by its id.
func (s *UserPhonesSuite) hookApp() int64 {
	pin, err := s.stream.Pin(context.Background(), s.customerID())
	s.Require().NoError(err)
	if pin == 0 {
		return suiteStreamApp
	}
	return pin
}

// chatOmniChannel is the omni-channel the test's user's chat row points at now.
func (s *UserPhonesSuite) chatOmniChannel() string {
	return s.chatOmniChannelOf(s.user)
}

// chatOmniChannelOf is chatOmniChannel for a user of the app.
func (s *UserPhonesSuite) chatOmniChannelOf(user string) string {
	person, err := omnichannel.StreamUser(s.hookApp(), user)
	s.Require().NoError(err)
	contact, err := s.store.Contact(context.Background(), s.customerID(), s.config.ID, store.ContactStreamUser, person.Address)
	s.Require().NoError(err)
	return strings.TrimPrefix(contact.ConversationID, "agent:")
}
