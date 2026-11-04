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

// Catches a link that leaves the person on two omni-channels, joins onto the newer one, or
// reads only the cards of the reader's own omni-channel: the SMS came first, so its
// omni-channel is the older, and the chat's card stays where it was written.
func (s *UserPhonesSuite) TestALinkJoinsTheChatOntoTheOlderSMSOmniChannel() {
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "hi"), time.Now()))
	s.NotContains(s.took(1)[0].text, chatSummary)
	sms := s.omniChannel(s.person)
	chat := s.chatCard()

	s.Require().Equal(http.StatusOK, s.put(s.user, `{"phone": "`+s.person+`"}`, nil))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, s.person, "is it still on?"), time.Now()))

	s.Contains(s.took(2)[1].text, chatSummary)
	s.Equal(sms, s.omniChannel(s.person))
	s.Equal(sms, s.chatOmniChannel())
	s.Len(s.chat.Stored(chat), 1, "the chat's card did not move")
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

// Catches the operation reachable from an end user's device: a user could link their own id to
// anybody's number and read that number's cards.
func (s *UserPhonesSuite) TestAnEndUserCannotLinkAUser() {
	status, _ := s.client.call(http.MethodPut, "/v1/agents/users/"+s.user, `{"phone": "`+s.person+`"}`)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.links())
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
	ctx := context.Background()
	cards, err := omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.stream})
	s.Require().NoError(err)
	person, err := omnichannel.StreamUser(s.user)
	s.Require().NoError(err)
	pin, err := s.stream.Pin(ctx, s.customerID())
	s.Require().NoError(err)
	opened, err := cards.Open(ctx, omnichannel.Episode{
		CustomerID: s.customerID(), AgentConfigID: s.config.ID, AgentName: s.config.Name, Person: person,
		Source: store.EpisodeChat, ThreadChannel: "agent:chat-" + s.utils.uuid(), StreamAppPK: pin,
	})
	s.Require().NoError(err)
	s.Require().NoError(cards.Write(ctx, opened))
	omni := strings.TrimPrefix(opened.Contact.ConversationID, "agent:")
	bound, err := s.stream.ForApp(ctx, s.customerID(), pin)
	s.Require().NoError(err)
	_, err = bound.Client.Chat().UpdateMessagePartial(ctx, opened.Episode.CardMessageID, &getstream.UpdateMessagePartialRequest{
		UserID: &omni, Set: map[string]any{"text": chatSummary, "status": store.EpisodeSummarized},
	})
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, "UPDATE episodes SET status = ?, ended_at = now() WHERE id = ?",
		store.EpisodeSummarized, opened.Episode.ID)
	s.Require().NoError(err)
	return omni
}

// chatOmniChannel is the omni-channel the test's user's chat row points at now.
func (s *UserPhonesSuite) chatOmniChannel() string {
	contact, err := s.store.Contact(context.Background(), s.customerID(), s.config.ID, store.ContactStreamUser, s.user)
	s.Require().NoError(err)
	return strings.TrimPrefix(contact.ConversationID, "agent:")
}
