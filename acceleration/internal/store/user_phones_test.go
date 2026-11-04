//go:build integration

package store

import (
	"fmt"
	"sync"
)

// streamUser is a contact map row of the acme-app customer's agent for a Stream user of app
// 4242, which a test changes one part of.
func streamUser(userID, conversationID string) *ContactMapEntry {
	entry := contact(userID, conversationID)
	entry.Kind = ContactStreamUser
	return entry
}

// link links a user of the acme-app customer's app 4242 to a number.
func (s *StoreSuite) link(userID, phone string) error {
	return s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: userID, Phone: phone})
}

// made makes a contact row and returns it as the store holds it.
func (s *StoreSuite) made(entry *ContactMapEntry) *ContactMapEntry {
	_, err := s.store.MapContact(s.ctx, entry)
	s.Require().NoError(err)
	return entry
}

// omniOf is the omni-channel a row points at now.
func (s *StoreSuite) omniOf(entry *ContactMapEntry) string {
	found, err := s.store.Contact(s.ctx, entry.CustomerID, entry.AgentConfigID, entry.Kind, entry.Address)
	s.Require().NoError(err)
	return found.ConversationID
}

// Catches a link that joins nothing, joins onto the newer omni-channel, or joins rows of one
// agent onto another agent's: under agent-one the number came first, under agent-two the user.
func (s *StoreSuite) TestALinkPointsTheUsersAndTheNumbersRowsAtTheOlderOmniChannel() {
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	s.Require().Equal("agent:omni-chat", s.omniOf(chat), "no link, no join")

	s.Require().NoError(s.link("anya", "+15550100"))

	s.Equal("agent:omni-sms", s.omniOf(sms))
	s.Equal("agent:omni-sms", s.omniOf(chat))
	s.Equal("agent:omni-chat-two", s.omniOf(chatTwo))
	s.Equal("agent:omni-chat-two", s.omniOf(smsTwo))
	var latest string
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT max(updated_at)::text FROM contact_map").Scan(&latest))
	s.Require().NoError(s.link("anya", "+15550100"))
	var again string
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT max(updated_at)::text FROM contact_map").Scan(&again))
	s.Equal(latest, again, "the same link again writes no row")
}

// Catches a row made after the link that keeps an omni-channel of its own, whichever half of
// the pair comes later, and a row of an agent the partner has no row under that joins anything.
func (s *StoreSuite) TestARowMadeAfterTheLinkTakesItsPartnersOmniChannel() {
	s.Require().NoError(s.link("anya", "+15550100"))
	s.made(streamUser("anya", "agent:omni-chat"))
	sms := contact("+15550100", "agent:omni-sms")

	created, err := s.store.MapContact(s.ctx, sms)

	s.Require().NoError(err)
	s.True(created, "the row is new, whatever omni-channel it was given")
	s.Equal("agent:omni-chat", sms.ConversationID, "what Cards.Write writes into")
	s.Equal("agent:omni-chat", s.omniOf(sms))

	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	s.Equal("agent:omni-sms-two", chatTwo.ConversationID)

	alone := contact("+15550100", "agent:omni-alone")
	alone.AgentConfigID = "agent-three"
	s.Equal("agent:omni-alone", s.made(alone).ConversationID)
}

// Catches a join that leaves the newer row in its own app: the omni-channel is in the app of
// the row that made it, so a row pointed at it takes that pin. The message hook pins a chat
// row of the deployment's own app to the app's id, an SMS row of it is pinned to none, and a
// link made by a request acting in it names 0: all three are one app.
func (s *StoreSuite) TestTheDeploymentsOwnAppIsOneAppWhateverItsPin() {
	s.store.SetStreamPins(StreamPins{Deployment: func() int64 { return 4242 }})
	s.T().Cleanup(func() { s.store.SetStreamPins(StreamPins{}) })
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	sms := contact("+15550100", "agent:omni-sms")
	sms.StreamAppPK = 0
	s.made(sms)

	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", UserID: "anya", Phone: "+15550100"}))

	s.Equal("agent:omni-chat", s.omniOf(sms))
	found, err := s.store.Contact(s.ctx, "acme-app", "agent-one", ContactPhone, "+15550100")
	s.Require().NoError(err)
	s.Equal(int64(4242), found.StreamAppPK, "the pin of the omni-channel it points at")
	s.Equal("agent:omni-chat", s.omniOf(chat))
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID, smsTwo.StreamAppPK = "agent-two", 0
	s.made(smsTwo)
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.Equal("agent:omni-sms-two", s.made(chatTwo).ConversationID, "a row made later, pinned to the app's id")
}

// Catches a link keyed by user id alone: anya of app 7 is not anya of app 4242.
func (s *StoreSuite) TestTheSameUserIdInAnotherAppIsAnotherPerson() {
	s.made(contact("+15550100", "agent:omni-sms"))
	other := streamUser("anya", "agent:omni-chat")
	other.StreamAppPK = 7
	s.made(other)

	s.Require().NoError(s.link("anya", "+15550100"))

	s.Equal("agent:omni-chat", s.omniOf(other))
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	later := streamUser("anya", "agent:omni-chat-two")
	later.AgentConfigID, later.StreamAppPK = "agent-two", 7
	s.Equal("agent:omni-chat-two", s.made(later).ConversationID, "a row made later in app 7")
}

// Catches a number linked to two users of one app, a second app refused the number, and a
// link sent again refused as taken.
func (s *StoreSuite) TestANumberIsOneUserOfAnApp() {
	s.Require().NoError(s.link("anya", "+15550100"))

	s.ErrorIs(s.link("bob", "+15550100"), ErrPhoneTaken)
	s.NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "bob", Phone: "+15550100"}))
	s.NoError(s.link("anya", "+15550100"), "the same link again")
	s.Equal(2, s.rows("SELECT count(*) FROM user_phones"))
}

// Catches a link of one customer that joins another customer's rows of the same user and
// number, and a row made later that finds the other customer's link rather than its own.
func (s *StoreSuite) TestAnotherCustomersLinkJoinsNothingOfThisCustomer() {
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))

	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "globex-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550199"}))

	s.Equal("agent:omni-sms", s.omniOf(sms))
	s.Equal("agent:omni-chat", s.omniOf(chat))
	s.Require().NoError(s.link("anya", "+15550100"))
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.Equal("agent:omni-sms-two", s.made(chatTwo).ConversationID, "acme's own link, not globex's")
}

// Catches an unlink that splits what was joined, or leaves the link to join the next row.
func (s *StoreSuite) TestAnUnlinkKeepsWhatWasJoinedAndJoinsNoMore() {
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	s.Require().NoError(s.link("anya", "+15550100"))
	s.Require().NoError(s.link("bob", "+15550199"))
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "anya", Phone: "+15550100"}))

	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "acme-app", 4242, "anya"))

	s.Equal("agent:omni-sms", s.omniOf(chat))
	s.Equal("agent:omni-sms", s.omniOf(sms))
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	later := contact("+15550100", "agent:omni-later")
	later.AgentConfigID = "agent-two"
	s.Equal("agent:omni-later", s.made(later).ConversationID)
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones WHERE user_id = 'bob'"), "another user's link")
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones WHERE stream_app_pk = 7"), "the user's link in another app")
	s.Equal(2, s.rows("SELECT count(*) FROM user_phones"))
}

// Catches a new number that leaves the old one linked, and a join that takes the older row
// rather than the older omni-channel: anya's row is the newest, but after the first link it
// points at the first number's omni-channel, older than the second number's.
func (s *StoreSuite) TestANewNumberMovesTheLinkAndKeepsTheOlderOmniChannel() {
	first := s.made(contact("+15550100", "agent:omni-first"))
	second := s.made(contact("+15550199", "agent:omni-second"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	s.Require().NoError(s.link("anya", "+15550100"))
	s.Require().Equal("agent:omni-first", s.omniOf(chat))

	s.Require().NoError(s.link("anya", "+15550199"))

	s.Equal("agent:omni-first", s.omniOf(chat))
	s.Equal("agent:omni-first", s.omniOf(second))
	s.Equal("agent:omni-first", s.omniOf(first))
	s.NoError(s.link("bob", "+15550100"), "the first number is free again")
}

// Catches a link and a first sighting at once that leave the pair on two omni-channels: each
// reads after its own write is committed, so one of them sees the other. Two routers, twenty
// pairs, the link and the chat row's first message at the same moment.
func (s *StoreSuite) TestALinkAndAFirstSightingAtOnceEndOnOneOmniChannel() {
	pools := s.pools(2)
	for i := range 20 {
		user, phone := fmt.Sprintf("user-%d", i), fmt.Sprintf("+1555010%04d", i)
		s.made(contact(phone, "agent:omni-sms-"+newID()))
		chat := streamUser(user, "agent:omni-chat-"+newID())
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			s.NoError(pools[0].LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: user, Phone: phone}))
		}()
		go func() {
			defer wg.Done()
			_, err := pools[1].MapContact(s.ctx, chat)
			s.NoError(err)
		}()
		wg.Wait()

		s.Equal(s.omniOf(contact(phone, "")), s.omniOf(chat), "pair %d", i)
	}
}

// rows is the one number a query answers.
func (s *StoreSuite) rows(query string) int {
	var n int
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, query).Scan(&n))
	return n
}
