//go:build integration

package store

import (
	"fmt"
	"slices"
	"sync"
	"time"
)

// streamUser is a contact map row of the acme-app customer's agent for a Stream user of app
// 4242, which a test changes one part of.
func streamUser(userID, conversationID string) *ContactMapEntry {
	return streamUserOf(4242, userID, conversationID)
}

// streamUserOf is streamUser for a user of the app given, its row pinned there, as the message
// hook makes it.
func streamUserOf(app int64, userID, conversationID string) *ContactMapEntry {
	entry := contact(StreamUserAddress(app, userID), conversationID)
	entry.Kind, entry.StreamAppPK = ContactStreamUser, app
	return entry
}

// link links a user of the acme-app customer's app 4242 to a number.
func (s *StoreSuite) link(userID, phone string) error {
	return s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: userID, Phone: phone})
}

// agentConfigs makes the acme-app customer's agent configs the tests' rows name: a link joins
// only the rows of the customer's own configs.
func (s *StoreSuite) agentConfigs() {
	for _, id := range []string{"agent-one", "agent-two", "agent-three"} {
		_, err := s.store.db.ExecContext(s.ctx, "INSERT INTO agent_configs (id, customer_id, name) VALUES (?, 'acme-app', ?)", id, id)
		s.Require().NoError(err)
	}
}

// chatEpisode opens a chat episode on a Stream user's row, the history its omni-channel holds.
func (s *StoreSuite) chatEpisode(row *ContactMapEntry) string {
	episode := &Episode{CustomerID: row.CustomerID, ContactID: row.ID, Source: EpisodeChat, ThreadChannel: "agent:chat-" + newID()}
	_, err := s.store.OpenEpisode(s.ctx, episode)
	s.Require().NoError(err)
	return episode.ID
}

// smsEpisode opens an SMS episode on a number's row, the history the number brings to a link.
func (s *StoreSuite) smsEpisode(row *ContactMapEntry) string {
	episode := &Episode{CustomerID: row.CustomerID, ContactID: row.ID, Source: EpisodeSMS, ThreadChannel: "agent:sms-" + newID()}
	_, err := s.store.OpenEpisode(s.ctx, episode)
	s.Require().NoError(err)
	return episode.ID
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
	s.agentConfigs()
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

// Catches a link that joins the number's earlier episodes to the user, under one agent config
// only; one that sets aside a row with no episode, a row in another app, another number's row,
// or a row of a config without episode_cards, which reads no card; and a set-aside row that loses its omni-channel,
// where its cards are, or a number's next row that does not join.
func (s *StoreSuite) TestALinkSetsAsideTheNumbersRowsThatHoldEpisodes() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "INSERT INTO agent_configs (id, customer_id, name) VALUES ('agent-four', 'acme-app', 'agent-four')")
	s.Require().NoError(err)
	_, err = s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true WHERE id <> 'agent-four'")
	s.Require().NoError(err)
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	anyas := s.chatEpisode(chat)
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	carls := s.smsEpisode(sms)
	other := contact("+15550100", "agent:omni-sms-7")
	other.AgentConfigID, other.StreamAppPK = "agent-two", 7
	s.smsEpisode(s.made(other))
	quiet := contact("+15550100", "agent:omni-quiet")
	quiet.AgentConfigID = "agent-three"
	s.made(quiet)
	off := contact("+15550100", "agent:omni-off")
	off.AgentConfigID = "agent-four"
	s.smsEpisode(s.made(off))
	another := s.made(contact("+15550199", "agent:omni-another"))
	s.smsEpisode(another)

	s.Require().NoError(s.link("anya", "+15550100"))

	s.Equal([]string{anyas}, ids(s.cardsOf(s.omniOf(chat), CardsQuery{})), "anya reads none of the number's earlier SMS")
	_, err = s.store.Contact(s.ctx, "acme-app", "agent-one", ContactPhone, "+15550100")
	s.ErrorIs(err, ErrNoContact, "the number has no row until it is seen again")
	var aside ContactMapEntry
	s.Require().NoError(s.store.db.NewSelect().Model(&aside).Where("id = ?", sms.ID).Scan(s.ctx))
	s.Equal("agent:omni-sms", aside.ConversationID, "where its card is")
	s.Equal(retiredPrefix+sms.ID+":+15550100", aside.Address)
	s.Equal([]string{carls}, ids(s.cardsOf("agent:omni-sms", CardsQuery{})))
	s.Equal("agent:omni-sms-7", s.omniOf(other), "a row of another app")
	s.Equal("agent:omni-quiet", s.omniOf(quiet), "a row with no episode")
	s.Equal("agent:omni-off", s.omniOf(off), "a row of a config without episode_cards")
	s.Equal("agent:omni-another", s.omniOf(another), "another number's row")
	next := contact("+15550100", "agent:omni-next")
	s.Equal(s.omniOf(chat), s.made(next).ConversationID, "the number seen again joins anya")
}

// Catches a set-aside row whose thread, call and cards no longer name the number: a Linq
// chat's replies would check opt-outs, the sandbox and the limits against retired:<id>
// (ThreadContact, channelbridge.replyTo), and a call going on at the link would close with no
// caller its lines are said by (the closed episodes' contact_address, omnichannel.callerOf).
func (s *StoreSuite) TestASetAsideRowIsStillReachedAtItsNumber() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true")
	s.Require().NoError(err)
	s.made(streamUser("anya", "agent:omni-chat"))
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	thread := &Episode{CustomerID: "acme-app", ContactID: sms.ID, Source: EpisodeSMS, ThreadChannel: "agent:thread-" + newID(),
		StartedAt: time.Now().Add(-2 * time.Hour)}
	_, err = s.store.OpenEpisode(s.ctx, thread)
	s.Require().NoError(err)
	callID := newID()
	call := &Episode{CustomerID: "acme-app", ContactID: sms.ID, Source: EpisodeCall, ThreadChannel: "agent:" + callID,
		CallID: callID, SessionID: newID()}
	_, err = s.store.OpenEpisode(s.ctx, call)
	s.Require().NoError(err)

	s.Require().NoError(s.link("anya", "+15550100"))

	_, err = s.store.Contact(s.ctx, "acme-app", "agent-one", ContactPhone, "+15550100")
	s.Require().ErrorIs(err, ErrNoContact, "the row was set aside")
	found, err := s.store.ThreadContact(s.ctx, "acme-app", "agent-one", thread.ThreadChannel)
	s.Require().NoError(err)
	s.Equal(sms.ID, found.ID)
	s.Equal("+15550100", found.Address, "a reply on the thread reaches the number")
	cards := s.cardsOf("agent:omni-sms", CardsQuery{})
	s.Require().Len(cards, 2)
	for _, card := range cards {
		s.Equal("+15550100", card.ContactAddress, card.Source)
	}
	now := time.Now()
	ended, err := s.store.EndCallEpisodes(s.ctx, AppScope{Unpinned: true}, callID, now, now.Add(-time.Minute))
	s.Require().NoError(err)
	s.Require().Len(ended, 1)
	s.Equal("+15550100", ended[0].ContactAddress, "the call's caller")
	idle, err := s.store.CloseIdleEpisodes(s.ctx, now.Add(-time.Hour), now, 1000, now.Add(-time.Minute))
	s.Require().NoError(err)
	s.Equal(map[string]string{thread.ID: "+15550100"}, s.addressesOf(idle, thread.ID))
	claimed, err := s.store.ClaimEpisodeSummaries(s.ctx, now, 1000, now.Add(time.Hour))
	s.Require().NoError(err)
	s.Equal(map[string]string{thread.ID: "+15550100", call.ID: "+15550100"}, s.addressesOf(claimed, thread.ID, call.ID))
}

// addressesOf are the contact addresses of the closed episodes named, by id.
func (s *StoreSuite) addressesOf(closed []ClosedEpisode, named ...string) map[string]string {
	addresses := map[string]string{}
	for _, episode := range closed {
		if slices.Contains(named, episode.ID) {
			addresses[episode.ID] = episode.ContactAddress
		}
	}
	return addresses
}

// Catches a row made after the link that keeps an omni-channel of its own, whichever half of
// the pair comes later, and a row of an agent the partner has no row under that joins anything.
func (s *StoreSuite) TestARowMadeAfterTheLinkTakesItsPartnersOmniChannel() {
	s.agentConfigs()
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
	s.agentConfigs()
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
	chatThree := streamUser("anya", "agent:omni-chat-three")
	chatThree.AgentConfigID = "agent-three"
	s.made(chatThree)
	pinned := contact("+15550100", "agent:omni-sms-three")
	pinned.AgentConfigID = "agent-three"
	s.Equal("agent:omni-chat-three", s.made(pinned).ConversationID,
		"a number's row made later, pinned to the app's id as a join leaves it")

	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "bob", Phone: "+15550199"}))

	s.ErrorIs(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", UserID: "carol", Phone: "+15550199"}), ErrPhoneTaken,
		"a link made with the app's id is the link of 0")
	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "acme-app", 0, "bob"))
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones"), "anya's link alone")
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", UserID: "bob", Phone: "+15550199"}))
	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "acme-app", 4242, "bob"))
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones"), "a link made with 0 is unlinked with the app's id")
}

// Catches a link keyed by user id alone: anya of app 7 is not anya of app 4242.
func (s *StoreSuite) TestTheSameUserIdInAnotherAppIsAnotherPerson() {
	s.agentConfigs()
	s.made(contact("+15550100", "agent:omni-sms"))
	other := streamUserOf(7, "anya", "agent:omni-chat")
	s.made(other)

	s.Require().NoError(s.link("anya", "+15550100"))

	s.Equal("agent:omni-chat", s.omniOf(other))
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	later := streamUserOf(7, "anya", "agent:omni-chat-two")
	later.AgentConfigID = "agent-two"
	s.Equal("agent:omni-chat-two", s.made(later).ConversationID, "a row made later in app 7")
	// App 7 sorts before 4242, as the links' primary key does.
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "anya", Phone: "+15550177"}))
	smsThree := contact("+15550100", "agent:omni-sms-three")
	smsThree.AgentConfigID = "agent-three"
	s.made(smsThree)
	chatThree := streamUser("anya", "agent:omni-chat-three")
	chatThree.AgentConfigID = "agent-three"
	s.Equal("agent:omni-sms-three", s.made(chatThree).ConversationID, "anya of app 4242 is found by app 4242's link")
}

// Catches a number's row of another app joined to this app's link: the link is the app's
// word about its user, and the number's row of app 7 is in app 7's omni-channel.
func (s *StoreSuite) TestANumbersRowInAnotherAppIsNotJoined() {
	s.agentConfigs()
	sms := contact("+15550100", "agent:omni-sms")
	sms.StreamAppPK = 7
	s.made(sms)
	chat := s.made(streamUser("anya", "agent:omni-chat"))

	s.Require().NoError(s.link("anya", "+15550100"))

	s.Equal("agent:omni-sms", s.omniOf(sms))
	s.Equal("agent:omni-chat", s.omniOf(chat))
	later := contact("+15550100", "agent:omni-later")
	later.AgentConfigID, later.StreamAppPK = "agent-two", 7
	chatTwo := streamUser("anya", "agent:omni-chat-two-4242")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	s.Equal("agent:omni-later", s.made(later).ConversationID, "a number's row made later in app 7")
	// App 7 sorts before 4242, as the links' index does.
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "bob", Phone: "+15550100"}))
	chatThree := streamUser("anya", "agent:omni-chat-three")
	chatThree.AgentConfigID = "agent-three"
	s.made(chatThree)
	smsThree := contact("+15550100", "agent:omni-sms-three")
	smsThree.AgentConfigID = "agent-three"
	s.Equal("agent:omni-chat-three", s.made(smsThree).ConversationID, "app 4242's link, not app 7's")
}

// Catches a number linked to two users of one app, a second app refused the number, and a
// link sent again refused as taken.
func (s *StoreSuite) TestANumberIsOneUserOfAnApp() {
	s.agentConfigs()
	s.Require().NoError(s.link("anya", "+15550100"))

	s.ErrorIs(s.link("bob", "+15550100"), ErrPhoneTaken)
	s.NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "bob", Phone: "+15550100"}))
	s.NoError(s.link("anya", "+15550100"), "the same link again")
	s.Equal(2, s.rows("SELECT count(*) FROM user_phones"))
}

// Catches a link of one customer that joins another customer's rows of the same user and
// number, and a row made later that finds the other customer's link rather than its own.
func (s *StoreSuite) TestAnotherCustomersLinkJoinsNothingOfThisCustomer() {
	s.agentConfigs()
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))

	// able-app sorts before acme-app, as the links' primary key does.
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "able-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"}))

	s.Equal("agent:omni-sms", s.omniOf(sms))
	s.Equal("agent:omni-chat", s.omniOf(chat))
	s.Require().NoError(s.link("anya", "+15550100"))
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.made(smsTwo)
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.Equal("agent:omni-sms-two", s.made(chatTwo).ConversationID, "acme's own link, not able's")
}

// Catches an unlink that leaves the number on the user's omni-channel, so whoever has the
// number next reads the user's cards (the number given to bob), or leaves the link to join
// the next row; and an unlink that takes another user's link or the user's link in another app.
func (s *StoreSuite) TestAnUnlinkTakesTheNumberOffTheUsersOmniChannel() {
	s.agentConfigs()
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	anya := s.made(streamUser("anya", "agent:omni-anya"))
	bob := s.made(streamUser("bob", "agent:omni-bob"))
	anyas := s.chatEpisode(anya)
	s.Require().NoError(s.link("anya", "+15550100"))
	s.Require().Equal("agent:omni-sms", s.omniOf(anya))
	s.Require().Equal([]string{anyas}, ids(s.cardsOf(s.omniOf(sms), CardsQuery{})), "joined, the number reads anya's card")
	s.Require().NoError(s.link("carol", "+15550199"))
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "anya", Phone: "+15550100"}))

	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "acme-app", 4242, "anya"))

	s.Equal("agent:omni-sms", s.omniOf(anya), "anya keeps her omni-channel")
	s.Equal([]string{anyas}, ids(s.cardsOf(s.omniOf(anya), CardsQuery{})))
	s.NotEqual(s.omniOf(anya), s.omniOf(sms))
	s.Regexp(`^agent:omni-[0-9a-f-]{36}$`, s.omniOf(sms), "an omni-channel id, as omnichannel.Open makes one")
	s.Empty(ids(s.cardsOf(s.omniOf(sms), CardsQuery{})), "the number reads none of anya's cards")
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	later := contact("+15550100", "agent:omni-later")
	later.AgentConfigID = "agent-two"
	s.Equal("agent:omni-later", s.made(later).ConversationID, "no link, no join")
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones WHERE user_id = 'carol'"), "another user's link")
	s.Equal(1, s.rows("SELECT count(*) FROM user_phones WHERE stream_app_pk = 7"), "the user's link in another app")
	s.Equal(2, s.rows("SELECT count(*) FROM user_phones"))

	s.Require().NoError(s.link("bob", "+15550100"))

	s.NotEqual(s.omniOf(anya), s.omniOf(bob))
	s.Empty(ids(s.cardsOf(s.omniOf(bob), CardsQuery{})), "bob, who has the number now, reads none of anya's cards")
}

// Catches an unlink keyed by app and user alone: every customer on the deployment's own app
// acts in pin 0, so another customer's unlink of its own anya would take this customer's link.
func (s *StoreSuite) TestAnotherCustomersUnlinkKeepsThisCustomersLink() {
	s.agentConfigs()
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	s.made(streamUser("anya", "agent:omni-chat"))
	s.Require().NoError(s.link("anya", "+15550100"))

	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "able-app", 4242, "anya"))

	s.Equal(1, s.rows("SELECT count(*) FROM user_phones WHERE customer_id = 'acme-app'"))
	s.Equal("agent:omni-sms", s.omniOf(sms))
	chatTwo := streamUser("anya", "agent:omni-chat-two")
	chatTwo.AgentConfigID = "agent-two"
	s.made(chatTwo)
	smsTwo := contact("+15550100", "agent:omni-sms-two")
	smsTwo.AgentConfigID = "agent-two"
	s.Equal("agent:omni-chat-two", s.made(smsTwo).ConversationID, "the link still joins")
}

// Catches a new number that leaves the old one linked or on the user's omni-channel, so the
// old number's next owner reads the user's cards, and a join that takes the older row rather
// than the older omni-channel: anya's row is the newest, but her omni-channel is the oldest,
// made by her Slack row.
func (s *StoreSuite) TestANewNumberMovesTheLinkAndTakesTheOldNumberOff() {
	s.agentConfigs()
	slack := contact("T1:U1", "agent:omni-chat")
	slack.Kind = ContactSlack
	s.made(slack)
	first := s.made(contact("+15550100", "agent:omni-first"))
	second := s.made(contact("+15550199", "agent:omni-second"))
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	anyas := s.chatEpisode(chat)
	s.Require().NoError(s.link("anya", "+15550100"))
	s.Require().Equal("agent:omni-chat", s.omniOf(first))

	s.Require().NoError(s.link("anya", "+15550199"))

	s.Equal("agent:omni-chat", s.omniOf(chat))
	s.Equal("agent:omni-chat", s.omniOf(second), "the older omni-channel, not the older row")
	s.NotEqual("agent:omni-chat", s.omniOf(first))
	s.Empty(ids(s.cardsOf(s.omniOf(first), CardsQuery{})), "the old number reads none of anya's cards")
	s.Contains(ids(s.cardsOf("agent:omni-chat", CardsQuery{})), anyas)
	s.NoError(s.link("bob", "+15550100"), "the first number is free again")
}

// Catches two PUTs at once for one user with two numbers that leave a number other than the
// stored one on the user's omni-channel: each join reads the link under its lock, and a number
// the user no longer has is taken off. Two routers, twenty users.
func (s *StoreSuite) TestTwoLinksOfOneUserAtOnceLeaveOnlyTheStoredNumberJoined() {
	s.agentConfigs()
	pools := s.pools(2)
	for i := range 20 {
		user := fmt.Sprintf("u-%d", i)
		phones := []string{fmt.Sprintf("+1555020%04d", i), fmt.Sprintf("+1555030%04d", i)}
		s.made(contact(phones[0], "agent:omni-p1-"+newID()))
		s.made(contact(phones[1], "agent:omni-p2-"+newID()))
		chat := s.made(streamUser(user, "agent:omni-c-"+newID()))
		var wg sync.WaitGroup
		for j, phone := range phones {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s.NoError(pools[j].LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: user, Phone: phone}))
			}()
		}
		wg.Wait()

		var stored string
		s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT phone FROM user_phones WHERE user_id = ?", user).Scan(&stored))
		for _, phone := range phones {
			if phone == stored {
				s.Equal(s.omniOf(chat), s.omniOf(contact(phone, "")), "user %d, the stored number", i)
			} else {
				s.NotEqual(s.omniOf(chat), s.omniOf(contact(phone, "")), "user %d, the other number", i)
			}
		}
	}
}

// Catches a link and a first sighting at once that leave the pair on two omni-channels: each
// reads after its own write is committed, so one of them sees the other. Two routers, twenty
// pairs, the link and the chat row's first message at the same moment.
func (s *StoreSuite) TestALinkAndAFirstSightingAtOnceEndOnOneOmniChannel() {
	s.agentConfigs()
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

// Catches a link and the number's next sighting at once that join the number's earlier SMS to
// the user, or leave the number's new row off the user's omni-channel: the sighting takes the
// set-aside row or makes one of its own, and either way the user reads none of the earlier
// episode. Two routers, twenty numbers.
func (s *StoreSuite) TestALinkAndTheNumbersNextSightingAtOnceJoinNoneOfItsHistory() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true")
	s.Require().NoError(err)
	pools := s.pools(2)
	for i := range 20 {
		user, phone := fmt.Sprintf("user-%d", i), fmt.Sprintf("+1555040%04d", i)
		chat := s.made(streamUser(user, "agent:omni-chat-"+newID()))
		earlier := s.made(contact(phone, "agent:omni-sms-"+newID()))
		s.smsEpisode(earlier)
		next := contact(phone, "agent:omni-next-"+newID())
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			s.NoError(pools[0].LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: user, Phone: phone}))
		}()
		go func() {
			defer wg.Done()
			_, err := pools[1].MapContact(s.ctx, next)
			s.NoError(err)
		}()
		wg.Wait()

		s.Empty(ids(s.cardsOf(s.omniOf(chat), CardsQuery{})), "pair %d", i)
		if next.ID != earlier.ID && next.ConversationID != s.omniOf(chat) {
			s.Fail("the number's new row is off the user's omni-channel", "pair %d", i)
		}
	}
}

// rows is the one number a query answers.
func (s *StoreSuite) rows(query string) int {
	var n int
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, query).Scan(&n))
	return n
}
