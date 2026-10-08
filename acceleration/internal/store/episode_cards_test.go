//go:build integration

package store

import (
	"errors"
	"time"
)

// episodeIn opens an episode of a contact in a thread channel, started minutes after the
// suite's base time.
func (s *StoreSuite) episodeIn(contactID, threadChannel string, minutes int) *Episode {
	episode := s.episodeOf(contactID, threadChannel)
	episode.StartedAt = s.base.Add(time.Duration(minutes) * time.Minute)
	_, err := s.store.OpenEpisode(s.ctx, episode)
	s.Require().NoError(err)
	return episode
}

// called opens the episode of a call session, as the session manager does.
func (s *StoreSuite) called(contactID, callChannel, sessionID string, minutes int) *Episode {
	episode := s.episodeOf(contactID, callChannel)
	episode.Source, episode.SessionID, episode.CallID = EpisodeCall, sessionID, "call-"+sessionID
	episode.StartedAt = s.base.Add(time.Duration(minutes) * time.Minute)
	_, err := s.store.OpenEpisode(s.ctx, episode)
	s.Require().NoError(err)
	return episode
}

// mappedAs is a contact row for address, in an omni-channel of the test's choosing, of a
// customer and an agent of the test's choosing.
func (s *StoreSuite) mappedAs(customerID, agentConfigID, kind, address, omni string) string {
	entry := contact(address, omni)
	entry.CustomerID, entry.AgentConfigID, entry.Kind = customerID, agentConfigID, kind
	_, err := s.store.MapContact(s.ctx, entry)
	s.Require().NoError(err)
	return entry.ID
}

// cardsOf reads the cards of the acme-app agent's omni-channel.
func (s *StoreSuite) cardsOf(omni string, query CardsQuery) []EpisodeCard {
	query.CustomerID, query.AgentConfigID, query.ConversationID = "acme-app", "agent-one", omni
	if query.Limit == 0 {
		query.Limit = 10
	}
	cards, err := s.store.EpisodeCards(s.ctx, query)
	s.Require().NoError(err)
	return cards
}

func ids(cards []EpisodeCard) []string {
	var out []string
	for _, card := range cards {
		out = append(out, card.ID)
	}
	return out
}

// A person's cards are theirs alone: another person of the same agent, the same omni-channel
// id under another customer, and the same one under another agent of the customer are none
// of theirs.
func (s *StoreSuite) TestAPersonsCardsAreOnlyTheirOwnNewestFirst() {
	ann := s.mapped("+15550100")
	first := s.episodeIn(ann, "agent:thread-one", 0)
	second := s.episodeIn(ann, "agent:thread-two", 10)
	s.episodeIn(s.mapped("+15550199"), "agent:thread-bob", 20)
	stranger := s.episodeOf(s.mappedAs("globex", "agent-one", ContactPhone, "+15550100", "agent:omni-+15550100"), "agent:thread-globex")
	stranger.CustomerID = "globex"
	_, err := s.store.OpenEpisode(s.ctx, stranger)
	s.Require().NoError(err)
	s.episodeIn(s.mappedAs("acme-app", "agent-two", ContactPhone, "+15550100", "agent:omni-+15550100"), "agent:thread-other-agent", 40)

	cards := s.cardsOf("agent:omni-+15550100", CardsQuery{})

	s.Equal([]string{second.ID, first.ID}, ids(cards))
}

// Account linking points a Slack row at a phone row's omni-channel; both rows are then the
// one person, and their cards are read together.
func (s *StoreSuite) TestEveryRowPointingAtTheOmniChannelIsThePerson() {
	phone := s.mapped("+15550100")
	slack := s.mappedAs("acme-app", "agent-one", ContactSlack, "T1:U1", "agent:omni-+15550100")
	sms := s.episodeIn(phone, "agent:thread-sms", 0)
	thread := s.episodeIn(slack, "agent:thread-slack", 10)

	cards := s.cardsOf("agent:omni-+15550100", CardsQuery{})
	s.Equal([]string{thread.ID, sms.ID}, ids(cards))
	s.Equal([2]string{ContactSlack, "T1:U1"}, [2]string{cards[0].ContactKind, cards[0].ContactAddress}, "each card says whose row it is")
	s.Equal([2]string{ContactPhone, "+15550100"}, [2]string{cards[1].ContactKind, cards[1].ContactAddress})
}

// A session reads its own thread word for word, and its own call is going on, so neither is
// a card.
func (s *StoreSuite) TestTheSessionsOwnThreadAndCallAreLeftOut() {
	ann := s.mapped("+15550100")
	sms := s.episodeIn(ann, "agent:thread-sms", 0)
	slack := s.episodeIn(ann, "agent:thread-slack", 10)
	call := s.called(ann, "agent:phone-1", "session-now", 20)

	s.Equal([]string{call.ID, sms.ID}, ids(s.cardsOf("agent:omni-+15550100", CardsQuery{ExceptThread: "agent:thread-slack"})))
	s.Equal([]string{slack.ID, sms.ID}, ids(s.cardsOf("agent:omni-+15550100", CardsQuery{ExceptSession: "session-now"})))
}

func (s *StoreSuite) TestNoMoreCardsThanTheLimitAreRead() {
	ann := s.mapped("+15550100")
	var newest []string
	for minutes := range 7 {
		newest = append([]string{s.episodeIn(ann, "agent:thread-"+string(rune('a'+minutes)), minutes).ID}, newest...)
	}

	s.Equal(newest[:5], ids(s.cardsOf("agent:omni-+15550100", CardsQuery{Limit: 5})))
}

// A call channel can hold several callers' calls, so a call's lines end where the next
// episode in the channel starts, when its session closed, or when it ended, whichever is
// first.
func (s *StoreSuite) TestACardsLinesEndAtTheFirstThingThatEndsIt() {
	ann, bob := s.mapped("+15550100"), s.mapped("+15550199")
	open := s.called(ann, "agent:phone-1", "session-ann", 0)
	s.called(bob, "agent:phone-1", "session-bob", 30)
	closed := s.called(ann, "agent:phone-2", "session-closed", 40)
	s.Require().NoError(s.store.SaveSession(s.ctx, &AgentSession{ID: "session-closed", CustomerID: "acme-app", CallID: "call-session-closed"}))
	s.Require().NoError(s.store.CloseSession(s.ctx, "session-closed", s.base.Add(45*time.Minute)))
	ended := s.episodeIn(ann, "agent:thread-ended", 50)
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE episodes SET status = 'ended', ended_at = ? WHERE id = ?", s.base.Add(55*time.Minute), ended.ID)
	s.Require().NoError(err)
	going := s.episodeIn(ann, "agent:thread-going", 60)

	until := map[string]*time.Time{}
	for _, card := range s.cardsOf("agent:omni-+15550100", CardsQuery{}) {
		until[card.ID] = card.Until
	}

	at := func(minutes int) *time.Time {
		t := s.base.Add(time.Duration(minutes) * time.Minute)
		return &t
	}
	s.Require().Len(until, 4)
	s.Equal(at(30).UTC(), until[open.ID].UTC(), "Bob's call in the same channel")
	s.Equal(at(45).UTC(), until[closed.ID].UTC(), "the session closed")
	s.Equal(at(55).UTC(), until[ended.ID].UTC(), "the episode ended")
	s.Nil(until[going.ID], "nothing ends it yet")
}

// The person of a thread is the one its episode is with, for the customer and agent asked
// about only.
func (s *StoreSuite) TestAThreadsPersonIsTheOneItsEpisodeIsWith() {
	ann := s.mapped("+15550100")
	s.episodeIn(ann, "agent:thread-sms", 0)

	found, err := s.store.ThreadContact(s.ctx, "acme-app", "agent-one", "agent:thread-sms")
	s.Require().NoError(err)
	s.Equal(ann, found.ID)

	for _, asked := range [][2]string{{"globex", "agent-one"}, {"acme-app", "agent-two"}} {
		_, err = s.store.ThreadContact(s.ctx, asked[0], asked[1], "agent:thread-sms")
		s.True(errors.Is(err, ErrNoContact), asked)
	}
	_, err = s.store.ThreadContact(s.ctx, "acme-app", "agent-one", "agent:thread-nobody")
	s.True(errors.Is(err, ErrNoContact))
}

// Reading a person's cards writes nothing: an address the contact map does not hold is none,
// and stays none.
func (s *StoreSuite) TestLookingUpAContactMakesNone() {
	_, err := s.store.Contact(s.ctx, "acme-app", "agent-one", ContactPhone, "+15550100")

	s.True(errors.Is(err, ErrNoContact))
	s.Zero(s.contactRows())

	id := s.mapped("+15550100")
	found, err := s.store.Contact(s.ctx, "acme-app", "agent-one", ContactPhone, "+15550100")
	s.Require().NoError(err)
	s.Equal(id, found.ID)
}
