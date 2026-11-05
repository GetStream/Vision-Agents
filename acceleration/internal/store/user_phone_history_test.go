//go:build integration

package store

import (
	"time"
)

// historyStory is the ticket's example under agent-one with the cards on, so the link sets the
// number's row aside: anya's chat, and the number's SMS on Sunday and Monday, before anya is
// linked to it.
func (s *StoreSuite) historyStory() (chat *ContactMapEntry, own, sunday, monday string) {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true WHERE id = 'agent-one'")
	s.Require().NoError(err)
	chat = s.made(streamUser("anya", "agent:omni-chat"))
	sms := s.made(contact("+15550100", "agent:omni-sms"))
	sunday, monday = s.smsAt(sms, linkedAt.Add(-36*time.Hour)), s.smsAt(sms, linkedAt.Add(-12*time.Hour))
	return chat, s.chatEpisode(chat), sunday, monday
}

// linkWith links anya to the number as a request with history does.
func (s *StoreSuite) linkWith(phone string, history *PhoneHistory) {
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "anya",
		Phone: phone, History: history}))
}

// boundary is anya's link's boundary as stored, nil for all of the number's past.
func (s *StoreSuite) boundary() *time.Time {
	_, since, linked, err := s.store.LinkedPhone(s.ctx, "acme-app", 4242, "anya")
	s.Require().NoError(err)
	s.Require().True(linked)
	return since
}

// Catches a link with all of the history that still sets the number's past aside, so anya
// reads none of it, or joins it with no contact address; the number's next row must join her too.
func (s *StoreSuite) TestALinkWithAllOfTheHistoryReadsWhatTheNumberSaidBeforeIt() {
	chat, own, sunday, monday := s.historyStory()

	s.linkWith("+15550100", &PhoneHistory{All: true})

	cards, err := s.store.EpisodeCards(s.ctx, CardsQuery{CustomerID: "acme-app", AgentConfigID: "agent-one",
		ConversationID: s.omniOf(chat), Limit: 10})
	s.Require().NoError(err)
	s.ElementsMatch([]string{own, sunday, monday}, ids(cards))
	for _, card := range cards {
		if card.Source == EpisodeSMS {
			s.Equal("+15550100", card.ContactAddress)
		}
	}
	s.Nil(s.boundary())
	s.Equal(s.omniOf(chat), s.made(contact("+15550100", "agent:omni-next")).ConversationID, "the number seen again")
}

// Catches a since that is read as all of it or as the link's time: anya reads Monday, not Sunday.
func (s *StoreSuite) TestALinkFromATimeReadsWhatTheNumberSaidFromThen() {
	chat, own, _, monday := s.historyStory()

	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(-24 * time.Hour)})

	s.ElementsMatch([]string{own, monday}, s.readFor(chat))
	s.Equal(linkedAt.Add(-24*time.Hour), s.boundary().UTC())
}

// Catches a request with neither field that resets the boundary, one with false that leaves it,
// a false that leaves set-aside rows joined but unbounded, and a new number that keeps the old
// number's.
func (s *StoreSuite) TestTheBoundaryStaysWithNoWordAndMovesWithFalseOrANewNumber() {
	chat, own, _, _ := s.historyStory()
	s.linkWith("+15550100", &PhoneHistory{All: true})

	s.linkWith("+15550100", nil)
	s.Nil(s.boundary(), "the same number with neither field")
	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt})
	s.linkWith("+15550100", nil)
	s.Equal(linkedAt, s.boundary().UTC())

	linked := func() time.Time {
		var at time.Time
		s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT updated_at FROM user_phones WHERE user_id = 'anya'").Scan(&at))
		return at
	}
	s.linkWith("+15550100", &PhoneHistory{})
	s.True(s.boundary().After(linked()), "false is from the request on, after the link")
	s.Equal([]string{own}, s.readFor(chat))
	s.linkWith("+15550199", &PhoneHistory{All: true})
	s.linkWith("+15550100", nil)
	s.Equal(linked(), *s.boundary(), "a new number with neither field is from its link on")
}

// Catches a join of set-aside rows for a boundary after the link, which a since after it gives,
// or for a row of another number, which a router that knows no history leaves.
func (s *StoreSuite) TestOnlyABoundaryBeforeTheLinkJoinsTheSetAsideRows() {
	chat, own, sunday, monday := s.historyStory()
	s.linkDated(&UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"})

	aside := func() string {
		var omni string
		s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT conversation_id FROM contact_map WHERE address LIKE 'retired:%'").Scan(&omni))
		return omni
	}
	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(time.Hour)})
	s.Equal([]string{own}, s.readFor(chat), "a boundary after the link")
	s.Equal("agent:omni-sms", aside(), "a boundary after the link")
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE user_phone_history SET phone = '+15550199', since = NULL")
	s.Require().NoError(err)
	s.linkWith("+15550100", nil)
	s.Equal([]string{own}, s.readFor(chat), "a row of another number")
	s.Equal("agent:omni-sms", aside(), "a row of another number")
	s.Equal(linkedAt, s.boundary().UTC())

	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(-time.Hour)})
	s.Equal([]string{own}, s.readFor(chat), "joined, and bounded")
	s.Equal(s.omniOf(chat), aside())
	s.linkWith("+15550100", &PhoneHistory{All: true})
	s.ElementsMatch([]string{own, sunday, monday}, s.readFor(chat))
}

// Catches an unlink that leaves the number's set-aside rows on anya's omni-channel, so whoever
// has the number next reads her cards; and an unlink that keeps the history row.
func (s *StoreSuite) TestAnUnlinkTakesTheJoinedSetAsideRowsOffTheUser() {
	chat, own, _, _ := s.historyStory()
	s.linkWith("+15550100", &PhoneHistory{All: true})

	s.Require().NoError(s.store.UnlinkUserPhone(s.ctx, "acme-app", 4242, "anya"))

	s.Equal([]string{own}, s.readFor(chat))
	s.Zero(s.rows("SELECT count(*) FROM contact_map WHERE conversation_id = '" + s.omniOf(chat) + "' AND address LIKE 'retired:%'"))
	s.Zero(s.rows("SELECT count(*) FROM user_phone_history"))
}

// The thread history's boundary: none for a thread of a number linked to nobody or by another
// customer, the link's for a linked number's, read through a row the link set aside too, and
// none for a link with all of it.
func (s *StoreSuite) TestAThreadsHistoryIsCutAtItsNumbersLink() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true")
	s.Require().NoError(err)
	thread := "messaging:thread-" + newID()
	open := func(number string) {
		row := s.made(contact(number, "agent:omni-"+newID()))
		episode := &Episode{CustomerID: "acme-app", ContactID: row.ID, Source: EpisodeSMS, ThreadChannel: thread,
			StartedAt: linkedAt.Add(-time.Hour)}
		_, err := s.store.OpenEpisode(s.ctx, episode)
		s.Require().NoError(err)
	}
	since := func() (time.Time, bool) {
		at, ok, err := s.store.ThreadHistorySince(s.ctx, "acme-app", thread)
		s.Require().NoError(err)
		return at.UTC(), ok
	}
	open("+15550100")
	s.linkDated(&UserPhone{CustomerID: "another-customer", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"})
	s.linkDated(&UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "anya", Phone: "+15550100"})
	_, cut := since()
	s.False(cut, "linked to nobody of the customer in the row's app")

	s.linkDated(&UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"})
	at, cut := since()
	s.True(cut, "read through the row set aside")
	s.Equal(linkedAt, at)
	s.linkWith("+15550100", &PhoneHistory{All: true})
	_, cut = since()
	s.False(cut, "all of it")
	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(-time.Minute)})
	at, cut = since()
	s.True(cut)
	s.Equal(linkedAt.Add(-time.Minute), at)
}
