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

// Catches a read of set-aside rows for a boundary after the link, which a since after it gives,
// or for a history row written for another link time, which a router that knows no history
// leaves; and set-aside rows moved onto anya's omni-channel, which a router that knows no
// boundary would read whole and would not take off her at an unlink.
func (s *StoreSuite) TestOnlyABoundaryBeforeTheLinkReadsTheSetAsideRows() {
	chat, own, sunday, monday := s.historyStory()
	s.linkDated(&UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"})

	aside := func() string {
		var omni string
		s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT conversation_id FROM contact_map WHERE address LIKE 'retired:%'").Scan(&omni))
		return omni
	}
	var setAside string
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT id FROM contact_map WHERE address LIKE 'retired:%'").Scan(&setAside))
	later := s.smsAt(&ContactMapEntry{ID: setAside, CustomerID: "acme-app"}, linkedAt.Add(2*time.Hour))
	s.Equal([]string{own}, s.readFor(chat), "the link's own time, as before (ba2c5e912)")
	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(time.Hour)})
	s.Equal([]string{own}, s.readFor(chat), "a boundary after the link")
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE user_phone_history SET linked_at = linked_at - interval '1 second', since = NULL")
	s.Require().NoError(err)
	s.linkWith("+15550100", nil)
	s.Equal([]string{own}, s.readFor(chat), "a row of another link time")
	s.Equal(linkedAt, s.boundary().UTC())

	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(-time.Hour)})
	s.ElementsMatch([]string{own, later}, s.readFor(chat), "read, and bounded")
	s.linkWith("+15550100", &PhoneHistory{All: true})
	s.ElementsMatch([]string{own, sunday, monday, later}, s.readFor(chat))
	s.Equal("agent:omni-sms", aside(), "the set-aside rows stay where they are")
}

// Catches a read of another app's set-aside rows of the number: carl of app 7 linked it first,
// with all of its past, which set aside its row there; the number then texted on app 4242, and
// anya of app 4242 reads only her app's past of it.
func (s *StoreSuite) TestASetAsideRowOfAnotherAppIsNotRead() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true WHERE id = 'agent-one'")
	s.Require().NoError(err)
	chat := s.made(streamUser("anya", "agent:omni-chat"))
	own := s.chatEpisode(chat)
	seven := contact("+15550100", "agent:omni-sms-7")
	seven.StreamAppPK = 7
	s.smsAt(s.made(seven), linkedAt.Add(-36*time.Hour))
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", StreamAppPK: 7, UserID: "carl", Phone: "+15550100",
		History: &PhoneHistory{All: true}}))
	s.Require().Equal(1, s.rows("SELECT count(*) FROM contact_map WHERE address LIKE 'retired:%' AND stream_app_pk = 7"), "app 7's row is set aside")
	monday := s.smsAt(s.made(contact("+15550100", "agent:omni-sms")), linkedAt.Add(-12*time.Hour))

	s.linkWith("+15550100", &PhoneHistory{All: true})

	s.ElementsMatch([]string{own, monday}, s.readFor(chat))
}

// Catches a read of the set-aside rows that finds the link only by the Stream user's row: anya
// never chatted with the agent, and the number's next row reads its past with all of it.
func (s *StoreSuite) TestALinkWithAllOfTheHistoryIsReadOnTheNumbersNextRow() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true WHERE id = 'agent-one'")
	s.Require().NoError(err)
	sunday := s.smsAt(s.made(contact("+15550100", "agent:omni-sms")), linkedAt.Add(-36*time.Hour))
	s.linkWith("+15550100", &PhoneHistory{All: true})

	s.Equal([]string{sunday}, s.readFor(s.made(contact("+15550100", "agent:omni-next"))))
}

// Catches a read of the set-aside rows that misses a link in the deployment's own app: the link
// names it 0, and the message hook's row of anya names it by its id.
func (s *StoreSuite) TestALinkWithAllOfTheHistoryInTheDeploymentsOwnApp() {
	s.store.SetStreamPins(StreamPins{Deployment: func() int64 { return 4242 }})
	s.T().Cleanup(func() { s.store.SetStreamPins(StreamPins{}) })
	chat, own, sunday, monday := s.historyStory()

	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &UserPhone{CustomerID: "acme-app", UserID: "anya", Phone: "+15550100",
		History: &PhoneHistory{All: true}}))

	s.ElementsMatch([]string{own, sunday, monday}, s.readFor(chat))
}

// Catches a link that leaves the router it replaced reading more than the user was given, on a
// rollback to it: the previous release's queries (v0.6.284, ba2c5e912) on the migrated schema.
// Its unlink detaches only the rows at the number, so set-aside rows moved onto anya's
// omni-channel would stay there; its cards read every row there, so the boundary would be lost.
func (s *StoreSuite) TestThePreviousReleaseReadsNoMoreThanTheLinkGives() {
	chat, own, _, monday := s.historyStory()
	s.linkWith("+15550100", &PhoneHistory{Since: linkedAt.Add(-24 * time.Hour)})
	s.Require().ElementsMatch([]string{own, monday}, s.readFor(chat))

	s.Equal([]string{own}, s.previousCards(s.omniOf(chat)), "its cards read none of the number's past")

	s.previousUnlink("+15550100")
	s.Equal([]string{own}, s.readFor(chat), "after its unlink, anya reads none of the number's cards")
	s.Equal([]string{own}, s.previousCards(s.omniOf(chat)))
}

// Catches a history row that outlives its link: the previous release moves anya's link to another
// number and back, which keeps the row; the link made again is from its own time on.
func (s *StoreSuite) TestALinkThePreviousReleaseMadeAgainIsFromItsTime() {
	chat, own, _, _ := s.historyStory()
	s.linkWith("+15550100", &PhoneHistory{All: true})

	s.previousRelink("+15550199")
	s.previousRelink("+15550100")

	var linked time.Time
	s.Require().NoError(s.store.db.QueryRowContext(s.ctx, "SELECT updated_at FROM user_phones WHERE user_id = 'anya'").Scan(&linked))
	s.Require().NotNil(s.boundary())
	s.Equal(linked, *s.boundary())
	s.Equal([]string{own}, s.readFor(chat))
}

// Catches the same stale history row on the thread read: after the previous release moves anya's
// link to another number and back, her number's thread is handed none of what it said before the
// link made again (ThreadHistorySince), as her cards read none of it.
func (s *StoreSuite) TestAThreadOfALinkThePreviousReleaseMadeAgainIsCutAtItsTime() {
	s.agentConfigs()
	_, err := s.store.db.ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = true")
	s.Require().NoError(err)
	thread := "messaging:thread-" + newID()
	row := s.made(contact("+15550100", "agent:omni-"+newID()))
	_, err = s.store.OpenEpisode(s.ctx, &Episode{CustomerID: "acme-app", ContactID: row.ID, Source: EpisodeSMS,
		ThreadChannel: thread, StartedAt: linkedAt.Add(-time.Hour)})
	s.Require().NoError(err)
	s.linkDated(&UserPhone{CustomerID: "acme-app", StreamAppPK: 4242, UserID: "anya", Phone: "+15550100"})
	s.linkWith("+15550100", &PhoneHistory{All: true})
	_, cut, err := s.store.ThreadHistorySince(s.ctx, "acme-app", thread)
	s.Require().NoError(err)
	s.Require().False(cut, "all of it")

	s.previousRelink("+15550199")
	s.previousRelink("+15550100")

	_, cut, err = s.store.ThreadHistorySince(s.ctx, "acme-app", thread)
	s.Require().NoError(err)
	s.True(cut, "cut at the link the previous release made again")
}

// previousCards is the previous release's EpisodeCards (ba2c5e912 episodes.go:384-412) for anya's
// agent: the rows at the omni-channel, less what a linked number said before its link's time.
func (s *StoreSuite) previousCards(omni string) []string {
	var found []string
	s.Require().NoError(s.store.db.NewRaw(`SELECT ep.id FROM episodes AS ep JOIN contact_map AS cm ON cm.id = ep.contact_id
WHERE ep.customer_id = 'acme-app' AND cm.customer_id = 'acme-app' AND cm.agent_config_id = 'agent-one' AND cm.conversation_id = ?
	AND NOT EXISTS (SELECT 1 FROM user_phones AS up WHERE up.customer_id = cm.customer_id
		AND up.stream_app_pk = COALESCE(NULLIF(cm.stream_app_pk, ?), 0) AND up.phone = cm.address
		AND ep.started_at < up.updated_at)
ORDER BY ep.started_at DESC`, omni, s.store.streamPins().deployment()).Scan(s.ctx, &found))
	return found
}

// previousUnlink is the previous release's UnlinkUserPhone of anya (ba2c5e912 user_phones.go:101
// and its detachLinked, :252): the link deleted, and only the rows at the number detached from
// her omni-channel.
func (s *StoreSuite) previousUnlink(phone string) {
	_, err := s.store.db.ExecContext(s.ctx, "DELETE FROM user_phones WHERE customer_id = 'acme-app' AND stream_app_pk = 4242 AND user_id = 'anya'")
	s.Require().NoError(err)
	_, err = s.store.db.ExecContext(s.ctx, `UPDATE contact_map AS cm
SET conversation_id = 'agent:omni-' || gen_random_uuid(), updated_at = now()
FROM contact_map AS su
WHERE su.customer_id = 'acme-app' AND su.kind = ? AND su.address IN (?)
	AND cm.customer_id = 'acme-app' AND cm.agent_config_id = su.agent_config_id AND cm.kind = ? AND cm.address = ?
	AND cm.conversation_id = su.conversation_id`, ContactStreamUser, StreamUserAddress(4242, "anya"), ContactPhone, phone)
	s.Require().NoError(err)
}

// previousRelink is the previous release's upsert of anya's link (ba2c5e912 user_phones.go:64-71):
// the number, and the link's time when the number is another.
func (s *StoreSuite) previousRelink(phone string) {
	_, err := s.store.db.ExecContext(s.ctx, `UPDATE user_phones SET phone = ?,
	updated_at = CASE WHEN phone = ? THEN updated_at ELSE statement_timestamp() END
WHERE customer_id = 'acme-app' AND stream_app_pk = 4242 AND user_id = 'anya'`, phone, phone)
	s.Require().NoError(err)
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
