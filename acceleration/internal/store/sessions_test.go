//go:build integration

package store

import (
	"time"
)

// opened records a session, defaulting the fields a test does not care about.
func (s *StoreSuite) opened(id, customerID string, at time.Time, change func(*AgentSession)) *AgentSession {
	session := &AgentSession{
		ID:         id,
		CustomerID: customerID,
		AgentName:  "docs",
		ConfigID:   "cfg-docs",
		UserID:     "jlahey",
		CallerKind: "user",
		State:      SessionRunning,
		CreatedAt:  at,
	}
	if change != nil {
		change(session)
	}
	s.Require().NoError(s.store.SaveSession(s.ctx, session))
	return session
}

func (s *StoreSuite) TestSavingASessionTwiceIsOneConversation() {
	s.opened("one", "app", s.base, func(session *AgentSession) { session.Title = "first" })
	s.opened("one", "app", s.base, func(session *AgentSession) { session.Title = "renamed" })

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Equal("renamed", found.Title)

	// The retry is the same conversation, not a second one.
	all, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{})
	s.Require().NoError(err)
	s.Len(all, 1)
}

func (s *StoreSuite) TestASessionIsOnlyItsOwnCustomers() {
	s.opened("mine", "app", s.base, nil)

	_, err := s.store.StoredSession(s.ctx, "somebody-else", "mine")
	s.Require().Error(err)

	theirs, err := s.store.QuerySessions(s.ctx, "somebody-else", SessionFilter{})
	s.Require().NoError(err)
	s.Empty(theirs)
}

func (s *StoreSuite) TestQueryNarrowsToOnePerson() {
	s.opened("mine", "app", s.base, func(session *AgentSession) { session.UserID = "jlahey" })
	s.opened("theirs", "app", s.base.Add(time.Minute), func(session *AgentSession) { session.UserID = "randy" })

	found, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{UserID: "jlahey"})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("mine", found[0].ID)
}

func (s *StoreSuite) TestQueryOrdersNewestFirst() {
	s.opened("older", "app", s.base, nil)
	s.opened("newer", "app", s.base.Add(time.Hour), nil)

	found, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{})
	s.Require().NoError(err)
	s.Require().Len(found, 2)
	s.Equal("newer", found[0].ID)
	s.Equal("older", found[1].ID)
}

func (s *StoreSuite) TestQueryNarrowsByProjectAndAgentAndState() {
	s.opened("health", "app", s.base, func(session *AgentSession) { session.Project = "Health" })
	s.opened("docs", "app", s.base.Add(time.Minute), func(session *AgentSession) { session.Project = "Docs" })
	s.opened("other-agent", "app", s.base.Add(2*time.Minute), func(session *AgentSession) {
		session.AgentName = "sales"
		session.Project = "Health"
	})
	s.Require().NoError(s.store.CloseSession(s.ctx, "docs", s.base.Add(time.Hour)))

	byProject, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Project: "Health"})
	s.Require().NoError(err)
	s.Len(byProject, 2)

	byAgent, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{AgentName: "docs"})
	s.Require().NoError(err)
	s.Len(byAgent, 2)

	running, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{State: SessionRunning})
	s.Require().NoError(err)
	s.Len(running, 2)

	closed, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{State: SessionClosed})
	s.Require().NoError(err)
	s.Require().Len(closed, 1)
	s.Equal("docs", closed[0].ID)
}

func (s *StoreSuite) TestQueryNarrowsByCustomLabels() {
	s.opened("both", "app", s.base, func(session *AgentSession) {
		session.Custom = map[string]any{"tenant": "acme", "plan": "pro"}
	})
	s.opened("one", "app", s.base.Add(time.Minute), func(session *AgentSession) {
		session.Custom = map[string]any{"tenant": "acme"}
	})

	// Containment, so asking for two labels wants the session carrying both.
	found, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{
		Custom: map[string]string{"tenant": "acme", "plan": "pro"},
	})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("both", found[0].ID)
}

func (s *StoreSuite) TestQueryPages() {
	for i, id := range []string{"a", "b", "c"} {
		s.opened(id, "app", s.base.Add(time.Duration(i)*time.Minute), nil)
	}

	first, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Limit: 2})
	s.Require().NoError(err)
	s.Require().Len(first, 2)
	s.Equal("c", first[0].ID)

	second, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{Limit: 2, Offset: 2})
	s.Require().NoError(err)
	s.Require().Len(second, 1)
	s.Equal("a", second[0].ID)
}

func (s *StoreSuite) TestSearchFindsWhatTheCallerNamedIt() {
	s.opened("billing", "app", s.base, func(session *AgentSession) {
		session.Title = "Refund for a duplicate charge"
		session.Description = "customer was billed twice in March"
	})
	s.opened("paginate", "app", s.base.Add(time.Minute), func(session *AgentSession) {
		session.Title = "Paginating a channel list"
	})

	found, err := s.store.SearchSessions(s.ctx, "app", "refund", SessionFilter{})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("billing", found[0].ID)

	// Stemmed, so the word a person types need not be the word they wrote.
	stemmed, err := s.store.SearchSessions(s.ctx, "app", "billed twice", SessionFilter{})
	s.Require().NoError(err)
	s.Require().Len(stemmed, 1)
	s.Equal("billing", stemmed[0].ID)
}

func (s *StoreSuite) TestSearchRanksTitlesAboveProjects() {
	s.opened("named", "app", s.base, func(session *AgentSession) { session.Title = "Health check" })
	s.opened("grouped", "app", s.base.Add(time.Minute), func(session *AgentSession) {
		session.Project = "Health"
		session.Title = "Something else"
	})

	found, err := s.store.SearchSessions(s.ctx, "app", "health", SessionFilter{})
	s.Require().NoError(err)
	s.Require().Len(found, 2)
	s.Equal("named", found[0].ID, "the conversation called Health should beat one merely in that project")
}

func (s *StoreSuite) TestSearchTakesPunctuationWithoutComplaining() {
	s.opened("apostrophe", "app", s.base, func(session *AgentSession) {
		session.Title = "Randy's trailer park questions"
	})

	// A search box takes whatever somebody types, so a quote must not become a syntax error.
	found, err := s.store.SearchSessions(s.ctx, "app", "randy's \"trailer park\"", SessionFilter{})
	s.Require().NoError(err)
	s.Len(found, 1)
}

func (s *StoreSuite) TestSearchWithoutWordsIsEverySession() {
	s.opened("one", "app", s.base, nil)
	s.opened("two", "app", s.base.Add(time.Minute), nil)

	found, err := s.store.SearchSessions(s.ctx, "app", "", SessionFilter{})
	s.Require().NoError(err)
	s.Len(found, 2)
}

func (s *StoreSuite) TestSearchHonoursTheSameFiltersAsQuery() {
	s.opened("mine", "app", s.base, func(session *AgentSession) {
		session.UserID = "jlahey"
		session.Title = "shared subject"
	})
	s.opened("theirs", "app", s.base.Add(time.Minute), func(session *AgentSession) {
		session.UserID = "randy"
		session.Title = "shared subject"
	})

	found, err := s.store.SearchSessions(s.ctx, "app", "shared", SessionFilter{UserID: "jlahey"})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("mine", found[0].ID)
}

func (s *StoreSuite) TestClosingASessionKeepsTheFirstTime() {
	s.opened("one", "app", s.base, nil)
	first := s.base.Add(time.Hour)
	s.Require().NoError(s.store.CloseSession(s.ctx, "one", first))
	s.Require().NoError(s.store.CloseSession(s.ctx, "one", first.Add(time.Hour)))

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().NotNil(found.ClosedAt)
	s.WithinDuration(first, *found.ClosedAt, time.Second)
	s.Equal(SessionClosed, found.State)
}

func (s *StoreSuite) TestDescribingASessionLongAfterItEnded() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "one", s.base.Add(time.Hour)))

	err := s.store.DescribeSession(s.ctx, "app", "one", "Named later", "by a person", map[string]any{"pinned": true})
	s.Require().NoError(err)

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Equal("Named later", found.Title)
	s.Equal("by a person", found.Description)
	s.Equal(true, found.Custom["pinned"])

	// Findable by the name it was given after the fact.
	searched, err := s.store.SearchSessions(s.ctx, "app", "named later", SessionFilter{})
	s.Require().NoError(err)
	s.Len(searched, 1)
}

func (s *StoreSuite) TestDescribingSomebodyElsesSessionFails() {
	s.opened("one", "app", s.base, nil)
	s.Require().Error(s.store.DescribeSession(s.ctx, "somebody-else", "one", "mine now", "", nil))
}

func (s *StoreSuite) TestModelOverwritesSurviveARoundTrip() {
	temperature := 0.2
	tokens := 4096
	s.opened("one", "app", s.base, func(session *AgentSession) {
		session.ModelOverwrites = ModelOverwrites{
			LLM: "llm-thinking", Thinking: "high",
			Temperature: &temperature, MaxOutputTokens: &tokens,
		}
	})

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Equal("llm-thinking", found.ModelOverwrites.LLM)
	s.Equal("high", found.ModelOverwrites.Thinking)
	s.Require().NotNil(found.ModelOverwrites.Temperature)
	s.InDelta(0.2, *found.ModelOverwrites.Temperature, 0.0001)
	s.Require().NotNil(found.ModelOverwrites.MaxOutputTokens)
	s.Equal(4096, *found.ModelOverwrites.MaxOutputTokens)
}

func (s *StoreSuite) TestAForkRemembersWhereItCameFrom() {
	s.opened("parent", "app", s.base, nil)
	s.opened("child", "app", s.base.Add(time.Minute), func(session *AgentSession) {
		session.ForkedFrom = "parent"
	})

	found, err := s.store.StoredSession(s.ctx, "app", "child")
	s.Require().NoError(err)
	s.Equal("parent", found.ForkedFrom)
}

// responded records one finished turn and its items.
func (s *StoreSuite) responded(sessionID, responseID, said string, at time.Time, kinds ...string) {
	s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
		ID: responseID, SessionID: sessionID, CustomerID: "app", Said: said, CreatedAt: at,
	}))
	items := make([]AgentResponseItem, 0, len(kinds))
	for i, kind := range kinds {
		items = append(items, AgentResponseItem{
			ResponseID: responseID, Ordinal: i, SessionID: sessionID,
			Kind: kind, Text: kind + " text", At: at.Add(time.Duration(i) * time.Second),
		})
	}
	s.Require().NoError(s.store.AppendResponseItems(s.ctx, items))
	s.Require().NoError(s.store.FinishResponse(s.ctx, responseID, ResponseCompleted, "", at.Add(time.Minute)))
}

func (s *StoreSuite) TestItemsUnwindInTheOrderTheyHappened() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "Is Stream better than Sendbird?", s.base, ItemSaid, ItemToolCall, ItemToolResult, ItemAnswer)
	s.responded("one", "second", "And cheaper?", s.base.Add(time.Hour), ItemSaid, ItemAnswer)

	items, err := s.store.SessionItems(s.ctx, "app", "one", "", 0, 0)
	s.Require().NoError(err)
	s.Require().Len(items, 6)
	kinds := make([]string, 0, len(items))
	for _, item := range items {
		kinds = append(kinds, item.Kind)
	}
	s.Equal([]string{ItemSaid, ItemToolCall, ItemToolResult, ItemAnswer, ItemSaid, ItemAnswer}, kinds)

	// One turn's worth, for a caller watching a single response.
	scoped, err := s.store.SessionItems(s.ctx, "app", "one", "second", 0, 0)
	s.Require().NoError(err)
	s.Len(scoped, 2)
}

func (s *StoreSuite) TestItemsAreOnlyTheirOwnCustomers() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "hello", s.base, ItemSaid, ItemAnswer)

	items, err := s.store.SessionItems(s.ctx, "somebody-else", "one", "", 0, 0)
	s.Require().NoError(err)
	s.Empty(items)
}

func (s *StoreSuite) TestWritingTheSameItemTwiceIsOneItem() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
		ID: "first", SessionID: "one", CustomerID: "app", CreatedAt: s.base,
	}))
	item := AgentResponseItem{
		ResponseID: "first", Ordinal: 0, SessionID: "one", Kind: ItemAnswer, Text: "once", At: s.base,
	}
	s.Require().NoError(s.store.AppendResponseItems(s.ctx, []AgentResponseItem{item}))
	item.Text = "again"
	s.Require().NoError(s.store.AppendResponseItems(s.ctx, []AgentResponseItem{item}))

	items, err := s.store.SessionItems(s.ctx, "app", "one", "", 0, 0)
	s.Require().NoError(err)
	s.Require().Len(items, 1)
	s.Equal("once", items[0].Text, "the first write wins, so a retried flush does not rewrite history")
}

func (s *StoreSuite) TestFinishingAResponseMovesTheSessionsClock() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "hello", s.base, ItemSaid, ItemAnswer)

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().NotNil(found.LastResponseAt)
	s.WithinDuration(s.base.Add(time.Minute), *found.LastResponseAt, time.Second)
}

func (s *StoreSuite) TestAnInterruptedResponseIsCancelledRatherThanFailed() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
		ID: "first", SessionID: "one", CustomerID: "app", CreatedAt: s.base,
	}))
	s.Require().NoError(s.store.FinishResponse(s.ctx, "first", ResponseCancelled, "", s.base.Add(time.Second)))

	responses, err := s.store.SessionResponses(s.ctx, "app", "one", 0, 0)
	s.Require().NoError(err)
	s.Require().Len(responses, 1)
	s.Equal(ResponseCancelled, responses[0].Status)
}

func (s *StoreSuite) TestDeletingASessionTakesItsTurnsWithIt() {
	s.opened("one", "app", s.base, nil)
	s.responded("one", "first", "hello", s.base, ItemSaid, ItemAnswer)

	_, err := s.store.DB().NewDelete().Model((*AgentSession)(nil)).Where("id = ?", "one").Exec(s.ctx)
	s.Require().NoError(err)

	items, err := s.store.SessionItems(s.ctx, "app", "one", "", 0, 0)
	s.Require().NoError(err)
	s.Empty(items)
}

func (s *StoreSuite) TestGettingAGuestTwiceIsOneGuest() {
	guest := &GuestUser{ID: "guest-1", CustomerID: "app", Name: "Guest", CreatedAt: s.base}
	s.Require().NoError(s.store.RecordGuest(s.ctx, guest))
	s.Require().NoError(s.store.RecordGuest(s.ctx, &GuestUser{ID: "guest-1", CustomerID: "app", Name: "Renamed"}))

	found, err := s.store.Guest(s.ctx, "app", "guest-1")
	s.Require().NoError(err)
	s.Equal("Guest", found.Name, "coming back is the same person, not a new one")
}

func (s *StoreSuite) TestClaimingAGuestMovesTheirConversations() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &GuestUser{ID: "guest-1", CustomerID: "app", CreatedAt: s.base}))
	s.opened("theirs", "app", s.base, func(session *AgentSession) {
		session.UserID = "guest-1"
		session.CallerKind = "guest"
	})
	s.opened("somebody-elses", "app", s.base, func(session *AgentSession) { session.UserID = "randy" })

	moved, err := s.store.ClaimGuest(s.ctx, "app", "guest-1", "jlahey")
	s.Require().NoError(err)
	s.Equal(int64(1), moved)

	found, err := s.store.StoredSession(s.ctx, "app", "theirs")
	s.Require().NoError(err)
	s.Equal("jlahey", found.UserID)

	// Now findable as the account's own history.
	mine, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{UserID: "jlahey"})
	s.Require().NoError(err)
	s.Len(mine, 1)

	guest, err := s.store.Guest(s.ctx, "app", "guest-1")
	s.Require().NoError(err)
	s.Equal("jlahey", guest.ClaimedBy)
	s.Require().NotNil(guest.ClaimedAt)
}

func (s *StoreSuite) TestAGuestIsClaimedOnce() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &GuestUser{ID: "guest-1", CustomerID: "app", CreatedAt: s.base}))
	_, err := s.store.ClaimGuest(s.ctx, "app", "guest-1", "jlahey")
	s.Require().NoError(err)

	// A second claim naming somebody else would move one person's conversations onto
	// another's account.
	_, err = s.store.ClaimGuest(s.ctx, "app", "guest-1", "randy")
	s.Require().Error(err)
	s.Contains(err.Error(), "already claimed by jlahey")
}

func (s *StoreSuite) TestClaimingSomethingThatWasNeverAGuestFails() {
	_, err := s.store.ClaimGuest(s.ctx, "app", "not-a-guest", "jlahey")
	s.Require().Error(err)
	s.Contains(err.Error(), "is not a guest of app")
}

func (s *StoreSuite) TestAGuestCannotBeClaimedByItself() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &GuestUser{ID: "guest-1", CustomerID: "app", CreatedAt: s.base}))
	_, err := s.store.ClaimGuest(s.ctx, "app", "guest-1", "guest-1")
	s.Require().Error(err)
}

func (s *StoreSuite) TestAGuestOfAnotherAppIsNotClaimable() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &GuestUser{ID: "guest-1", CustomerID: "app", CreatedAt: s.base}))
	_, err := s.store.ClaimGuest(s.ctx, "somebody-else", "guest-1", "jlahey")
	s.Require().Error(err)
}
