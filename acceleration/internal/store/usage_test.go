//go:build integration

package store

import (
	"fmt"
	"time"
)

// spent records a request that cost money, which is what the spend paths aggregate.
func (s *StoreSuite) spent(
	customerID, modality string,
	at time.Time,
	costMicros int64,
	tags map[string]string,
) {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality:   modality,
		CustomerID: customerID,
		Provider:   "openai",
		Model:      "gpt-5.6-sol",
		Tags:       tags,
		StartedAt:  at,
		CostMicros: costMicros,
		Success:    true,
	}))
}

func (s *StoreSuite) TestSpendCoversEveryModalityAtOnce() {
	s.spent("acme", "llm", s.base.Add(time.Minute), 1200, nil)
	s.spent("acme", "tts", s.base.Add(2*time.Minute), 800, nil)
	s.spent("acme", "llm", s.base.Add(26*time.Hour), 400, nil)

	buckets, err := s.store.CustomerSpend(
		s.ctx, "acme", "modality", Daily, s.base, s.base.Add(48*time.Hour), 6, nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 3)

	firstDay := time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC)
	s.Equal(firstDay, buckets[0].Bucket.UTC())
	s.Equal("llm", buckets[0].Value, "the biggest group of the day comes first")
	s.EqualValues(1200, buckets[0].CostMicrosTotal)
	s.Equal(firstDay, buckets[1].Bucket.UTC())
	s.Equal("tts", buckets[1].Value)
	s.EqualValues(800, buckets[1].CostMicrosTotal)
	s.Equal(firstDay.Add(24*time.Hour), buckets[2].Bucket.UTC())
	s.EqualValues(400, buckets[2].CostMicrosTotal)
}

func (s *StoreSuite) TestSpendKeepsTheBiggestGroupsAndSumsTheRestIntoOther() {
	// A label naming an end customer has as many values as the customer has customers, so
	// a chart of every one of them says nothing. What is left has to still add up.
	for i, cost := range []int64{5000, 4000, 3000, 2000, 1000} {
		s.spent("acme", "llm", s.base.Add(time.Duration(i)*time.Minute), cost,
			map[string]string{"customer_id": fmt.Sprintf("c-%d", i)})
	}
	s.spent("acme", "llm", s.base.Add(10*time.Minute), 900, nil)

	buckets, err := s.store.CustomerSpend(
		s.ctx, "acme", "customer_id", Daily, s.base, s.base.Add(24*time.Hour), 2, nil)
	s.Require().NoError(err)

	byValue := map[string]int64{}
	for _, bucket := range buckets {
		byValue[bucket.Value] += bucket.CostMicrosTotal
	}
	s.Equal(map[string]int64{"c-0": 5000, "c-1": 4000, "other": 6000, "": 900}, byValue,
		"the two biggest keep a group, the rest are other, and the unlabelled spend is its own")
}

func (s *StoreSuite) TestSpendCanBeNarrowedToOneLabel() {
	s.spent("acme", "llm", s.base.Add(time.Minute), 1000,
		map[string]string{"product": "support", "environment": "production"})
	s.spent("acme", "llm", s.base.Add(2*time.Minute), 9000,
		map[string]string{"product": "support", "environment": "staging"})

	buckets, err := s.store.CustomerSpend(s.ctx, "acme", "product", Daily,
		s.base, s.base.Add(24*time.Hour), 6, map[string]string{"environment": "production"})
	s.Require().NoError(err)

	s.Require().Len(buckets, 1)
	s.Equal("support", buckets[0].Value)
	s.EqualValues(1000, buckets[0].CostMicrosTotal, "only the production request counts")
}

func (s *StoreSuite) TestSpendIsOnlyTheCallingCustomers() {
	s.spent("acme", "llm", s.base.Add(time.Minute), 1000, nil)
	s.spent("globex", "llm", s.base.Add(time.Minute), 9000, nil)

	buckets, err := s.store.CustomerSpend(
		s.ctx, "acme", "modality", Daily, s.base, s.base.Add(24*time.Hour), 6, nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1000, buckets[0].CostMicrosTotal)
}

func (s *StoreSuite) TestTagKeysTellABreakdownFromAPieceOfContext() {
	// environment is on everything with one value, which is context. product is what the
	// spend actually splits by, and only one of the two is worth charting.
	s.spent("acme", "llm", s.base.Add(time.Minute), 6000,
		map[string]string{"product": "support", "environment": "production"})
	s.spent("acme", "llm", s.base.Add(2*time.Minute), 2000,
		map[string]string{"product": "sales", "environment": "production"})
	s.spent("acme", "llm", s.base.Add(3*time.Minute), 1000, nil)

	keys, err := s.store.CustomerTagKeys(s.ctx, "acme", s.base, s.base.Add(24*time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(keys, 2)

	byKey := map[string]TagKeySummary{}
	for _, key := range keys {
		byKey[key.Key] = key
	}

	product := byKey["product"]
	s.EqualValues(2, product.ValueCount)
	s.EqualValues(8000, product.CostMicrosTotal)
	s.EqualValues(2, product.RequestCount)
	s.InDelta(2.0/3.0, product.Coverage, 0.001, "one of the three requests carries no label at all")
	s.Require().Len(product.TopValues, 2)
	s.Equal("support", product.TopValues[0].Value, "biggest spend first")
	s.InDelta(0.75, product.TopValues[0].Share, 0.001)

	s.EqualValues(1, byKey["environment"].ValueCount, "one value is context, not a breakdown")
}

func (s *StoreSuite) TestTagKeysReportOnlyTheLargestValues() {
	for i := range 15 {
		s.spent("acme", "llm", s.base.Add(time.Duration(i)*time.Minute), int64(100*(i+1)),
			map[string]string{"customer_id": fmt.Sprintf("c-%02d", i)})
	}

	keys, err := s.store.CustomerTagKeys(s.ctx, "acme", s.base, s.base.Add(24*time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(keys, 1)
	s.EqualValues(15, keys[0].ValueCount, "the count is of every value, not of the ones listed")
	s.Len(keys[0].TopValues, 10)
	s.Equal("c-14", keys[0].TopValues[0].Value)
}

func (s *StoreSuite) TestTagKeysAreEmptyWhenNothingWasRecorded() {
	keys, err := s.store.CustomerTagKeys(s.ctx, "acme", s.base, s.base.Add(24*time.Hour), nil)
	s.Require().NoError(err)
	s.Empty(keys)
}

func (s *StoreSuite) TestActivityCountsWhatWasAskedOfTheAgents() {
	session := s.opened("one", "acme", s.base, nil)
	for i := range 3 {
		s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
			ID:         fmt.Sprintf("r-%d", i),
			SessionID:  session.ID,
			CustomerID: "acme",
			Said:       "how much is it",
			CreatedAt:  s.base.Add(time.Duration(i) * time.Minute),
		}))
	}

	buckets, err := s.store.CustomerActivity(
		s.ctx, "acme", ActivityDaily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].Sessions)
	s.EqualValues(3, buckets[0].Messages)
	s.EqualValues(1, buckets[0].ActiveUsers)
}

func (s *StoreSuite) TestAClaimedGuestIsTheSamePersonAsTheUserWhoClaimedThem() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &User{
		ID: "guest-1", CustomerID: "acme", Name: "Randy", CreatedAt: s.base,
	}))
	_, err := s.store.ClaimGuest(s.ctx, "acme", "guest-1", "randy")
	s.Require().NoError(err)

	s.opened("as-a-guest", "acme", s.base.Add(time.Minute), func(session *AgentSession) {
		session.UserID = "guest-1"
		session.CallerKind = "guest"
	})
	s.opened("signed-in", "acme", s.base.Add(2*time.Minute), func(session *AgentSession) {
		session.UserID = "randy"
	})

	buckets, err := s.store.CustomerActivity(
		s.ctx, "acme", ActivityDaily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(2, buckets[0].Sessions)
	s.EqualValues(1, buckets[0].ActiveUsers, "two sessions, one person")
}

func (s *StoreSuite) TestAnAnonymousNameIsNotCountedAsAPerson() {
	// An anonymous name is a claim nothing verified, so counting it would make guessing a
	// name enough to inflate this.
	s.opened("claimed", "acme", s.base.Add(time.Minute), func(session *AgentSession) {
		session.UserID = "whoever"
		session.CallerKind = "anonymous"
	})
	s.opened("nameless", "acme", s.base.Add(2*time.Minute), func(session *AgentSession) {
		session.UserID = ""
		session.CallerKind = "anonymous"
	})
	s.opened("verified", "acme", s.base.Add(3*time.Minute), func(session *AgentSession) {
		session.UserID = "randy"
		session.CallerKind = "authenticated"
	})

	buckets, err := s.store.CustomerActivity(
		s.ctx, "acme", ActivityDaily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(3, buckets[0].Sessions)
	s.EqualValues(1, buckets[0].ActiveUsers)
}

func (s *StoreSuite) TestAMonthOfUsersIsWhoCameBackRatherThanTheSumOfItsDays() {
	s.opened("monday", "acme", s.base, func(session *AgentSession) { session.UserID = "randy" })
	s.opened("tuesday", "acme", s.base.Add(24*time.Hour), func(session *AgentSession) {
		session.UserID = "randy"
	})
	s.opened("wednesday", "acme", s.base.Add(48*time.Hour), func(session *AgentSession) {
		session.UserID = "lahey"
	})

	from, to := s.base, s.base.Add(72*time.Hour)

	daily, err := s.store.CustomerActivity(s.ctx, "acme", ActivityDaily, from, to)
	s.Require().NoError(err)
	s.Require().Len(daily, 3)
	var summed int64
	for _, bucket := range daily {
		summed += bucket.ActiveUsers
	}
	s.EqualValues(3, summed)

	monthly, err := s.store.CustomerActivity(s.ctx, "acme", ActivityMonthly, from, to)
	s.Require().NoError(err)
	s.Require().Len(monthly, 1)
	s.EqualValues(2, monthly[0].ActiveUsers, "randy on two days is one person for the month")
}

func (s *StoreSuite) TestPhoneMinutesAreThePartOfTheTalkingThatArrivedOnANumber() {
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{
		ID: "over-the-web", CustomerID: "acme", CallID: "c-1", AgentID: "a-1", StartedAt: s.base,
	}))
	s.Require().NoError(s.store.FinishCall(s.ctx, "over-the-web", s.base.Add(5*time.Minute)))

	s.Require().NoError(s.store.StartCall(s.ctx, &Call{
		ID: "over-a-number", CustomerID: "acme", CallID: "c-2", AgentID: "a-2",
		FromNumber: "+15125551234", StartedAt: s.base.Add(time.Minute),
	}))
	s.Require().NoError(s.store.FinishCall(s.ctx, "over-a-number", s.base.Add(4*time.Minute)))

	buckets, err := s.store.CustomerActivity(
		s.ctx, "acme", ActivityDaily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(2, buckets[0].Calls)
	s.InDelta(8, buckets[0].VoiceMinutes, 0.001)
	s.InDelta(3, buckets[0].PhoneMinutes, 0.001, "only the call that arrived on a number")
}

func (s *StoreSuite) TestActivityIsOnlyTheCallingCustomers() {
	s.opened("mine", "acme", s.base, nil)
	s.opened("theirs", "globex", s.base, nil)

	buckets, err := s.store.CustomerActivity(
		s.ctx, "acme", ActivityDaily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].Sessions)
}

// ranTool records one tool call and its result, the way a session writes them, on a
// response of its own.
func (s *StoreSuite) ranTool(session *AgentSession, tool string, at time.Time, took time.Duration, result string, failed bool) {
	responseID := fmt.Sprintf("%s-%s-%d", session.ID, tool, at.UnixNano())
	s.Require().NoError(s.store.StartResponse(s.ctx, &AgentResponse{
		ID: responseID, SessionID: session.ID, CustomerID: session.CustomerID, CreatedAt: at,
	}))
	payload := map[string]any{"call_id": "call-1", "arguments": `{"q":"x"}`}
	if failed {
		payload["error"] = "it broke"
	}
	s.Require().NoError(s.store.AppendResponseItems(s.ctx, []AgentResponseItem{
		{ResponseID: responseID, Ordinal: 0, SessionID: session.ID, Kind: ItemToolCall,
			ToolName: tool, Payload: map[string]any{"call_id": "call-1"}, At: at},
		{ResponseID: responseID, Ordinal: 1, SessionID: session.ID, Kind: ItemToolResult,
			ToolName: tool, Text: result, Payload: payload, At: at.Add(took)},
	}))
}

func (s *StoreSuite) TestThePeakIsTheMostSessionsOpenAtOnce() {
	s.opened("first", "acme", s.base, nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "first", s.base.Add(10*time.Minute)))
	s.opened("second", "acme", s.base.Add(2*time.Minute), nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "second", s.base.Add(5*time.Minute)))
	s.opened("third", "acme", s.base.Add(3*time.Minute), nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "third", s.base.Add(4*time.Minute)))
	s.opened("later", "acme", s.base.Add(20*time.Minute), nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "later", s.base.Add(21*time.Minute)))

	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.EqualValues(3, peak.Sessions)
	s.Require().NotNil(peak.At)
	s.True(peak.At.Equal(s.base.Add(3 * time.Minute)))
}

func (s *StoreSuite) TestASessionThatClosedAsAnotherOpenedDidNotOverlapIt() {
	s.opened("before", "acme", s.base, nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "before", s.base.Add(time.Minute)))
	s.opened("after", "acme", s.base.Add(time.Minute), nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "after", s.base.Add(2*time.Minute)))

	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.EqualValues(1, peak.Sessions)
}

func (s *StoreSuite) TestASessionOpenedBeforeTheWindowStillCountsInIt() {
	s.opened("overnight", "acme", s.base.Add(-time.Hour), nil)
	s.opened("morning", "acme", s.base.Add(time.Minute), nil)

	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.EqualValues(2, peak.Sessions, "both still running")
}

func (s *StoreSuite) TestThereIsNoPeakWhenNothingRan() {
	s.opened("theirs", "globex", s.base, nil)

	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Zero(peak.Sessions)
	s.Nil(peak.At)
}

func (s *StoreSuite) TestToolUsageIsMostCalledFirstWithHowLongEachTook() {
	session := s.opened("one", "acme", s.base, nil)
	s.ranTool(session, "lookup_order", s.base.Add(time.Minute), 100*time.Millisecond, "shipped", false)
	s.ranTool(session, "lookup_order", s.base.Add(2*time.Minute), 300*time.Millisecond, "shipped", false)
	s.ranTool(session, "refund", s.base.Add(3*time.Minute), 2*time.Second, "", true)

	overall, tools, err := s.store.CustomerToolUsage(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Require().Len(tools, 2)

	s.Equal("lookup_order", tools[0].Tool)
	s.EqualValues(2, tools[0].Calls)
	s.Zero(tools[0].Failed)
	s.Require().NotNil(tools[0].AvgMs)
	s.InDelta(200, *tools[0].AvgMs, 0.01)
	// lookup_order, {"q":"x"} and shipped are 12, 9 and 7 bytes: 28, at four a token.
	s.InDelta(7, tools[0].AvgTokens, 0.01)

	s.Equal("refund", tools[1].Tool)
	s.EqualValues(1, tools[1].Failed)

	s.Empty(overall.Tool)
	s.EqualValues(3, overall.Calls)
	s.EqualValues(1, overall.Failed)
	s.Require().NotNil(overall.P99Ms)
	s.Greater(*overall.P99Ms, 1900.0, "the slow call is the tail")
}

func (s *StoreSuite) TestToolUsageIsOnlyTheCallingCustomers() {
	theirs := s.opened("theirs", "globex", s.base, nil)
	s.ranTool(theirs, "lookup_order", s.base.Add(time.Minute), time.Second, "shipped", false)

	overall, tools, err := s.store.CustomerToolUsage(s.ctx, "acme", s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	s.Empty(tools)
	s.Zero(overall.Calls)
	s.Nil(overall.AvgMs)
}

func (s *StoreSuite) TestUsageQueriesRejectBadArguments() {
	_, err := s.store.CustomerSpend(s.ctx, "", "modality", Daily, s.base, s.base.Add(time.Hour), 6, nil)
	s.ErrorContains(err, "customer id is required")

	_, err = s.store.CustomerSpend(
		s.ctx, "acme", "modality", Granularity("weekly"), s.base, s.base.Add(time.Hour), 6, nil)
	s.ErrorContains(err, "unknown granularity")

	_, err = s.store.CustomerSpend(s.ctx, "acme", "modality", Daily, s.base, s.base.Add(time.Hour), 0, nil)
	s.ErrorContains(err, "limit must be at least 1")

	_, err = s.store.CustomerTagKeys(s.ctx, "", s.base, s.base.Add(time.Hour), nil)
	s.ErrorContains(err, "customer id is required")

	_, err = s.store.CustomerActivity(s.ctx, "acme", ActivityGranularity("weekly"), s.base, s.base.Add(time.Hour))
	s.ErrorContains(err, "unknown activity granularity")

	_, err = s.store.CustomerActivity(s.ctx, "", ActivityDaily, s.base, s.base.Add(time.Hour))
	s.ErrorContains(err, "customer id is required")
}
