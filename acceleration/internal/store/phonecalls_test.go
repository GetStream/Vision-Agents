//go:build integration

package store

import (
	"sync"
	"time"
)

// placedPhoneCall stores a phone call placed at s.base, defaulting the fields a test does not
// care about.
func (s *StoreSuite) placedPhoneCall(token string, change func(*PhoneCall)) PhoneCall {
	call := PhoneCall{
		Token:        token,
		CustomerID:   "app",
		CallType:     "default",
		CallID:       "call",
		Vendor:       "telnyx",
		LocalNumber:  "+15125550100",
		RemoteNumber: "+15125550199",
		Reports:      true,
		PlacedAt:     s.base,
	}
	if change != nil {
		change(&call)
	}
	s.Require().NoError(s.store.InsertPhoneCall(s.ctx, &call))
	return call
}

// unchanged reads a phone call without writing it.
func unchanged(*PhoneCall) bool { return false }

func (s *StoreSuite) TestAPhoneCallIsFoundByItsToken() {
	s.placedPhoneCall("one", nil)
	s.placedPhoneCall("two", func(call *PhoneCall) { call.CallID = "other" })

	found, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "two"}, unchanged)
	s.Require().NoError(err)
	s.Equal("other", found.CallID)
	s.Equal(s.base, found.PlacedAt.UTC())

	_, err = s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "three"}, unchanged)
	s.ErrorIs(err, ErrNoPhoneCall)
}

func (s *StoreSuite) TestAPhoneCallIsFoundByItsCallAndNumbers() {
	s.placedPhoneCall("first", nil)
	s.placedPhoneCall("again", nil)
	s.placedPhoneCall("transfer", func(call *PhoneCall) { call.RemoteNumber = "+15125550142" })
	where := PhoneCallWhere{CustomerID: "app", CallType: "default", CallID: "call", LocalNumber: "+15125550100", RemoteNumber: "+15125550199"}

	found, err := s.store.ChangePhoneCall(s.ctx, where, unchanged)
	s.Require().NoError(err)
	s.Equal("again", found.Token, "the newest match")

	where.StreamAppPK = 7
	_, err = s.store.ChangePhoneCall(s.ctx, where, unchanged)
	s.ErrorIs(err, ErrNoPhoneCall, "another Stream app's call of the same id")
}

func (s *StoreSuite) TestAChangeThatSaysNoWritesNothing() {
	s.placedPhoneCall("one", nil)

	_, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "one"}, func(call *PhoneCall) bool {
		call.Outcome = "busy"
		return false
	})
	s.Require().NoError(err)

	found, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "one"}, unchanged)
	s.Require().NoError(err)
	s.Empty(found.Outcome)
}

func (s *StoreSuite) TestTwoChangesAtOnceSeeEachOther() {
	s.placedPhoneCall("one", nil)

	var wg sync.WaitGroup
	for _, word := range []string{"a", "b"} {
		wg.Go(func() {
			_, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "one"}, func(call *PhoneCall) bool {
				call.VendorStatus += word
				return true
			})
			s.NoError(err)
		})
	}
	wg.Wait()

	found, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "one"}, unchanged)
	s.Require().NoError(err)
	s.ElementsMatch([]rune("ab"), []rune(found.VendorStatus), "neither write lost")
}

func (s *StoreSuite) TestDeletingAPhoneCallLeavesNoRow() {
	s.placedPhoneCall("one", nil)

	s.Require().NoError(s.store.DeletePhoneCall(s.ctx, "one"))

	calls, err := s.store.PhoneCalls(s.ctx, "app", []string{"call"})
	s.Require().NoError(err)
	s.Empty(calls)
}

func (s *StoreSuite) TestClearingTokensKeepsTheRows() {
	s.placedPhoneCall("old", func(call *PhoneCall) { call.CallID = "old" })
	s.placedPhoneCall("new", func(call *PhoneCall) {
		call.CallID = "new"
		call.PlacedAt = s.base.Add(time.Hour)
	})

	cleared, err := s.store.ClearPhoneCallTokens(s.ctx, s.base.Add(time.Minute))
	s.Require().NoError(err)
	s.EqualValues(1, cleared)

	_, err = s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "old"}, unchanged)
	s.ErrorIs(err, ErrNoPhoneCall)
	calls, err := s.store.PhoneCalls(s.ctx, "app", []string{"old", "new"})
	s.Require().NoError(err)
	s.Len(calls, 2)
}

func (s *StoreSuite) TestPhoneCallsReadsTheNewestPerCall() {
	s.placedPhoneCall("first", nil)
	s.placedPhoneCall("second", func(call *PhoneCall) { call.PlacedAt = s.base.Add(time.Minute) })
	s.placedPhoneCall("other", func(call *PhoneCall) { call.CallID = "other" })
	s.placedPhoneCall("theirs", func(call *PhoneCall) {
		call.CustomerID = "someone else"
		call.PlacedAt = s.base.Add(time.Hour)
	})

	calls, err := s.store.PhoneCalls(s.ctx, "app", []string{"call", "other", "unknown"})
	s.Require().NoError(err)
	tokens := map[string]string{}
	for _, call := range calls {
		tokens[call.CallID] = call.Token
	}
	s.Equal(map[string]string{"call": "second", "other": "other"}, tokens)
}

func (s *ShardedStoreSuite) TestAPhoneCallAnotherShardPlacedIsChangedThere() {
	call := PhoneCall{Token: "there", CustomerID: "app", CallType: "default", CallID: "call", Vendor: "telnyx",
		LocalNumber: "+15125550100", RemoteNumber: "+15125550199", PlacedAt: s.base}
	s.Require().NoError(s.other.InsertPhoneCall(s.ctx, &call))

	_, err := s.store.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "there"}, func(call *PhoneCall) bool {
		call.Outcome = "busy"
		return true
	})
	s.Require().NoError(err)

	found, err := s.other.ChangePhoneCall(s.ctx, PhoneCallWhere{Token: "there"}, unchanged)
	s.Require().NoError(err)
	s.Equal("busy", found.Outcome)
}
