//go:build integration

package store

import (
	"context"
	"sync"
	"time"
)

func pointerTo[T any](value T) *T { return &value }

func (s *StoreSuite) changeQuota(customerID, by string, messages, tokens *int64) AppQuotaChange {
	change, err := s.store.ChangeAppQuota(s.ctx, customerID, by, "a reason", &Quota{MessagesPerUserPerDay: messages, TokensPerUserPerDay: tokens})
	s.Require().NoError(err)
	return change
}

func (s *StoreSuite) quotaRows(customerID string) int {
	count, err := s.store.DB().NewSelect().Model((*AppQuotaChange)(nil)).Where("customer_id = ?", customerID).Count(s.ctx)
	s.Require().NoError(err)
	return count
}

func (s *StoreSuite) TestChangingAnAppsQuotaRecordsWhoWhenAndTheNumbers() {
	before := time.Now().UTC().Add(-time.Second)
	yes := true
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeApp, "app-1", PolicyDocument{PromptInjection: &yes}))

	change := s.changeQuota("app-1", "ops@getstream.io", pointerTo(int64(5)), nil)

	s.Equal("ops@getstream.io", change.ChangedBy)
	s.Equal("a reason", change.Reason)
	s.Equal("app-1", change.CustomerID)
	s.Nil(change.PreviousMessagesPerUserPerDay)
	s.Nil(change.PreviousTokensPerUserPerDay)
	s.Equal(pointerTo(int64(5)), change.NewMessagesPerUserPerDay)
	s.Nil(change.NewTokensPerUserPerDay)
	s.True(change.ChangedAt.After(before))
	s.Equal(1, s.quotaRows("app-1"))

	document, err := s.store.Policy(s.ctx, ScopeApp, "app-1")
	s.Require().NoError(err)
	s.Equal(&Quota{MessagesPerUserPerDay: pointerTo(int64(5))}, document.Quota)
	s.Equal(&yes, document.PromptInjection, "the rest of the policy is left as it was")
}

func (s *StoreSuite) TestASecondChangeRecordsTheFirstOnesNumbersAsItsPrevious() {
	first := s.changeQuota("app-1", "ops@getstream.io", pointerTo(int64(5)), pointerTo(int64(0)))
	s.Equal(pointerTo(int64(0)), first.NewTokensPerUserPerDay, "zero is no cap, kept apart from unset")
	second := s.changeQuota("app-1", "other@getstream.io", pointerTo(int64(9)), nil)

	s.Equal(pointerTo(int64(5)), second.PreviousMessagesPerUserPerDay)
	s.Equal(pointerTo(int64(0)), second.PreviousTokensPerUserPerDay, "zero is no cap, kept apart from unset")
	s.Equal(pointerTo(int64(9)), second.NewMessagesPerUserPerDay)
	s.Nil(second.NewTokensPerUserPerDay)

	cleared, err := s.store.ChangeAppQuota(s.ctx, "app-1", "ops@getstream.io", "", nil)
	s.Require().NoError(err)
	s.Equal(pointerTo(int64(9)), cleared.PreviousMessagesPerUserPerDay)
	s.Nil(cleared.NewMessagesPerUserPerDay, "no quota is every number unset")
}

func (s *StoreSuite) TestChangesMadeAtOnceFormOneChainOfPreviousAndNew() {
	const changes = 8
	var wait sync.WaitGroup
	for i := int64(1); i <= changes; i++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := s.store.ChangeAppQuota(s.ctx, "app-1", "ops@getstream.io", "", &Quota{MessagesPerUserPerDay: pointerTo(i)})
			s.NoError(err)
		}()
	}
	wait.Wait()

	listed, err := s.store.AppQuotaChanges(s.ctx, "app-1", changes, nil)
	s.Require().NoError(err)
	s.Require().Len(listed, changes)
	for i := 0; i < changes-1; i++ {
		s.Equal(listed[i+1].NewMessagesPerUserPerDay, listed[i].PreviousMessagesPerUserPerDay,
			"each change's previous is the one before its new")
	}
	s.Nil(listed[changes-1].PreviousMessagesPerUserPerDay)
}

func (s *StoreSuite) TestAChangeNobodyMadeIsRefused() {
	_, err := s.store.ChangeAppQuota(s.ctx, "app-1", "", "", &Quota{})
	s.Error(err)
	_, err = s.store.ChangeAppQuota(s.ctx, "", "ops@getstream.io", "", &Quota{})
	s.Error(err)
	s.Equal(0, s.quotaRows("app-1"))
}

func (s *StoreSuite) TestAChangeWhoseRecordCannotBeWrittenLeavesTheQuotaAsItWas() {
	s.changeQuota("app-1", "ops@getstream.io", pointerTo(int64(5)), nil)
	// Another transaction holds the record of changes, so the change saves the quota and
	// then cannot write its row, and gives up.
	holder, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = holder.Rollback() }()
	_, err = holder.ExecContext(s.ctx, "LOCK TABLE app_quota_changes IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(s.ctx, 500*time.Millisecond)
	defer cancel()
	_, err = s.store.ChangeAppQuota(ctx, "app-1", "ops@getstream.io", "", &Quota{MessagesPerUserPerDay: pointerTo(int64(9))})

	s.Require().Error(err)
	s.Require().NoError(holder.Rollback())
	document, err := s.store.Policy(s.ctx, ScopeApp, "app-1")
	s.Require().NoError(err)
	s.Equal(pointerTo(int64(5)), document.Quota.MessagesPerUserPerDay)
	s.Equal(1, s.quotaRows("app-1"))
}

func (s *StoreSuite) TestAChangeWhoseSaveFailsWritesNoRecord() {
	s.changeQuota("app-1", "ops@getstream.io", pointerTo(int64(5)), nil)
	// Another transaction holds the app's policy, so the change cannot save and gives up.
	holder, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = holder.Rollback() }()
	_, err = holder.ExecContext(s.ctx, "SELECT 1 FROM policies WHERE scope = 'app' AND scope_id = 'app-1' FOR UPDATE")
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(s.ctx, 500*time.Millisecond)
	defer cancel()
	_, err = s.store.ChangeAppQuota(ctx, "app-1", "ops@getstream.io", "", &Quota{MessagesPerUserPerDay: pointerTo(int64(9))})

	s.Require().Error(err)
	s.Require().NoError(holder.Rollback())
	s.Equal(1, s.quotaRows("app-1"))
	document, err := s.store.Policy(s.ctx, ScopeApp, "app-1")
	s.Require().NoError(err)
	s.Equal(pointerTo(int64(5)), document.Quota.MessagesPerUserPerDay)
}

func (s *StoreSuite) TestQuotaChangesAreListedNewestFirstAndPageByCursor() {
	for i := int64(1); i <= 3; i++ {
		s.changeQuota("app-1", "ops@getstream.io", pointerTo(i), nil)
	}
	s.changeQuota("app-2", "ops@getstream.io", pointerTo(int64(99)), nil)

	first, err := s.store.AppQuotaChanges(s.ctx, "app-1", 2, nil)
	s.Require().NoError(err)
	s.Require().Len(first, 3, "one more than the page, to say there is another")
	s.Equal(pointerTo(int64(3)), first[0].NewMessagesPerUserPerDay)
	s.Equal(pointerTo(int64(2)), first[1].NewMessagesPerUserPerDay)

	rest, err := s.store.AppQuotaChanges(s.ctx, "app-1", 2, &QuotaChangePosition{ChangedAt: first[1].ChangedAt, ID: first[1].ID})
	s.Require().NoError(err)
	s.Require().Len(rest, 1)
	s.Equal(pointerTo(int64(1)), rest[0].NewMessagesPerUserPerDay)

	_, err = s.store.AppQuotaChanges(s.ctx, "", 2, nil)
	s.Error(err)
}
