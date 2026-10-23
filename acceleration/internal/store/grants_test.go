//go:build integration

package store

import (
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func (s *StoreSuite) grant(grant LimitGrant) LimitGrant {
	if grant.OrganizationID == "" {
		grant.OrganizationID = "org-1"
	}
	if grant.Subject == "" {
		grant.Subject, grant.Category = usage.SubjectCategory, "team:finance"
	}
	if grant.AmountMicros == 0 {
		grant.AmountMicros = 5
	}
	grant.Reason, grant.Actor = "support", "someone"
	s.Require().NoError(s.store.AddGrant(s.ctx, &grant, LimitAudit{CustomerID: "app-" + uuid.NewString(), Source: AuditSourceAPI}))
	return grant
}

func (s *StoreSuite) TestAGrantNamesOneSubject() {
	audit := LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI}
	for name, grant := range map[string]LimitGrant{
		"a user grant with no user":           {Subject: usage.SubjectUser, AppID: "app-1"},
		"a key grant with a category":         {Subject: usage.SubjectAPIKey, AppID: "app-1", APIKey: "k", Category: "x"},
		"an organization grant naming an app": {Subject: usage.SubjectOrganization, AppID: "app-1"},
	} {
		s.Run(name, func() {
			grant.OrganizationID, grant.AmountMicros, grant.Reason, grant.Actor = "org-1", 5, "r", "a"
			s.Error(s.store.AddGrant(s.ctx, &grant, audit))
		})
	}
	past := time.Now().Add(-time.Hour)
	expired := LimitGrant{OrganizationID: "org-1", Subject: usage.SubjectOrganization, AmountMicros: 5, ExpiresAt: &past, Reason: "r", Actor: "a"}
	s.Error(s.store.AddGrant(s.ctx, &expired, audit), "limit_grants_expiry_check")
}

func (s *StoreSuite) TestLiveGrantsLeaveOutRevokedAndExpired() {
	soon := time.Now().Add(500 * time.Millisecond)
	live := s.grant(LimitGrant{})
	expiring := s.grant(LimitGrant{ExpiresAt: &soon})
	revoked := s.grant(LimitGrant{})
	_, err := s.store.RevokeGrant(s.ctx, "org-1", revoked.ID, "someone", "no longer needed", LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI})
	s.Require().NoError(err)

	grants, err := s.store.LiveGrants(s.ctx, []string{"org-1"}, time.Now())
	s.Require().NoError(err)
	s.Equal([]int64{live.ID, expiring.ID}, idsOf(grants))
	grants, err = s.store.LiveGrants(s.ctx, []string{"org-1"}, soon)
	s.Require().NoError(err)
	s.Equal([]int64{live.ID}, idsOf(grants), "at expires_at it is expired")
}

func (s *StoreSuite) TestRevokingTwiceIsANoOp() {
	grant := s.grant(LimitGrant{})
	audit := LimitAudit{CustomerID: "app-" + uuid.NewString(), Source: AuditSourceAPI}

	first, err := s.store.RevokeGrant(s.ctx, "org-1", grant.ID, "someone", "done", audit)
	s.Require().NoError(err)
	second, err := s.store.RevokeGrant(s.ctx, "org-1", grant.ID, "someone else", "again", audit)
	s.Require().NoError(err)

	s.Equal(first.RevokedAt, second.RevokedAt)
	s.Equal("someone", second.RevokedBy)
	entries, err := s.store.AuditEntries(s.ctx, audit.CustomerID, AuditEntryFilter{ResourceType: AuditSpendLimit})
	s.Require().NoError(err)
	s.Len(entries, 1, "the second records nothing")
	_, err = s.store.RevokeGrant(s.ctx, "org-2", grant.ID, "x", "x", audit)
	s.ErrorIs(err, ErrNoGrant, "another organization's grant is not there")
}

func (s *StoreSuite) TestAnOrganizationHasAtMostSoManyLiveGrants() {
	for range MaxLiveGrants {
		s.grant(LimitGrant{OrganizationID: "org-full"})
	}
	extra := LimitGrant{OrganizationID: "org-full", Subject: usage.SubjectOrganization, AmountMicros: 5, Reason: "r", Actor: "a"}

	s.ErrorIs(s.store.AddGrant(s.ctx, &extra, LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI}), ErrTooManyGrants)
	s.grant(LimitGrant{OrganizationID: "org-other"})
}

func (s *StoreSuite) TestGrantsAreListedNewestFirst() {
	a, b, c := s.grant(LimitGrant{}), s.grant(LimitGrant{}), s.grant(LimitGrant{})

	page, err := s.store.ListGrants(s.ctx, GrantFilter{OrganizationID: "org-1"}, 0, 2)
	s.Require().NoError(err)
	s.Equal([]int64{c.ID, b.ID, a.ID}, idsOf(page), "two and one more")
	next, err := s.store.ListGrants(s.ctx, GrantFilter{OrganizationID: "org-1"}, b.ID, 2)
	s.Require().NoError(err)
	s.Equal([]int64{a.ID}, idsOf(next))
}

// grantRow is a request row that names grants, written as the writer writes it.
func (s *StoreSuite) grantRow(store *Store, grants []int64, cost int64, billable bool) Request {
	row := Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "m",
		StartedAt: s.base, CostMicros: cost, Billable: billable, Success: true, LimitGrants: grants,
	}
	s.Require().NoError(store.RecordRequest(s.ctx, &row))
	return row
}

// used is what the grant has paid for, as its row says.
func (s *StoreSuite) used(id int64) int64 {
	grant, err := s.store.Grant(s.ctx, "", id)
	s.Require().NoError(err)
	return grant.UsedMicros
}

func (s *StoreSuite) TestAWrittenRowCountsTowardEachGrantItNames() {
	a, b := s.grant(LimitGrant{AmountMicros: 10_000}), s.grant(LimitGrant{AmountMicros: 10_000})

	s.grantRow(s.store, []int64{a.ID}, 100, true)
	s.grantRow(s.store, []int64{a.ID, b.ID}, 200, true)
	s.grantRow(s.store, []int64{a.ID}, 400, false)
	s.grantRow(s.store, nil, 800, true)

	s.EqualValues(300, s.used(a.ID), "billable rows only")
	s.EqualValues(200, s.used(b.ID), "a row naming two counts for both")
}

func (s *StoreSuite) TestARowWrittenTwiceCountsOnceTowardItsGrant() {
	grant := s.grant(LimitGrant{AmountMicros: 10_000})
	row := s.grantRow(s.store, []int64{grant.ID}, 300, true)

	again := row
	again.ID = 0
	written, err := s.store.RecordRequests(s.ctx, []Request{again, again})
	s.Require().NoError(err)
	s.Empty(written, "the shard has it")
	counted, err := s.store.AddGrantUses(s.ctx, []GrantUse{{GrantID: grant.ID, UsageID: row.UsageID, Micros: 300}})
	s.Require().NoError(err)
	s.Zero(counted, "nor does counting it again")

	s.EqualValues(300, s.used(grant.ID))
	fresh := Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "m",
		StartedAt: s.base, CostMicros: 50, Billable: true, Success: true, LimitGrants: []int64{grant.ID},
	}
	_, err = s.store.RecordRequests(s.ctx, []Request{fresh, fresh})
	s.Require().NoError(err)
	s.EqualValues(350, s.used(grant.ID), "a row sent twice in one batch counts once")
}

func (s *StoreSuite) TestAUsedUpGrantIsNotLiveAndLeavesRoomForAnother() {
	for range MaxLiveGrants - 1 {
		s.grant(LimitGrant{OrganizationID: "org-used"})
	}
	used := s.grant(LimitGrant{OrganizationID: "org-used", AmountMicros: 300})
	extra := LimitGrant{OrganizationID: "org-used", Subject: usage.SubjectOrganization, AmountMicros: 5, Reason: "r", Actor: "a"}
	s.Require().ErrorIs(s.store.AddGrant(s.ctx, &extra, LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI}), ErrTooManyGrants)

	s.grantRow(s.store, []int64{used.ID}, 300, true)

	live, err := s.store.LiveGrants(s.ctx, []string{"org-used"}, time.Now())
	s.Require().NoError(err)
	s.NotContains(idsOf(live), used.ID, "used up, it is not read")
	s.Len(live, MaxLiveGrants-1)
	s.Require().NoError(s.store.AddGrant(s.ctx, &extra, LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI}),
		"nor counted toward the cap")
	yes := true
	listed, err := s.store.ListGrants(s.ctx, GrantFilter{OrganizationID: "org-used", Used: &yes}, 0, 50)
	s.Require().NoError(err)
	s.Equal([]int64{used.ID}, idsOf(listed))
}

func (s *ShardedStoreSuite) TestRowsInEveryShardCountTowardTheirGrant() {
	grant := s.grant(LimitGrant{AmountMicros: 10_000})
	for _, store := range []*Store{s.store, s.other} {
		s.grantRow(store, []int64{grant.ID}, 300, true)
	}

	s.EqualValues(600, s.used(grant.ID))
}

func (s *ShardedStoreSuite) TestAReconcileCountsARowItsWriterDidNot() {
	grant := s.grant(LimitGrant{AmountMicros: 10_000})
	s.grantRow(s.store, []int64{grant.ID}, 300, true)
	// Written to the shard by a process that stopped before counting it.
	lost := Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "m",
		StartedAt: s.base, CostMicros: 70, Billable: true, Success: true, LimitGrants: []int64{grant.ID},
	}
	s.Require().NoError(prepareRequest(&lost))
	_, err := s.other.ShardDB().NewInsert().Model(&lost).Exec(s.ctx)
	s.Require().NoError(err)
	s.EqualValues(300, s.used(grant.ID))

	added, err := s.store.ReconcileGrantUses(s.ctx, []int64{grant.ID})
	s.Require().NoError(err)
	s.Equal(1, added, "the row the writer counted is not counted again")
	s.EqualValues(370, s.used(grant.ID))
	added, err = s.store.ReconcileGrantUses(s.ctx, []int64{grant.ID})
	s.Require().NoError(err)
	s.Zero(added)
	s.EqualValues(370, s.used(grant.ID))
}

func idsOf(grants []LimitGrant) []int64 {
	ids := make([]int64, len(grants))
	for i, grant := range grants {
		ids[i] = grant.ID
	}
	return ids
}
