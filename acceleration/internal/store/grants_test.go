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

func (s *StoreSuite) TestGrantUsedSumsTheBillableRowsNamingEachGrant() {
	row := func(store *Store, grants []int64, cost int64, billable bool) {
		s.Require().NoError(store.RecordRequest(s.ctx, &Request{
			UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "m",
			StartedAt: s.base, CostMicros: cost, Billable: billable, Success: true, LimitGrants: grants,
		}))
	}
	row(s.store, []int64{7}, 100, true)
	row(s.store, []int64{7, 8}, 200, true)
	row(s.store, []int64{7}, 400, false)
	row(s.store, nil, 800, true)

	used, err := s.store.GrantUsed(s.ctx, []int64{7, 8, 9})

	s.Require().NoError(err)
	s.Equal(map[int64]int64{7: 300, 8: 200}, used, "billable rows only, a row naming two counts for both")
}

func (s *ShardedStoreSuite) TestGrantUsedSumsAcrossShards() {
	for _, store := range []*Store{s.store, s.other} {
		s.Require().NoError(store.RecordRequest(s.ctx, &Request{
			UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "m",
			StartedAt: s.base, CostMicros: 300, Billable: true, Success: true, LimitGrants: []int64{42},
		}))
	}

	used, err := s.store.GrantUsed(s.ctx, []int64{42})

	s.Require().NoError(err)
	s.Equal(int64(600), used[42])
}

func idsOf(grants []LimitGrant) []int64 {
	ids := make([]int64, len(grants))
	for i, grant := range grants {
		ids[i] = grant.ID
	}
	return ids
}
