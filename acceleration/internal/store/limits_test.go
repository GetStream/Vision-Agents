//go:build integration

package store

import (
	"errors"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

var limitAudit = LimitAudit{CustomerID: "app-1", Source: AuditSourceAPI, ActorID: "someone"}

func (s *StoreSuite) newLimit(limit Limit) Limit {
	if limit.Interval == "" {
		limit.Interval = usage.Daily
	}
	if limit.OrganizationID == "" {
		limit.OrganizationID = "org-1"
	}
	limit.UpdatedBy = "someone"
	s.Require().NoError(s.store.CreateLimit(s.ctx, &limit, limitAudit))
	return limit
}

func (s *StoreSuite) TestTheLevelChecksRefuseAMismatchedRow() {
	for name, limit := range map[string]Limit{
		"a key row with no key":      {Subject: usage.SubjectAPIKey, Level: LevelAPIKey, AppID: "app-1"},
		"an app row with no app":     {Subject: usage.SubjectApp, Level: LevelApp},
		"a row with no organization": {Subject: usage.SubjectOrganization, Level: LevelOrganization, OrganizationID: ""},
		"a global level":             {Subject: usage.SubjectOrganization, Level: "global"},
		"a tier level":               {Subject: usage.SubjectOrganization, Level: "tier"},
		"an app limited per user":    {Subject: usage.SubjectApp, Level: LevelUser, AppID: "app-1", UserID: "u1"},
		"a category with none":       {Subject: usage.SubjectCategory, Level: LevelOrganization},
		"an hourly limit":            {Subject: usage.SubjectOrganization, Level: LevelOrganization, Interval: "hourly"},
	} {
		s.Run(name, func() {
			if limit.Interval == "" {
				limit.Interval = usage.Daily
			}
			if name != "a row with no organization" {
				limit.OrganizationID = "org-1"
			}
			limit.UpdatedBy = "someone"
			s.Error(s.store.CreateLimit(s.ctx, &limit, limitAudit))
		})
	}
}

func (s *StoreSuite) TestOneRowPerLevelAndInterval() {
	first := s.newLimit(Limit{Subject: usage.SubjectAPIKey, Level: LevelApp, AppID: "app-1", LimitMicros: 5})
	s.newLimit(Limit{Subject: usage.SubjectAPIKey, Level: LevelApp, AppID: "app-1", LimitMicros: 5, Interval: usage.Weekly})

	again := Limit{Subject: usage.SubjectAPIKey, Level: LevelApp, OrganizationID: "org-1", AppID: "app-1", Interval: usage.Daily, LimitMicros: 9, UpdatedBy: "x"}
	err := s.store.CreateLimit(s.ctx, &again, limitAudit)

	var exists *ErrLimitExists
	s.Require().True(errors.As(err, &exists), "%v", err)
	s.Equal(first.ID, exists.ID)
}

func (s *StoreSuite) TestALimitsWritesAreAudited() {
	audit := LimitAudit{CustomerID: "app-" + uuid.NewString(), Source: AuditSourceManage, ActorID: "staff"}
	limit := Limit{Subject: usage.SubjectApp, Level: LevelApp, OrganizationID: "org-1", AppID: "app-1", Interval: usage.Daily, LimitMicros: 5, UpdatedBy: "x"}
	s.Require().NoError(s.store.CreateLimit(s.ctx, &limit, audit))
	more, note := int64(7), "raised"
	updated, err := s.store.UpdateLimit(s.ctx, "org-1", limit.ID, LimitChange{LimitMicros: &more, Note: &note, UpdatedBy: "other"}, audit)
	s.Require().NoError(err)
	s.EqualValues(7, updated.LimitMicros)
	s.Equal("other", updated.UpdatedBy)
	_, err = s.store.UpdateLimit(s.ctx, "org-2", limit.ID, LimitChange{LimitMicros: &more}, audit)
	s.ErrorIs(err, ErrNoLimit, "another organization's limit is not there")
	_, err = s.store.DeleteLimit(s.ctx, "", limit.ID, audit)
	s.Require().NoError(err, "with no organization named, any organization's")
	_, err = s.store.Limit(s.ctx, "org-1", limit.ID)
	s.ErrorIs(err, ErrNoLimit)

	entries, err := s.store.AuditEntries(s.ctx, audit.CustomerID, AuditEntryFilter{ResourceType: AuditSpendLimit})
	s.Require().NoError(err)
	s.Require().Len(entries, 3)
	s.Equal([]string{AuditDeleted, AuditUpdated, AuditCreated}, []string{entries[0].Action, entries[1].Action, entries[2].Action})
	s.Equal([]AuditChange{{Field: "limit_micros", Before: float64(5), After: float64(7)}, {Field: "note", After: "raised"}}, entries[1].Changes)
}

func (s *StoreSuite) TestLimitsAreListedByOrganizationAndFilter() {
	a := s.newLimit(Limit{Subject: usage.SubjectCategory, Level: LevelOrganization, Category: "team:finance", LimitMicros: 1})
	b := s.newLimit(Limit{Subject: usage.SubjectCategory, Level: LevelOrganization, Category: "project:x", LimitMicros: 1})
	s.newLimit(Limit{Subject: usage.SubjectCategory, Level: LevelOrganization, OrganizationID: "org-2", Category: "team:finance", LimitMicros: 1})

	teams, err := s.store.ListLimits(s.ctx, LimitFilter{OrganizationID: "org-1", Subject: usage.SubjectCategory, CategoryPrefix: "team:"}, 0, 10)
	s.Require().NoError(err)
	s.Require().Len(teams, 1)
	s.Equal(a.ID, teams[0].ID)
	others, err := s.store.ListLimits(s.ctx, LimitFilter{OrganizationID: "org-1", Subject: usage.SubjectCategory, NotCategoryPrefix: "team:"}, 0, 10)
	s.Require().NoError(err)
	s.Require().Len(others, 1)
	s.Equal(b.ID, others[0].ID)
	after, err := s.store.ListLimits(s.ctx, LimitFilter{OrganizationID: "org-1"}, a.ID, 10)
	s.Require().NoError(err)
	s.Require().Len(after, 1, "the page after the first row")
	s.Equal(b.ID, after[0].ID)

	both, err := s.store.LimitsFor(s.ctx, []string{"org-1", "org-2"})
	s.Require().NoError(err)
	s.Len(both, 3)
}

func (s *StoreSuite) TestResetsAreReadFromTheWindowsAsked() {
	today := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	audit := LimitAudit{CustomerID: "app-" + uuid.NewString(), Source: AuditSourceAPI}
	s.Require().NoError(s.store.AddResets(s.ctx, []UsageReset{
		{OrganizationID: "org-1", Subject: usage.SubjectApp, AppID: "app-1", Interval: usage.Daily, WindowStart: today, BaselineMicros: 5, Reason: "r", Actor: "a"},
		{OrganizationID: "org-1", Subject: usage.SubjectApp, AppID: "app-1", Interval: usage.Daily, WindowStart: today.AddDate(0, 0, -40), BaselineMicros: 5, Reason: "r", Actor: "a"},
	}, audit))

	resets, err := s.store.Resets(s.ctx, []string{"org-1"}, today.AddDate(0, 0, -1))

	s.Require().NoError(err)
	s.Require().Len(resets, 1)
	s.True(resets[0].WindowStart.Equal(today))
	entries, err := s.store.AuditEntries(s.ctx, audit.CustomerID, AuditEntryFilter{ResourceType: AuditSpendLimit})
	s.Require().NoError(err)
	s.Len(entries, 2)
	s.Contains(entries[0].ResourceID, "reset:")
}

func (s *StoreSuite) TestASubjectsSpendIsItsBillableRowsAlone() {
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-1", "org-1"))
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-2", "org-1"))
	row := func(customer, key, user string, categories []string, cost int64, billable bool) {
		s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
			UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: customer, Provider: "openai", Model: "m",
			StartedAt: s.base.Add(-time.Minute), APIKey: key, UserID: user, UsageCategories: categories,
			CostMicros: cost, Billable: billable, Success: true,
		}))
	}
	row("app-1", "k1", "u1", []string{"team:finance"}, 100, true)
	row("app-1", "k2", "u2", nil, 200, true)
	row("app-2", "k1", "u1", []string{"team:finance"}, 400, true)
	row("app-1", "k1", "u1", []string{"team:finance"}, 800, false)
	since := s.base.Add(-time.Hour)

	for ref, want := range map[usage.SubjectRef]int64{
		{OrganizationID: "org-1", Subject: usage.SubjectOrganization}:                                          700,
		{OrganizationID: "org-1", Subject: usage.SubjectApp, AppID: "app-1"}:                                   300,
		{OrganizationID: "org-1", Subject: usage.SubjectAPIKey, AppID: "app-1", APIKey: "k1"}:                  100,
		{OrganizationID: "org-1", Subject: usage.SubjectUser, AppID: "app-2", UserID: "u1"}:                    400,
		{OrganizationID: "org-1", Subject: usage.SubjectCategory, Category: "team:finance"}:                    500,
		{OrganizationID: "org-1", Subject: usage.SubjectAppCategory, AppID: "app-1", Category: "team:finance"}: 100,
	} {
		spent, err := s.store.SubjectSpendSince(s.ctx, ref, since)
		s.Require().NoError(err)
		s.Equal(want, spent, "%+v", ref)
	}
}
