//go:build integration

package store

import (
	"time"

	"github.com/google/uuid"
)

func (s *StoreSuite) limitNotice(organization string) LimitNotice {
	return LimitNotice{
		OrganizationID: organization, Kind: LimitNoCredit, AppID: "app-1", Tier: "free",
		Enforced: true, Message: "the organization has no credit left",
	}
}

func (s *StoreSuite) TestALimitNoticeIsWrittenOncePerOrganizationAndKind() {
	organization := "org-" + uuid.NewString()
	notice := s.limitNotice(organization)

	written, err := s.store.NoteLimit(s.ctx, notice)
	s.Require().NoError(err)
	s.True(written, "the first")
	written, err = s.store.NoteLimit(s.ctx, notice)
	s.Require().NoError(err)
	s.False(written, "the same organization and kind again")

	notice.Kind = LimitSpendHardLimit
	written, err = s.store.NoteLimit(s.ctx, notice)
	s.Require().NoError(err)
	s.True(written, "another kind")

	notice.Kind = "suspended"
	_, err = s.store.NoteLimit(s.ctx, notice)
	s.Error(err, "not a limit")
}

func (s *StoreSuite) TestAnUnpostedLimitNoticeIsClaimedOnceItsClaimIsStale() {
	organization := "org-" + uuid.NewString()
	_, err := s.store.NoteLimit(s.ctx, s.limitNotice(organization))
	s.Require().NoError(err)

	claimed, err := s.store.ClaimUnpostedLimits(s.ctx, time.Hour, 100)
	s.Require().NoError(err)
	s.NotContains(organizations(claimed), organization, "its writer's claim stands")

	_, err = s.store.DB().NewUpdate().Model((*LimitNotice)(nil)).Set("claimed_at = now() - interval '2 hours'").
		Where("organization_id = ?", organization).Exec(s.ctx)
	s.Require().NoError(err)
	claimed, err = s.store.ClaimUnpostedLimits(s.ctx, time.Hour, 100)
	s.Require().NoError(err)
	s.Require().Contains(organizations(claimed), organization)
	for _, notice := range claimed {
		if notice.OrganizationID == organization {
			s.Equal("the organization has no credit left", notice.Message)
			s.True(notice.Enforced)
		}
	}
	claimed, err = s.store.ClaimUnpostedLimits(s.ctx, time.Hour, 100)
	s.Require().NoError(err)
	s.NotContains(organizations(claimed), organization, "claimed again just now")

	s.Require().NoError(s.store.MarkLimitPosted(s.ctx, organization, LimitNoCredit))
	_, err = s.store.DB().NewUpdate().Model((*LimitNotice)(nil)).Set("claimed_at = now() - interval '2 hours'").
		Where("organization_id = ?", organization).Exec(s.ctx)
	s.Require().NoError(err)
	claimed, err = s.store.ClaimUnpostedLimits(s.ctx, time.Hour, 100)
	s.Require().NoError(err)
	s.NotContains(organizations(claimed), organization, "posted")
}

func organizations(notices []LimitNotice) []string {
	names := make([]string, 0, len(notices))
	for _, notice := range notices {
		names = append(names, notice.OrganizationID)
	}
	return names
}
