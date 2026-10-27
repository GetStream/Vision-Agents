//go:build integration

package ledger_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// devTiers is the development tier file, which the local stack seeds too.
const devTiers = "../../../tiers/dev.json"

var staff = ledger.Actor{ID: "staff-1", Name: "Ada"}

func ptr[T any](v T) *T { return &v }

// newKey is an idempotency key no other write uses.
func newKey() string { return uuid.NewString() }

// audits is every manage_audit row, oldest first.
func (s *LedgerSuite) audits() []ledger.ManageAudit {
	var rows []ledger.ManageAudit
	s.Require().NoError(s.ledger.DB().NewSelect().Model(&rows).Order("id").Scan(s.ctx))
	return rows
}

// account is the organization's account row, which the test expects there to be.
func (s *LedgerSuite) account(organization string) ledger.Account {
	account, found, err := s.ledger.Account(s.ctx, organization)
	s.Require().NoError(err)
	s.Require().True(found, organization)
	return account
}

func (s *LedgerSuite) credit(organization string, kind usage.CreditKind, amount int64, key string) (ledger.CreditEntry, bool, error) {
	return s.ledger.AddCredit(s.ctx, ledger.CreditEntry{
		OrganizationID: organization, Kind: kind, AmountMicros: amount, IdempotencyKey: key, Actor: staff.ID, ActorName: staff.Name,
	})
}

func (s *LedgerSuite) setTier(tier usage.Tier) usage.Tier {
	written, err := s.ledger.SetTier(s.ctx, tier, staff)
	s.Require().NoError(err)
	return written
}

func (s *LedgerSuite) TestTheMigrationWritesNoTier() {
	fresh := usagetest.Database(s.T(), "usage_ledger")
	opened, err := ledger.Open(ledger.Options{DSN: fresh})
	s.Require().NoError(err)
	defer opened.Close() //nolint:errcheck // a pool the test is done with

	tiers, err := opened.Tiers(s.ctx)

	s.Require().NoError(err)
	s.Empty(tiers)
}

func (s *LedgerSuite) TestSyncTiersAddsWhatTheFileHas() {
	file, err := ledger.ReadTierFile(devTiers)
	s.Require().NoError(err)

	added, err := s.ledger.SyncTiers(s.ctx, file, "v1.2.3")

	s.Require().NoError(err)
	s.Equal([]string{"default", "tier-a"}, added)
	tiers, err := s.ledger.Tiers(s.ctx)
	s.Require().NoError(err)
	s.Equal(file.Tiers, tiers, "every setting round-trips")
	s.Equal(usage.FailOpen, tiers[1].FailMode)
	s.Equal(ptr(int64(25_000_000)), tiers[1].Limits.SpendHardLimit.DailyMicros)
	s.Equal([]string{"NL", "IT"}, tiers[0].Limits.Channels["voice"].Countries)
	s.Equal(ptr(usage.NetworkNone), tiers[0].Limits.Sandbox.Network)

	changed := *file
	changed.Tiers = []usage.Tier{{Name: "tier-a", FailMode: usage.FailClosed}}
	again, err := s.ledger.SyncTiers(s.ctx, &changed, "v1.2.4")
	s.Require().NoError(err)
	s.Empty(again)
	tier, found, err := s.ledger.Tier(s.ctx, "tier-a")
	s.Require().NoError(err)
	s.True(found)
	s.Equal(file.Tiers[1], tier, "a tier the database has is never touched")

	audits := s.audits()
	s.Require().Len(audits, 2)
	for i, audit := range audits {
		s.Equal(ledger.ActionSeedTier, audit.Action)
		s.Equal("usage v1.2.3", audit.Actor)
		s.Empty(audit.OrganizationID)
		var after usage.Tier
		s.Require().NoError(json.Unmarshal(audit.After, &after))
		s.Equal(added[i], after.Name)
	}
}

func (s *LedgerSuite) TestWithNoFileTheBuiltInDefaultIsWritten() {
	added, err := s.ledger.SyncTiers(s.ctx, nil, "")

	s.Require().NoError(err)
	s.Equal([]string{"default"}, added)
	tier, found, err := s.ledger.Tier(s.ctx, usage.DefaultTier)
	s.Require().NoError(err)
	s.True(found)
	s.Equal(usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed}, tier)
	s.Equal("usage", s.audits()[0].Actor)

	usagetest.Empty(s.T(), s.ledger)
	added, err = s.ledger.SyncTiers(s.ctx, &ledger.TierFile{Tiers: []usage.Tier{{Name: "tier-a", FailMode: usage.FailOpen}}}, "")
	s.Require().NoError(err)
	s.Equal([]string{"tier-a", "default"}, added, "a file without default gets the built-in one too")
}

func (s *LedgerSuite) TestPlanTierSyncWritesNothing() {
	file, err := ledger.ReadTierFile(devTiers)
	s.Require().NoError(err)
	s.setTier(usage.Tier{Name: "tier-a", FailMode: usage.FailClosed})

	planned, migrated, err := s.ledger.PlanTierSync(s.ctx, file)

	s.Require().NoError(err)
	s.True(migrated)
	s.Equal([]string{"default"}, planned)
	tiers, err := s.ledger.Tiers(s.ctx)
	s.Require().NoError(err)
	s.Len(tiers, 1)
}

func (s *LedgerSuite) TestPlanTierSyncOnADatabaseNotMigratedYetPlansEveryTier() {
	opened, err := ledger.Open(ledger.Options{DSN: usagetest.EmptyDatabase(s.T(), "usage_unmigrated")})
	s.Require().NoError(err)
	defer opened.Close() //nolint:errcheck // a pool the test is done with
	file, err := ledger.ReadTierFile(devTiers)
	s.Require().NoError(err)

	planned, migrated, err := opened.PlanTierSync(s.ctx, file)

	s.Require().NoError(err)
	s.False(migrated)
	s.Equal([]string{"default", "tier-a"}, planned)
}

func (s *LedgerSuite) TestABadTierFileStopsTheStart() {
	_, err := ledger.ParseTierFile([]byte(`{"tiers": [{"name": "tier-a", "fail_mode": "open"}, {"name": "tier-a", "fail_mode": "closed"}]}`))
	s.Require().ErrorContains(err, `tier "tier-a" is named twice`)

	_, err = ledger.ParseTierFile([]byte(`{"tiers": [{"name": "tier-b", "fail_mode": "maybe"}]}`))
	s.Require().ErrorContains(err, `tier "tier-b" (tiers[0]): fail_mode "maybe" is not closed or open`)

	_, err = ledger.ParseTierFile([]byte(`{"tiers": [{"name": "tier-b", "fail_mode": "open", "limits": {"spend_limit": {}}}]}`))
	s.Require().ErrorContains(err, `unknown field "spend_limit"`, "a misspelt setting is not silently no limit")

	_, err = ledger.ReadTierFile("/nonexistent/tiers.json")
	s.Require().ErrorContains(err, "/nonexistent/tiers.json")
}

func (s *LedgerSuite) TestATierWithNoRowIsDefault() {
	tier, found, err := s.ledger.Tier(s.ctx, "tier-z")
	s.Require().NoError(err)
	s.False(found)
	s.Equal(usage.BuiltInDefaultTier(), tier, "before any default row, the built-in one")

	s.setTier(usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailOpen, Limits: usage.Limits{LowBalanceMicros: ptr(int64(7))}})
	for _, name := range []string{"tier-z", ""} {
		tier, found, err = s.ledger.Tier(s.ctx, name)
		s.Require().NoError(err)
		s.False(found, name)
		s.Equal(usage.FailOpen, tier.FailMode, name)
		s.Equal(ptr(int64(7)), tier.Limits.LowBalanceMicros, name)
	}
}

func (s *LedgerSuite) TestTheDefaultTierCannotBeDeleted() {
	s.setTier(usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed})
	s.setTier(usage.Tier{Name: "tier-a", FailMode: usage.FailOpen})

	_, err := s.ledger.DeleteTier(s.ctx, usage.DefaultTier, staff)
	s.Require().ErrorIs(err, ledger.ErrDefaultTier)
	deleted, err := s.ledger.DeleteTier(s.ctx, "tier-a", staff)
	s.Require().NoError(err)
	s.True(deleted)
	deleted, err = s.ledger.DeleteTier(s.ctx, "tier-a", staff)
	s.Require().NoError(err)
	s.False(deleted, "deleting it again finds nothing")

	tiers, err := s.ledger.Tiers(s.ctx)
	s.Require().NoError(err)
	s.Equal([]usage.Tier{{Name: usage.DefaultTier, FailMode: usage.FailClosed}}, tiers)
	audits := s.audits()
	s.Equal(ledger.ActionDeleteTier, audits[len(audits)-1].Action)
	s.Len(audits, 3, "a delete that found nothing changed nothing")
}

func (s *LedgerSuite) TestASetTierIsValidated() {
	_, err := s.ledger.SetTier(s.ctx, usage.Tier{Name: "tier a", FailMode: usage.FailOpen}, staff)
	var refused *ledger.InvalidError
	s.Require().ErrorAs(err, &refused)

	_, err = s.ledger.SetTier(s.ctx, usage.Tier{Name: "tier-a", FailMode: usage.FailOpen, Limits: usage.Limits{
		SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(-1))},
	}}, staff)
	s.Require().ErrorAs(err, &refused)
	s.Contains(refused.Problem, "spend_hard_limit.daily_micros is -1")

	_, err = s.ledger.SetTier(s.ctx, usage.Tier{Name: "tier-a", FailMode: usage.FailOpen, Limits: usage.Limits{
		Channels: map[string]usage.ChannelLimits{"sms": {Countries: []string{"usa"}}},
	}}, staff)
	s.Require().ErrorAs(err, &refused)
	s.Contains(refused.Problem, `"usa" is not an ISO 3166-1 alpha-2 code`)
}

func (s *LedgerSuite) TestTierOverridesListsOrganizationsLastSeenOnTheTier() {
	overrides := usage.Limits{Sandbox: &usage.SandboxLimits{MaxConcurrent: ptr(3)}}
	earlier := time.Now().Add(-time.Hour)
	s.setTier(usage.Tier{Name: "tier-b", FailMode: usage.FailClosed})
	s.setTier(usage.Tier{Name: "tier-c", FailMode: usage.FailClosed})
	s.ingest(
		usagetest.Event("now-b", func(e *usage.Event) { e.OrganizationTier, e.OccurredAt = "tier-c", earlier }),
		usagetest.Event("now-b", func(e *usage.Event) { e.OrganizationTier = "tier-b" }),
		usagetest.Event("now-c", func(e *usage.Event) { e.OrganizationTier, e.OccurredAt = "tier-b", earlier }),
		usagetest.Event("now-c", func(e *usage.Event) { e.OrganizationTier = "tier-c" }),
		usagetest.Event("plain-b", func(e *usage.Event) { e.OrganizationTier = "tier-b" }),
	)
	for _, organization := range []string{"now-b", "now-c", "no-events"} {
		_, _, err := s.ledger.SetAccountLimits(s.ctx, organization, overrides, newKey(), staff)
		s.Require().NoError(err)
	}
	_, _, err := s.ledger.SetAccountLimits(s.ctx, "plain-b", usage.Limits{Sandbox: &usage.SandboxLimits{}}, newKey(), staff)
	s.Require().NoError(err)

	tierB, more, err := s.ledger.TierOverrides(s.ctx, "tier-b", "", 10)

	s.Require().NoError(err)
	s.False(more)
	s.Require().Len(tierB, 1, "an empty override is none, and an organization with no event is on no tier")
	s.Equal("now-b", tierB[0].OrganizationID)
	s.Equal(overrides, tierB[0].LimitOverrides)
}

func (s *LedgerSuite) TestTheDefaultTiersOverridesIncludeTiersWithNoRow() {
	s.setTier(usage.Tier{Name: "tier-b", FailMode: usage.FailClosed})
	s.ingest(
		usagetest.Event("on-default", func(e *usage.Event) { e.OrganizationTier = usage.DefaultTier }),
		usagetest.Event("on-unknown", func(e *usage.Event) { e.OrganizationTier = "tier-z" }),
		usagetest.Event("on-b", func(e *usage.Event) { e.OrganizationTier = "tier-b" }),
	)
	for _, organization := range []string{"on-default", "on-unknown", "on-b"} {
		_, _, err := s.ledger.SetAccountLimits(s.ctx, organization, usage.Limits{LowBalanceMicros: ptr(int64(1))}, newKey(), staff)
		s.Require().NoError(err)
	}

	listed, _, err := s.ledger.TierOverrides(s.ctx, usage.DefaultTier, "", 10)

	s.Require().NoError(err)
	ids := make([]string, 0, len(listed))
	for _, account := range listed {
		ids = append(ids, account.OrganizationID)
	}
	s.Equal([]string{"on-default", "on-unknown"}, ids)
}

func (s *LedgerSuite) TestTheNewestEventOfTwoAtOneInstantIsTheOneWithTheGreaterUsageID() {
	at := time.Now().UTC().Truncate(time.Microsecond)
	first, second := usagetest.Event("o1"), usagetest.Event("o1")
	if bytes.Compare(first.UsageID[:], second.UsageID[:]) > 0 {
		first, second = second, first
	}
	first.OccurredAt, first.OrganizationTier = at, "tier-b"
	second.OccurredAt, second.OrganizationTier = at, "tier-c"
	s.ingest(second, first)

	tier, found, err := s.ledger.LastTier(s.ctx, "o1")

	s.Require().NoError(err)
	s.True(found)
	s.Equal("tier-c", tier)
}

func (s *LedgerSuite) TestTierOverridesPages() {
	for _, organization := range []string{"o1", "o2", "o3"} {
		s.ingest(usagetest.Event(organization, func(e *usage.Event) { e.OrganizationTier = "tier-b" }))
		_, _, err := s.ledger.SetAccountLimits(s.ctx, organization, usage.Limits{LowBalanceMicros: ptr(int64(1))}, newKey(), staff)
		s.Require().NoError(err)
	}

	first, more, err := s.ledger.TierOverrides(s.ctx, "tier-b", "", 2)
	s.Require().NoError(err)
	s.True(more)
	second, more, err := s.ledger.TierOverrides(s.ctx, "tier-b", first[1].OrganizationID, 2)
	s.Require().NoError(err)
	s.False(more)

	s.Equal([]string{"o1", "o2", "o3"}, []string{first[0].OrganizationID, first[1].OrganizationID, second[0].OrganizationID})
}

func (s *LedgerSuite) TestAnOverrideReplacesOnlyTheSettingsItSets() {
	tier := usage.Limits{
		Channels: map[string]usage.ChannelLimits{
			"sms":   {PerDay: ptr(10), Countries: []string{"NL"}},
			"voice": {PerDay: ptr(5), Concurrent: ptr(1)},
		},
		Sandbox:          &usage.SandboxLimits{Network: ptr(usage.NetworkNone), MaxCPU: ptr(1)},
		LowBalanceMicros: ptr(int64(100)),
	}
	overrides := usage.Limits{
		Channels: map[string]usage.ChannelLimits{"voice": {Concurrent: ptr(4)}, "rcs": {PerDay: ptr(7)}},
		Sandbox:  &usage.SandboxLimits{MaxCPU: ptr(2)},
	}

	effective := tier.Over(overrides)

	s.Equal(usage.Limits{
		Channels: map[string]usage.ChannelLimits{
			"sms":   {PerDay: ptr(10), Countries: []string{"NL"}},
			"voice": {PerDay: ptr(5), Concurrent: ptr(4)},
			"rcs":   {PerDay: ptr(7)},
		},
		Sandbox:          &usage.SandboxLimits{Network: ptr(usage.NetworkNone), MaxCPU: ptr(2)},
		LowBalanceMicros: ptr(int64(100)),
	}, effective)
	s.Equal(ptr(1), tier.Channels["voice"].Concurrent, "the tier's own limits are left as they were")
	s.Equal(tier, tier.Over(usage.Limits{}))
}

func (s *LedgerSuite) TestASpendHardLimitOverrideReplacesTheTiers() {
	tier := usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(5_000_000_000)), MonthlyMicros: ptr(int64(50_000_000_000))}}
	limit := func(limits usage.Limits, interval usage.Interval) any {
		if amount, ok := limits.SpendHardLimitMicros(interval); ok {
			return amount
		}
		return "none"
	}

	raised := tier.Over(usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(20_000_000_000))}})
	s.Equal(int64(20_000_000_000), limit(raised, usage.Daily))
	s.Equal(int64(50_000_000_000), limit(raised, usage.Monthly))

	same := tier.Over(usage.Limits{SpendHardLimit: &usage.SpendHardLimit{}})
	s.Equal(int64(5_000_000_000), limit(same, usage.Daily))
	s.Equal(int64(50_000_000_000), limit(same, usage.Monthly))

	lifted := tier.Over(usage.Limits{SpendHardLimit: &usage.SpendHardLimit{MonthlyMicros: ptr(int64(0))}})
	s.Equal(int64(5_000_000_000), limit(lifted, usage.Daily))
	s.Equal("none", limit(lifted, usage.Monthly))
}

func (s *LedgerSuite) TestACreditIsAddedOnceUnderItsKey() {
	first, written, err := s.credit("o1", usage.CreditGrant, 5_000_000, "grant-1")
	s.Require().NoError(err)
	s.True(written)
	second, written, err := s.credit("o1", usage.CreditGrant, 5_000_000, "grant-1")
	s.Require().NoError(err)

	s.False(written)
	s.Equal(first.ID, second.ID)
	s.Equal(int64(5_000_000), s.account("o1").GrantedMicros)
	entries, more, err := s.ledger.CreditEntries(s.ctx, "o1", nil, 10)
	s.Require().NoError(err)
	s.False(more)
	s.Len(entries, 1)
	s.Equal(staff.ID, entries[0].Actor)
	s.Equal(staff.Name, entries[0].ActorName)
}

func (s *LedgerSuite) TestAKeyHoldingAnotherEntryIsRefused() {
	_, _, err := s.credit("o1", usage.CreditGrant, 5_000_000, "grant-1")
	s.Require().NoError(err)

	_, _, err = s.credit("o2", usage.CreditGrant, 5_000_000, "grant-1")
	s.Require().ErrorIs(err, ledger.ErrIdempotencyKeyReused)
	_, _, err = s.credit("o1", usage.CreditGrant, 6_000_000, "grant-1")
	s.Require().ErrorIs(err, ledger.ErrIdempotencyKeyReused)

	s.Equal(int64(5_000_000), s.account("o1").GrantedMicros)
	_, found, err := s.ledger.Account(s.ctx, "o2")
	s.Require().NoError(err)
	s.False(found, "nothing written for the other organization")
}

func (s *LedgerSuite) TestACreditForASuspendedAccountIsWritten() {
	_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "chargeback", newKey(), staff)
	s.Require().NoError(err)

	_, topped, err := s.credit("o1", usage.CreditTopUp, 10_000_000, "top-up-1")
	s.Require().NoError(err)
	_, refunded, err := s.credit("o1", usage.CreditRefund, -4_000_000, "refund-1")
	s.Require().NoError(err)

	s.True(topped)
	s.True(refunded)
	account := s.account("o1")
	s.Equal(int64(6_000_000), account.GrantedMicros)
	s.Equal(usage.AccountSuspended, account.Status)
}

func (s *LedgerSuite) TestACreditWithTheWrongSignIsRefused() {
	for _, entry := range []struct {
		kind   usage.CreditKind
		amount int64
	}{{usage.CreditTopUp, -1}, {usage.CreditDispute, 1}, {usage.CreditAdjustment, 0}, {"gift", 1}} {
		_, _, err := s.credit("o1", entry.kind, entry.amount, fmt.Sprintf("%s-%d", entry.kind, entry.amount))
		var refused *ledger.InvalidError
		s.Require().ErrorAs(err, &refused, "%s %d", entry.kind, entry.amount)
	}

	entries, _, err := s.ledger.CreditEntries(s.ctx, "o1", nil, 10)
	s.Require().NoError(err)
	s.Empty(entries)
	_, found, err := s.ledger.Account(s.ctx, "o1")
	s.Require().NoError(err)
	s.False(found, "no account written either")
}

func (s *LedgerSuite) TestAnAdjustmentMayTakeCreditAway() {
	_, _, err := s.credit("o1", usage.CreditGrant, 10, "grant")
	s.Require().NoError(err)
	_, _, err = s.credit("o1", usage.CreditAdjustment, -3, "adjust")
	s.Require().NoError(err)

	s.Equal(int64(7), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestCreditEntriesPageNewestFirst() {
	for i := range 5 {
		_, _, err := s.credit("o1", usage.CreditGrant, int64(i+1), fmt.Sprintf("grant-%d", i))
		s.Require().NoError(err)
	}

	var amounts []int64
	var after *ledger.CreditPosition
	for {
		entries, more, err := s.ledger.CreditEntries(s.ctx, "o1", after, 2)
		s.Require().NoError(err)
		for _, entry := range entries {
			amounts = append(amounts, entry.AmountMicros)
		}
		if !more {
			break
		}
		last := entries[len(entries)-1]
		after = &ledger.CreditPosition{CreatedAt: last.CreatedAt, ID: last.ID}
	}

	s.Equal([]int64{5, 4, 3, 2, 1}, amounts)
}

// reverse writes an adjustment of the organization reversing the entry of id by amount.
func (s *LedgerSuite) reverse(organization string, id, amount int64, key string) (ledger.CreditEntry, bool, error) {
	return s.ledger.AddCredit(s.ctx, ledger.CreditEntry{
		OrganizationID: organization, Kind: usage.CreditAdjustment, AmountMicros: amount, ReversesID: id,
		IdempotencyKey: key, Actor: staff.ID, ActorName: staff.Name,
	})
}

// balance is the organization's balance, as its account has it.
func (s *LedgerSuite) balance(organization string) int64 {
	account := s.account(organization)
	return account.GrantedMicros - account.SpentMicros
}

func (s *LedgerSuite) TestAGrantIsRevokedUnrevokedAndRevokedAgain() {
	grant, _, err := s.credit("o1", usage.CreditGrant, 5_000, "grant")
	s.Require().NoError(err)
	s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 1_000 }))

	revoke, written, err := s.reverse("o1", grant.ID, -5_000, "revoke-1")
	s.Require().NoError(err)
	s.True(written)
	s.Equal(int64(-1_000), s.balance("o1"), "what was spent of it takes the balance below 0")
	unrevoke, _, err := s.reverse("o1", revoke.ID, 5_000, "unrevoke-1")
	s.Require().NoError(err)
	s.Equal(int64(4_000), s.balance("o1"))
	again, _, err := s.reverse("o1", unrevoke.ID, -5_000, "revoke-2")
	s.Require().NoError(err)
	s.Equal(int64(-1_000), s.balance("o1"))

	entries, _, err := s.ledger.CreditEntries(s.ctx, "o1", nil, 10)
	s.Require().NoError(err)
	s.Require().Len(entries, 4)
	reverses := map[int64][2]int64{}
	for _, entry := range entries {
		reverses[entry.ID] = [2]int64{entry.ReversesID, entry.ReversedByID}
	}
	s.Equal(map[int64][2]int64{
		grant.ID:    {0, revoke.ID},
		revoke.ID:   {grant.ID, unrevoke.ID},
		unrevoke.ID: {revoke.ID, again.ID},
		again.ID:    {unrevoke.ID, 0},
	}, reverses)
}

func (s *LedgerSuite) TestOnlyAGrantOrAReversalCanBeReversed() {
	topUp, _, err := s.credit("o1", usage.CreditTopUp, 100, "top-up")
	s.Require().NoError(err)
	refund, _, err := s.credit("o1", usage.CreditRefund, -10, "refund")
	s.Require().NoError(err)
	adjustment, _, err := s.credit("o1", usage.CreditAdjustment, 7, "adjust")
	s.Require().NoError(err)

	for _, entry := range []ledger.CreditEntry{topUp, refund, adjustment} {
		_, _, err := s.reverse("o1", entry.ID, -entry.AmountMicros, newKey())
		s.ErrorIs(err, ledger.ErrNotReversible, "%s", entry.Kind)
	}
	s.Equal(int64(97), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestAnEntryIsReversedOnce() {
	grant, _, err := s.credit("o1", usage.CreditGrant, 100, "grant")
	s.Require().NoError(err)
	_, _, err = s.reverse("o1", grant.ID, -100, "revoke-1")
	s.Require().NoError(err)

	_, _, err = s.reverse("o1", grant.ID, -100, "revoke-2")

	s.ErrorIs(err, ledger.ErrAlreadyReversed)
	s.Equal(int64(0), s.account("o1").GrantedMicros)
	entries, _, err := s.ledger.CreditEntries(s.ctx, "o1", nil, 10)
	s.Require().NoError(err)
	s.Len(entries, 2)
}

func (s *LedgerSuite) TestTwoReversalsOfOneEntryAtOnceWriteOne() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	for round := range 10 {
		grant, _, err := s.credit("o1", usage.CreditGrant, 100, fmt.Sprintf("grant-%d", round))
		s.Require().NoError(err)
		var wg sync.WaitGroup
		errs := make([]error, 2)
		for i, pool := range []*ledger.Ledger{s.ledger, other} {
			wg.Go(func() {
				_, _, errs[i] = pool.AddCredit(s.ctx, ledger.CreditEntry{
					OrganizationID: "o1", Kind: usage.CreditAdjustment, AmountMicros: -100, ReversesID: grant.ID,
					IdempotencyKey: fmt.Sprintf("revoke-%d-%d", round, i), Actor: "staff",
				})
			})
		}
		wg.Wait()
		s.True((errs[0] == nil) != (errs[1] == nil), "exactly one wrote it, round %d: %v", round, errs)
		s.True(errors.Is(errs[0], ledger.ErrAlreadyReversed) || errors.Is(errs[1], ledger.ErrAlreadyReversed), "%v", errs)
	}

	s.Equal(int64(0), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestAReversalIsAnAdjustmentOfMinusTheAmount() {
	grant, _, err := s.credit("o1", usage.CreditGrant, 100, "grant")
	s.Require().NoError(err)
	var refused *ledger.InvalidError

	_, _, err = s.reverse("o1", grant.ID, -99, newKey())
	s.Require().ErrorAs(err, &refused)
	s.Contains(refused.Problem, "amount_micros must be -100, not -99")
	_, _, err = s.ledger.AddCredit(s.ctx, ledger.CreditEntry{
		OrganizationID: "o1", Kind: usage.CreditRefund, AmountMicros: -100, ReversesID: grant.ID, IdempotencyKey: newKey(),
	})
	s.Require().ErrorAs(err, &refused)
	s.Contains(refused.Problem, "kind must be adjustment")

	s.Equal(int64(100), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestAnotherOrganizationsEntryIsNotThereToReverse() {
	grant, _, err := s.credit("o1", usage.CreditGrant, 100, "grant")
	s.Require().NoError(err)

	_, _, err = s.reverse("o2", grant.ID, -100, newKey())
	s.ErrorIs(err, ledger.ErrCreditNotFound)
	_, _, err = s.reverse("o1", grant.ID+1000, -100, newKey())
	s.ErrorIs(err, ledger.ErrCreditNotFound)

	s.Equal(int64(100), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestAReversalIsWrittenOnceUnderItsKey() {
	grant, _, err := s.credit("o1", usage.CreditGrant, 100, "grant")
	s.Require().NoError(err)
	first, _, err := s.reverse("o1", grant.ID, -100, "revoke")
	s.Require().NoError(err)
	unrevoke, _, err := s.reverse("o1", first.ID, 100, "unrevoke")
	s.Require().NoError(err)

	replayed, written, err := s.reverse("o1", grant.ID, -100, "revoke")
	s.Require().NoError(err)
	s.False(written)
	s.Equal(first.ID, replayed.ID)
	s.Equal(unrevoke.ID, replayed.ReversedByID, "answered as it is now")
	s.Equal(int64(100), s.account("o1").GrantedMicros)
	_, _, err = s.credit("o1", usage.CreditAdjustment, -100, "revoke")
	s.ErrorIs(err, ledger.ErrIdempotencyKeyReused, "the key without reverses_id is another request")
	_, _, err = s.reverse("o1", grant.ID, 100, "unrevoke")
	s.ErrorIs(err, ledger.ErrIdempotencyKeyReused, "nor with another reverses_id")
}

func (s *LedgerSuite) TestAnIngestedEventIsDebitedFromItsOrganization() {
	s.ingest(
		usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 1000 }),
		usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 2500 }),
		usagetest.Event("o2", func(e *usage.Event) { e.CostMicros = 10 }),
	)

	s.Equal(int64(3500), s.account("o1").SpentMicros)
	s.Equal(int64(10), s.account("o2").SpentMicros)
}

func (s *LedgerSuite) TestABatchIngestedTwiceIsDebitedOnce() {
	batch := []usage.Event{
		usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 1000 }),
		usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 2500 }),
	}

	s.ingest(batch...)
	s.ingest(batch...)

	s.Equal(int64(3500), s.account("o1").SpentMicros)
}

func (s *LedgerSuite) TestAFreeEventTouchesNoAccount() {
	s.ingest(
		usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 0 }),
		usagetest.Event("o1", func(e *usage.Event) { e.Billable = false }),
	)

	_, found, err := s.ledger.Account(s.ctx, "o1")
	s.Require().NoError(err)
	s.False(found)
}

func (s *LedgerSuite) TestEveryWriteIsAudited() {
	s.setTier(usage.Tier{Name: "tier-a", FailMode: usage.FailOpen})
	_, _, err := s.ledger.SetAccountLimits(s.ctx, "o1", usage.Limits{LowBalanceMicros: ptr(int64(5))}, newKey(), staff)
	s.Require().NoError(err)
	_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "a dispute", newKey(), staff)
	s.Require().NoError(err)
	_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountActive, "resolved", newKey(), staff)
	s.Require().NoError(err)
	_, _, err = s.credit("o1", usage.CreditGrant, 9, "grant")
	s.Require().NoError(err)

	audits := s.audits()

	s.Require().Len(audits, 5)
	actions := make([]string, 0, len(audits))
	for _, audit := range audits {
		actions = append(actions, audit.Action)
		s.Equal(staff.ID, audit.Actor, audit.Action)
		s.Equal(staff.Name, audit.ActorName, audit.Action)
	}
	s.Equal([]string{ledger.ActionSetTier, ledger.ActionSetLimits, ledger.ActionSuspend, ledger.ActionRestore, ledger.ActionAddCredit}, actions)
	s.Empty(audits[0].OrganizationID)
	s.Equal("o1", audits[1].OrganizationID)
	s.Equal([]string{"a dispute", "resolved"}, []string{audits[2].Reason, audits[3].Reason})
	var before, after struct {
		Account ledger.Account     `json:"account"`
		Entry   *usage.CreditEntry `json:"entry"`
	}
	s.Require().NoError(json.Unmarshal(audits[4].Before, &before))
	s.Require().NoError(json.Unmarshal(audits[4].After, &after))
	s.Zero(before.Account.GrantedMicros)
	s.Nil(before.Entry)
	s.Equal(int64(9), after.Account.GrantedMicros)
	s.Require().NotNil(after.Entry, "the audit keeps the entry written")
	s.Equal("grant", after.Entry.IdempotencyKey)
	s.Equal(int64(9), after.Entry.AmountMicros)
}

func (s *LedgerSuite) TestTextPostgresWouldChangeIsRefused() {
	var refused *ledger.InvalidError
	_, _, err := s.credit("o1", usage.CreditGrant, 1, "k\xfeey")
	s.Require().ErrorAs(err, &refused, "keys k\\xfeey and k\\xffey would both be stored as k\\uFFFDey")
	_, _, err = s.credit("o1", usage.CreditGrant, 1, "key\n")
	s.Require().ErrorAs(err, &refused)
	_, _, err = s.ledger.AddCredit(s.ctx, ledger.CreditEntry{
		OrganizationID: "o1", Kind: usage.CreditGrant, AmountMicros: 1, IdempotencyKey: "key", Note: "a\x00b",
	})
	s.Require().ErrorAs(err, &refused)
	_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "a\x00b", newKey(), staff)
	s.Require().ErrorAs(err, &refused)
	for _, key := range []string{"", "k\xfeey", "key\n", strings.Repeat("k", 256)} {
		_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "fraud", key, staff)
		s.Require().ErrorAs(err, &refused, "%q", key)
		_, _, err = s.ledger.SetAccountLimits(s.ctx, "o1", usage.Limits{}, key, staff)
		s.Require().ErrorAs(err, &refused, "%q", key)
	}

	_, found, err := s.ledger.Account(s.ctx, "o1")
	s.Require().NoError(err)
	s.False(found, "nothing written")
}

func (s *LedgerSuite) TestTheDatabaseRefusesAnAmountOfTheWrongSign() {
	_, _, err := s.credit("o1", usage.CreditGrant, 1, "grant")
	s.Require().NoError(err)

	for kind, amount := range map[usage.CreditKind]int64{
		usage.CreditGrant: -1, usage.CreditRefund: 1, usage.CreditAdjustment: 0, usage.CreditDisputeReversal: -1,
	} {
		_, err := s.ledger.DB().ExecContext(s.ctx, `INSERT INTO credit_entries (organization_id, kind, amount_micros, idempotency_key)
			VALUES ('o1', ?, ?, ?)`, string(kind), amount, fmt.Sprintf("raw-%s", kind))
		s.Require().ErrorContains(err, "credit_entries_sign_check", "%s %d", kind, amount)
	}
}

func (s *LedgerSuite) TestTwoWritersCreatingOneTierAuditACreateAndAnUpdate() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	for round := range 5 {
		name := fmt.Sprintf("tier-%d", round)
		var wg sync.WaitGroup
		errs := make([]error, 2)
		for i, pool := range []*ledger.Ledger{s.ledger, other} {
			wg.Go(func() {
				_, errs[i] = pool.SetTier(s.ctx, usage.Tier{Name: name, FailMode: usage.FailClosed}, staff)
			})
		}
		wg.Wait()
		s.Require().NoError(errors.Join(errs...))
	}

	created := map[string]int{}
	for _, audit := range s.audits() {
		if string(audit.Before) == "{}" {
			var after usage.Tier
			s.Require().NoError(json.Unmarshal(audit.After, &after))
			created[after.Name]++
		}
	}
	s.Equal(map[string]int{"tier-0": 1, "tier-1": 1, "tier-2": 1, "tier-3": 1, "tier-4": 1}, created,
		"one create per tier; the second writer audits what it replaced")
}

func (s *LedgerSuite) TestSuspendNeedsAReasonAndRestoreClearsIt() {
	_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "", newKey(), staff)
	var refused *ledger.InvalidError
	s.Require().ErrorAs(err, &refused)

	suspended, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "first", newKey(), staff)
	s.Require().NoError(err)
	again, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "second", newKey(), staff)
	s.Require().NoError(err)
	restored, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountActive, "lifted", newKey(), staff)
	s.Require().NoError(err)

	s.Equal("first", suspended.StatusReason)
	s.Equal("second", again.StatusReason, "suspending a suspended account replaces the reason")
	s.Equal(usage.AccountActive, restored.Status)
	s.Empty(restored.StatusReason)
}

func (s *LedgerSuite) TestConcurrentDebitsAndGrantsLoseNothing() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	const writers, rounds = 4, 10
	var wg sync.WaitGroup
	errs := make(chan error, 2*writers*rounds)
	for w := range writers {
		pool := s.ledger
		if w%2 == 1 {
			pool = other
		}
		wg.Go(func() {
			for range rounds {
				batch := []usage.Event{
					usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 7 }),
					usagetest.Event("o2", func(e *usage.Event) { e.CostMicros = 3 }),
				}
				if _, err := pool.Ingest(s.ctx, batch); err != nil {
					errs <- err
				}
			}
		})
		wg.Go(func() {
			for r := range rounds {
				if _, _, err := pool.AddCredit(s.ctx, ledger.CreditEntry{
					OrganizationID: "o1", Kind: usage.CreditTopUp, AmountMicros: 100, IdempotencyKey: fmt.Sprintf("w%d-r%d", w, r), Actor: "proxy",
				}); err != nil {
					errs <- err
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		s.Require().NoError(err)
	}

	o1 := s.account("o1")
	s.Equal(int64(writers*rounds*7), o1.SpentMicros)
	s.Equal(int64(writers*rounds*100), o1.GrantedMicros)
	s.Equal(int64(writers*rounds*3), s.account("o2").SpentMicros)
}

func (s *LedgerSuite) TestOneKeySentByTwoWritersAtOnceCreditsOnce() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	for round := range 10 {
		key := fmt.Sprintf("payment-%d", round)
		var wg sync.WaitGroup
		written := make([]bool, 2)
		errs := make([]error, 2)
		for i, pool := range []*ledger.Ledger{s.ledger, other} {
			wg.Go(func() {
				_, written[i], errs[i] = pool.AddCredit(s.ctx, ledger.CreditEntry{
					OrganizationID: "o1", Kind: usage.CreditTopUp, AmountMicros: 1000, IdempotencyKey: key, Actor: "proxy",
				})
			})
		}
		wg.Wait()
		s.Require().NoError(errors.Join(errs...))
		s.True(written[0] != written[1], "exactly one wrote it, round %d", round)
	}

	s.Equal(int64(10_000), s.account("o1").GrantedMicros)
}

func (s *LedgerSuite) TestStandingsAnswerAccountsTiersAndSpendWindows() {
	s.setTier(usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed, Limits: usage.Limits{LowBalanceMicros: ptr(int64(1))}})
	s.setTier(usage.Tier{Name: "tier-a", FailMode: usage.FailOpen, Limits: usage.Limits{
		SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(100))},
	}})
	_, _, err := s.credit("o1", usage.CreditGrant, 10_000, "grant")
	s.Require().NoError(err)
	_, _, err = s.ledger.SetAccountLimits(s.ctx, "o1", usage.Limits{SpendHardLimit: &usage.SpendHardLimit{MonthlyMicros: ptr(int64(900))}}, newKey(), staff)
	s.Require().NoError(err)
	s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.CostMicros = 2500 }))

	standings, err := s.ledger.Standings(s.ctx, []usage.StandingOrganization{
		{OrganizationID: "o1", OrganizationTier: "tier-a"}, {OrganizationID: "never", OrganizationTier: "tier-z"},
	})

	s.Require().NoError(err)
	s.Require().Len(standings, 2)
	o1, never := standings[0], standings[1]
	s.Equal("tier-a", o1.Tier.Name)
	s.False(o1.FallbackTier)
	s.Equal(usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(100)), MonthlyMicros: ptr(int64(900))}}, o1.Limits)
	s.Equal(int64(10_000), o1.GrantedMicros)
	s.Equal(int64(2500), o1.SpentMicros)
	s.Equal(int64(7500), o1.BalanceMicros)
	start, _ := usage.Window(usage.Daily, s.clock())
	s.Equal(start, o1.SpendWindows.Daily.WindowStart)
	s.Equal(int64(2500), o1.SpendWindows.Daily.SpentMicros)
	s.Equal(int64(2500), o1.SpendWindows.Monthly.SpentMicros)

	s.Equal("tier-z", never.OrganizationTier)
	s.Equal(usage.DefaultTier, never.Tier.Name)
	s.True(never.FallbackTier)
	s.Equal(usage.AccountActive, never.Status)
	s.Zero(never.BalanceMicros)
	s.Equal(usage.Monthly, never.SpendWindows.Monthly.Interval)
	s.Zero(never.SpendWindows.Monthly.SpentMicros)
	_, found, err := s.ledger.Account(s.ctx, "never")
	s.Require().NoError(err)
	s.False(found, "reading a standing writes no account")
}

func (s *LedgerSuite) TestUsageFiltersAndPages() {
	s.ingest(
		usagetest.Event("o1", func(e *usage.Event) { e.AppID, e.UsageCategories = "a1", []string{"team:finance"} }),
		usagetest.Event("o1", func(e *usage.Event) { e.AppID, e.Meter = "a2", "runtime" }),
		usagetest.Event("o1", func(e *usage.Event) { e.AppID, e.UserID = "a1", "u1" }),
		usagetest.Event("o2"),
	)

	finance, _, err := s.ledger.Usage(s.ctx, ledger.UsageFilter{OrganizationID: "o1", Category: "team:finance", Limit: 10})
	s.Require().NoError(err)
	s.Len(finance, 1)
	runtime, _, err := s.ledger.Usage(s.ctx, ledger.UsageFilter{OrganizationID: "o1", Meter: "runtime", Limit: 10})
	s.Require().NoError(err)
	s.Len(runtime, 1)
	user, _, err := s.ledger.Usage(s.ctx, ledger.UsageFilter{OrganizationID: "o1", AppID: "a1", UserID: "u1", Limit: 10})
	s.Require().NoError(err)
	s.Len(user, 1)

	page, more, err := s.ledger.Usage(s.ctx, ledger.UsageFilter{OrganizationID: "o1", Limit: 2})
	s.Require().NoError(err)
	s.True(more)
	last := page[1]
	rest, more, err := s.ledger.Usage(s.ctx, ledger.UsageFilter{OrganizationID: "o1", Limit: 2, After: &ledger.UsagePosition{OccurredAt: last.OccurredAt, UsageID: last.UsageID}})
	s.Require().NoError(err)
	s.False(more)
	s.Len(rest, 1)
	s.True(page[0].OccurredAt.After(rest[0].OccurredAt) || page[0].OccurredAt.Equal(rest[0].OccurredAt), "newest first")
}

func (s *LedgerSuite) TestAStatusChangeIsMadeOnceUnderItsKey() {
	suspended, applied, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "chargeback", "suspend-1", staff)
	s.Require().NoError(err)
	s.True(applied)
	replayed, applied, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "chargeback", "suspend-1", staff)
	s.Require().NoError(err)

	s.False(applied)
	s.Equal(suspended, replayed, "a replay answers the account as it is")
	s.Len(s.audits(), 1, "a replay is not audited")
}

func (s *LedgerSuite) TestALimitsChangeIsMadeOnceUnderItsKey() {
	first := usage.Limits{LowBalanceMicros: ptr(int64(5))}
	second := usage.Limits{LowBalanceMicros: ptr(int64(9))}
	_, applied, err := s.ledger.SetAccountLimits(s.ctx, "o1", first, "limits-1", staff)
	s.Require().NoError(err)
	s.True(applied)
	_, _, err = s.ledger.SetAccountLimits(s.ctx, "o1", second, "limits-2", staff)
	s.Require().NoError(err)

	replayed, applied, err := s.ledger.SetAccountLimits(s.ctx, "o1", first, "limits-1", staff)

	s.Require().NoError(err)
	s.False(applied)
	s.Equal(second, replayed.LimitOverrides, "the replay did not put the first overrides back")
	s.Equal(second, s.account("o1").LimitOverrides)
	s.Len(s.audits(), 2)
}

func (s *LedgerSuite) TestASuspensionReplayedAfterARestoreLeavesTheAccountActive() {
	_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "fraud", "k1", staff)
	s.Require().NoError(err)
	_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountActive, "reviewed", "k2", staff)
	s.Require().NoError(err)

	account, applied, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "fraud", "k1", staff)

	s.Require().NoError(err)
	s.False(applied)
	s.Equal(usage.AccountActive, account.Status)
	s.Equal(usage.AccountActive, s.account("o1").Status)
	s.Len(s.audits(), 2)
}

func (s *LedgerSuite) TestAKeyHoldingAnotherRequestIsRefusedWhateverTheOperation() {
	_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "fraud", "held", staff)
	s.Require().NoError(err)
	_, _, err = s.credit("o1", usage.CreditGrant, 5, "credited")
	s.Require().NoError(err)
	before := s.account("o1")

	for name, write := range map[string]func(key string) error{
		"another reason": func(key string) error {
			_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "not fraud", key, staff)
			return err
		},
		"another organization": func(key string) error {
			_, _, err := s.ledger.SetAccountStatus(s.ctx, "o2", usage.AccountSuspended, "fraud", key, staff)
			return err
		},
		"a restore": func(key string) error {
			_, _, err := s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountActive, "fraud", key, staff)
			return err
		},
		"a limits change": func(key string) error {
			_, _, err := s.ledger.SetAccountLimits(s.ctx, "o1", usage.Limits{}, key, staff)
			return err
		},
		"a credit": func(key string) error {
			_, _, err := s.credit("o1", usage.CreditGrant, 5, key)
			return err
		},
	} {
		s.Require().ErrorIs(write("held"), ledger.ErrIdempotencyKeyReused, name)
		if name != "a credit" {
			s.Require().ErrorIs(write("credited"), ledger.ErrIdempotencyKeyReused, "%s under a credit's key", name)
		}
	}

	s.Equal(before, s.account("o1"), "nothing changed")
	s.Len(s.audits(), 2)
	_, found, err := s.ledger.Account(s.ctx, "o2")
	s.Require().NoError(err)
	s.False(found, "nothing written for the other organization")
}

func (s *LedgerSuite) TestACreditWrittenBeforeManageRequestsStillHoldsItsKey() {
	_, err := s.ledger.DB().ExecContext(s.ctx, `INSERT INTO accounts (organization_id, granted_micros) VALUES ('o1', 7)`)
	s.Require().NoError(err)
	_, err = s.ledger.DB().ExecContext(s.ctx, `INSERT INTO credit_entries (organization_id, kind, amount_micros, idempotency_key)
		VALUES ('o1', 'grant', 7, 'earlier')`)
	s.Require().NoError(err)

	_, _, err = s.ledger.SetAccountStatus(s.ctx, "o1", usage.AccountSuspended, "fraud", "earlier", staff)
	s.Require().ErrorIs(err, ledger.ErrIdempotencyKeyReused)
	_, _, err = s.credit("o1", usage.CreditGrant, 8, "earlier")
	s.Require().ErrorIs(err, ledger.ErrIdempotencyKeyReused)
	entry, written, err := s.credit("o1", usage.CreditGrant, 7, "earlier")
	s.Require().NoError(err)

	s.False(written, "the same entry is a replay")
	s.Equal("earlier", entry.IdempotencyKey)
	s.Equal(int64(7), s.account("o1").GrantedMicros)
	s.Equal(usage.AccountActive, s.account("o1").Status)
	s.Empty(s.audits())
}

func (s *LedgerSuite) TestOneStatusKeySentByTwoWritersAtOnceAppliesOnce() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	for round := range 10 {
		key := fmt.Sprintf("suspend-%d", round)
		var wg sync.WaitGroup
		applied := make([]bool, 2)
		errs := make([]error, 2)
		for i, pool := range []*ledger.Ledger{s.ledger, other} {
			wg.Go(func() {
				_, applied[i], errs[i] = pool.SetAccountStatus(s.ctx, fmt.Sprintf("o%d", round), usage.AccountSuspended, "fraud", key, staff)
			})
		}
		wg.Wait()
		s.Require().NoError(errors.Join(errs...))
		s.True(applied[0] != applied[1], "exactly one applied it, round %d", round)
	}

	s.Len(s.audits(), 10, "one audit row per key")
}
