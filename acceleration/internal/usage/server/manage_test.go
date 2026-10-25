//go:build integration

package server_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// devTiers is the development tier file, which the local stack seeds too.
const devTiers = "../../../tiers/dev.json"

// failure is a management API failure as answered.
type failure struct {
	Status int    `json:"status"`
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

func ptr[T any](v T) *T { return &v }

// key is the Idempotency-Key header of a key no other write uses, for manage's headers.
func key() []string { return []string{"Idempotency-Key", uuid.NewString()} }

// manage calls the management API with the suite's manage key and an actor; headers, in
// pairs, replace those, and an empty value leaves the header out.
func (s *ServerSuite) manage(method, path string, body any, headers ...string) (int, []byte) {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		s.Require().NoError(err)
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(s.ctx, method, s.service.URL+path, payload)
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Manage-Key", s.service.ManageKey)
	request.Header.Set("X-Stream-Actor-Id", "staff-1")
	request.Header.Set("X-Stream-Actor-Name", "Ada")
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Del(headers[i])
		if headers[i+1] != "" {
			request.Header.Set(headers[i], headers[i+1])
		}
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body
	answer, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, answer
}

// ok calls the management API, expects status, and decodes the answer into out.
func (s *ServerSuite) ok(status int, out any, method, path string, body any, headers ...string) {
	got, answer := s.manage(method, path, body, headers...)
	s.Require().Equal(status, got, "%s %s: %s", method, path, answer)
	if out != nil {
		s.Require().NoError(json.Unmarshal(answer, out), string(answer))
	}
}

// refused calls the management API and expects a failure of status.
func (s *ServerSuite) refused(status int, method, path string, body any, headers ...string) failure {
	got, answer := s.manage(method, path, body, headers...)
	s.Require().Equal(status, got, "%s %s: %s", method, path, answer)
	var answered failure
	s.Require().NoError(json.Unmarshal(answer, &answered), string(answer))
	return answered
}

func (s *ServerSuite) seedDevTiers() ledger.TierFile {
	file, err := ledger.ReadTierFile(devTiers)
	s.Require().NoError(err)
	_, err = s.service.Ledger.SyncTiers(s.ctx, file, "test")
	s.Require().NoError(err)
	return *file
}

func (s *ServerSuite) account(organization string, query ...string) usage.Standing {
	var standing usage.Standing
	path := "/v1/manage/accounts/" + organization
	if len(query) > 0 {
		path += "?" + query[0]
	}
	s.ok(http.StatusOK, &standing, http.MethodGet, path, nil)
	return standing
}

func (s *ServerSuite) standing(organization, tier string) usage.Standing {
	standings, err := s.service.Client.Standings(s.ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: organization, OrganizationTier: tier},
	}})
	s.Require().NoError(err)
	s.Require().Len(standings, 1)
	return standings[0]
}

func (s *ServerSuite) grant(organization string, amount int64, key string) (int, usage.CreditEntry) {
	status, answer := s.manage(http.MethodPost, "/v1/manage/accounts/"+organization+"/credits",
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: amount, Note: "contract"}, "Idempotency-Key", key)
	var entry usage.CreditEntry
	s.Require().NoError(json.Unmarshal(answer, &entry), string(answer))
	return status, entry
}

func (s *ServerSuite) audits() []ledger.ManageAudit {
	var rows []ledger.ManageAudit
	s.Require().NoError(s.service.Ledger.DB().NewSelect().Model(&rows).Order("id").Scan(s.ctx))
	return rows
}

func (s *ServerSuite) TestTheManageKeyIsRequired() {
	routes := [][2]string{
		{http.MethodGet, "/v1/manage/tiers"},
		{http.MethodPut, "/v1/manage/tiers/tier-a"},
		{http.MethodPost, "/v1/manage/accounts/" + s.organization + "/credits"},
		{http.MethodGet, "/v1/manage/accounts/" + s.organization},
	}
	for _, route := range routes {
		for _, headers := range [][]string{
			{"X-Manage-Key", ""},
			{"X-Manage-Key", "wrong"},
			{"X-Manage-Key", s.service.ManageKey + "x"},
			{"X-Manage-Key", s.service.Token},
			{"X-Manage-Key", "", "Authorization", "Bearer " + s.service.Token},
		} {
			status, answer := s.manage(route[0], route[1], map[string]any{"fail_mode": "open"}, append(headers, "Idempotency-Key", "k")...)
			s.Equal(http.StatusUnauthorized, status, "%s %s with %v: %s", route[0], route[1], headers, answer)
			s.Contains(string(answer), `"code":"unauthenticated"`)
			s.NotContains(string(answer), s.service.ManageKey)
		}
	}
	s.Empty(s.audits())
	tiers, err := s.service.Ledger.Tiers(s.ctx)
	s.Require().NoError(err)
	s.Empty(tiers)
}

func (s *ServerSuite) TestTheManageKeyDoesNotOpenIngest() {
	for _, headers := range []map[string]string{
		{"X-Manage-Key": s.service.ManageKey},
		{"Authorization": "Bearer " + s.service.ManageKey},
	} {
		body, err := json.Marshal(usage.Batch{Events: []usage.Event{usagetest.Event(s.organization)}})
		s.Require().NoError(err)
		request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, s.service.URL+usage.PathEvents, bytes.NewReader(body))
		s.Require().NoError(err)
		for name, value := range headers {
			request.Header.Set(name, value)
		}
		response, err := http.DefaultClient.Do(request)
		s.Require().NoError(err)
		s.Require().NoError(response.Body.Close())
		s.Equal(http.StatusUnauthorized, response.StatusCode, "%v", headers)
	}
	s.Empty(s.stored())
}

func (s *ServerSuite) TestWithNoManageKeyTheRoutesAreGone() {
	off := usagetest.Start(s.T(), usagetest.Options{DSN: s.dsn})

	for _, key := range []string{"", s.service.ManageKey, off.Token} {
		request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, off.URL+"/v1/manage/tiers", nil)
		s.Require().NoError(err)
		request.Header.Set("X-Manage-Key", key)
		request.Header.Set("X-Stream-Actor-Id", "staff-1")
		response, err := http.DefaultClient.Do(request)
		s.Require().NoError(err)
		answer, err := io.ReadAll(response.Body)
		s.Require().NoError(err)
		s.Require().NoError(response.Body.Close())

		s.Equal(http.StatusNotFound, response.StatusCode, "with %q: %s", key, answer)
		s.Contains(string(answer), `"code":"not_configured"`)
	}
}

func (s *ServerSuite) TestAWriteWithoutAnActorIsRefused() {
	for _, headers := range [][]string{
		{"X-Stream-Actor-Id", ""},
		{"X-Stream-Actor-Id", "   "},
		{"X-Stream-Actor-Id", strings.Repeat("a", 257)},
		{"X-Stream-Actor-Name", "%ZZ"},
	} {
		answered := s.refused(http.StatusBadRequest, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{FailMode: usage.FailOpen}, headers...)
		s.Equal("invalid_request", answered.Code, "%v", headers)
	}
	s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/tiers", nil, "X-Stream-Actor-Id", "")
	s.Empty(s.audits())
}

func (s *ServerSuite) TestEveryChangeIsRecordedAgainstTheActor() {
	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{FailMode: usage.FailOpen},
		"X-Stream-Actor-Id", "operator:42", "X-Stream-Actor-Name", url.PathEscape("Łukasz Nowak"))
	status, _ := s.grant(s.organization, 100, "grant-1")
	s.Require().Equal(http.StatusCreated, status)

	audits := s.audits()
	s.Require().Len(audits, 2)
	s.Equal("operator:42", audits[0].Actor)
	s.Equal("Łukasz Nowak", audits[0].ActorName)
	s.Equal(ledger.ActionSetTier, audits[0].Action)
	s.Equal("staff-1", audits[1].Actor)
	s.Equal(ledger.ActionAddCredit, audits[1].Action)
	entries, _, err := s.service.Ledger.CreditEntries(s.ctx, s.organization, nil, 10)
	s.Require().NoError(err)
	s.Equal([]string{"staff-1", "Ada"}, []string{entries[0].Actor, entries[0].ActorName})
}

func (s *ServerSuite) TestAStandingAnswersTheTierItWasAskedAbout() {
	file := s.seedDevTiers()

	tierA := s.standing(s.organization, "tier-a")
	unknown := s.standing(s.organization, "tier-c")

	s.Equal(file.Tiers[1], tierA.Tier)
	s.False(tierA.FallbackTier)
	s.Equal(file.Tiers[1].Limits, tierA.Limits)
	s.Equal(file.Tiers[0], unknown.Tier)
	s.Equal("tier-c", unknown.OrganizationTier)
	s.True(unknown.FallbackTier)
}

func (s *ServerSuite) TestATierWithNoRowAnswersTheBuiltInDefaultBeforeAnyIsWritten() {
	standing := s.standing(s.organization, "tier-a")

	s.Equal(usage.BuiltInDefaultTier(), standing.Tier)
	s.True(standing.FallbackTier)
	s.Equal(usage.FailClosed, standing.Tier.FailMode)
}

func (s *ServerSuite) TestASetTierReachesTheNextStanding() {
	var written usage.Tier
	s.ok(http.StatusOK, &written, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{
		FailMode: usage.FailOpen, Limits: usage.Limits{LowBalanceMicros: ptr(int64(42))},
	})

	standing := s.standing(s.organization, "tier-a")

	s.Equal(usage.Tier{Name: "tier-a", FailMode: usage.FailOpen, Limits: usage.Limits{LowBalanceMicros: ptr(int64(42))}}, written)
	s.Equal(written, standing.Tier)
	s.Equal(ptr(int64(42)), standing.Limits.LowBalanceMicros)
}

func (s *ServerSuite) TestATierNeedsNoSpendHardLimit() {
	for _, limits := range []usage.Limits{
		{},
		{SpendHardLimit: &usage.SpendHardLimit{}},
		{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(0)), MonthlyMicros: ptr(int64(0))}},
	} {
		s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{FailMode: usage.FailClosed, Limits: limits})
		standing := s.standing(s.organization, "tier-a")
		for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
			_, limited := standing.Limits.SpendHardLimitMicros(interval)
			s.False(limited, "%+v %s", limits, interval)
		}
	}

	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{
		FailMode: usage.FailClosed, Limits: usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(5_000_000))}},
	})
	daily, limited := s.standing(s.organization, "tier-a").Limits.SpendHardLimitMicros(usage.Daily)
	s.True(limited)
	s.Equal(int64(5_000_000), daily)
}

func (s *ServerSuite) TestASetTierThatIsNotValidIsRefused() {
	answered := s.refused(http.StatusBadRequest, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{
		FailMode: usage.FailOpen, Limits: usage.Limits{Channels: map[string]usage.ChannelLimits{"sms": {Countries: []string{"usa"}}}},
	})
	s.Contains(answered.Detail, `"usa" is not an ISO 3166-1 alpha-2 code`)

	s.refused(http.StatusBadRequest, http.MethodPut, "/v1/manage/tiers/"+strings.Repeat("t", 65), server.TierSettings{FailMode: usage.FailOpen})
	for _, body := range []any{
		map[string]any{"fail_mode": "sometimes"},
		server.TierSettings{FailMode: usage.FailOpen, Limits: usage.Limits{Sandbox: &usage.SandboxLimits{Network: ptr("internet")}}},
		server.TierSettings{FailMode: usage.FailOpen, Limits: usage.Limits{LowBalanceMicros: ptr(int64(-1))}},
	} {
		status, answer := s.manage(http.MethodPut, "/v1/manage/tiers/tier-a", body)
		s.Equal(http.StatusUnprocessableEntity, status, "the schema refuses it: %s", answer)
	}
	s.Empty(s.audits())
}

func (s *ServerSuite) TestGetTierAnswersTheFallbackForAnUnknownName() {
	file := s.seedDevTiers()

	var known, unknown server.TierAnswer
	s.ok(http.StatusOK, &known, http.MethodGet, "/v1/manage/tiers/tier-a", nil)
	s.ok(http.StatusOK, &unknown, http.MethodGet, "/v1/manage/tiers/tier-c", nil)

	s.Equal(server.TierAnswer{Tier: file.Tiers[1]}, known)
	s.Equal(server.TierAnswer{Tier: file.Tiers[0], FallbackTier: true}, unknown)
	var list server.TierList
	s.ok(http.StatusOK, &list, http.MethodGet, "/v1/manage/tiers", nil)
	s.Equal(file.Tiers, list.Tiers)
}

func (s *ServerSuite) TestDeletingTheDefaultTierIsAConflict() {
	s.seedDevTiers()

	answered := s.refused(http.StatusConflict, http.MethodDelete, "/v1/manage/tiers/default", nil)
	s.ok(http.StatusNoContent, nil, http.MethodDelete, "/v1/manage/tiers/tier-a", nil)

	s.Equal("default_tier", answered.Code)
	s.True(s.standing(s.organization, "tier-a").FallbackTier, "its organizations fall back to default")
}

func (s *ServerSuite) TestLimitOverridesAreLaidOverTheTier() {
	file := s.seedDevTiers()
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.OrganizationTier = "tier-a" }))

	var standing usage.Standing
	s.ok(http.StatusOK, &standing, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits",
		usage.Limits{Channels: map[string]usage.ChannelLimits{"voice": {Concurrent: ptr(100)}}}, key()...)

	tier := file.Tiers[1].Limits
	s.Equal("tier-a", standing.OrganizationTier, "the tier its newest event carried")
	s.Equal(ptr(100), standing.Limits.Channels["voice"].Concurrent)
	s.Equal(tier.Channels["voice"].PerDay, standing.Limits.Channels["voice"].PerDay)
	s.Equal(tier.Channels["sms"], standing.Limits.Channels["sms"])
	s.Equal(tier.Sandbox, standing.Limits.Sandbox)
	s.Equal(usage.Limits{Channels: map[string]usage.ChannelLimits{"voice": {Concurrent: ptr(100)}}}, standing.Overrides)
	s.Equal(standing, s.account(s.organization))

	var cleared usage.Standing
	s.ok(http.StatusOK, &cleared, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits", usage.Limits{}, key()...)
	s.Equal(tier, cleared.Limits, "{} clears the overrides")
	s.True(cleared.Overrides.IsZero())
}

func (s *ServerSuite) TestOneOrganizationsOverrideLeavesItsTierAlone() {
	file := s.seedDevTiers()
	other := usagetest.Organization()

	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits",
		usage.Limits{Channels: map[string]usage.ChannelLimits{"voice": {Concurrent: ptr(100)}}}, key()...)

	s.Equal(ptr(100), s.standing(s.organization, "tier-a").Limits.Channels["voice"].Concurrent)
	s.Equal(file.Tiers[1].Limits, s.standing(other, "tier-a").Limits)
	tier, _, err := s.service.Ledger.Tier(s.ctx, "tier-a")
	s.Require().NoError(err)
	s.Equal(file.Tiers[1], tier)
}

func (s *ServerSuite) TestListTierOverridesAnswersThoseLastSeenOnIt() {
	other := usagetest.Organization()
	defer usagetest.Forget(s.T(), s.redis, other)
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.OrganizationTier = "tier-a" }),
		usagetest.Event(other, func(e *usage.Event) { e.OrganizationTier = "tier-b" }))
	overrides := usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(1))}}
	for _, organization := range []string{s.organization, other} {
		s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+organization+"/limits", overrides, key()...)
	}

	var page server.TierOverridePage
	s.ok(http.StatusOK, &page, http.MethodGet, "/v1/manage/tiers/tier-a/overrides", nil)

	s.Equal(server.TierOverridePage{Items: []server.TierOverride{{OrganizationID: s.organization, LimitOverrides: overrides}}}, page)
}

func (s *ServerSuite) TestAGrantRaisesTheBalanceAndIsListed() {
	status, entry := s.grant(s.organization, 20_000_000, "contract-2026-10")
	s.Require().Equal(http.StatusCreated, status)
	again, replayed := s.grant(s.organization, 20_000_000, "contract-2026-10")

	s.Equal(http.StatusOK, again)
	s.Equal(entry, replayed)
	s.Equal(usage.CreditEntry{
		ID: entry.ID, OrganizationID: s.organization, Kind: usage.CreditGrant, AmountMicros: 20_000_000,
		IdempotencyKey: "contract-2026-10", Note: "contract", Actor: "staff-1", ActorName: "Ada", CreatedAt: entry.CreatedAt,
	}, entry)
	s.Equal(int64(20_000_000), s.account(s.organization).BalanceMicros)
	s.Equal(int64(20_000_000), s.standing(s.organization, "").BalanceMicros)
	var page server.CreditEntryPage
	s.ok(http.StatusOK, &page, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/credits", nil)
	s.Equal(server.CreditEntryPage{Items: []usage.CreditEntry{entry}}, page)

	answered := s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 1})
	s.Contains(answered.Detail, "Idempotency-Key")
	s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 1}, "Idempotency-Key", strings.Repeat("k", 256))
	s.Equal(int64(20_000_000), s.account(s.organization).GrantedMicros)
}

func (s *ServerSuite) TestACreditKeyReusedForAnotherEntryIsAConflict() {
	status, _ := s.grant(s.organization, 100, "key-1")
	s.Require().Equal(http.StatusCreated, status)

	answered := s.refused(http.StatusConflict, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 200}, "Idempotency-Key", "key-1")

	s.Equal("idempotency_key_reused", answered.Code)
	s.Equal(int64(100), s.account(s.organization).GrantedMicros)
}

func (s *ServerSuite) TestACreditWithTheWrongSignIsABadRequest() {
	answered := s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
		server.CreditRequest{Kind: usage.CreditRefund, AmountMicros: 100}, "Idempotency-Key", "refund-1")

	s.Contains(answered.Detail, "a refund takes credit away")
	_, found, err := s.service.Ledger.Account(s.ctx, s.organization)
	s.Require().NoError(err)
	s.False(found)
}

func (s *ServerSuite) TestAddCreditHasNoTopUpRule() {
	topUp := func(key string) int {
		status, _ := s.manage(http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
			server.CreditRequest{Kind: usage.CreditTopUp, AmountMicros: 10_000_000_000, Reference: "ref-" + key}, "Idempotency-Key", key)
		return status
	}

	s.Equal(http.StatusCreated, topUp("payment-1"))
	s.Equal(http.StatusCreated, topUp("payment-2"))
	s.ok(http.StatusOK, nil, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{Reason: "a dispute"}, key()...)
	s.Equal(http.StatusCreated, topUp("payment-3"))

	s.Equal(int64(30_000_000_000), s.account(s.organization).GrantedMicros)
}

func (s *ServerSuite) TestSuspendNeedsAReasonAndRestoreClearsIt() {
	s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{}, key()...)
	s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{Reason: strings.Repeat("r", 501)}, key()...)

	var suspended, restored usage.Standing
	s.ok(http.StatusOK, &suspended, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{Reason: "early fraud warning"}, key()...)
	s.Equal(usage.AccountSuspended, s.standing(s.organization, "").Status, "a suspension shows in the next standing")
	s.ok(http.StatusOK, &restored, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/restore", server.StatusChange{Reason: "reviewed"}, key()...)

	s.Equal(usage.AccountSuspended, suspended.Status)
	s.Equal("early fraud warning", suspended.StatusReason)
	s.Equal(usage.AccountActive, restored.Status)
	s.Empty(restored.StatusReason)
	audits := s.audits()
	s.Require().Len(audits, 2)
	s.Equal([]string{"suspend", "restore"}, []string{audits[0].Action, audits[1].Action})
	s.Equal([]string{"early fraud warning", "reviewed"}, []string{audits[0].Reason, audits[1].Reason})
}

func (s *ServerSuite) TestAnOrganizationWithNoAccountAnswersItsVirtualAccount() {
	standing := s.account(s.organization)

	s.Equal(s.organization, standing.OrganizationID)
	s.Equal(usage.AccountActive, standing.Status)
	s.Zero(standing.GrantedMicros)
	s.Zero(standing.BalanceMicros)
	s.True(standing.FallbackTier)
	_, found, err := s.service.Ledger.Account(s.ctx, s.organization)
	s.Require().NoError(err)
	s.False(found, "reading it writes nothing")
}

func (s *ServerSuite) TestGetAccountAnswersOnTheTierAskedAbout() {
	s.seedDevTiers()
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.OrganizationTier = "tier-a" }))

	s.Equal("tier-a", s.account(s.organization).Tier.Name)
	s.Equal(usage.DefaultTier, s.account(s.organization, "organization_tier=tier-z").Tier.Name)
}

func (s *ServerSuite) TestSpendFromAnEventShowsInTheStanding() {
	_, _ = s.grant(s.organization, 10_000, "grant")

	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.CostMicros = 2_500 }))

	for _, standing := range []usage.Standing{s.standing(s.organization, ""), s.account(s.organization)} {
		s.Equal(int64(2_500), standing.SpentMicros)
		s.Equal(int64(7_500), standing.BalanceMicros)
	}
}

func (s *ServerSuite) TestTheStandingCarriesTheOrganizationsSpendWindows() {
	never := usagetest.Organization()
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.AppID, e.CostMicros = "a1", 300 }),
		usagetest.Event(s.organization, func(e *usage.Event) { e.AppID, e.CostMicros = "a2", 700 }))

	standings, err := s.service.Client.Standings(s.ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: s.organization}, {OrganizationID: never},
	}})

	s.Require().NoError(err)
	now := time.Now()
	for interval, window := range map[usage.Interval]usage.WindowSpend{
		usage.Daily: standings[0].SpendWindows.Daily, usage.Monthly: standings[0].SpendWindows.Monthly,
	} {
		start, end := usage.Window(interval, now)
		s.Equal(usage.WindowSpend{
			SubjectRef: usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectOrganization},
			Interval:   interval, WindowStart: start, WindowEnd: end, SpentMicros: 1000, Events: 2,
		}, window)
	}
	s.Zero(standings[1].SpendWindows.Daily.SpentMicros)
	s.Zero(standings[1].SpendWindows.Monthly.SpentMicros)
	start, _ := usage.Window(usage.Monthly, now)
	s.Equal(start, standings[1].SpendWindows.Monthly.WindowStart)
	s.Equal(standings[0].SpendWindows, s.account(s.organization).SpendWindows)
}

func (s *ServerSuite) TestANewDayStartsTheDailyWindowAtZero() {
	var mu sync.Mutex
	now := time.Date(2026, 10, 21, 23, 59, 0, 0, time.UTC)
	clock := func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return now
	}
	moved := usagetest.Start(s.T(), usagetest.Options{DSN: s.dsn, Now: clock, ManageKey: "moved-key"})
	_, err := moved.Client.Ship(s.ctx, []usage.Event{usagetest.Event(s.organization, func(e *usage.Event) {
		e.OccurredAt, e.CostMicros = time.Date(2026, 10, 21, 23, 59, 0, 0, time.UTC), 1234
	})})
	s.Require().NoError(err)
	mu.Lock()
	now = time.Date(2026, 10, 22, 0, 0, 1, 0, time.UTC)
	mu.Unlock()

	standings, err := moved.Client.Standings(s.ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{{OrganizationID: s.organization}}})

	s.Require().NoError(err)
	windows := standings[0].SpendWindows
	s.Equal(time.Date(2026, 10, 22, 0, 0, 0, 0, time.UTC), windows.Daily.WindowStart)
	s.Zero(windows.Daily.SpentMicros)
	s.Equal(time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC), windows.Monthly.WindowStart)
	s.Equal(int64(1234), windows.Monthly.SpentMicros)
}

func (s *ServerSuite) TestASpendHardLimitOverrideShowsInTheStanding() {
	overrides := usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(1_000_000))}}

	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits", overrides, key()...)

	for _, standing := range []usage.Standing{s.account(s.organization), s.standing(s.organization, "")} {
		s.Equal(overrides, standing.Overrides)
		daily, limited := standing.Limits.SpendHardLimitMicros(usage.Daily)
		s.True(limited)
		s.Equal(int64(1_000_000), daily)
	}
}

func (s *ServerSuite) TestListUsageFiltersByCategory() {
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.UsageCategories = []string{"project:x"} }),
		usagetest.Event(s.organization, func(e *usage.Event) { e.UsageCategories = []string{"project:y"} }),
		usagetest.Event(s.organization))

	var page server.UsageEventPage
	s.ok(http.StatusOK, &page, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?category=project:x", nil)

	s.Require().Len(page.Items, 1)
	s.Equal([]string{"project:x"}, page.Items[0].UsageCategories)
}

func (s *ServerSuite) TestListUsageFiltersByTeam() {
	finance := usagetest.Event(s.organization, func(e *usage.Event) {
		e.UserID, e.UsageCategories = "u1", []string{"team:finance", "project:x"}
	})
	s.ship(finance, usagetest.Event(s.organization, func(e *usage.Event) { e.UsageCategories = []string{"team:sales"} }))

	var page server.UsageEventPage
	s.ok(http.StatusOK, &page, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?team=finance", nil)

	s.Require().Len(page.Items, 1)
	s.Equal(finance.UsageID, page.Items[0].UsageID)
	s.Equal(s.organization, page.Items[0].OrganizationID)
	s.Equal("u1", page.Items[0].UserID)
	s.Equal(usage.DefaultTier, page.Items[0].OrganizationTier)
	s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?team=finance&category=team:finance", nil)
	s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?team=fin:ance", nil)
	s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?user_id=u1", nil)
}

func (s *ServerSuite) TestListUsageTakesATimeRange() {
	base := time.Now().Add(-time.Hour).UTC().Truncate(time.Second)
	inside := usagetest.Event(s.organization, func(e *usage.Event) { e.OccurredAt = base.Add(10 * time.Minute) })
	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.OccurredAt = base }), inside,
		usagetest.Event(s.organization, func(e *usage.Event) { e.OccurredAt = base.Add(20 * time.Minute) }))

	var page server.UsageEventPage
	query := url.Values{"from": {base.Add(time.Minute).Format(time.RFC3339)}, "to": {base.Add(20 * time.Minute).Format(time.RFC3339)}}
	s.ok(http.StatusOK, &page, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?"+query.Encode(), nil)

	s.Require().Len(page.Items, 1)
	s.Equal(inside.UsageID, page.Items[0].UsageID)
}

func (s *ServerSuite) TestListUsagePagesTheEvents() {
	base := time.Now().Add(-time.Minute).UTC().Truncate(time.Microsecond)
	var shipped []usage.Event
	for i := range 5 {
		shipped = append(shipped, usagetest.Event(s.organization, func(e *usage.Event) { e.OccurredAt = base.Add(time.Duration(i) * time.Second) }))
	}
	s.ship(shipped...)

	var seen []string
	cursor := ""
	for range 5 {
		path := "/v1/manage/accounts/" + s.organization + "/usage?limit=2"
		if cursor != "" {
			path += "&cursor=" + cursor
		}
		var page server.UsageEventPage
		s.ok(http.StatusOK, &page, http.MethodGet, path, nil)
		for _, event := range page.Items {
			seen = append(seen, event.UsageID.String())
		}
		if !page.HasMore {
			break
		}
		cursor = page.NextCursor
	}

	expected := make([]string, 0, len(shipped))
	for i := len(shipped) - 1; i >= 0; i-- {
		expected = append(expected, shipped[i].UsageID.String())
	}
	s.Equal(expected, seen, "newest first, each once")
	s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/usage?cursor=not-a-cursor", nil)
}

func (s *ServerSuite) TestAnOrganizationIDNoEventCouldCarryIsRefused() {
	for _, id := range []string{url.PathEscape("a|b"), strings.Repeat("o", 256), "a%00b", "a%FFb", "a%0Ab"} {
		s.refused(http.StatusBadRequest, http.MethodGet, fmt.Sprintf("/v1/manage/accounts/%s", id), nil)
		s.refused(http.StatusBadRequest, http.MethodPost, fmt.Sprintf("/v1/manage/accounts/%s/credits", id),
			server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 1}, "Idempotency-Key", "k-"+id)
	}
	_, found, err := s.service.Ledger.Account(s.ctx, "ab")
	s.Require().NoError(err)
	s.False(found, "a NUL stripped would have credited another organization")
}

func (s *ServerSuite) TestAnIdempotencyKeyPostgresWouldChangeIsRefused() {
	for _, key := range []string{"k\xfeey", "k\xffey"} {
		answered := s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
			server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 1}, "Idempotency-Key", key)
		s.Contains(answered.Detail, "not UTF-8")
	}
	s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/credits",
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 1, Note: "a\x00b"}, "Idempotency-Key", "k")
	s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{Reason: "a\x00b"}, key()...)
	for _, path := range []string{"/suspend", "/restore"} {
		answered := s.refused(http.StatusBadRequest, http.MethodPost, "/v1/manage/accounts/"+s.organization+path,
			server.StatusChange{Reason: "fraud"}, "Idempotency-Key", "k\xfeey")
		s.Contains(answered.Detail, "not UTF-8", path)
	}
	answered := s.refused(http.StatusBadRequest, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits",
		usage.Limits{}, "Idempotency-Key", "k\xffey")
	s.Contains(answered.Detail, "not UTF-8")
	_, found, err := s.service.Ledger.Account(s.ctx, s.organization)
	s.Require().NoError(err)
	s.False(found)
}

func (s *ServerSuite) TestAMisspeltFieldIsRefusedAndChangesNothing() {
	overrides := usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: ptr(int64(1_000_000))}}
	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits", overrides, key()...)
	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/tiers/tier-a", server.TierSettings{FailMode: usage.FailOpen})
	audits := len(s.audits())

	for _, call := range []struct {
		method, path string
		body         map[string]any
	}{
		{http.MethodPut, "/v1/manage/accounts/" + s.organization + "/limits", map[string]any{"spend_hard_limt": map[string]any{"daily_micros": 5}}},
		{http.MethodPut, "/v1/manage/accounts/" + s.organization + "/limits", map[string]any{"spend_hard_limit": map[string]any{"daily_mircos": 5}}},
		{http.MethodPut, "/v1/manage/accounts/" + s.organization + "/limits", map[string]any{"channels": map[string]any{"sms": map[string]any{"perday": 5}}}},
		{http.MethodPut, "/v1/manage/accounts/" + s.organization + "/limits", map[string]any{"sandbox": map[string]any{"cpu": 5}}},
		{http.MethodPut, "/v1/manage/tiers/tier-a", map[string]any{"fail_mode": "closed", "limits": map[string]any{"spend_hard_limit": map[string]any{"daily_mircos": 5}}}},
		{http.MethodPut, "/v1/manage/tiers/tier-a", map[string]any{"fail_mode": "closed", "limit": map[string]any{}}},
		{http.MethodPost, "/v1/manage/accounts/" + s.organization + "/credits", map[string]any{"kind": "grant", "amount_micros": 1, "amout": 2}},
		{http.MethodPost, "/v1/manage/accounts/" + s.organization + "/suspend", map[string]any{"reason": "x", "reson": "x"}},
	} {
		status, answer := s.manage(call.method, call.path, call.body, "Idempotency-Key", "typo")
		s.Equal(http.StatusUnprocessableEntity, status, "%s %s %v: %s", call.method, call.path, call.body, answer)
	}

	standing := s.account(s.organization)
	s.Equal(overrides, standing.Overrides, "the overrides there are untouched")
	s.Equal(usage.AccountActive, standing.Status)
	s.Zero(standing.GrantedMicros)
	tier, _, err := s.service.Ledger.Tier(s.ctx, "tier-a")
	s.Require().NoError(err)
	s.Equal(usage.FailOpen, tier.FailMode)
	s.Len(s.audits(), audits, "nothing written")
}

// keyedWrite is one account write that answers the standing, made under an Idempotency-Key.
type keyedWrite struct {
	method, path string
	body         any
}

func (s *ServerSuite) keyedWrites(organization string) map[string]keyedWrite {
	account := "/v1/manage/accounts/" + organization
	return map[string]keyedWrite{
		"suspend": {http.MethodPost, account + "/suspend", server.StatusChange{Reason: "fraud"}},
		"restore": {http.MethodPost, account + "/restore", server.StatusChange{Reason: "reviewed"}},
		"limits":  {http.MethodPut, account + "/limits", usage.Limits{LowBalanceMicros: ptr(int64(5))}},
	}
}

func (s *ServerSuite) TestAStateChangeWithoutAnIdempotencyKeyIsRefused() {
	for name, write := range s.keyedWrites(s.organization) {
		for _, headers := range [][]string{nil, {"Idempotency-Key", strings.Repeat("k", 256)}} {
			answered := s.refused(http.StatusBadRequest, write.method, write.path, write.body, headers...)
			s.Equal("invalid_request", answered.Code, name)
			s.Contains(answered.Detail, "Idempotency-Key", name)
		}
	}

	s.Empty(s.audits())
	_, found, err := s.service.Ledger.Account(s.ctx, s.organization)
	s.Require().NoError(err)
	s.False(found)
}

func (s *ServerSuite) TestAReplayedStateChangeChangesNothing() {
	for name, write := range s.keyedWrites(s.organization) {
		audits := len(s.audits())
		headers := key()
		var first, replayed usage.Standing
		s.ok(http.StatusOK, &first, write.method, write.path, write.body, headers...)
		s.ok(http.StatusOK, &replayed, write.method, write.path, write.body, headers...)

		s.Equal(first, replayed, name)
		s.Len(s.audits(), audits+1, "%s: one audit row", name)
	}
}

func (s *ServerSuite) TestASuspensionReplayedAfterARestoreLeavesTheOrganizationActive() {
	account := "/v1/manage/accounts/" + s.organization
	s.ok(http.StatusOK, nil, http.MethodPost, account+"/suspend", server.StatusChange{Reason: "fraud"}, "Idempotency-Key", "k1")
	s.ok(http.StatusOK, nil, http.MethodPost, account+"/restore", server.StatusChange{Reason: "reviewed"}, "Idempotency-Key", "k2")

	var replayed usage.Standing
	s.ok(http.StatusOK, &replayed, http.MethodPost, account+"/suspend", server.StatusChange{Reason: "fraud"}, "Idempotency-Key", "k1")

	s.Equal(usage.AccountActive, replayed.Status, "the replay answers the standing as it is now")
	s.Equal(usage.AccountActive, s.standing(s.organization, "").Status)
	s.Len(s.audits(), 2)
}

func (s *ServerSuite) TestAnIdempotencyKeyIsOneSpaceAcrossTheAccountWrites() {
	other := usagetest.Organization()
	account := "/v1/manage/accounts/" + s.organization
	s.ok(http.StatusOK, nil, http.MethodPost, account+"/suspend", server.StatusChange{Reason: "fraud"}, "Idempotency-Key", "suspend-1")
	status, _ := s.grant(s.organization, 100, "credit-1")
	s.Require().Equal(http.StatusCreated, status)
	audits := len(s.audits())

	for name, call := range map[string]keyedWrite{
		"another body":         {http.MethodPost, account + "/suspend", server.StatusChange{Reason: "not fraud"}},
		"another organization": {http.MethodPost, "/v1/manage/accounts/" + other + "/suspend", server.StatusChange{Reason: "fraud"}},
		"another operation":    {http.MethodPost, account + "/restore", server.StatusChange{Reason: "fraud"}},
		"limits":               {http.MethodPut, account + "/limits", usage.Limits{}},
		"a credit":             {http.MethodPost, account + "/credits", server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 100}},
	} {
		answered := s.refused(http.StatusConflict, call.method, call.path, call.body, "Idempotency-Key", "suspend-1")
		s.Equal("idempotency_key_reused", answered.Code, name)
	}
	for name, write := range s.keyedWrites(s.organization) {
		answered := s.refused(http.StatusConflict, write.method, write.path, write.body, "Idempotency-Key", "credit-1")
		s.Equal("idempotency_key_reused", answered.Code, "%s under a credit's key", name)
	}
	again, replayed := s.grant(s.organization, 100, "credit-1")

	s.Equal(http.StatusOK, again, "a credit replay is still answered with its entry")
	s.Equal("credit-1", replayed.IdempotencyKey)
	standing := s.account(s.organization)
	s.Equal(usage.AccountSuspended, standing.Status)
	s.Equal("fraud", standing.StatusReason)
	s.Equal(int64(100), standing.GrantedMicros)
	s.True(standing.Overrides.IsZero())
	s.Len(s.audits(), audits)
	_, found, err := s.service.Ledger.Account(s.ctx, other)
	s.Require().NoError(err)
	s.False(found)
}

func (s *ServerSuite) TestOneStateChangeSentAtOnceIsMadeOnce() {
	const callers = 8
	headers := key()
	var wg sync.WaitGroup
	statuses := make([]int, callers)
	for i := range callers {
		wg.Go(func() {
			statuses[i], _ = s.manage(http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend", server.StatusChange{Reason: "fraud"}, headers...)
		})
	}
	wg.Wait()

	for _, status := range statuses {
		s.Equal(http.StatusOK, status)
	}
	audits := s.audits()
	s.Require().Len(audits, 1, "one change, one audit row")
	s.Equal(ledger.ActionSuspend, audits[0].Action)
	s.Equal(usage.AccountSuspended, s.account(s.organization).Status)
}

func (s *ServerSuite) TestACursorThatIsNotAPositionIsRefused() {
	for _, cursor := range []string{"bnVsbA", "WzFd", "e30x"} { // null, [1], {}1
		s.refused(http.StatusBadRequest, http.MethodGet, "/v1/manage/accounts/"+s.organization+"/credits?cursor="+cursor, nil)
	}
}
