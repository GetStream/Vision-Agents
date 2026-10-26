//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// AccountSuite reads the organization's account, credit and teams as the hosted router does:
// from the real usage service, which the tests write to as the operator and the routers would.
type AccountSuite struct {
	RouterSuite
	staff *testClient
}

func TestAccountSuite(t *testing.T) {
	runSuite(t, new(AccountSuite))
}

func (s *AccountSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an organization of its own.
func (s *AccountSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
	}}
}

func (s *AccountSuite) organization() string { return s.app.organization.ID }

// account reads the organization's account as the app's backend, with the answer's headers.
func (s *AccountSuite) account() (Account, http.Header) {
	response := s.serverClient.raw(http.MethodGet, "/v1/account", nil)
	defer response.Body.Close() //nolint:errcheck // read-only body
	var answer bytes.Buffer
	_, err := answer.ReadFrom(response.Body)
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, response.StatusCode, answer.String())
	var account Account
	s.Require().NoError(json.Unmarshal(answer.Bytes(), &account))
	return account, response.Header
}

// accountUntil reads the account until done says it is the one wanted.
func (s *AccountSuite) accountUntil(done func(Account, http.Header) bool, why string) (Account, http.Header) {
	var account Account
	var header http.Header
	s.Require().Eventually(func() bool {
		account, header = s.account()
		return done(account, header)
	}, 10*time.Second, 50*time.Millisecond, why)
	return account, header
}

// hold sets micros aside for work of the app, as a request in flight does, until the test ends.
func (s *AccountSuite) hold(micros int64) {
	id := uuid.New()
	owner := routing.Owner{CustomerID: s.customerID(), Attribution: routing.Attribution{OrganizationID: s.organization()}}
	s.Require().NoError(s.enforcer.Reserve(owner, id, micros, time.Minute))
	s.T().Cleanup(func() { s.enforcer.Release(id) })
}

// credit writes a credit entry through the usage service's management API as actor.
func (s *AccountSuite) credit(actor string, entry map[string]any) {
	body, err := json.Marshal(entry)
	s.Require().NoError(err)
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		s.usage.URL+usage.PathManage+"/accounts/"+s.organization()+"/credits", bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Manage-Key", s.usage.ManageKey)
	request.Header.Set("X-Stream-Actor-Id", actor)
	request.Header.Set("X-Stream-Actor-Name", "Ada Lovelace")
	request.Header.Set("Idempotency-Key", uuid.NewString())
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusCreated, response.StatusCode)
}

// spend records billable work of the app that cost micros, as the router ships it.
func (s *AccountSuite) spend(micros int64, change func(*usage.Event)) {
	s.usage.Spend(s.T(), s.organization(), micros, func(event *usage.Event) {
		event.AppID = s.customerID()
		if change != nil {
			change(event)
		}
	})
}

// teams lists the organization's teams as client, through path.
func (s *AccountSuite) teams(client *testClient, path string) []Team {
	var teams []Team
	status, payload := client.call(http.MethodGet, path, nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &teams))
	return teams
}

func (s *AccountSuite) TestTheAccountIsServerSideOnly() {
	for _, path := range []string{"/v1/account", "/v1/account/credits", "/v1/teams"} {
		s.assertPosture(serverOnly, func(as *testClient) int {
			return as.do(http.MethodGet, path, nil, nil)
		})
	}
}

func (s *AccountSuite) TestTheBalanceSubtractsHolds() {
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	s.accountUntil(func(account Account, header http.Header) bool {
		return header.Get(headerBalanceMicros) == "100000000"
	}, "the router has read the account")

	s.hold(5_000_000)

	account, header := s.accountUntil(func(account Account, header http.Header) bool {
		return account.BalanceMicros == 95_000_000
	}, "the hold lowers the balance")
	s.Equal(strconv.FormatInt(account.BalanceMicros, 10), header.Get(headerBalanceMicros), "the header agrees")
	s.Equal(int64(100_000_000), account.GrantedMicros)
	s.Zero(account.SpentMicros, "the usage service has not had the work")
	s.Equal("ok", account.BalanceState)
}

func (s *AccountSuite) TestTheAccountNamesOwnLimitsNotTheirReason() {
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{Channels: map[string]usage.ChannelLimits{"voice": {Concurrent: pointerTo(100)}}})
	s.usage.Suspend(s.T(), s.organization(), "chargebacks under review")

	status, payload := s.serverClient.call(http.MethodGet, "/v1/account", nil)
	s.Require().Equal(http.StatusOK, status, "a suspended organization reads its account: %s", payload)
	var account Account
	s.Require().NoError(json.Unmarshal(payload, &account))

	s.Equal("suspended", account.Status)
	s.NotContains(string(payload), "chargebacks", "the reason is the operator's")
	s.NotContains(string(payload), "status_reason")
	s.Require().Contains(account.Limits.Channels, "voice")
	s.Equal(100, *account.Limits.Channels["voice"].Concurrent)
	s.Equal([]string{"channels.voice.concurrent"}, account.OwnLimits)
	s.Equal(s.organization(), account.OrganizationId)
	s.NotEmpty(account.Tier)
}

func (s *AccountSuite) TestTheAccountShowsTheSpendHardLimit() {
	daily := int64(20_000_000_000)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &daily}})
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	s.spend(41_200_000, nil)
	s.hold(5_000_000)

	account, _ := s.accountUntil(func(account Account, _ http.Header) bool {
		return account.SpendHardLimit != nil && account.SpendHardLimit.Daily != nil &&
			account.SpendHardLimit.Daily.RoomMicros == daily-41_200_000-5_000_000
	}, "the room is the limit less the spend and the hold")

	limit := account.SpendHardLimit.Daily
	s.Equal(daily, limit.LimitMicros)
	s.Equal(int64(41_200_000), limit.SpentMicros)
	_, midnight := usage.Window(usage.Daily, time.Now())
	s.True(limit.WindowEnd.Equal(midnight), "%s is the next midnight UTC", limit.WindowEnd)
	s.Nil(account.SpendHardLimit.Monthly, "no monthly limit")

	zero := int64(0)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &zero, MonthlyMicros: &zero}})
	account, _ = s.account()
	s.Nil(account.SpendHardLimit, "0 is no limit")
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{})
	account, _ = s.account()
	s.Nil(account.SpendHardLimit, "the tier sets none")
}

func (s *AccountSuite) TestCreditsLeaveOutStaffNotes() {
	s.credit("staff:ada", map[string]any{"kind": "grant", "amount_micros": 10_000_000, "reference": "contract 7", "note": "a goodwill grant"})
	s.credit("staff:ada", map[string]any{"kind": "refund", "amount_micros": -1_000_000, "reference": "support ticket 4417"})
	s.credit(usage.StripeActor, map[string]any{"kind": "top_up", "amount_micros": 50_000_000, "reference": "pi_3Nabc"})

	status, payload := s.serverClient.call(http.MethodGet, "/v1/account/credits", nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	var page CreditPage
	s.Require().NoError(json.Unmarshal(payload, &page))

	s.Require().Len(page.Items, 3)
	s.Equal("top_up", page.Items[0].Kind, "newest first")
	s.Equal("pi_3Nabc", page.Items[0].Reference, "a Stripe top-up names its PaymentIntent")
	s.Equal(int64(-1_000_000), page.Items[1].AmountMicros)
	s.Empty(page.Items[1].Reference, "a staff refund's reference is the operator's")
	s.Empty(page.Items[2].Reference)
	for _, hidden := range []string{"ticket 4417", "contract 7", "goodwill", "staff:ada", "Ada Lovelace"} {
		s.NotContains(string(payload), hidden)
	}
}

func (s *AccountSuite) TestCreditsPage() {
	for range 3 {
		s.usage.Credit(s.T(), s.organization(), 1_000_000)
	}
	var first CreditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/account/credits?limit=2", nil, &first))
	s.Require().Len(first.Items, 2)
	s.True(first.HasMore)
	s.Require().NotNil(first.NextCursor)

	var second CreditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/account/credits?limit=2&cursor="+url.QueryEscape(*first.NextCursor), nil, &second))
	s.Require().Len(second.Items, 1)
	s.False(second.HasMore)

	status, failure := s.serverClient.failure(http.MethodGet, "/v1/account/credits?cursor=nonsense", nil)
	s.Equal(http.StatusBadRequest, status, failure)
}

func (s *AccountSuite) TestTheThresholdComesFromThePolicyFirst() {
	s.usage.SetTier(s.T(), usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed, Limits: usage.Limits{LowBalanceMicros: pointerTo(int64(1_000_000))}})
	s.T().Cleanup(func() { s.usage.SetTier(s.T(), usage.Tier{Name: usage.DefaultTier, FailMode: usage.FailClosed}) })
	s.usage.Credit(s.T(), s.organization(), 2_500_000)

	account, _ := s.account()
	s.Equal(int64(1_000_000), account.BalanceWarningMicros)
	s.Equal(warningFromTier, account.BalanceWarningSource)
	s.Equal("ok", account.BalanceState)

	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{LowBalanceMicros: pointerTo(int64(2_000_000))})
	account, _ = s.account()
	s.Equal(int64(2_000_000), account.BalanceWarningMicros)
	s.Equal(warningFromAccount, account.BalanceWarningSource, "the override Stream set for the organization")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization",
		Policy{BalanceWarningMicros: pointerTo(int64(3_000_000))}, nil))
	account, _ = s.accountUntil(func(account Account, _ http.Header) bool {
		return account.BalanceWarningSource == warningFromOrganizationPolicy
	}, "the organization policy's threshold wins")
	s.Equal(int64(3_000_000), account.BalanceWarningMicros)
	s.Equal("low", account.BalanceState, "2.5 is at or below 3")
}

func (s *AccountSuite) TestTeamsListsEveryTeamWithALimitOrSpend() {
	sibling := store.App{OrganizationID: s.organization(), Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &sibling))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "team", "team": "legal", "interval": "weekly", "limit_micros": 250_000_000}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "app_team", "team": "ops", "app_id": sibling.ID, "interval": "daily", "limit_micros": 1_000_000}, nil))
	s.spend(1_000_000, func(event *usage.Event) { event.UsageCategories = []string{"team:finance"} })
	s.spend(2_000_000, func(event *usage.Event) { event.AppID, event.UsageCategories = sibling.ID, []string{"team:finance"} })
	lastMonth, _ := usage.Window(usage.Monthly, time.Now())
	s.spend(9_000_000, func(event *usage.Event) {
		event.UsageCategories, event.OccurredAt = []string{"team:old"}, lastMonth.Add(-48*time.Hour)
	})

	teams := s.teams(s.serverClient, "/v1/teams")

	s.Require().Len(teams, 3, "a team that spent last month only is left out: %+v", teams)
	finance, legal, ops := teams[0], teams[1], teams[2]
	s.Equal("finance", finance.Name)
	s.ElementsMatch([]string{s.customerID(), sibling.ID}, finance.Apps)
	s.Require().Len(finance.Spent, 4)
	s.Equal(LimitInterval(usage.Monthly), finance.Spent[3].Interval)
	s.Equal(int64(3_000_000), finance.Spent[3].SpentMicros)
	s.Empty(finance.Limits)
	s.Equal("legal", legal.Name)
	s.Empty(legal.Apps, "it has a limit and no spend")
	s.Require().Len(legal.Limits, 1)
	s.Equal(LimitSubjectTeam, legal.Limits[0].Subject)
	s.Equal(int64(250_000_000), legal.Limits[0].EffectiveLimitMicros)
	s.Zero(legal.Spent[0].SpentMicros)
	s.Equal("ops", ops.Name)

	within := s.teams(s.serverClient, "/v1/teams?app_id="+url.QueryEscape(sibling.ID))

	s.Require().Len(within, 2, "only what the sibling saw: %+v", within)
	s.Equal("finance", within[0].Name)
	s.Equal([]string{sibling.ID}, within[0].Apps)
	s.Equal(int64(2_000_000), within[0].Spent[0].SpentMicros, "the sibling's spend alone")
	s.Equal("ops", within[1].Name)
	s.Require().Len(within[1].Limits, 1)
	s.Equal(LimitSubjectAppTeam, within[1].Limits[0].Subject)
}

func (s *AccountSuite) TestStaffListTeams() {
	s.spend(1_000_000, func(event *usage.Event) { event.UsageCategories = []string{"team:finance"} })
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "team", "team": "finance", "interval": "daily", "limit_micros": 50_000_000}, nil))

	customer := s.teams(s.serverClient, "/v1/teams")
	staff := s.teams(s.staff, "/v1/manage/teams?organization_id="+url.QueryEscape(s.organization()))

	s.Require().Len(customer, 1)
	s.Equal(customer, staff)
	without := &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{actorIDHeader: {"staff:x"}}}
	s.Equal(http.StatusUnauthorized, without.do(http.MethodGet, "/v1/manage/teams?organization_id="+url.QueryEscape(s.organization()), nil, nil))
	s.Equal(http.StatusBadRequest, s.staff.do(http.MethodGet, "/v1/manage/teams", nil, nil), "no organization_id")
}

// SelfHostedAccountSuite reads the account and the teams on a router with no usage service.
type SelfHostedAccountSuite struct {
	RouterSuite
}

func TestSelfHostedAccountSuite(t *testing.T) {
	runSuite(t, new(SelfHostedAccountSuite))
}

func (s *SelfHostedAccountSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SelfHostedAccountSuite) TestNoUsageServiceAnswersNotEnabled() {
	for _, path := range []string{"/v1/account", "/v1/account/credits"} {
		status, payload := s.serverClient.call(http.MethodGet, path, nil)
		s.Equal(http.StatusNotFound, status, path)
		var answered ErrorResponse
		s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
		s.Equal(codeBillingNotEnabled, answered.Error.Code, path)
	}
}

func (s *SelfHostedAccountSuite) TestTeamsSumTheRequestsThisMonth() {
	record := func(at time.Time, micros int64, team string) {
		s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
			Modality: "llm", CustomerID: s.customerID(), OrganizationID: s.app.organization.ID, Provider: "stub", Model: "stub-llm",
			StartedAt: at, CostMicros: micros, Billable: true, Success: true, UsageCategories: []string{"support", "team:" + team},
		}))
	}
	record(time.Now(), 1_500_000, "finance")
	record(time.Now(), 500_000, "finance")
	lastMonth, _ := usage.Window(usage.Monthly, time.Now())
	record(lastMonth.Add(-48*time.Hour), 9_000_000, "old")

	var teams []Team
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/teams", nil, &teams))

	s.Require().Len(teams, 1, "%+v", teams)
	s.Equal("finance", teams[0].Name)
	s.Equal([]string{s.customerID()}, teams[0].Apps)
	s.Equal(int64(2_000_000), teams[0].Spent[3].SpentMicros)
}
