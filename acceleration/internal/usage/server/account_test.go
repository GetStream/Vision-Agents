//go:build integration

package server_test

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// read sends a GET to path with headers, in pairs, and returns the status and the answer.
func (s *ServerSuite) read(path string, headers ...string) (int, []byte) {
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, s.service.URL+path, nil)
	s.Require().NoError(err)
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Set(headers[i], headers[i+1])
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body
	answer, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, answer
}

// customerView reads path with the ingest token, expects a 200 and decodes it into out.
func (s *ServerSuite) customerView(path string, out any) []byte {
	status, answer := s.read(path, "Authorization", "Bearer "+s.service.Token)
	s.Require().Equal(http.StatusOK, status, "%s: %s", path, answer)
	s.Require().NoError(json.Unmarshal(answer, out), string(answer))
	return answer
}

func (s *ServerSuite) TestTheIngestTokenReadsAnAccount() {
	daily := int64(20_000_000_000)
	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits", usage.Limits{
		SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &daily},
		Channels:       map[string]usage.ChannelLimits{"voice": {Concurrent: ptr(100)}},
	}, key()...)
	s.ok(http.StatusOK, nil, http.MethodPost, "/v1/manage/accounts/"+s.organization+"/suspend",
		server.StatusChange{Reason: "chargebacks under review"}, key()...)
	s.ship(usagetest.Event(s.organization, func(event *usage.Event) { event.CostMicros = 41_200_000 }))

	var account usage.Account
	answer := s.customerView(usage.PathAccounts+"/"+s.organization, &account)

	s.NotContains(string(answer), "status_reason")
	s.NotContains(string(answer), "chargebacks")
	s.Equal(usage.AccountSuspended, account.Status)
	s.Require().NotNil(account.Limits.SpendHardLimit)
	s.Equal(daily, *account.Limits.SpendHardLimit.DailyMicros)
	s.Equal(int64(41_200_000), account.SpendWindows.Daily.SpentMicros)
	s.Equal(int64(41_200_000), account.SpentMicros)
	s.Equal([]string{"channels.voice.concurrent", "spend_hard_limit.daily_micros"}, account.OwnLimits)
	s.Nil(account.LowBalanceMicros, "neither the tier nor the account sets a threshold")
	s.Empty(account.LowBalanceSource)
}

func (s *ServerSuite) TestTheThresholdIsTheAccountsOwnElseTheTiers() {
	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/tiers/"+usage.DefaultTier,
		server.TierSettings{FailMode: usage.FailClosed, Limits: usage.Limits{LowBalanceMicros: ptr(int64(5_000_000))}})

	var account usage.Account
	s.customerView(usage.PathAccounts+"/"+s.organization, &account)
	s.Require().NotNil(account.LowBalanceMicros)
	s.Equal(int64(5_000_000), *account.LowBalanceMicros)
	s.Equal(usage.ThresholdTier, account.LowBalanceSource)
	s.Empty(account.OwnLimits)

	s.ok(http.StatusOK, nil, http.MethodPut, "/v1/manage/accounts/"+s.organization+"/limits",
		usage.Limits{LowBalanceMicros: ptr(int64(8_000_000))}, key()...)
	account = usage.Account{}
	s.customerView(usage.PathAccounts+"/"+s.organization, &account)
	s.Equal(int64(8_000_000), *account.LowBalanceMicros)
	s.Equal(usage.ThresholdAccount, account.LowBalanceSource)
	s.Equal([]string{"low_balance_micros"}, account.OwnLimits)
}

func (s *ServerSuite) TestTheCustomerViewOfCreditsLeavesOutWhatStaffWrote() {
	path := "/v1/manage/accounts/" + s.organization + "/credits"
	s.ok(http.StatusCreated, nil, http.MethodPost, path,
		server.CreditRequest{Kind: usage.CreditGrant, AmountMicros: 10_000_000, Reference: "contract 7", Note: "a goodwill grant"}, key()...)
	s.ok(http.StatusCreated, nil, http.MethodPost, path,
		server.CreditRequest{Kind: usage.CreditRefund, AmountMicros: -1_000_000, Reference: "support ticket 4417"}, key()...)
	s.ok(http.StatusCreated, nil, http.MethodPost, path,
		server.CreditRequest{Kind: usage.CreditTopUp, AmountMicros: 50_000_000, Reference: "pi_3Nabc"},
		append(key(), "X-Stream-Actor-Id", usage.StripeActor)...)

	var page usage.CustomerCreditPage
	answer := s.customerView(usage.PathAccounts+"/"+s.organization+"/credits", &page)

	s.Require().Len(page.Items, 3)
	s.Equal(usage.CreditTopUp, page.Items[0].Kind, "newest first")
	s.Equal("pi_3Nabc", page.Items[0].Reference)
	s.Empty(page.Items[1].Reference, "a staff refund's reference is the operator's")
	s.Empty(page.Items[2].Reference)
	for _, hidden := range []string{"ticket 4417", "contract 7", "goodwill", "staff-1", "Ada", "actor", "note", "idempotency_key"} {
		s.NotContains(string(answer), hidden)
	}
}

func (s *ServerSuite) TestTheCustomerViewOfCreditsPages() {
	for range 3 {
		s.grant(s.organization, 1_000_000, uuid.NewString())
	}
	var first usage.CustomerCreditPage
	s.customerView(usage.PathAccounts+"/"+s.organization+"/credits?limit=2", &first)
	s.Require().Len(first.Items, 2)
	s.True(first.HasMore)

	var second usage.CustomerCreditPage
	s.customerView(usage.PathAccounts+"/"+s.organization+"/credits?limit=2&cursor="+first.NextCursor, &second)
	s.Require().Len(second.Items, 1)
	s.False(second.HasMore)
	s.Less(second.Items[0].ID, first.Items[1].ID)
}

func (s *ServerSuite) TestTheIngestTokenReadsTeams() {
	team := func(app string, micros int64, categories ...string) usage.Event {
		return usagetest.Event(s.organization, func(event *usage.Event) {
			event.AppID, event.CostMicros, event.UsageCategories = app, micros, categories
		})
	}
	lastMonth, _ := usage.Window(usage.Monthly, time.Now())
	old := team("a1", 9_000_000, "team:old")
	old.OccurredAt = lastMonth.Add(-48 * time.Hour)
	s.ship(
		team("a1", 1_000_000, "team:finance", "support"),
		team("a2", 2_000_000, "team:finance"),
		team("a1", 3_000_000, "team:sales"),
		team("a2", 4_000_000, "support"),
		old,
	)

	var teams usage.Teams
	s.customerView(usage.PathAccounts+"/"+s.organization+"/teams", &teams)

	s.Require().Len(teams.Teams, 2, "a category that is no team's, and a team that spent last month only, are left out")
	finance, sales := teams.Teams[0], teams.Teams[1]
	s.Equal("finance", finance.Team)
	s.Equal("sales", sales.Team)
	s.Require().Len(finance.Windows, len(usage.Intervals))
	for i, window := range finance.Windows {
		s.Equal(usage.Intervals[i], window.Interval)
		s.Equal(int64(3_000_000), window.SpentMicros, "%s: both apps' work", window.Interval)
		s.Equal(usage.SubjectCategory, window.Subject)
	}
	s.Require().Len(finance.Apps, 2)
	s.Equal("a1", finance.Apps[0].AppID)
	s.Equal(int64(1_000_000), finance.Apps[0].Windows[0].SpentMicros)
	s.Equal("a2", finance.Apps[1].AppID)
	s.Equal(int64(2_000_000), finance.Apps[1].Windows[3].SpentMicros)
	s.Require().Len(sales.Apps, 1)
	s.Equal(int64(3_000_000), sales.Windows[3].SpentMicros)
}

func (s *ServerSuite) TestTheCustomerViewOfUsageLeavesOutWhatTheOperatorReads() {
	event := usagetest.Event(s.organization, func(event *usage.Event) {
		event.UserID, event.UsageCategories, event.PriceVersion = "u1", []string{"support", "team:finance"}, "prices-v7"
		event.Attributes, event.Tags = map[string]string{"country": "nl"}, map[string]string{"project": "alpha"}
		// 13 digits: longer than any digit run a usage id (12 at most) or a time can hold.
		event.BaseCostMicros, event.CostMicros = 7_777_777_777_777, 1_234
	})
	bare := usagetest.Event(s.organization, func(event *usage.Event) { event.OccurredAt = event.OccurredAt.Add(-time.Second) })
	s.ship(event, bare)

	var page usage.CustomerUsagePage
	answer := s.customerView(usage.PathAccounts+"/"+s.organization+"/usage", &page)

	s.Require().Len(page.Items, 2)
	listed := page.Items[0]
	s.Equal(event.UsageID, listed.UsageID, "newest first")
	s.Equal("u1", listed.UserID)
	s.Equal([]string{"support", "team:finance"}, listed.UsageCategories)
	s.Equal(int64(1_234), listed.CostMicros)
	s.Equal(event.Quantities, listed.Quantities)
	s.Equal(map[string]string{"country": "nl"}, listed.Attributes)
	s.Equal(map[string]string{"project": "alpha"}, listed.Tags)
	s.Equal("llm", listed.Meter)
	s.Equal([]string{}, page.Items[1].UsageCategories, "none is an empty list")
	for _, hidden := range []string{"organization_tier", "organization_id", "base_cost_micros", "7777777777777", "price_version", "prices-v7", "source", "router/test/pod"} {
		s.NotContains(string(answer), hidden)
	}
}

func (s *ServerSuite) TestTheCustomerViewOfUsageFiltersAndPagesAsListUsageDoes() {
	base := time.Now().Add(-time.Minute).UTC().Truncate(time.Microsecond)
	var finance []usage.Event
	for i := range 3 {
		finance = append(finance, usagetest.Event(s.organization, func(event *usage.Event) {
			event.OccurredAt, event.AppID, event.UserID = base.Add(time.Duration(i)*time.Second), "a1", "u1"
			event.UsageCategories = []string{"team:finance"}
		}))
	}
	s.ship(append(finance,
		usagetest.Event(s.organization, func(event *usage.Event) { event.AppID, event.UserID = "a2", "u1" }),
		usagetest.Event(s.organization, func(event *usage.Event) { event.AppID, event.APIKey = "a1", "hook:chat" }),
	)...)
	path := usage.PathAccounts + "/" + s.organization + "/usage?app_id=a1&user_id=u1&team=finance&limit=2"

	var first, second usage.CustomerUsagePage
	s.customerView(path, &first)
	s.Require().Len(first.Items, 2)
	s.True(first.HasMore)
	s.customerView(path+"&cursor="+first.NextCursor, &second)
	s.Require().Len(second.Items, 1)
	s.False(second.HasMore)
	s.Equal([]uuid.UUID{finance[2].UsageID, finance[1].UsageID, finance[0].UsageID},
		[]uuid.UUID{first.Items[0].UsageID, first.Items[1].UsageID, second.Items[0].UsageID})

	var keyed usage.CustomerUsagePage
	s.customerView(usage.PathAccounts+"/"+s.organization+"/usage?api_key=hook:chat", &keyed)
	s.Require().Len(keyed.Items, 1)
	s.Equal("hook:chat", keyed.Items[0].APIKey)

	for _, refused := range []string{"?user_id=u1", "?team=finance&category=support", "?cursor=not-a-cursor"} {
		status, answer := s.read(usage.PathAccounts+"/"+s.organization+"/usage"+refused, "Authorization", "Bearer "+s.service.Token)
		s.Equal(http.StatusBadRequest, status, "%s: %s", refused, answer)
	}
}

func (s *ServerSuite) TestTheIngestTokenWritesNothingElse() {
	account := "/v1/manage/accounts/" + s.organization
	writes := [][3]string{
		{http.MethodPost, account + "/credits", `{"kind":"grant","amount_micros":1000000}`},
		{http.MethodPost, account + "/suspend", `{"reason":"x"}`},
		{http.MethodPost, account + "/restore", `{"reason":"x"}`},
		{http.MethodPut, account + "/limits", `{}`},
		{http.MethodPost, usage.PathAccounts + "/" + s.organization + "/credits", `{"kind":"grant","amount_micros":1000000}`},
		{http.MethodPut, usage.PathAccounts + "/" + s.organization, `{}`},
	}
	for _, write := range writes {
		status, answer := s.manage(write[0], write[1], json.RawMessage(write[2]),
			"X-Manage-Key", "", "Authorization", "Bearer "+s.service.Token, "Idempotency-Key", usagetest.Organization())
		s.GreaterOrEqual(status, http.StatusBadRequest, "%s %s: %s", write[0], write[1], answer)
	}

	s.Empty(s.audits())
	var credited usage.CustomerCreditPage
	s.customerView(usage.PathAccounts+"/"+s.organization+"/credits", &credited)
	s.Empty(credited.Items)
	s.Equal(usage.AccountActive, s.account(s.organization).Status)
}

func (s *ServerSuite) TestTheManageKeyDoesNotOpenTheCustomerView() {
	for _, path := range []string{
		usage.PathAccounts + "/" + s.organization,
		usage.PathAccounts + "/" + s.organization + "/credits",
		usage.PathAccounts + "/" + s.organization + "/teams",
		usage.PathAccounts + "/" + s.organization + "/usage",
	} {
		for _, headers := range [][]string{
			{"X-Manage-Key", s.service.ManageKey},
			{"Authorization", "Bearer " + s.service.ManageKey},
			{},
		} {
			status, answer := s.read(path, headers...)
			s.Equal(http.StatusUnauthorized, status, "%s with %v: %s", path, headers, answer)
		}
	}
}
