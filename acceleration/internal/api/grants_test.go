//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// grant makes a grant as the app's backend, and answers it.
func (s *LimitsSuite) grant(body map[string]any) LimitGrant {
	var made LimitGrant
	status, payload := s.serverClient.call(http.MethodPost, "/v1/spend/limits/grants", body)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &made))
	return made
}

// grants are the organization's grants as the app's backend lists them.
func (s *LimitsSuite) grants(query string) []LimitGrant {
	var page LimitGrantPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits/grants"+query, nil, &page))
	return page.Items
}

// spentTeam has team finance's monthly limit of $1 spent.
func (s *LimitsSuite) spentTeam() *testClient {
	s.create(map[string]any{"subject": "team", "team": "finance", "interval": "monthly", "limit_micros": 1_000_000})
	s.spend(1_000_000, func(event *usage.Event) { event.UsageCategories = []string{"team:finance"} })
	finance := s.backend(s.app, jwt.MapClaims{"team": "finance"})
	s.searchedUntil(finance, http.StatusPaymentRequired)
	return finance
}

func (s *LimitsSuite) TestAOneTimeGrantForATeam() {
	finance := s.spentTeam()
	tomorrow := time.Now().Add(24 * time.Hour).UTC()

	// One search costs $0.005, so the grant pays for two.
	made := s.grant(map[string]any{"subject": "team", "team": "finance", "amount_micros": 10_000, "expires_at": tomorrow, "reason": "quarter close"})
	s.Equal(GrantLive, made.State)
	s.Equal("finance", made.Team)
	s.Equal(int64(10_000), made.RemainingMicros)

	s.searchedUntil(finance, http.StatusOK)
	s.searchedUntil(finance, http.StatusOK)
	failure := s.searchedUntil(finance, http.StatusPaymentRequired)
	s.Equal(codeSpendLimitReached, failure.Code)
	s.Contains(failure.Message, "team finance", "once it is used, the limit's refusal as before")

	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).
			Where("? = ANY(limit_grants)", made.Id).Scan(context.Background())
		return err == nil && len(rows) >= 2
	}, 5*time.Second, 50*time.Millisecond, "the rows admitted on it name it")
	var cost int64
	for _, row := range rows {
		cost += row.CostMicros
	}
	listed := s.grants("?subject=team&team=finance")
	s.Require().Len(listed, 1)
	s.Equal(cost, listed[0].UsedMicros)
	s.Equal(GrantUsed, listed[0].State)
	s.Equal(int64(0), listed[0].RemainingMicros)
	s.Len(s.grants("?state=used"), 1)
	s.Empty(s.grants("?state=live"))
}

func (s *LimitsSuite) TestAGrantUntilMidnightLetsTheUserSpendAgainToday() {
	s.create(map[string]any{"subject": "user", "app_id": s.customerID(), "interval": "daily", "limit_micros": 1_000_000})
	s.spend(1_000_000, func(event *usage.Event) { event.UserID = "u1" })
	u1 := s.signedIn(jwt.MapClaims{"user_id": "u1"}, "jwt", user, "u1")
	s.searchedUntil(u1, http.StatusTooManyRequests)
	_, midnight := usage.Window(usage.Daily, time.Now())

	made := s.grant(map[string]any{"subject": "user", "app_id": s.customerID(), "user_id": "u1", "amount_micros": 1_000_000,
		"expires_at": midnight, "reason": "a support case"})

	s.Require().NotNil(made.ExpiresAt)
	s.True(made.ExpiresAt.Equal(midnight))
	s.searchedUntil(u1, http.StatusOK)
	var read SubjectUsage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/spend/limits/usage?subject=user&app_id="+s.customerID()+"&user_id=u1", nil, &read))
	s.True(read.Windows[0].OnGrant, "the usage says the user is running on a grant")
	s.Require().Len(read.Grants, 1)
	s.Equal(made.Id, read.Grants[0].Id)
}

func (s *LimitsSuite) TestARevokedGrantStopsAtOnce() {
	finance := s.spentTeam()
	made := s.grant(map[string]any{"subject": "team", "team": "finance", "amount_micros": 1_000_000, "reason": "close"})
	s.searchedUntil(finance, http.StatusOK)

	var revoked LimitGrant
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/spend/limits/grants/"+strconv.FormatInt(made.Id, 10)+"/revoke",
		map[string]any{"reason": "the close is done"}, &revoked))

	s.Equal(GrantRevoked, revoked.State)
	s.NotNil(revoked.RevokedAt)
	status, _ := finance.call(http.MethodPost, "/v1/search", SearchRequest{Query: "what is the time"})
	s.Equal(http.StatusPaymentRequired, status, "the router that revoked it refuses at once")
	s.Require().Eventually(func() bool {
		return s.grants("?state=revoked")[0].UsedMicros > 0
	}, 5*time.Second, 50*time.Millisecond, "it keeps what it paid for")
	again := LimitGrant{}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/spend/limits/grants/"+strconv.FormatInt(made.Id, 10)+"/revoke",
		map[string]any{"reason": "twice"}, &again))
	s.True(again.RevokedAt.Equal(*revoked.RevokedAt), "revoking again changes nothing")
}

func (s *LimitsSuite) TestAGrantNeedsALimit() {
	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits/grants",
		map[string]any{"subject": "team", "team": "nobody", "amount_micros": 5, "reason": "x"})

	s.Equal(http.StatusNotFound, status)
	s.Equal(codeNoLimit, failure.Code)
}

func (s *LimitsSuite) TestAGrantMoreThanAYearAwayOrPastIsRefused() {
	s.create(map[string]any{"subject": "organization", "interval": "monthly", "limit_micros": 5})
	for _, expires := range []time.Time{time.Now().Add(367 * 24 * time.Hour), time.Now().Add(-time.Minute)} {
		status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits/grants",
			map[string]any{"subject": "organization", "amount_micros": 5, "expires_at": expires, "reason": "x"})

		s.Equal(http.StatusBadRequest, status)
		s.Equal(codeInvalidExpiry, failure.Code)
	}
}

func (s *LimitsSuite) TestTheTwentyFirstLiveGrantIsAConflict() {
	s.create(map[string]any{"subject": "organization", "interval": "monthly", "limit_micros": 5})
	for range store.MaxLiveGrants {
		s.grant(map[string]any{"subject": "organization", "amount_micros": 5, "reason": "x"})
	}

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits/grants",
		map[string]any{"subject": "organization", "amount_micros": 5, "reason": "x"})

	s.Equal(http.StatusConflict, status)
	s.Equal(codeTooManyGrants, failure.Code)
}

func (s *LimitsSuite) TestASystemFieldOnAGrantIsAnUnknownField() {
	s.create(map[string]any{"subject": "organization", "interval": "monthly", "limit_micros": 5})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits/grants",
		map[string]any{"subject": "organization", "amount_micros": 5, "reason": "x", "system": true})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeValidationFailed, failure.Code)
}

func (s *LimitsSuite) TestACustomerGrantDoesNotLiftTheSpendHardLimit() {
	daily := int64(1_000_000)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &daily}})
	s.create(map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 1_000_000})
	s.spend(1_000_000, nil)
	s.grant(map[string]any{"subject": "organization", "amount_micros": 5_000_000, "reason": "x"})

	failure := s.searchedUntil(s.serverClient, http.StatusPaymentRequired)

	s.Equal(codeSystemLimitReached, failure.Code, "the operator's limit refuses first, and no grant lifts it")
}

func (s *LimitsSuite) TestStaffGrantForACustomerThroughManage() {
	s.create(map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 5})

	status, payload := s.staff.call(http.MethodPost, "/v1/manage/spend/limits/grants", map[string]any{
		"organization_id": s.organization(), "subject": "app", "app_id": s.customerID(), "amount_micros": 5, "reason": "support",
	})

	s.Require().Equal(http.StatusCreated, status, string(payload))
	listed := s.grants("")
	s.Require().Len(listed, 1, "the customer reads staff's grant")
	s.Contains(listed[0].CreatedBy, "manage:")
	entries, err := s.store.AuditEntries(context.Background(), s.organization(), store.AuditEntryFilter{ResourceType: store.AuditSpendLimit})
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Equal("grant:"+strconv.FormatInt(listed[0].Id, 10), entries[0].ResourceID)
	s.Equal(store.AuditSourceManage, entries[0].Source)
}
