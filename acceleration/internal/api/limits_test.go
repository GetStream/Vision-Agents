//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// LimitsSuite runs the limits API as the hosted router does: spend is read from the real usage
// service, which the tests spend in directly, as the rows the router ships would.
type LimitsSuite struct {
	RouterSuite
	staff *testClient
}

func TestLimitsSuite(t *testing.T) {
	runSuite(t, new(LimitsSuite))
}

func (s *LimitsSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an organization of its own, with credit, so only its limits refuse.
func (s *LimitsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	s.staff = &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
	}}
}

func (s *LimitsSuite) organization() string { return s.app.organization.ID }

// create writes a limit as the app's backend, and answers it.
func (s *LimitsSuite) create(body map[string]any) Limit {
	var created Limit
	status, payload := s.serverClient.call(http.MethodPost, "/v1/spend/limits", body)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &created))
	return created
}

// refusal is the failure a write is answered with.
func (s *LimitsSuite) refusal(client *testClient, method, path string, body any) (int, ErrorDetail) {
	status, payload := client.call(method, path, body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

// backend is a server-side client of app, signed with its own credential, naming claims.
func (s *LimitsSuite) backend(app testApp, claims jwt.MapClaims) *testClient {
	claims["server"], claims["exp"] = true, time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(app.secret))
	s.Require().NoError(err)
	header := http.Header{}
	header.Set(auth.APIKeyHeader, app.key)
	header.Set("Authorization", "Bearer "+token)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)
	return &testClient{suite: &s.RouterSuite, header: header, kind: server, token: token}
}

// searchedUntil waits for a search as client to be answered status, once the money is read.
func (s *LimitsSuite) searchedUntil(client *testClient, status int) ErrorDetail {
	var failure ErrorDetail
	s.Require().Eventually(func() bool {
		answered, payload := client.call(http.MethodPost, "/v1/search", SearchRequest{Query: "what is the time"})
		if answered != status {
			return false
		}
		if status >= http.StatusBadRequest {
			var body ErrorResponse
			s.Require().NoError(json.Unmarshal(payload, &body), string(payload))
			failure = body.Error
		}
		return true
	}, 10*time.Second, 50*time.Millisecond, "searching answers %d", status)
	return failure
}

func (s *LimitsSuite) spend(micros int64, change func(*usage.Event)) {
	s.usage.Spend(s.T(), s.organization(), micros, func(event *usage.Event) {
		event.AppID = s.customerID()
		if change != nil {
			change(event)
		}
	})
}

func (s *LimitsSuite) TestAClientTokenCannotReadOrWriteLimits() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/spend/limits", nil, nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/spend/limits", map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 1}, nil)
	})
	status, failure := s.refusal(s.client, http.MethodGet, "/v1/spend/limits/usage?subject=organization", nil)
	s.Equal(http.StatusForbidden, status)
	s.Equal(codeServerSideOnly, failure.Code)
}

func (s *LimitsSuite) TestOneAppsServerCredentialManagesTheWholeOrganization() {
	sibling := store.App{OrganizationID: s.organization(), Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &sibling))

	created := s.create(map[string]any{"subject": "app", "app_id": sibling.ID, "interval": "daily", "limit_micros": 1_000_000})

	s.Equal(LimitSubjectApp, created.Subject)
	s.Equal(LimitLevel(store.LevelApp), created.Level, "the level an app's id implies")
	s.Equal(sibling.ID, created.AppId)
	var listed LimitPage
	s.Require().Equal(http.StatusOK, s.backend(s.data.keyed(s.app.organization, sibling), jwt.MapClaims{}).
		do(http.MethodGet, "/v1/spend/limits", nil, &listed))
	s.Require().Len(listed.Items, 1, "the sibling's credential reads the organization's limits")
	s.Equal(created.Id, listed.Items[0].Id)
	s.Equal("api", created.UpdatedBy)
}

func (s *LimitsSuite) TestAnAppLimitAboveItsOrganizationsIsRefused() {
	organization := s.create(map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 10_000_000})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 11_000_000})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeLimitAboveParent, failure.Code)
	s.Contains(failure.Message, "the organization (limit "+strconv.FormatInt(organization.Id, 10)+")")
	s.create(map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "weekly", "limit_micros": 11_000_000})
}

func (s *LimitsSuite) TestAKeyLimitIsBoundByItsAppAndOrganization() {
	s.create(map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 10_000_000})
	app := s.create(map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 5_000_000})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "api_key", "app_id": s.customerID(), "api_key": "key_a", "interval": "daily", "limit_micros": 6_000_000})
	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeLimitAboveParent, failure.Code)
	s.Contains(failure.Message, "app "+s.customerID()+" (limit "+strconv.FormatInt(app.Id, 10)+")")

	status, failure = s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "api_key", "level": "organization", "interval": "daily", "limit_micros": 11_000_000})
	s.Equal(http.StatusBadRequest, status, "a default for every key of the organization is held to the organization")
	s.Contains(failure.Message, "the organization")
}

func (s *LimitsSuite) TestAnAppCategoryIsBoundByTheOrganizationsCategory() {
	category := s.create(map[string]any{"subject": "category", "category": "project:onboarding", "interval": "monthly", "limit_micros": 2_000_000})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits", map[string]any{
		"subject": "app_category", "app_id": s.customerID(), "category": "project:onboarding", "interval": "monthly", "limit_micros": 3_000_000,
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "category project:onboarding (limit "+strconv.FormatInt(category.Id, 10)+")")
}

func (s *LimitsSuite) TestLoweringTheOrganizationBelowAnAppIsAllowedAndShown() {
	organization := s.create(map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 10_000_000})
	app := s.create(map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 8_000_000})
	s.Equal(int64(8_000_000), app.EffectiveLimitMicros)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/spend/limits/"+strconv.FormatInt(organization.Id, 10),
		map[string]any{"limit_micros": 3_000_000}, nil))

	var read Limit
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits/"+strconv.FormatInt(app.Id, 10), nil, &read))
	s.Equal(int64(8_000_000), read.LimitMicros, "the app's own stays")
	s.Equal(int64(3_000_000), read.EffectiveLimitMicros, "and what it can spend is the organization's")
}

func (s *LimitsSuite) TestADuplicateIsAConflictNamingTheFirst() {
	first := s.create(map[string]any{"subject": "user", "app_id": s.customerID(), "interval": "daily", "limit_micros": 1})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "user", "app_id": s.customerID(), "interval": "daily", "limit_micros": 2})

	s.Equal(http.StatusConflict, status)
	s.Equal(codeLimitExists, failure.Code)
	s.Contains(failure.Message, "limit "+strconv.FormatInt(first.Id, 10))
}

func (s *LimitsSuite) TestAKeyLimitRefusesThatKeyOnly() {
	other := s.data.keyed(s.app.organization, s.app.app)
	s.create(map[string]any{"subject": "api_key", "app_id": s.customerID(), "api_key": s.app.key, "interval": "daily", "limit_micros": 1_000_000})
	s.spend(1_000_000, func(event *usage.Event) { event.APIKey = s.app.key })

	failure := s.searchedUntil(s.serverClient, http.StatusPaymentRequired)

	s.Equal(codeSpendLimitReached, failure.Code)
	s.Contains(failure.Message, "on key "+s.app.key+" of app "+s.customerID())
	s.searchedUntil(s.backend(other, jwt.MapClaims{}), http.StatusOK)
}

func (s *LimitsSuite) TestATeamLimitRefusesItsTeamOnly() {
	created := s.create(map[string]any{"subject": "team", "team": "finance", "interval": "weekly", "limit_micros": 1_000_000})
	s.Equal(LimitSubjectTeam, created.Subject)
	s.Equal("finance", created.Team)
	s.Empty(created.Category)
	s.spend(1_000_000, func(event *usage.Event) { event.UsageCategories = []string{"team:finance"} })

	failure := s.searchedUntil(s.backend(s.app, jwt.MapClaims{"team": "finance"}), http.StatusPaymentRequired)

	s.Equal(codeSpendLimitReached, failure.Code)
	s.Contains(failure.Message, "the weekly limit of $1.00 on team finance is spent; it resets at")
	s.searchedUntil(s.backend(s.app, jwt.MapClaims{"team": "sales"}), http.StatusOK)
}

func (s *LimitsSuite) TestACategoryLimitRefusesWorkCarryingIt() {
	s.create(map[string]any{"subject": "category", "category": "project:onboarding", "interval": "daily", "limit_micros": 1_000_000})
	s.spend(1_000_000, func(event *usage.Event) { event.UsageCategories = []string{"project:onboarding"} })

	failure := s.searchedUntil(s.backend(s.app, jwt.MapClaims{"usage_categories": []string{"project:onboarding"}}), http.StatusPaymentRequired)

	s.Contains(failure.Message, "on category project:onboarding is spent")
	s.searchedUntil(s.backend(s.app, jwt.MapClaims{"usage_categories": []string{"project:support"}}), http.StatusOK)
}

func (s *LimitsSuite) TestATeamIsStoredAsACategoryAndReadAsATeam() {
	team := s.create(map[string]any{"subject": "team", "team": "finance", "interval": "daily", "limit_micros": 5})
	s.create(map[string]any{"subject": "category", "category": "project:x", "interval": "daily", "limit_micros": 5})

	stored, err := s.store.Limit(context.Background(), s.organization(), team.Id)
	s.Require().NoError(err)
	s.Equal(usage.SubjectCategory, stored.Subject)
	s.Equal("team:finance", stored.Category)
	var teams, categories LimitPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits?subject=team", nil, &teams))
	s.Require().Len(teams.Items, 1)
	s.Equal("finance", teams.Items[0].Team)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits?subject=category", nil, &categories))
	s.Require().Len(categories.Items, 1, "the teams are not categories on the API")
	s.Equal("project:x", categories.Items[0].Category)
}

func (s *LimitsSuite) TestACategoryNamingATeamIsRefused() {
	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "category", "category": "team:finance", "interval": "daily", "limit_micros": 1})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeUseTeam, failure.Code)
}

func (s *LimitsSuite) TestATeamWithinAnAppIsBoundByTheTeam() {
	team := s.create(map[string]any{"subject": "team", "team": "finance", "interval": "weekly", "limit_micros": 100_000_000})

	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits", map[string]any{
		"subject": "app_team", "app_id": s.customerID(), "team": "finance", "interval": "weekly", "limit_micros": 101_000_000,
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "team finance (limit "+strconv.FormatInt(team.Id, 10)+")")
}

func (s *LimitsSuite) TestAnOriginCannotHaveAKeyLimit() {
	status, _ := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "api_key", "app_id": s.customerID(), "api_key": "hook:chat", "interval": "daily", "limit_micros": 1})

	s.Equal(http.StatusBadRequest, status)
}

func (s *LimitsSuite) TestAUserIDFollowsStreamChatsRule() {
	for _, id := range []string{"jane.doe", "Jane Doe", "jane@example.com", "u_1-2"} {
		limit := s.create(map[string]any{"subject": "user", "app_id": s.customerID(), "user_id": id, "interval": "daily", "limit_micros": 1})
		s.Equal(id, limit.UserId)
	}
	for _, id := range []string{"auth0|abc", "jane+doe", "jané", "a/b", strings.Repeat("u", 256)} {
		status, _ := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
			map[string]any{"subject": "user", "app_id": s.customerID(), "user_id": id, "interval": "daily", "limit_micros": 1})
		s.Equal(http.StatusBadRequest, status, id)
	}
}

func (s *LimitsSuite) TestASystemFieldIsAnUnknownField() {
	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits",
		map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 1, "system": true})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeValidationFailed, failure.Code)
	s.Contains(failure.Message, "system")
}

func (s *LimitsSuite) TestACustomerResetLetsItSpendAgain() {
	s.create(map[string]any{"subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 1_000_000})
	s.spend(1_000_000, nil)
	s.searchedUntil(s.serverClient, http.StatusPaymentRequired)

	var after SubjectUsage
	status, payload := s.serverClient.call(http.MethodPost, "/v1/spend/limits/usage/reset",
		map[string]any{"subject": "app", "app_id": s.customerID(), "intervals": []string{"daily"}, "reason": "a runaway loop, fixed"})
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &after))

	daily := after.Windows[0]
	s.Equal(LimitInterval(usage.Daily), daily.Interval)
	s.Equal(int64(1_000_000), daily.SpentMicros, "the usage records are untouched")
	s.Equal(int64(1_000_000), daily.BaselineMicros)
	s.Require().NotNil(daily.RoomMicros)
	s.Equal(int64(1_000_000), *daily.RoomMicros, "the whole limit is back")
	s.searchedUntil(s.serverClient, http.StatusOK)
	entries, err := s.store.AuditEntries(context.Background(), s.customerID(), store.AuditEntryFilter{ResourceType: store.AuditSpendLimit})
	s.Require().NoError(err)
	s.Contains(entries[0].ResourceID, "reset:")
}

func (s *LimitsSuite) TestAResetOfASubjectWithNoLimitIsNotFound() {
	status, failure := s.refusal(s.serverClient, http.MethodPost, "/v1/spend/limits/usage/reset",
		map[string]any{"subject": "user", "app_id": s.customerID(), "user_id": "u1", "reason": "x"})

	s.Equal(http.StatusNotFound, status)
	s.Equal(codeNoLimit, failure.Code)
}

func (s *LimitsSuite) TestAUsersDefaultIsReadAsTheUsersLimit() {
	created := s.create(map[string]any{"subject": "user", "app_id": s.customerID(), "interval": "daily", "limit_micros": 2_000_000})
	s.spend(500_000, func(event *usage.Event) { event.UserID = "u1" })

	var read SubjectUsage
	s.Require().Eventually(func() bool {
		read = SubjectUsage{}
		return s.serverClient.do(http.MethodGet, "/v1/spend/limits/usage?subject=user&app_id="+url.QueryEscape(s.customerID())+"&user_id=u1", nil, &read) == http.StatusOK &&
			read.Windows[0].SpentMicros == 500_000
	}, 5*time.Second, 50*time.Millisecond)

	daily := read.Windows[0]
	s.Require().NotNil(daily.Limit)
	s.Equal(created.Id, daily.Limit.Id)
	s.Equal(LimitLevel(store.LevelApp), daily.Limit.Level)
	s.Equal(int64(1_500_000), *daily.RoomMicros)
	s.Nil(read.Windows[1].Limit, "no weekly limit")
	s.Nil(read.SpendHardLimit, "the operator set none")
}

func (s *LimitsSuite) TestSubjectUsageShowsTheSpendHardLimit() {
	daily := int64(5_000_000)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &daily}})
	s.spend(1_000_000, nil)

	var read SubjectUsage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits/usage?subject=organization", nil, &read))

	s.Require().NotNil(read.SpendHardLimit)
	s.Require().NotNil(read.SpendHardLimit.Daily)
	s.Nil(read.SpendHardLimit.Monthly)
	s.Equal(int64(5_000_000), read.SpendHardLimit.Daily.LimitMicros)
	s.Equal(int64(1_000_000), read.SpendHardLimit.Daily.SpentMicros)
	s.Equal(int64(4_000_000), read.SpendHardLimit.Daily.RoomMicros)
	_, end := usage.Window(usage.Daily, time.Now())
	s.True(read.SpendHardLimit.Daily.WindowEnd.Equal(end))

	zero := int64(0)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &zero, MonthlyMicros: &zero}})
	read = SubjectUsage{}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/spend/limits/usage?subject=organization", nil, &read))
	s.Nil(read.SpendHardLimit, "0 is no limit")
}

func (s *LimitsSuite) TestStaffEditACustomerLimitThroughManage() {
	status, payload := s.staff.call(http.MethodPost, "/v1/manage/spend/limits", map[string]any{
		"organization_id": s.organization(), "subject": "app", "app_id": s.customerID(), "interval": "daily", "limit_micros": 1_000_000,
	})
	s.Require().Equal(http.StatusCreated, status, string(payload))
	var created Limit
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Contains(created.UpdatedBy, "manage:")

	entries, err := s.store.AuditEntries(context.Background(), s.organization(), store.AuditEntryFilter{ResourceType: store.AuditSpendLimit})
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Equal(store.AuditSourceManage, entries[0].Source)
	s.Equal(s.staff.header.Get(actorIDHeader), entries[0].ActorID)

	var changed Limit
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/spend/limits/"+strconv.FormatInt(created.Id, 10),
		map[string]any{"limit_micros": 2_000_000}, &changed), "the customer's credential changes it afterwards")
	s.Equal(int64(2_000_000), changed.LimitMicros)

	var listed LimitPage
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/manage/spend/limits?organization_id="+url.QueryEscape(s.organization()), nil, &listed))
	s.Len(listed.Items, 1)

	without := &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{actorIDHeader: {"staff:x"}}}
	s.Equal(http.StatusUnauthorized, without.do(http.MethodGet, "/v1/manage/spend/limits?organization_id="+url.QueryEscape(s.organization()), nil, nil))
	status, _ = s.staff.call(http.MethodGet, "/v1/manage/spend/limits", nil)
	s.Equal(http.StatusBadRequest, status, "no organization_id")
	status, _ = s.staff.call(http.MethodPost, "/v1/manage/spend/limits", map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 1})
	s.Equal(http.StatusBadRequest, status, "no organization_id")

	s.Equal(http.StatusNoContent, s.staff.do(http.MethodDelete, "/v1/manage/spend/limits/"+strconv.FormatInt(created.Id, 10), nil, nil))
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/spend/limits/"+strconv.FormatInt(created.Id, 10), nil, nil))
}

func (s *LimitsSuite) TestAnotherOrganizationsLimitIsNotThere() {
	created := s.create(map[string]any{"subject": "organization", "interval": "daily", "limit_micros": 1})

	stranger := s.data.backendOfAnotherApp()

	s.Equal(http.StatusNotFound, stranger.do(http.MethodGet, "/v1/spend/limits/"+strconv.FormatInt(created.Id, 10), nil, nil))
	s.Equal(http.StatusNotFound, stranger.do(http.MethodPatch, "/v1/spend/limits/"+strconv.FormatInt(created.Id, 10), map[string]any{"limit_micros": 5}, nil))
	s.Equal(http.StatusNotFound, stranger.do(http.MethodDelete, "/v1/spend/limits/"+strconv.FormatInt(created.Id, 10), nil, nil))
}
