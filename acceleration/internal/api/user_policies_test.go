//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// userPath is where one end user's policy is.
func userPath(userID string) string { return "/v1/policies/users/" + url.PathEscape(userID) }

// putUser stores a user's own policy as the app's backend.
func (s *PoliciesSuite) putUser(userID string, body map[string]any) UserPolicy {
	var stored UserPolicy
	status, payload := s.serverClient.call(http.MethodPut, userPath(userID), body)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &stored))
	return stored
}

// getUser reads a user's policy as the app's backend.
func (s *PoliciesSuite) getUser(userID string) UserPolicy {
	var read UserPolicy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, userPath(userID), nil, &read))
	return read
}

// messages is a quota of so many messages a day.
func messages(perDay int64) map[string]any {
	return map[string]any{"quota": map[string]any{"messages_per_user_per_day": perDay}}
}

// heldTo is the daily message limit a user's device is told it is held to.
func (s *PoliciesSuite) heldTo(client *testClient) string {
	response := client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))
	s.Require().Equal(http.StatusCreated, response.StatusCode)
	return response.Header.Get("X-Quota-Limit")
}

// sibling is another app of the suite's app's organization, with a key of its own.
func (s *PoliciesSuite) sibling() testApp {
	app := store.App{OrganizationID: s.app.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	return s.data.keyed(s.app.organization, app)
}

func (s *PoliciesSuite) TestAUsersOwnQuotaWinsOverTheDefault() {
	s.put(messages(1))
	vip := s.data.createUser()

	stored := s.putUser(vip.userID, messages(3))

	s.Require().NotNil(stored.Quota)
	s.Equal(pointerTo(int64(3)), stored.Quota.MessagesPerUserPerDay)
	s.Equal(EffectiveQuota{
		MessagesPerUserPerDay: 3, MessagesSetBy: QuotaLevelUser,
		TokensPerUserPerDay: 0, TokensSetBy: QuotaLevelDefault,
	}, stored.Effective)
	s.Equal("3", s.heldTo(vip), "the user's own")
	s.Equal("1", s.heldTo(s.data.createUser()), "another user keeps the app's")
}

func (s *PoliciesSuite) TestTheOrganizationsQuotaIsTheDefaultForEveryApp() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization", messages(4), nil))
	s.Equal("4", s.heldTo(s.data.createUser()))

	s.useApp(s.sibling())
	s.Equal("4", s.heldTo(s.data.createUser()), "another app of the organization")
	s.put(messages(2))
	s.Equal("2", s.heldTo(s.data.createUser()), "an app's own wins over its organization's")
	s.Equal(QuotaLevelApp, s.getUser("someone").Effective.MessagesSetBy)
}

func (s *PoliciesSuite) TestAnOrganizationsQuotaIsStoredAndReadBack() {
	var written Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization", messages(5_000), &written))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &read))
	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(5_000)), read.Quota.MessagesPerUserPerDay)
	s.Equal(read.Quota, written.Quota)
}

func (s *PoliciesSuite) TestTheSameUserIDInTwoAppsIsTwoUsers() {
	s.putUser("ann", messages(3))

	s.useApp(s.data.createApp())
	read := s.getUser("ann")

	s.Nil(read.Quota, "ann of another app has none of her own")
	s.Equal(QuotaLevelDefault, read.Effective.MessagesSetBy)
	var listed UserPolicyPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/users", nil, &listed))
	s.Empty(listed.Items)
}

func (s *PoliciesSuite) TestDeletingAUserPolicyFallsBackToTheDefault() {
	s.put(messages(2))
	vip := s.data.createUser()
	s.putUser(vip.userID, messages(5))

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, userPath(vip.userID), nil, nil))

	read := s.getUser(vip.userID)
	s.Nil(read.Quota)
	s.Equal(int64(2), read.Effective.MessagesPerUserPerDay)
	s.Equal(QuotaLevelApp, read.Effective.MessagesSetBy)
	s.Equal("2", s.heldTo(vip))
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, userPath(vip.userID), nil, nil),
		"a user with none of their own is deleted all the same")
}

func (s *PoliciesSuite) TestAUserPolicyHoldsOnlyAQuota() {
	status, body := s.serverClient.call(http.MethodPut, userPath("ann"),
		map[string]any{"data_policy": map[string]any{"retention": "none"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"validation_failed"`)
	s.Nil(s.getUser("ann").Quota)
}

func (s *PoliciesSuite) TestAUserPolicyWithNoNumberRemovesTheUsersOwn() {
	s.putUser("ann", messages(3))

	cleared := s.putUser("ann", map[string]any{"quota": map[string]any{}})

	s.Nil(cleared.Quota)
	var listed UserPolicyPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/users", nil, &listed))
	s.Empty(listed.Items, "a user whose quota sets nothing has no policy to list")
}

func (s *PoliciesSuite) TestANullQuotaRemovesTheUsersOwn() {
	s.putUser("ann", messages(3))

	cleared := s.putUser("ann", map[string]any{"quota": nil})

	s.Nil(cleared.Quota)
	s.Nil(s.getUser("ann").Quota)
}

func (s *PoliciesSuite) TestAUserIDStreamChatWouldRefuseIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPut, userPath("ann/lee"), messages(3))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "user_id")
}

func (s *PoliciesSuite) TestOnlyTheAppsOwnBackendMaySetAUsersQuota() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, userPath("ann"), messages(3), nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, userPath("ann"), nil, nil)
	})
}

func (s *PoliciesSuite) TestAnAppsUsersWithAPolicyAreListedAPageAtATime() {
	for _, user := range []string{"cal", "ann", "bea lee"} {
		s.putUser(user, messages(3))
	}

	var first, rest UserPolicyPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/users?limit=2", nil, &first))
	s.Require().True(first.HasMore)
	s.Require().NotNil(first.NextCursor)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/policies/users?limit=2&cursor="+url.QueryEscape(*first.NextCursor), nil, &rest))

	s.Equal([]string{"ann", "bea lee"}, userIDs(first.Items))
	s.Equal([]string{"cal"}, userIDs(rest.Items))
	s.False(rest.HasMore)
	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodGet, "/v1/policies/users?cursor=nonsense", nil, nil))
}

func (s *PoliciesSuite) TestAUserIsToldWhatTheirBackendSpentForThemToday() {
	s.put(messages(10))
	backend := routing.Caller{UserID: "ann", ServerSide: true}
	s.limiter.Debit(s.T().Context(), s.customerID(), backend, 40)
	s.limiter.Debit(s.T().Context(), s.customerID(), routing.Caller{UserID: "ann", Verified: true}, 2)

	read := s.getUser("ann")

	s.EqualValues(2, read.Messages)
	s.EqualValues(42, read.Tokens)
	s.WithinDuration(time.Now().UTC().Truncate(24*time.Hour).Add(24*time.Hour), read.ResetsAt, time.Second,
		"the next UTC midnight")
}

func (s *PoliciesSuite) TestAUsersQuotaChangeIsAudited() {
	s.putUser("ann", messages(3))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, userPath("ann"), nil, nil))

	entries := s.auditOf("user:" + store.UserScopeID(s.customerID(), "ann"))

	s.Require().Len(entries, 2)
	s.Equal(AuditAction(store.AuditDeleted), entries[0].Action)
	s.Equal(AuditAction(store.AuditUpdated), entries[1].Action)
	s.Equal("user", entries[1].ResourceName)
}

// auditOf is the app's changes to one resource, newest first.
func (s *PoliciesSuite) auditOf(resourceID string) []AuditEntry {
	var answered AuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/audit/query",
		map[string]any{"filter": AuditFilter{ResourceID: &resourceID}, "limit": 200}, &answered))
	return answered.Items
}

func userIDs(entries []UserPolicyEntry) []string {
	ids := make([]string, len(entries))
	for i, entry := range entries {
		ids[i] = entry.UserId
	}
	return ids
}

// staff is Stream staff calling the management API with the manage key.
func (s *PoliciesSuite) staff() *testClient {
	return &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
	}}
}

// customerPolicyPath is where staff reach one of the suite's organization's policies.
func (s *PoliciesSuite) customerPolicyPath(organizationID, scope, id string) string {
	return "/v1/manage/organizations/" + url.PathEscape(organizationID) + "/policies/" + scope + "/" + url.PathEscape(id)
}

func (s *PoliciesSuite) TestStaffSetAQuotaThroughManage() {
	path := s.customerPolicyPath(s.app.organization.ID, "user", store.UserScopeID(s.customerID(), "ann lee"))
	staff := s.staff()

	var written, read Policy
	status, payload := staff.call(http.MethodPut, path, messages(7))
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &written))
	s.Require().Equal(http.StatusOK, staff.do(http.MethodGet, path, nil, &read))

	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(7)), read.Quota.MessagesPerUserPerDay)
	s.Equal(written, read)
	s.Equal(pointerTo(int64(7)), s.getUser("ann lee").Quota.MessagesPerUserPerDay, "the customer reads what staff wrote")
	entries := s.auditOf("user:" + store.UserScopeID(s.customerID(), "ann lee"))
	s.Require().Len(entries, 1)
	s.Equal(AuditSource(store.AuditSourceManage), entries[0].Source)
	s.Equal(http.StatusUnauthorized, staff.with(manageKeyHeader, "").do(http.MethodGet, path, nil, nil))
	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, path, nil, nil), "an app's own credential is no manage key")
}

func (s *PoliciesSuite) TestStaffSetAnAppsAndAnOrganizationsPolicyThroughManage() {
	staff := s.staff()
	s.Require().Equal(http.StatusOK, staff.do(http.MethodPut,
		s.customerPolicyPath(s.app.organization.ID, "organization", s.app.organization.ID), messages(9), nil))
	s.Require().Equal(http.StatusOK, staff.do(http.MethodPut,
		s.customerPolicyPath(s.app.organization.ID, "app", s.customerID()), map[string]any{"prompt_injection": true}, nil))

	var organization, app Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &organization))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &app))
	s.Require().NotNil(organization.Quota)
	s.Equal(pointerTo(int64(9)), organization.Quota.MessagesPerUserPerDay)
	s.Require().NotNil(app.PromptInjection)
	s.True(*app.PromptInjection)
}

func (s *PoliciesSuite) TestStaffWriteAUsersPolicyByTheCustomersRules() {
	path := s.customerPolicyPath(s.app.organization.ID, "user", store.UserScopeID(s.customerID(), "ann"))

	status, failure := s.staff().failure(http.MethodPut, path, map[string]any{"prompt_injection": true})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "a user's policy holds a quota only")
	status, body := s.staff().call(http.MethodPut, path, map[string]any{"budget": map[string]any{"limit_micros": 1}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"`+codeBudgetMoved+`"`)
	s.Nil(s.getUser("ann").Quota)
}

func (s *PoliciesSuite) TestStaffCannotReachAnotherOrganizationsPolicy() {
	mine := s.app
	other := s.data.createApp()
	staff := s.staff()

	for _, path := range []string{
		s.customerPolicyPath(mine.organization.ID, "app", other.app.ID),
		s.customerPolicyPath(mine.organization.ID, "user", store.UserScopeID(other.app.ID, "ann")),
		s.customerPolicyPath(mine.organization.ID, "organization", other.organization.ID),
	} {
		s.Equal(http.StatusNotFound, staff.do(http.MethodGet, path, nil, nil), path)
		s.Equal(http.StatusNotFound, staff.do(http.MethodPut, path, messages(1), nil), path)
	}
	s.useApp(other)
	s.Nil(s.getUser("ann").Quota, "nothing was written to the other organization's app")
}
