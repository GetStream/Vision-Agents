//go:build integration

package api

import (
	"net/http"
	"testing"
)

// CustomerQuotaSuite is Stream staff changing an app's daily limit, and reading who did.
type CustomerQuotaSuite struct {
	RouterSuite
	// staff is Stream's own tool, holding the ops key and no app's credentials.
	staff *testClient
}

func TestCustomerQuotaSuite(t *testing.T) {
	runSuite(t, new(CustomerQuotaSuite))
}

func (s *CustomerQuotaSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

func (s *CustomerQuotaSuite) path(customerID string) string {
	return "/v1/ops/customers/" + customerID + "/quota"
}

func (s *CustomerQuotaSuite) change(customerID, by string, messages int64) {
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodPut, s.path(customerID), SetCustomerQuotaRequest{
		MessagesPerUserPerDay: &messages, ChangedBy: by, Reason: "asked for by the customer",
	}, nil))
}

func (s *CustomerQuotaSuite) changes(customerID, query string) QuotaChangePage {
	var listed QuotaChangePage
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, s.path(customerID)+"/changes"+query, nil, &listed))
	return listed
}

func (s *CustomerQuotaSuite) TestAChangeRecordsWhoWhyWhenAndTheNumbersBeforeAndAfter() {
	s.change(s.customerID(), "ops@getstream.io", 5)
	s.change(s.customerID(), "other@getstream.io", 9)

	listed := s.changes(s.customerID(), "")

	s.Require().Len(listed.Items, 2)
	newest, oldest := listed.Items[0], listed.Items[1]
	s.Equal("other@getstream.io", newest.ChangedBy)
	s.Equal("asked for by the customer", newest.Reason)
	s.False(newest.ChangedAt.IsZero())
	s.Equal(pointerTo(int64(5)), newest.Previous.MessagesPerUserPerDay, "the second change's previous is the first one's new")
	s.Equal(pointerTo(int64(9)), newest.New.MessagesPerUserPerDay)
	s.Equal("ops@getstream.io", oldest.ChangedBy)
	s.Nil(oldest.Previous.MessagesPerUserPerDay, "an app with no limit set had the default")
	s.Equal(oldest.New, newest.Previous)
}

func (s *CustomerQuotaSuite) TestAChangeNamingNobodyIsRefused() {
	for name, body := range map[string]map[string]any{
		"left out": {"messages_per_user_per_day": 5},
		"empty":    {"messages_per_user_per_day": 5, "changed_by": ""},
	} {
		status, failure := s.staff.call(http.MethodPut, s.path(s.customerID()), body)

		s.Equal(http.StatusBadRequest, status, name)
		s.Contains(string(failure), `"code":"validation_failed"`, name)
	}
	status, failure := s.staff.call(http.MethodPut, s.path(s.customerID()),
		map[string]any{"messages_per_user_per_day": 5, "changed_by": "  "})
	s.Equal(http.StatusBadRequest, status, "blank")
	s.Contains(string(failure), "say who is changing the limit")
	s.Empty(s.changes(s.customerID(), "").Items)
	var read Quota
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, s.path(s.customerID()), nil, &read))
	s.Equal(Quota{}, read, "a refused change stores nothing")
}

func (s *CustomerQuotaSuite) TestAChangeWithNoBodyIsRefused() {
	status, failure := s.staff.call(http.MethodPut, s.path(s.customerID()), nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(failure), "a request body is required")
	s.Empty(s.changes(s.customerID(), "").Items)
}

func (s *CustomerQuotaSuite) TestAChangeNamesWhoWithinTheirLimit() {
	status, failure := s.staff.call(http.MethodPut, s.path(s.customerID()), map[string]any{
		"changed_by": string(make([]byte, 256)), "reason": "x",
	})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(failure), `"code":"validation_failed"`)
}

func (s *CustomerQuotaSuite) TestChangesAreListedNewestFirstAndPageByCursor() {
	for messages := int64(1); messages <= 3; messages++ {
		s.change(s.customerID(), "ops@getstream.io", messages)
	}

	first := s.changes(s.customerID(), "?limit=2")
	s.Require().Len(first.Items, 2)
	s.True(first.HasMore)
	s.Require().NotNil(first.NextCursor)
	s.Equal(pointerTo(int64(3)), first.Items[0].New.MessagesPerUserPerDay)
	s.Equal(pointerTo(int64(2)), first.Items[1].New.MessagesPerUserPerDay)

	last := s.changes(s.customerID(), "?limit=2&cursor="+*first.NextCursor)
	s.Require().Len(last.Items, 1)
	s.False(last.HasMore)
	s.Nil(last.NextCursor)
	s.Equal(pointerTo(int64(1)), last.Items[0].New.MessagesPerUserPerDay)

	s.Equal(http.StatusBadRequest, s.staff.do(http.MethodGet, s.path(s.customerID())+"/changes?cursor=nonsense", nil, nil))
}

func (s *CustomerQuotaSuite) TestAnotherAppsChangesNeverAppear() {
	other := s.data.createApp()
	s.change(other.app.ID, "ops@getstream.io", 77)
	s.change(s.customerID(), "ops@getstream.io", 5)

	listed := s.changes(s.customerID(), "")

	s.Require().Len(listed.Items, 1)
	s.Equal(pointerTo(int64(5)), listed.Items[0].New.MessagesPerUserPerDay)
}

func (s *CustomerQuotaSuite) TestOnlyStaffReadTheChanges() {
	s.change(s.customerID(), "ops@getstream.io", 5)
	wrong := &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {"guess"}}, kind: noCredential}
	path := s.path(s.customerID()) + "/changes"

	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, path, nil, nil), "the app's own backend")
	s.Equal(http.StatusUnauthorized, wrong.do(http.MethodGet, path, nil, nil))
	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodGet, path, nil, nil))
}
