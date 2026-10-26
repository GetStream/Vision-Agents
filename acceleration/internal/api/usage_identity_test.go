//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

// UsageIdentitySuite checks that every usage row names its organization, tier, app, key,
// user and categories, with the router authenticating keys as a deployment of its own does.
type UsageIdentitySuite struct {
	RouterSuite
}

func TestUsageIdentitySuite(t *testing.T) {
	runSuite(t, new(UsageIdentitySuite))
}

func (s *UsageIdentitySuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *UsageIdentitySuite) TestARequestRecordsAllSixFields() {
	s.Require().NoError(s.store.SetOrganizationTier(context.Background(), s.app.organization.ID, "tier-a", time.Now()))
	asking := s.userWith([]string{"project:onboarding"}, "finance")

	row := s.searched(asking)

	s.Equal(s.app.organization.ID, row.OrganizationID)
	s.Equal("tier-a", row.OrganizationTier, "the tier last recorded for the organization, which no request named")
	s.Equal(s.customerID(), row.CustomerID)
	s.Equal(s.app.key, row.APIKey)
	s.Equal(asking.userID, row.UserID)
	s.Equal([]string{"project:onboarding", "team:finance"}, row.UsageCategories)
	s.Equal(routing.CallerUser, row.Attributes[routing.CallerAttribute])
}

func (s *UsageIdentitySuite) TestAnOrganizationWithNoTierRecordsTheDefault() {
	row := s.searched(s.client)

	s.Equal("default", row.OrganizationTier)
}

func (s *UsageIdentitySuite) TestABackendActingForAUserRecordsTheUser() {
	owner := s.data.createUser()

	row := s.searched(s.serverClient.actingFor(owner))

	s.Equal(owner.userID, row.UserID)
	s.Equal(routing.CallerServer, row.Attributes[routing.CallerAttribute])
}

func (s *UsageIdentitySuite) TestAnAnonymousCallerRecordsNoUser() {
	row := s.searched(s.anonymousClient)

	s.Empty(row.UserID, "a name nobody checked is not recorded")
	s.Equal(routing.CallerAnonymous, row.Attributes[routing.CallerAttribute])
}

func (s *UsageIdentitySuite) TestABackendActingForAUserTheUsageServiceWouldRefuseRecordsNoUser() {
	before := userIDsDropped(&s.RouterSuite)

	row := s.searched(s.serverClient.actingFor(&testClient{userID: "auth0|" + s.utils.uuid()}))

	s.Empty(row.UserID, "the request is served and billed, without its user")
	s.Equal(routing.CallerServer, row.Attributes[routing.CallerAttribute])
	s.Greater(userIDsDropped(&s.RouterSuite), before)
}

func (s *UsageIdentitySuite) TestAUserTokenTheUsageServiceWouldRefuseRecordsNoUser() {
	before := userIDsDropped(&s.RouterSuite)
	id := "a{b}" + s.utils.uuid()

	row := s.searched(s.signedIn(jwt.MapClaims{"user_id": id}, auth.AuthTypeJWT, user, id))

	s.Empty(row.UserID)
	s.Equal(routing.CallerUser, row.Attributes[routing.CallerAttribute], "who asked is still said")
	s.Greater(userIDsDropped(&s.RouterSuite), before)
}

func (s *UsageIdentitySuite) TestInvalidCategoriesAreRefused() {
	asking := s.userWith([]string{"not a category"}, "")

	status, payload := asking.call(http.MethodPost, "/v1/search", SearchRequest{Query: "what is the time"})

	s.Equal(http.StatusBadRequest, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	s.Equal(codeInvalidUsageCategories, answered.Error.Code)
}

func (s *UsageIdentitySuite) TestASessionKeepsTheKeyAndCategoriesItWasOpenedWith() {
	opener := s.userWith([]string{"project:onboarding"}, "finance")
	opened := opener.createSession(textSession(nil))
	s.respond(opener, opened.Id, "hello")
	backend := s.serverClient.actingFor(opener)
	backend.header.Set(auth.UsageCategoriesHeader, "project:elsewhere")

	s.respond(backend, opened.Id, "and again")

	rows := s.sessionRows(opened.Id, 2)
	for _, row := range rows {
		s.Equal(s.app.key, row.APIKey)
		s.Equal(opener.userID, row.UserID)
		s.Equal([]string{"project:onboarding", "team:finance"}, row.UsageCategories,
			"the categories a later request names are not the session's")
	}
}

func (s *UsageIdentitySuite) TestAChatCarriedOnInAnotherProcessKeepsItsKeyAndCategories() {
	opener := s.userWith([]string{"project:onboarding"}, "finance")
	opened := opener.createSession(textSession(nil))
	s.respond(opener, opened.Id, "hello")
	s.sessionRows(opened.Id, 1)
	opener.stopSession(opened.Id)
	_, other := s.otherRouter(nil)
	// What the request carrying it on there says: its own key and categories.
	ctx := context.WithValue(context.Background(), customerContextKey{}, s.customerID())
	ctx = context.WithValue(ctx, kindContextKey{}, auth.KindAuthenticated)
	ctx = routing.WithCaller(ctx, routing.Caller{UserID: opener.userID})
	ctx = routing.WithAttribution(ctx, routing.Attribution{
		APIKey: "key-of-whoever-carries-it-on", UserID: opener.userID, UsageCategories: []string{"project:elsewhere"},
	})

	// As sessionToAnswer carries on a chat that ended in another process.
	spec, failure := other.reopenedFromRow(ctx, opened.Id)
	s.Require().NoError(failure)
	spec.Caller, spec.CallerKind = CallerFrom(ctx), KindFrom(ctx)
	carried, err := other.sessions.Create(ctx, spec)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = other.sessions.Close(carried.ID(), session.OwnerOf(carried.Spec())) })
	asked := time.Now()
	_, err = carried.Ask(ctx, "and again")
	s.Require().NoError(err)

	var answered store.Request
	s.Require().Eventually(func() bool {
		for _, row := range s.rowsWhere("session_id = ? AND modality = 'llm' AND started_at >= ?", opened.Id, asked) {
			answered = row
			return true
		}
		return false
	}, settleFor, 25*time.Millisecond, "the answer in the other process recorded nothing")
	s.Equal(s.app.key, answered.APIKey, "the key the chat was opened with")
	s.Equal([]string{"project:onboarding", "team:finance"}, answered.UsageCategories)
}

func (s *UsageIdentitySuite) TestEveryOwnerSiteSetsAKey() {
	// A request, a session's turns and what the router does for the session in the
	// background, its title among them.
	s.searched(s.client)
	opened := s.client.createSession(textSession(nil))
	s.respond(s.client, opened.Id, "hello")
	s.sessionRows(opened.Id, 1)

	for _, row := range s.rowsWhere("customer_id = ?", s.customerID()) {
		s.Equal(s.app.key, row.APIKey, "%s %s %s", row.Modality, row.Purpose, row.Model)
	}
}

func (s *UsageIdentitySuite) TestSpendIsNarrowedToAKeyAUserATeamAndACategory() {
	finance := s.userWith([]string{"project:onboarding"}, "finance")
	sales := s.userWith(nil, "sales")
	s.searched(finance)
	s.searched(sales)

	s.Equal(int64(1), s.spentCount("team=finance"))
	s.Equal(int64(1), s.spentCount("category=project:onboarding"))
	s.Equal(int64(1), s.spentCount("user_id="+url.QueryEscape(sales.userID)))
	s.Equal(int64(2), s.spentCount("api_key="+url.QueryEscape(s.app.key)))
	s.Equal(int64(0), s.spentCount("api_key=hook:chat"))
}

func (s *UsageIdentitySuite) TestASpendFilterNamingTwoTeamsIsRefused() {
	status, payload := s.serverClient.call(http.MethodGet, s.spendPath("team=finance&category=team:sales"), nil)

	s.Equal(http.StatusBadRequest, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	s.Equal(codeInvalidUsageCategories, answered.Error.Code)
}

// userWith is a new user of the app whose token names usage categories and a team.
func (s *UsageIdentitySuite) userWith(categories []string, team string) *testClient {
	id := s.utils.uuid()
	claims := jwt.MapClaims{"user_id": id}
	if categories != nil {
		claims["usage_categories"] = categories
	}
	if team != "" {
		claims["team"] = team
	}
	return s.signedIn(claims, auth.AuthTypeJWT, user, id)
}

// searched has client search once and returns the row it recorded.
func (s *UsageIdentitySuite) searched(client *testClient) store.Request {
	return searchedBy(&s.RouterSuite, client)
}

// respond asks a session a question as client.
func (s *UsageIdentitySuite) respond(client *testClient, id, text string) {
	respondIn(&s.RouterSuite, client, id, text)
}

// sessionRows waits for a session's model rows, at least n of them, and returns every row it has.
func (s *UsageIdentitySuite) sessionRows(id string, n int) []store.Request {
	return sessionRowsOf(&s.RouterSuite, id, n)
}

func (s *UsageIdentitySuite) rowsWhere(query string, args ...any) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Where(query, args...).Scan(context.Background()))
	return rows
}

func (s *UsageIdentitySuite) spendPath(filter string) string {
	from := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	to := time.Now().UTC().Add(time.Hour).Format(time.RFC3339)
	return "/v1/stats/spend?from=" + url.QueryEscape(from) + "&to=" + url.QueryEscape(to) + "&" + filter
}

// spentCount is how many requests the app's spend counts under a filter.
func (s *UsageIdentitySuite) spentCount(filter string) int64 {
	var buckets []SpendBucket
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, s.spendPath(filter), nil, &buckets))
	var count int64
	for _, bucket := range buckets {
		count += bucket.RequestCount
	}
	return count
}

// ShippedUsageSuite checks what reaches the usage service, which the conversation model's rows
// are shipped to.
type ShippedUsageSuite struct {
	RouterSuite
}

func TestShippedUsageSuite(t *testing.T) {
	shipped := new(ShippedUsageSuite)
	shipped.shipping = true
	runSuite(t, shipped)
}

func (s *ShippedUsageSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ShippedUsageSuite) TestEveryRowOfASessionCarriesItsID() {
	asking := s.data.createUser()
	opened := asking.createSession(textSession(nil))

	respondIn(&s.RouterSuite, asking, opened.Id, "hello")

	rows := sessionRowsOf(&s.RouterSuite, opened.Id, 1)
	for _, row := range rows {
		s.Equal(opened.Id, row.SessionID)
	}
	var events []ledger.UsageEvent
	s.Require().Eventually(func() bool {
		events = nil
		err := s.usage.Ledger.DB().NewSelect().Model(&events).Where("session_id = ?", opened.Id).Scan(context.Background())
		return err == nil && !slices.ContainsFunc(rows, func(row store.Request) bool {
			return !slices.ContainsFunc(events, func(event ledger.UsageEvent) bool { return event.UsageID == row.UsageID })
		})
	}, 10*time.Second, 50*time.Millisecond, "every row of the session is shipped with its id")
	for _, event := range events {
		s.Equal(s.customerID(), event.AppID)
		s.Equal(s.app.key, event.APIKey)
		s.Equal(asking.userID, event.UserID)
	}
}

// ProxiedUsageSuite checks the same behind Stream's proxy, which names the key, the tier and
// the categories in headers of its own.
type ProxiedUsageSuite struct {
	RouterSuite
}

func TestProxiedUsageSuite(t *testing.T) {
	proxied := new(ProxiedUsageSuite)
	proxied.proxied = true
	runSuite(t, proxied)
}

func (s *ProxiedUsageSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ProxiedUsageSuite) TestTheProxysHeadersAreRecorded() {
	asking := s.proxiedClient(s.utils.uuid(), "tier-a", "project:onboarding", "team:finance")

	row := searchedBy(&s.RouterSuite, asking)

	s.Equal(s.app.organization.ID, row.OrganizationID)
	s.Equal("tier-a", row.OrganizationTier)
	s.Equal(s.app.key, row.APIKey)
	s.Equal(asking.userID, row.UserID)
	s.Equal([]string{"project:onboarding", "team:finance"}, row.UsageCategories)
}

func (s *ProxiedUsageSuite) TestAProxiedUserTheUsageServiceWouldRefuseRecordsNoUser() {
	before := userIDsDropped(&s.RouterSuite)

	row := searchedBy(&s.RouterSuite, s.proxiedClient(strings.Repeat("u", 300), "tier-a"))

	s.Empty(row.UserID, "an id over 255 bytes")
	s.Equal(s.app.key, row.APIKey)
	s.Greater(userIDsDropped(&s.RouterSuite), before)
}

func (s *ProxiedUsageSuite) TestATierChangeReachesARunningSession() {
	opener := s.proxiedClient(s.utils.uuid(), "tier-a")
	opened := opener.createSession(textSession(nil))
	respondIn(&s.RouterSuite, opener, opened.Id, "hello")
	s.Require().Equal("tier-a", newest(sessionRowsOf(&s.RouterSuite, opened.Id, 1)).OrganizationTier)

	// Stream changed the tier, and another request for the organization says so. The session's
	// next turn is asked for by a request naming no tier at all.
	searchedBy(&s.RouterSuite, s.proxiedClient(s.utils.uuid(), "tier-b"))
	respondIn(&s.RouterSuite, opener.withTier(""), opened.Id, "and again")

	rows := sessionRowsOf(&s.RouterSuite, opened.Id, 2)
	s.Equal("tier-b", newest(rows).OrganizationTier, "the session's next row carries the organization's tier now")
}

// proxiedClient is a signed-in user of the app as the proxy names one.
func (s *ProxiedUsageSuite) proxiedClient(userID, tier string, categories ...string) *testClient {
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)
	header.Set(auth.UserHeader, userID)
	header.Set(auth.APIKeyIDHeader, s.app.key)
	header.Set(auth.TierHeader, tier)
	if len(categories) > 0 {
		header.Set(auth.UsageCategoriesHeader, strings.Join(categories, ","))
	}
	return &testClient{suite: &s.RouterSuite, header: header, kind: user, userID: userID}
}

// withTier is the same proxied caller with the tier the proxy names for its organization, none
// when it is empty.
func (c *testClient) withTier(tier string) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header.Del(auth.TierHeader)
	if tier != "" {
		named.header.Set(auth.TierHeader, tier)
	}
	return &named
}

// userIDsDropped is router_usage_user_id_dropped_total over every reason. Suites run beside
// each other in one process, so a test asserts only that it rose.
func userIDsDropped(s *RouterSuite) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	var total float64
	for _, family := range families {
		if family.GetName() != "router_usage_user_id_dropped_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			total += metric.GetCounter().GetValue()
		}
	}
	return total
}

func searchedBy(s *RouterSuite, client *testClient) store.Request {
	probe := s.utils.uuid()
	status, payload := client.call(http.MethodPost, "/v1/search",
		SearchRequest{Query: "what is the time", Tags: &map[string]string{"probe": probe}})
	s.Require().Equal(http.StatusOK, status, string(payload))
	var recorded store.Request
	s.Require().Eventually(func() bool {
		var rows []store.Request
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("tags->>'probe' = ?", probe).Scan(context.Background())
		if err != nil || len(rows) != 1 {
			return false
		}
		recorded = rows[0]
		return true
	}, 5*time.Second, 50*time.Millisecond)
	return recorded
}

// respondIn asks a session a question as client, once the answer before it has finished.
func respondIn(s *RouterSuite, client *testClient, id, text string) {
	request := CreateResponseRequest{Text: text, RequestId: pointerTo(s.utils.uuid())}
	var status int
	var failure string
	s.Require().Eventually(func() bool {
		status, failure = client.failure(http.MethodPost, "/v1/agents/sessions/"+id+"/responses", request)
		return status != http.StatusBadRequest || !strings.Contains(failure, "already running")
	}, 5*time.Second, 50*time.Millisecond)
	s.Require().Equal(http.StatusAccepted, status, failure)
}

// sessionRowsOf waits for a session's replies, at least n of them, and returns every row it has.
func sessionRowsOf(s *RouterSuite, id string, n int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("session_id = ?", id).Order("started_at").Scan(context.Background())
		return err == nil && len(slices.DeleteFunc(slices.Clone(rows), func(row store.Request) bool { return !isReply(row) })) >= n
	}, 10*time.Second, 50*time.Millisecond, "want %d replies for session %s", n, id)
	return rows
}

// newest is the reply that started last.
func newest(rows []store.Request) store.Request {
	var last store.Request
	for _, row := range rows {
		if isReply(row) && !row.StartedAt.Before(last.StartedAt) {
			last = row
		}
	}
	return last
}

func isReply(row store.Request) bool { return row.Modality == "llm" && row.Purpose == "reply" }
