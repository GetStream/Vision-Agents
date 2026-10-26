//go:build integration

package store

import (
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// spend stores a request that cost the given amount, some minutes before the suite's base.
func (s *StoreSuite) spend(customerID string, costMicros int64, minutesAgo int) {
	request := &Request{
		Modality:   "llm",
		CustomerID: customerID,
		Provider:   "openai",
		Model:      "gpt-5.6-luna",
		StartedAt:  s.base.Add(-time.Duration(minutesAgo) * time.Minute),
		CostMicros: costMicros,
		Success:    true,
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

func (s *StoreSuite) TestAScopeNobodyWroteAPolicyForHasTheEmptyOne() {
	document, err := s.store.Policy(s.ctx, ScopeApp, "app-1")

	s.Require().NoError(err)
	s.Equal(PolicyDocument{}, document)
}

func (s *StoreSuite) TestSavingAPolicyReplacesWhatWasThere() {
	no, yes := false, true
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{
		Tags:            map[string]string{"team": "a"},
		DataPolicy:      options.DataPolicy{AllowTraining: &no, Retention: "30d"},
		PromptInjection: &yes,
	}))
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{
		PromptInjection: &no,
	}))

	document, err := s.store.Policy(s.ctx, ScopeOrganization, "org-1")
	s.Require().NoError(err)
	s.Nil(document.Tags)
	s.False(document.DataPolicy.Asks())
	s.Require().NotNil(document.PromptInjection)
	s.False(*document.PromptInjection)

	app, err := s.store.Policy(s.ctx, ScopeApp, "org-1")
	s.Require().NoError(err)
	s.Nil(app.PromptInjection, "an app and an organization sharing an id are different scopes")
}

func (s *StoreSuite) TestSavingAnOrganizationsPolicyWithoutTheRequirementKeepsTheStoredOne() {
	// A save over HTTP leaves the requirement out, whatever the document it read said, so
	// one landing after the operator's write keeps what the operator set.
	yes, no := true, false
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{RequireOwnStreamApp: &yes}))
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{PromptInjection: &yes}))

	document, err := s.store.Policy(s.ctx, ScopeOrganization, "org-1")
	s.Require().NoError(err)
	s.Require().NotNil(document.PromptInjection)
	s.True(*document.PromptInjection)
	s.Require().NotNil(document.RequireOwnStreamApp, "a save that leaves it out keeps the operator's")
	s.True(*document.RequireOwnStreamApp)

	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{RequireOwnStreamApp: &no}))
	document, err = s.store.Policy(s.ctx, ScopeOrganization, "org-1")
	s.Require().NoError(err)
	s.Require().NotNil(document.RequireOwnStreamApp)
	s.False(*document.RequireOwnStreamApp, "the operator still turns it off")
}

func (s *StoreSuite) TestAUsersPolicyIsKeptUnderTheirApp() {
	messages := int64(3)
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeUser, UserScopeID("app-1", "ann"), PolicyDocument{
		Quota: &Quota{MessagesPerUserPerDay: &messages},
	}))

	own, err := s.store.Policy(s.ctx, ScopeUser, UserScopeID("app-1", "ann"))
	s.Require().NoError(err)
	s.Require().NotNil(own.Quota)
	s.Equal(&messages, own.Quota.MessagesPerUserPerDay)
	other, err := s.store.Policy(s.ctx, ScopeUser, UserScopeID("app-2", "ann"))
	s.Require().NoError(err)
	s.Nil(other.Quota, "the same user id in another app is another user")
}

func (s *StoreSuite) TestAnAppsUsersAreListedByIDAndOnlyTheirs() {
	for _, id := range []string{
		UserScopeID("app_1", "bea"), UserScopeID("app_1", "ann"), UserScopeID("app_1", "Cal"),
		UserScopeID("app_1x", "dan"), UserScopeID("appa1", "eve"),
	} {
		s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeUser, id, PolicyDocument{}))
	}
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeApp, "app_1", PolicyDocument{}))

	first, err := s.store.UserPolicies(s.ctx, "app_1", "", 2)
	s.Require().NoError(err)
	rest, err := s.store.UserPolicies(s.ctx, "app_1", "Cal", 2)
	s.Require().NoError(err)

	s.Equal([]string{"app_1/Cal", "app_1/ann", "app_1/bea"}, scopeIDs(first), "byte order, and one past the page")
	s.Equal([]string{"app_1/ann", "app_1/bea"}, scopeIDs(rest))
	has, err := s.store.HasUserPolicies(s.ctx, "app_1")
	s.Require().NoError(err)
	s.True(has)
	none, err := s.store.HasUserPolicies(s.ctx, "app")
	s.Require().NoError(err)
	s.False(none, "an _ in an app id is matched as itself")
}

func (s *StoreSuite) TestDeletingAPolicySaysWhetherThereWasOne() {
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeUser, UserScopeID("app-1", "ann"), PolicyDocument{}))

	deleted, err := s.store.DeletePolicy(s.ctx, ScopeUser, UserScopeID("app-1", "ann"))
	s.Require().NoError(err)
	again, err := s.store.DeletePolicy(s.ctx, ScopeUser, UserScopeID("app-1", "ann"))
	s.Require().NoError(err)

	s.True(deleted)
	s.False(again)
	has, err := s.store.HasUserPolicies(s.ctx, "app-1")
	s.Require().NoError(err)
	s.False(has)
}

func (s *StoreSuite) TestAUserScopeIDSplitsAtItsLastSlash() {
	app, user, ok := SplitUserScopeID(UserScopeID("app-1", "ann lee"))
	s.True(ok)
	s.Equal("app-1", app)
	s.Equal("ann lee", user)
	for _, bad := range []string{"", "app-1", "/ann", "app-1/"} {
		_, _, ok := SplitUserScopeID(bad)
		s.False(ok, bad)
	}
}

func (s *StoreSuite) TestAnAppIsInTheOrganizationItIsRegisteredOrSeenUnder() {
	org := Organization{Name: "registering"}
	s.Require().NoError(s.store.CreateOrganization(s.ctx, &org))
	registered := App{OrganizationID: org.ID, Name: "registered"}
	s.Require().NoError(s.store.CreateApp(s.ctx, &registered))
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-seen", org.ID))

	for app, want := range map[string]bool{registered.ID: true, "app-seen": true, "app-never": false} {
		in, err := s.store.AppInOrganization(s.ctx, app, org.ID)
		s.Require().NoError(err)
		s.Equal(want, in, app)
		other, err := s.store.AppInOrganization(s.ctx, app, "org-other")
		s.Require().NoError(err)
		s.False(other, app)
	}
}

func scopeIDs(policies []Policy) []string {
	ids := make([]string, len(policies))
	for i, policy := range policies {
		ids[i] = policy.ScopeID
	}
	return ids
}

func (s *StoreSuite) TestAnAppBelongsToTheOrganizationItWasLastSeenUnder() {
	unknown, err := s.store.OrganizationOf(s.ctx, "app-1")
	s.Require().NoError(err)
	s.Empty(unknown)

	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-1", "org-1"))
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-1", "org-2"))

	organization, err := s.store.OrganizationOf(s.ctx, "app-1")
	s.Require().NoError(err)
	s.Equal("org-2", organization)
}

func (s *StoreSuite) TestAnOrganizationHasTheTierItWasLastSeenWith() {
	organization := "org-" + uuid.NewString()
	unknown, err := s.store.TierOf(s.ctx, organization)
	s.Require().NoError(err)
	s.Empty(unknown)

	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, organization, "tier-a", s.base))
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, organization, "tier-b", s.base.Add(time.Second)))

	tier, err := s.store.TierOf(s.ctx, organization)
	s.Require().NoError(err)
	s.Equal("tier-b", tier)
}

func (s *StoreSuite) TestATierSeenEarlierThanTheRecordedOneChangesNothing() {
	// Routers write what they saw in the background, so the older write may land last.
	organization := "org-" + uuid.NewString()
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, organization, "tier-b", s.base.Add(time.Second)))
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, organization, "tier-a", s.base))

	tier, err := s.store.TierOf(s.ctx, organization)
	s.Require().NoError(err)
	s.Equal("tier-b", tier)
}

func (s *StoreSuite) TestSpendSinceCountsOnlyTheCustomerAndTheWindow() {
	s.spend("app-1", 1_000_000, 10)
	s.spend("app-1", 250_000, 90)
	s.spend("app-2", 7_000_000, 10)

	spent, err := s.store.SpendSince(s.ctx, "app-1", s.base.Add(-time.Hour))

	s.Require().NoError(err)
	s.Equal(int64(1_000_000), spent)
}
