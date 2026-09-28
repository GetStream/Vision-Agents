//go:build integration

package store

import (
	"time"

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
		Budget:          &Budget{LimitMicros: 5_000_000, Interval: BudgetDaily},
		DataPolicy:      options.DataPolicy{AllowTraining: &no, Retention: "30d"},
		PromptInjection: &yes,
	}))
	s.Require().NoError(s.store.SavePolicy(s.ctx, ScopeOrganization, "org-1", PolicyDocument{
		PromptInjection: &no,
	}))

	document, err := s.store.Policy(s.ctx, ScopeOrganization, "org-1")
	s.Require().NoError(err)
	s.Nil(document.Budget)
	s.False(document.DataPolicy.Asks())
	s.Require().NotNil(document.PromptInjection)
	s.False(*document.PromptInjection)

	app, err := s.store.Policy(s.ctx, ScopeApp, "org-1")
	s.Require().NoError(err)
	s.Nil(app.PromptInjection, "an app and an organization sharing an id are different scopes")
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

func (s *StoreSuite) TestSpendSinceCountsOnlyTheCustomerAndTheWindow() {
	s.spend("app-1", 1_000_000, 10)
	s.spend("app-1", 250_000, 90)
	s.spend("app-2", 7_000_000, 10)

	spent, err := s.store.SpendSince(s.ctx, "app-1", s.base.Add(-time.Hour))

	s.Require().NoError(err)
	s.Equal(int64(1_000_000), spent)
}

func (s *StoreSuite) TestAnOrganizationSpendsWhatEveryAppSeenUnderItSpent() {
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-1", "org-1"))
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-2", "org-1"))
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "app-3", "org-2"))
	s.spend("app-1", 1_000_000, 10)
	s.spend("app-2", 2_000_000, 10)
	s.spend("app-3", 4_000_000, 10)

	spent, err := s.store.OrganizationSpendSince(s.ctx, "org-1", s.base.Add(-time.Hour))

	s.Require().NoError(err)
	s.Equal(int64(3_000_000), spent)
}
