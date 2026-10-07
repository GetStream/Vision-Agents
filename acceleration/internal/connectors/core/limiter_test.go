package core_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// RateLimitKeySuite is which calls share a provider's rate limit: the key a manifest's
// rate_limit.per gives a connection.
type RateLimitKeySuite struct {
	suite.Suite
}

func TestRateLimitKeySuite(t *testing.T) {
	suite.Run(t, new(RateLimitKeySuite))
}

// manifest is a connector limited per scope, whose accounts are a team, then a user.
func manifest(per core.RateLimitScope) core.ResolvedManifest {
	return core.ResolvedManifest{ConnectorID: "chat", Identity: []string{"team_id", "user_id"},
		RateLimit: core.RateLimitRule{Per: per}}
}

// member is a connection to user of team, which the consent captured.
func member(id, team, user string) core.Connection {
	return core.Connection{ID: id, AccountID: team + ":" + user,
		Metadata: map[string]string{"team_id": team, "user_id": user}}
}

func (s *RateLimitKeySuite) TestAConnectorWhoseManifestNamesNoScopeHasNoKey() {
	s.Empty(manifest("").RateLimitKey("customer-1", member("c1", "T1", "U1")))
}

func (s *RateLimitKeySuite) TestPerAppEveryConnectionOfTheCustomerSharesOneKey() {
	m := manifest(core.RateLimitPerApp)

	s.Equal(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T2", "U2")))
}

func (s *RateLimitKeySuite) TestAnotherCustomerNeverSharesAKey() {
	for _, per := range []core.RateLimitScope{core.RateLimitPerApp, core.RateLimitPerTenant, core.RateLimitPerUser} {
		m := manifest(per)
		s.NotEqual(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-2", member("c1", "T1", "U1")), per)
	}
}

func (s *RateLimitKeySuite) TestAnotherConnectorNeverSharesAKey() {
	m, other := manifest(core.RateLimitPerApp), manifest(core.RateLimitPerApp)
	other.ConnectorID = "crm"

	s.NotEqual(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), other.RateLimitKey("customer-1", member("c1", "T1", "U1")))
}

func (s *RateLimitKeySuite) TestPerTenantTwoPeopleOfOneTeamShareAKeyAndAnotherTeamHasItsOwn() {
	m := manifest(core.RateLimitPerTenant)

	first, second := m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T1", "U2"))
	elsewhere := m.RateLimitKey("customer-1", member("c3", "T2", "U1"))

	s.Equal(first, second)
	s.NotEqual(first, elsewhere)
}

// TestPerTenantTheTenantMayBeAnInput: a Shopify-style shop the connection was created for.
func (s *RateLimitKeySuite) TestPerTenantTheTenantMayBeAnInput() {
	m := manifest(core.RateLimitPerTenant)
	m.Identity = []string{"shop"}
	shop := func(id, name string) core.Connection {
		return core.Connection{ID: id, Inputs: map[string]string{"shop": name}}
	}

	s.Equal(m.RateLimitKey("customer-1", shop("c1", "one")), m.RateLimitKey("customer-1", shop("c2", "one")))
	s.NotEqual(m.RateLimitKey("customer-1", shop("c1", "one")), m.RateLimitKey("customer-1", shop("c2", "two")))
}

// TestPerTenantWithoutATenantIsPerAccount: a manifest with no identity limits each account
// alone, which holds less than the provider's limit, never more.
func (s *RateLimitKeySuite) TestPerTenantWithoutATenantIsPerAccount() {
	m := manifest(core.RateLimitPerTenant)
	m.Identity = nil

	s.NotEqual(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T1", "U2")))
	s.Equal(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T1", "U1")))
}

func (s *RateLimitKeySuite) TestPerUserTwoConnectionsToOneAccountShareAKeyAndAnotherAccountHasItsOwn() {
	m := manifest(core.RateLimitPerUser)

	s.Equal(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T1", "U1")))
	s.NotEqual(m.RateLimitKey("customer-1", member("c1", "T1", "U1")), m.RateLimitKey("customer-1", member("c2", "T1", "U2")))
}

// TestPerUserAConnectionWithoutAnAccountIdIsItsOwnKey: a provider that gives no id.
func (s *RateLimitKeySuite) TestPerUserAConnectionWithoutAnAccountIdIsItsOwnKey() {
	m := manifest(core.RateLimitPerUser)

	s.NotEqual(m.RateLimitKey("customer-1", core.Connection{ID: "c1"}), m.RateLimitKey("customer-1", core.Connection{ID: "c2"}))
}

// TestAColonInAnAccountIdCannotMakeTwoKeysOne: the parts are escaped before they are joined.
func (s *RateLimitKeySuite) TestAColonInAnAccountIdCannotMakeTwoKeysOne() {
	m := manifest(core.RateLimitPerUser)

	// Joined unescaped, both would be a:chat:user:account:b:chat:user:account:c.
	s.NotEqual(m.RateLimitKey("a", core.Connection{ID: "c1", AccountID: "b:chat:user:account:c"}),
		m.RateLimitKey("a:chat:user:account:b", core.Connection{ID: "c1", AccountID: "c"}))
}
