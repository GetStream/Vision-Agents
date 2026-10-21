//go:build integration

package egress_test

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ResolverTokensSuite is the proxy's tokens as the router gives them: the connections
// resolver over Postgres, asked on every call.
type ResolverTokensSuite struct {
	suite.Suite
	ctx         context.Context
	store       *store.Store
	credentials core.CredentialStore
	registry    core.Registry
	tokens      egress.Tokens
	grant       egress.Grant
}

func TestResolverTokensSuite(t *testing.T) {
	suite.Run(t, new(ResolverTokensSuite))
}

func (s *ResolverTokensSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = db.Close() })
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "egress test key"})
	s.Require().NoError(err)
	s.credentials, err = pgsealed.New(db, sealer)
	s.Require().NoError(err)
	s.registry = core.Registry{Schemes: map[string]core.Scheme{bearer.Name: bearer.New(), none.Name: none.New(),
		"empty_bearer": sends{Scheme: bearer.New(), authorization: "Bearer "}, "basic": sends{Scheme: bearer.New(), authorization: "Basic eDp5"}}}
	connections, err := resolver.New(resolver.Config{Store: db, Credentials: s.credentials, Schemes: s.registry.Schemes})
	s.Require().NoError(err)
	s.tokens = egress.ResolverTokens(connections, s.registry.Schemes)
}

// SetupTest makes a connection of a new customer that holds gho_first.
func (s *ResolverTokensSuite) SetupTest() {
	customer := "egress-" + uuid.NewString()
	manifest, err := core.ParseManifest([]byte(`
id: custom_forge
revision: 1
name: Forge
endpoints:
  mcp: https://forge.example/mcp
schemes: [bearer, none]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, customer, manifest)
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{CustomerID: customer, ConnectorID: definition.ID, DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerApp, AuthScheme: bearer.Name}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, s.registry, connection))
	s.grant = egress.Grant{Connection: core.ConnectionRef{CustomerID: customer, ConnectionID: connection.ID}, Binding: "forge",
		Scheme: bearer.Name, Placeholder: "va_ph_forge", Hosts: []string{"api.forge.example"}}
	s.hold("gho_first", store.ConnectionConnected)
}

func (s *ResolverTokensSuite) hold(token, status string) {
	stored, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: token}})
	s.Require().NoError(err)
	s.Require().NoError(s.credentials.Update(s.ctx, s.grant.Connection, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials, state.Status = stored, status
		return true, nil
	}))
}

// TestEachCallGivesTheTokenTheConnectionHoldsThen: a new credential is used on the next call.
func (s *ResolverTokensSuite) TestEachCallGivesTheTokenTheConnectionHoldsThen() {
	first, err := s.tokens(s.ctx, s.grant)
	s.Require().NoError(err)
	s.hold("gho_second", store.ConnectionConnected)
	second, err := s.tokens(s.ctx, s.grant)
	s.Require().NoError(err)

	s.Equal([]string{"gho_first", "gho_second"}, []string{first, second})
}

// TestAConnectionThatIsNotConnectedOrIsGoneGivesNoCredential: the proxy answers 401.
func (s *ResolverTokensSuite) TestAConnectionThatIsNotConnectedOrIsGoneGivesNoCredential() {
	s.hold("gho_first", store.ConnectionNeedsReauthorization)
	_, reauthorize := s.tokens(s.ctx, s.grant)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.grant.Connection.CustomerID, s.grant.Connection.ConnectionID))
	_, deleted := s.tokens(s.ctx, s.grant)

	s.ErrorIs(reauthorize, sandbox.ErrNoCredential)
	s.ErrorIs(deleted, sandbox.ErrNoCredential)
}

// TestASchemeThatIsNotRegisteredOrSendsNoBearerGivesNoCredential.
func (s *ResolverTokensSuite) TestASchemeThatIsNotRegisteredOrSendsNoBearerGivesNoCredential() {
	unknown := s.grant
	unknown.Scheme = "gone"
	_, unregistered := s.tokens(s.ctx, unknown)
	unsent := s.grant
	unsent.Scheme = none.Name
	_, notBearer := s.tokens(s.ctx, unsent)
	for _, name := range []string{"empty_bearer", "basic"} {
		other := s.grant
		other.Scheme = name
		_, err := s.tokens(s.ctx, other)
		s.ErrorIs(err, sandbox.ErrNotBearer, name)
	}

	s.ErrorIs(unregistered, sandbox.ErrNoCredential)
	s.ErrorIs(notBearer, sandbox.ErrNotBearer)
}

// sends is a scheme whose requests carry authorization as their Authorization.
type sends struct {
	core.Scheme
	authorization string
}

func (s sends) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return roundTrip(func(r *http.Request) (*http.Response, error) {
		r = r.Clone(r.Context())
		r.Header.Set("Authorization", s.authorization)
		return base.RoundTrip(r)
	})
}

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
