//go:build integration

package streamapp

import (
	"context"
	"errors"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// StoredSuite runs app mode's source against Postgres. The deployment's own app is app 1,
// and each test's customer is an app id nothing else uses.
type StoredSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	sealer     *auth.Sealer
	deployment *Deployment
	clock      *clock
	customer   string
	app        int64
	// stream is the Stream every app in the suite is reached at.
	stream *chattest.Server
}

func TestStoredSuite(t *testing.T) {
	suite.Run(t, new(StoredSuite))
}

func (s *StoredSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "streamapp"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.sealer, err = auth.NewSealerWithKeyring(1, map[int]string{1: "first-key"})
	s.Require().NoError(err)
}

func (s *StoredSuite) SetupTest() {
	// The database is this suite's own, and an api key is the same key in every test.
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE stream_fallback_uses, stream_app_tokens CASCADE")
	s.Require().NoError(err)
	s.stream = chattest.NewServer(s.T())
	s.deployment = NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", App: 1, Strict: true, BaseURL: s.stream.URL})
	s.clock = &clock{now: time.Unix(1_700_000_000, 0)}
	// An app id of the test's own, so a registration left by an earlier run is not this one.
	s.app = int64(uuid.New().ID()) + 1_000_000
	s.customer = strconv.FormatInt(s.app, 10)
}

func (s *StoredSuite) source(fallback bool) *Stored {
	source, err := NewStored(StoredOptions{
		Store: s.store, Deployment: s.deployment, FallbackToDeployment: fallback, Forwarded: s.forwarded(nil), Now: s.clock.Now,
	})
	s.Require().NoError(err)
	return source
}

// ownToken keeps a server token for the test's customer's own app, as the gateway forwards
// one with each request it vouches for.
func (s *StoredSuite) ownToken() {
	_, err := s.forwarded(nil).Keep(s.ctx, s.customer, forwardedKey, s.serverToken(24*time.Hour))
	s.Require().NoError(err)
}

func (s *StoredSuite) TestTheDeploymentAppWritesWithTheEnvironmentCredential() {
	identity, err := s.source(false).For(s.ctx, "1")

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
	s.Equal(int64(1), identity.StreamApp, "app mode writes the deployment app's real id")
}

func (s *StoredSuite) TestOnlyTheCanonicalDeploymentIdGetsTheEnvironmentIdentity() {
	for _, customer := range []string{"01", "1 ", "+1", "1.0"} {
		_, err := s.source(false).For(s.ctx, customer)
		s.ErrorIs(err, ErrNoIdentity, "%q is not app 1", customer)
	}
}

func (s *StoredSuite) TestAnUnregisteredAppFallsBackOnlyWhenAllowed() {
	identity, err := s.source(true).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
	s.Equal(int64(1), identity.StreamApp)
	uses, err := s.store.StreamFallbackUses(s.ctx, s.clock.Now().Add(-time.Minute))
	s.Require().NoError(err)
	s.Contains(customersOf(uses), s.customer)
}

func (s *StoredSuite) TestAnUnregisteredAppIsRefusedWhenFallbackIsRefuse() {
	_, err := s.source(false).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity)
}

func (s *StoredSuite) TestWorkIsFinishedInTheAppItWasPinnedTo() {
	s.ownToken()

	identity, err := s.source(false).ForApp(s.ctx, s.customer, s.app)

	s.Require().NoError(err)
	s.Equal(forwardedKey, identity.APIKey)
}

func (s *StoredSuite) TestARecordPinnedToAnotherAppIsParked() {
	s.ownToken()

	_, err := s.source(true).ForApp(s.ctx, s.customer, 999)
	s.ErrorIs(err, ErrStreamAppMoved)
	_, err = s.source(true).ForApp(s.ctx, s.customer, store.ForeignStreamApp)
	s.ErrorIs(err, ErrStreamAppMoved)
}

func (s *StoredSuite) TestAnotherCustomersLegacyWorkIsReadOnlyUnderRefusal() {
	// Written into the shared app before the customer had one of its own: still read back
	// from there, never added to.
	s.ownToken()
	source := s.source(false)

	for _, pin := range []int64{0, 1} {
		_, err := source.ForApp(s.ctx, s.customer, pin)
		s.ErrorIs(err, ErrReadOnly)
		identity, err := source.ForAppReading(s.ctx, s.customer, pin)
		s.Require().NoError(err)
		s.Equal("deploy-key", identity.APIKey)
	}
}

func (s *StoredSuite) TestLegacyWorkStaysWritableWhileTheFallbackAllows() {
	identity, err := s.source(true).ForApp(s.ctx, s.customer, 0)

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
}

func (s *StoredSuite) TestTheDeploymentCustomersLegacyWorkStaysWritable() {
	identity, err := s.source(false).ForApp(s.ctx, "1", 0)

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
}

func (s *StoredSuite) TestDeploymentAppWorkWaitsUntilItsIdIsKnown() {
	s.deployment = NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", Strict: true})
	source := s.source(false)

	_, err := source.For(s.ctx, s.customer)
	s.ErrorIs(err, ErrDeploymentAppUnknown, "this may be the deployment's own customer")
	_, err = source.ForApp(s.ctx, s.customer, 0)
	s.ErrorIs(err, ErrDeploymentAppUnknown)
	_, err = source.ForApp(s.ctx, s.customer, 999)
	s.ErrorIs(err, ErrDeploymentAppUnknown, "999 may be the deployment's own app")
	_, err = source.For(s.ctx, "not-an-app")
	s.ErrorIs(err, ErrNoIdentity)

	s.deployment.SetApp(1)
	identity, err := source.For(s.ctx, "1")
	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
}

func (s *StoredSuite) TestTheFallbackIsRecordedAtMostOnceAMinute() {
	source := s.source(true)
	for range 3 {
		_, err := source.For(s.ctx, s.customer)
		s.Require().NoError(err)
	}
	s.Equal(int64(1), s.usesOf(s.customer))

	s.clock.Advance(time.Minute)
	_, err := source.For(s.ctx, s.customer)
	s.Require().NoError(err)

	s.Equal(int64(4), s.usesOf(s.customer), "uses in between are counted when next written")
}

func (s *StoredSuite) TestForwardedTokensAreNotKeptWithoutAKeyring() {
	_, err := NewForwarded(ForwardedOptions{Store: s.store})

	s.ErrorContains(err, "keyring")
}

func (s *StoredSuite) usesOf(customer string) int64 {
	uses, err := s.store.StreamFallbackUses(s.ctx, time.Unix(0, 0))
	s.Require().NoError(err)
	for _, use := range uses {
		if use.CustomerID == customer {
			return use.Uses
		}
	}
	return 0
}

func customersOf(uses []store.StreamFallbackUse) []string {
	customers := make([]string, 0, len(uses))
	for _, use := range uses {
		customers = append(customers, use.CustomerID)
	}
	return customers
}

func (s *StoredSuite) TestADeploymentSignedHookWaitsForTheDeploymentApp() {
	s.deployment = NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", Strict: true})
	clients := NewClients(s.source(false), ClientsOptions{})

	_, err := clients.Verifiers(s.ctx, "", 0)
	s.ErrorIs(err, ErrDeploymentAppUnknown)
	_, err = clients.Verifiers(s.ctx, "", 999)
	s.ErrorIs(err, ErrDeploymentAppUnknown, "999 may be the deployment's own app")
}

// floored is app mode's source with the fallback on and a floor answering as given.
func (s *StoredSuite) floored(required bool, err error) *Stored {
	source := s.source(true)
	source.SetFloor(func(context.Context, string) (bool, error) { return required, err })
	return source
}

func (s *StoredSuite) TestAnOrganizationThatRequiresItsOwnAppRefusesTheDeploymentApp() {
	_, err := s.floored(true, nil).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity, "the fallback is never used for an app that must have its own")
}

func (s *StoredSuite) TestARequiredOwnAppParksLegacyWrites() {
	source := s.floored(true, nil)

	_, err := source.ForApp(s.ctx, s.customer, 0)
	s.ErrorIs(err, ErrReadOnly)
	identity, err := source.ForAppReading(s.ctx, s.customer, 0)
	s.Require().NoError(err, "what it wrote there is still read back")
	s.Equal("deploy-key", identity.APIKey)
}

func (s *StoredSuite) TestAPolicyReadErrorRefusesTheFallback() {
	_, err := s.floored(false, errors.New("the policies could not be read")).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity)
}

func (s *StoredSuite) TestTheDeploymentAppIsUnaffectedByTheRequirement() {
	identity, err := s.floored(true, nil).For(s.ctx, "1")

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
}

func (s *StoredSuite) TestAnAppWithATokenOfItsOwnIsUnaffectedByTheRequirement() {
	s.ownToken()

	identity, err := s.floored(true, nil).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal(forwardedKey, identity.APIKey)
}

func (s *StoredSuite) TestADeploymentWithoutASecretChecksNoHookWithAnEmptyOne() {
	// An empty secret is one anybody can sign with.
	s.deployment = NewDeployment(DeploymentOptions{App: 1, Strict: true})
	clients := NewClients(s.source(false), ClientsOptions{})

	for _, pathApp := range []int64{0, 1} {
		verifiers, err := clients.Verifiers(s.ctx, "", pathApp)
		s.Require().NoError(err)
		s.Empty(verifiers, "app %d", pathApp)
	}
}

func (s *StoredSuite) TestTheDeploymentAppsHooksAreCheckedWithItsOwnSecret() {
	clients := NewClients(s.source(false), ClientsOptions{})

	verifiers, err := clients.Verifiers(s.ctx, "", 1)

	s.Require().NoError(err)
	s.Require().Len(verifiers, 1)
	s.True(verifiers[0].Deployment)
	s.Equal("deploy-secret", verifiers[0].Secret.Reveal())
}

func (s *StoredSuite) TestACustomersOwnAppsHooksAreCheckedAgainstNoSecretHere() {
	// The router holds none of the app's secrets: the gateway in front vouches for them.
	s.ownToken()
	clients := NewClients(s.source(false), ClientsOptions{})

	verifiers, err := clients.Verifiers(s.ctx, forwardedKey, s.app)

	s.Require().NoError(err)
	s.Empty(verifiers)
}

func (s *StoredSuite) TestADeploymentWithNoKeyOfItsOwnMakesNobodyWait() {
	// Its own app can never be learned, so nobody's work can turn out to be that app's.
	s.deployment = NewDeployment(DeploymentOptions{Strict: true})
	source := s.source(false)

	_, err := source.For(s.ctx, s.customer)
	s.ErrorIs(err, ErrNoIdentity)
	_, err = source.ForApp(s.ctx, s.customer, 999)
	s.ErrorIs(err, ErrStreamAppMoved)
}
