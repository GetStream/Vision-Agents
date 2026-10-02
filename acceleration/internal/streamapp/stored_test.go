//go:build integration

package streamapp

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
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
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE stream_apps, stream_fallback_uses CASCADE")
	s.Require().NoError(err)
	s.deployment = NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", App: 1, Strict: true})
	s.clock = &clock{now: time.Unix(1_700_000_000, 0)}
	// An app id of the test's own, so a registration left by an earlier run is not this one.
	s.app = int64(uuid.New().ID()) + 1_000_000
	s.customer = strconv.FormatInt(s.app, 10)
}

func (s *StoredSuite) source(fallback bool) *Stored {
	source, err := NewStored(StoredOptions{
		Store: s.store, Sealer: s.sealer, Deployment: s.deployment, FallbackToDeployment: fallback, Now: s.clock.Now,
	})
	s.Require().NoError(err)
	return source
}

// register gives the test's customer its own app with the keys named, the first primary.
func (s *StoredSuite) register(keys ...string) store.StreamApp {
	return s.registerWith(s.sealer, keys...)
}

func (s *StoredSuite) registerWith(sealer *auth.Sealer, keys ...string) store.StreamApp {
	sealed := make([]store.StreamAppKey, 0, len(keys))
	for _, key := range keys {
		one, err := SealKey(sealer, s.customer, s.app, key, key+"-secret-value")
		s.Require().NoError(err)
		sealed = append(sealed, one)
	}
	app, err := s.store.PutStreamApp(s.ctx, store.StreamAppRegistration{
		CustomerID: s.customer, StreamAppPK: s.app, Keys: sealed, PrimaryKey: keys[0], VerifiedAt: time.Now(),
	})
	s.Require().NoError(err)
	return app
}

func (s *StoredSuite) TestAnAppWithItsOwnCredentialActsInItsOwnApp() {
	s.register("own-key")

	identity, err := s.source(true).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal("own-key", identity.APIKey)
	s.Equal("own-key-secret-value", identity.Secret.Reveal())
	s.Equal(s.app, identity.StreamApp)
	s.True(identity.Registered)
	s.False(identity.MintsGuests(), "a registered app mints no guests until it says so")
}

func (s *StoredSuite) TestTheDeploymentAppWritesWithTheEnvironmentCredential() {
	identity, err := s.source(false).For(s.ctx, "1")

	s.Require().NoError(err)
	s.Equal("deploy-key", identity.APIKey)
	s.Equal(int64(1), identity.StreamApp, "app mode writes the deployment app's real id")
	s.True(identity.MintsGuests())
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

func (s *StoredSuite) TestADisconnectedAppNeverFallsBack() {
	s.register("own-key")
	_, err := s.store.DisconnectStreamApp(s.ctx, s.customer, nil, "test")
	s.Require().NoError(err)

	_, err = s.source(true).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrStreamAppDisconnected)
}

func (s *StoredSuite) TestABlockedAppNeverFallsBack() {
	s.register("own-key")
	_, err := s.store.BlockStreamApp(s.ctx, s.customer, "suspended")
	s.Require().NoError(err)

	_, err = s.source(true).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrStreamAppDisconnected)
}

func (s *StoredSuite) TestAnAppWhoseKeysAreAllRejectedNeverFallsBack() {
	s.register("own-key")
	s.Require().NoError(s.store.RejectStreamAppKey(s.ctx, s.customer, "own-key", "401", time.Now()))

	_, err := s.source(true).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrStreamAppDisconnected)
}

func (s *StoredSuite) TestARejectedPrimaryIsPassedOverForTheOldestAcceptedKey() {
	s.register("primary", "secondary")
	s.Require().NoError(s.store.RejectStreamAppKey(s.ctx, s.customer, "primary", "401", time.Now()))

	identity, err := s.source(false).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal("secondary", identity.APIKey)
}

func (s *StoredSuite) TestWorkIsFinishedInTheAppItWasPinnedTo() {
	s.register("own-key")

	identity, err := s.source(false).ForApp(s.ctx, s.customer, s.app)

	s.Require().NoError(err)
	s.Equal("own-key", identity.APIKey)
}

func (s *StoredSuite) TestARecordPinnedToAnotherAppIsParked() {
	s.register("own-key")

	_, err := s.source(true).ForApp(s.ctx, s.customer, 999)
	s.ErrorIs(err, ErrStreamAppMoved)
	_, err = s.source(true).ForApp(s.ctx, s.customer, store.ForeignStreamApp)
	s.ErrorIs(err, ErrStreamAppMoved)
}

func (s *StoredSuite) TestAnotherCustomersLegacyWorkIsReadOnlyUnderRefusal() {
	// Written into the shared app before the customer had one of its own: still read back
	// from there, never added to.
	s.register("own-key")
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

func (s *StoredSuite) TestAKeySealedUnderAnOldVersionIsRewrappedOnUse() {
	s.register("own-key")
	s.sealer, _ = auth.NewSealerWithKeyring(2, map[int]string{1: "first-key", 2: "second-key"})
	defer func() { s.sealer, _ = auth.NewSealerWithKeyring(1, map[int]string{1: "first-key"}) }()

	identity, err := s.source(false).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal("own-key-secret-value", identity.Secret.Reveal())
	held, err := s.store.StreamApp(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal(2, held.Keys[0].KEKVersion)
	reopened, err := s.source(false).For(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal("own-key-secret-value", reopened.Secret.Reveal())
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

func (s *StoredSuite) TestAppModeRefusesToStartWithoutAKeyring() {
	_, err := NewStored(StoredOptions{Store: s.store, Deployment: s.deployment})

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
