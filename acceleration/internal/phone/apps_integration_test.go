//go:build integration

package phone

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// AppsSuite runs the phone service against two Stream apps in memory, the deployment's and
// a customer's own, with a stub vendor, and checks that every trunk and rule is made and
// removed in the app the call it serves is in.
type AppsSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store

	deployment *chattest.Server
	own        *chattest.Server
	apps       *testApps
	service    *Service
	customer   string
	e164       string
}

func TestAppsSuite(t *testing.T) { suite.Run(t, new(AppsSuite)) }

func (s *AppsSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "phone_apps"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *AppsSuite) SetupTest() {
	s.deployment, s.own = chattest.NewServer(s.T()), chattest.NewServer(s.T())
	s.apps = &testApps{deployment: s.deployment, own: map[string]*chattest.Server{}, pins: map[string]int64{}}
	s.customer = "customer-" + uuid.NewString()
	s.e164 = "+1512" + uuid.NewString()[:7]

	vendor := &stub{vendor: "twilio"}
	registry := NewRegistry(s.config())
	registry.Register(vendor.vendor, func() (Provider, error) { return vendor, nil })
	for _, name := range s.credentials(vendor.vendor) {
		s.T().Setenv(name, "set")
	}
	service, err := NewService(ServiceOptions{Registry: registry, Store: s.store, Apps: s.apps})
	s.Require().NoError(err)
	s.service = service

	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: s.e164, Vendor: vendor.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))
}

func (s *AppsSuite) config() Config {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	return config
}

func (s *AppsSuite) credentials(vendor string) []string {
	declared, ok := s.config().Lookup(vendor)
	s.Require().True(ok)
	return declared.Credentials
}

// ownApp gives the suite's customer an app of its own from now on.
func (s *AppsSuite) ownApp() {
	s.apps.give(s.customer, 4242, s.own)
}

func (s *AppsSuite) number() store.PhoneNumber {
	held, err := s.store.Number(s.ctx, s.customer, s.e164)
	s.Require().NoError(err)
	return held
}

func (s *AppsSuite) TestATrunkIsCreatedInTheAppAttachingTheNumber() {
	s.ownApp()

	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
	s.Equal([]string{attached.RouteID}, s.own.Rules())
	s.Empty(s.deployment.Trunks())
	held := s.number()
	s.Equal(int64(4242), held.StreamAppPK)
	s.Equal(attached.RouteID, held.StreamRouteID)
}

func (s *AppsSuite) TestReleasingANumberDeletesItsTrunkAndRouteInItsApp() {
	// A trunk left behind is billed, and points a number nobody holds at a call.
	s.ownApp()
	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Empty(s.own.Trunks())
	s.Empty(s.own.Rules())
}

func (s *AppsSuite) TestReattachingAfterRegistrationMovesTheTrunk() {
	// The number was attached while the customer acted in the deployment's app, and is
	// attached again once it has its own: the old trunk goes from the old app.
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.Require().Equal([]string{first.TrunkID}, s.deployment.Trunks())
	s.ownApp()

	second, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	s.Empty(s.deployment.Trunks())
	s.Empty(s.deployment.Rules())
	s.Equal([]string{second.TrunkID}, s.own.Trunks())
}

func (s *AppsSuite) TestReleasingAfterRegistrationDeletesTheLinesLeftInTheDeploymentApp() {
	// The customer may no longer add to the deployment's app, and what the router made
	// there for it is still the router's to take down.
	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.Require().Len(s.deployment.Trunks(), 1)
	s.ownApp()

	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Empty(s.deployment.Trunks())
	s.Empty(s.deployment.Rules())
}

func (s *AppsSuite) TestACallEndedAfterRegistrationDeletesItsLinesInTheDeploymentApp() {
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)
	s.Require().Len(s.deployment.Trunks(), 1)
	s.ownApp()

	s.Require().NoError(s.service.ReleaseCall(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID))

	s.Empty(s.deployment.Trunks())
	s.Empty(s.deployment.Rules())
}

func (s *AppsSuite) TestACampaignCallAndItsSessionShareAnAppAndACall() {
	s.ownApp()

	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
		CallID: "campaign-contact-1", CallType: "agent",
	})
	s.Require().NoError(err)

	s.Equal(int64(4242), placed.StreamApp)
	s.Len(s.own.Trunks(), 1)
	pin, found, err := s.store.CallPin(s.ctx, s.customer, "agent", "campaign-contact-1")
	s.Require().NoError(err)
	s.True(found)
	s.Equal(int64(4242), pin, "the session joining it finds the app the call's lines are in")
}

func (s *AppsSuite) TestReleasingDeletesEachTrunkInTheAppThatMadeIt() {
	s.ownApp()
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	s.Require().NoError(s.service.ReleaseCall(s.ctx, store.AppScope{App: 4242}, placed.CallType, placed.CallID))

	s.Empty(s.own.Trunks())
	s.Empty(s.own.Rules())
}

func (s *AppsSuite) TestAnotherAppsEndedCallReleasesNothing() {
	// A call's id is only unique within an app. An event about a call in the deployment's
	// app must not take down the lines of a call of the same name in a customer's.
	s.ownApp()
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	s.Require().NoError(s.service.ReleaseCall(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID))

	s.Len(s.own.Trunks(), 1)
}

func (s *AppsSuite) TestALegacyNumberStillReleasesInTheDeploymentApp() {
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)
	s.Require().Len(s.deployment.Trunks(), 1)

	s.Require().NoError(s.service.ReleaseCall(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID))

	s.Empty(s.deployment.Trunks())
}

func (s *AppsSuite) TestAMidCallTransferCreatesItsTrunkInTheSessionsApp() {
	s.ownApp()

	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550002222",
		CallID: "call-1", StreamApp: 4242,
	})
	s.Require().NoError(err)

	s.Len(s.own.Trunks(), 1)
	s.Empty(s.deployment.Trunks())
}

// testApps is the apps the suite's customers make lines in: their own when given one, the
// deployment's otherwise. As app mode does once it no longer falls back, a customer given an
// app of its own can read what it left in the deployment's app and not add to it.
type testApps struct {
	mu         sync.Mutex
	deployment *chattest.Server
	own        map[string]*chattest.Server
	pins       map[string]int64
}

func (a *testApps) give(customer string, pin int64, server *chattest.Server) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.own[customer], a.pins[customer] = server, pin
}

func (a *testApps) For(_ context.Context, customer string) (*Stream, int64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if server, ok := a.own[customer]; ok {
		return NewStreamFromClient(server.Client), a.pins[customer], nil
	}
	return NewStreamFromClient(a.deployment.Client), 0, nil
}

func (a *testApps) ForApp(ctx context.Context, customer string, app int64) (*Stream, error) {
	a.mu.Lock()
	_, owns := a.own[customer]
	a.mu.Unlock()
	if owns && app == 0 {
		return nil, streamapp.ErrReadOnly
	}
	return a.ForAppRemoving(ctx, customer, app)
}

func (a *testApps) ForAppRemoving(_ context.Context, customer string, app int64) (*Stream, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if server, ok := a.own[customer]; ok && a.pins[customer] == app {
		return NewStreamFromClient(server.Client), nil
	}
	return NewStreamFromClient(a.deployment.Client), nil
}
