//go:build integration

package phone

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
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
	vendor     *stub
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
	s.apps = &testApps{deployment: s.deployment, own: map[string]*chattest.Server{}, pins: map[string]int64{}, blocked: map[string]bool{}}
	s.customer = "customer-" + uuid.NewString()
	s.e164 = "+1512" + uuid.NewString()[:7]

	vendor := &stub{vendor: "twilio"}
	s.vendor = vendor
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

func (s *AppsSuite) TestAttachingAnAttachedNumberAgainChangesNothing() {
	// The SDKs attach on every WaitForCall, so the same request arrives on every run.
	s.ownApp()
	s.own.OneTrunkPerNumber()
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.vendor.inbound = Inbound{}

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal(first.RouteID, again.RouteID)
	s.Equal(first.Bridge.URI, again.Bridge.URI)
	s.Equal([]string{first.TrunkID}, s.own.Trunks())
	s.Equal([]string{first.RouteID}, s.own.Rules())
	s.Empty(s.vendor.inbound.E164, "the vendor is not told anything new")
}

func (s *AppsSuite) TestAttachingAnAttachedNumberAgainInTheDeploymentsAppChangesNothing() {
	// The deployment's app is stored as no app at all, and read back as zero.
	s.deployment.OneTrunkPerNumber()
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.vendor.inbound = Inbound{}

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal(first.RouteID, again.RouteID)
	s.Equal([]string{first.TrunkID}, s.deployment.Trunks())
	s.Equal([]string{first.RouteID}, s.deployment.Rules())
	s.Empty(s.vendor.inbound.E164, "the vendor is not told anything new")
}

// Numbers attached before every caller had a call of their own route into one fixed call,
// which attaching again used to replace.
func (s *AppsSuite) TestAttachingANumberThatStillNamesAFixedCallChangesNothing() {
	s.ownApp()
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	held := s.number()
	s.Require().NoError(s.store.AttachNumber(s.ctx, s.customer, s.e164, store.NumberAttachment{
		TrunkID: held.StreamTrunkID, RouteID: held.StreamRouteID, StreamAppPK: held.StreamAppPK,
		CallType: held.StreamCallType, CallID: "phone-" + s.e164, BridgeURI: held.BridgeURI,
	}))
	s.vendor.inbound = Inbound{}

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal([]string{first.TrunkID}, s.own.Trunks())
	s.Equal([]string{first.RouteID}, s.own.Rules())
	s.Empty(s.vendor.inbound.E164, "the vendor is not told anything new")
	s.Equal("phone-"+s.e164, s.number().StreamCallID, "the row is left as it was")
}

func (s *AppsSuite) TestAttachingATelnyxNumberWhosePasswordWasNotKeptChangesNothing() {
	telnyx := &stub{vendor: "telnyx"}
	registry := NewRegistry(s.config())
	registry.Register(telnyx.vendor, func() (Provider, error) { return telnyx, nil })
	for _, name := range s.credentials(telnyx.vendor) {
		s.T().Setenv(name, "set")
	}
	sealer, err := auth.NewSealer("apps suite")
	s.Require().NoError(err)
	service, err := NewService(ServiceOptions{Registry: registry, Store: s.store, Apps: s.apps, Sealer: sealer})
	s.Require().NoError(err)
	e164 := "+1512" + uuid.NewString()[:7]
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: telnyx.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))
	first, err := service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: e164})
	s.Require().NoError(err)
	s.Require().NoError(s.store.AttachNumber(s.ctx, s.customer, e164, store.NumberAttachment{
		TrunkID: first.TrunkID, RouteID: first.RouteID, CallType: defaultCallType,
		BridgeURI: first.Bridge.URI, BridgeUsername: first.Bridge.Username,
	}))
	telnyx.inbound = Inbound{}

	again, err := service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal([]string{first.TrunkID}, s.deployment.Trunks())
	s.Equal([]string{first.RouteID}, s.deployment.Rules())
	s.Empty(telnyx.inbound.E164, "telnyx is not told anything new")
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
	})
	s.Require().NoError(err)

	s.Equal(int64(4242), placed.StreamApp)
	s.Equal(placed.SessionID, placed.CallID, "the answered leg lands in the session's own call")
	s.Equal("agent", placed.CallType)
	s.Len(s.own.Trunks(), 1)
	pin, found, err := s.store.CallPin(s.ctx, s.customer, "agent", placed.SessionID)
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
	blocked    map[string]bool
}

func (a *testApps) block(customer string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.blocked[customer] = true
}

func (a *testApps) give(customer string, pin int64, server *chattest.Server) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.own[customer], a.pins[customer] = server, pin
}

func (a *testApps) For(_ context.Context, customer string) (*Stream, int64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.blocked[customer] {
		return nil, 0, streamapp.ErrStreamAppDisconnected
	}
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

func (s *AppsSuite) TestAnAppStreamBlockedCannotBuyANumber() {
	s.apps.block(s.customer)

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: "+15550002222", Owner: routing.Owner{CustomerID: s.customer}})

	s.ErrorIs(err, streamapp.ErrStreamAppDisconnected)
}

func (s *AppsSuite) TestEveryLegOfAnEndedCallCountsTowardTheSandboxsMinutes() {
	// A transfer is a second leg alongside the call's own, billed for as long as it lasts.
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(redis.Close)
	gate := dlc.NewGate(s.store, redis, dlc.Sandbox{Enabled: true}, nil)
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config()), Store: s.store, Apps: s.apps, Gate: gate})
	s.Require().NoError(err)

	callID := "call-" + uuid.NewString()
	now := time.Now().UTC()
	for _, began := range []time.Duration{10 * time.Minute, 4 * time.Minute} {
		s.Require().NoError(s.store.RecordCallResource(s.ctx, &store.CallResource{
			TrunkID: "trunk-" + uuid.NewString(), RouteID: "route-" + uuid.NewString(),
			CallType: "default", CallID: callID, CustomerID: s.customer, CreatedAt: now.Add(-began),
		}))
	}

	s.Require().NoError(service.ReleaseCall(s.ctx, store.AppScope{App: 1, Unpinned: true}, "default", callID))

	usage, err := gate.Usage(s.ctx, s.customer)
	s.Require().NoError(err)
	s.InDelta(14*60, usage.AudioSeconds, 5, "both legs, not only the one that began first")
}

func (s *AppsSuite) TestTheVendorIsGivenATrunkAddressStreamCanFindByTheCallingNumber() {
	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	s.Equal("sip:"+s.e164+"@sip.example.test", s.vendor.dialed.Bridge.URI)
}

func (s *AppsSuite) TestATransferGivesTheVendorATrunkAddressStreamCanFindByTheCallingNumber() {
	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550002222", CallID: "call-1",
	})
	s.Require().NoError(err)

	s.Equal("sip:"+s.e164+"@sip.example.test", s.vendor.dialed.Bridge.URI)
}

func (s *AppsSuite) TestAttachingGivesTheVendorATrunkAddressStreamCanFindByTheNumber() {
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	want := "sip:" + s.e164 + "@sip.example.test"
	s.Equal(want, s.vendor.inbound.Bridge.URI)
	s.Equal(want, attached.Bridge.URI, "the api shows the address the vendor was given")
}

func (s *AppsSuite) TestAnAttachedNumbersCallersEachLandInACallOfTheirOwn() {
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	rule := s.deployment.Rule(attached.RouteID)
	s.Require().NotNil(rule)
	direct := rule["direct_routing_configs"].(map[string]any)
	s.Equal("{{uuid}}", direct["call_id"], "each caller's call is named for the session answering them")
	s.Equal("agent", direct["call_type"])
	custom := rule["call_configs"].(map[string]any)["custom_data"].(map[string]any)
	s.Equal(s.e164, custom[NumberKey], "the call says which number it rang")
}

func (s *AppsSuite) TestAPlacedCallLandsInTheCallOfTheSessionItWasPlacedFor() {
	id := "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e"

	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111", SessionID: id,
	})
	s.Require().NoError(err)

	s.Equal(id, placed.SessionID)
	s.Equal(id, placed.CallID)
}

func (s *AppsSuite) TestAnAttachedNumbersTrunkCredentialsAreKeptForItsCalls() {
	sealer, err := auth.NewSealer("apps suite")
	s.Require().NoError(err)
	s.service.sealer = sealer

	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	bridge, err := s.service.InboundBridge(s.ctx, "twilio", s.e164)
	s.Require().NoError(err)
	s.Equal("sip:"+s.e164+"@sip.example.test", bridge.URI, "the address names the number")
	s.Equal(attached.TrunkID, bridge.Username)
	s.Equal("secret", bridge.Password, "the password Stream returned at attach")
	s.NotContains(string(s.number().BridgePasswordSealed), "secret", "the password is stored sealed")
}

func (s *AppsSuite) TestWithoutASealerANumberStillAttachesButKeepsNoCredentials() {
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	_, err = s.service.InboundBridge(s.ctx, "twilio", s.e164)

	s.ErrorIs(err, ErrNoSealer)
	s.Empty(s.number().BridgePasswordSealed)
	s.Equal(attached.Bridge.URI, s.number().BridgeURI, "the address is not secret, so it is kept")
}

// Telnyx asks per call where to send a caller, and the answer is the stored trunk, so a
// Telnyx number attached with nowhere to keep the trunk's password could never be answered.
func (s *AppsSuite) TestWithoutASealerATelnyxNumberIsNotAttached() {
	telnyx := &stub{vendor: "telnyx"}
	registry := NewRegistry(s.config())
	registry.Register(telnyx.vendor, func() (Provider, error) { return telnyx, nil })
	for _, name := range s.credentials(telnyx.vendor) {
		s.T().Setenv(name, "set")
	}
	service, err := NewService(ServiceOptions{Registry: registry, Store: s.store, Apps: s.apps})
	s.Require().NoError(err)
	e164 := "+1512" + uuid.NewString()[:7]
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: telnyx.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))

	_, err = service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: e164})

	s.ErrorIs(err, ErrNoSealer)
	s.ErrorContains(err, "ROUTER_AUTH_KEK", "the error names what to set")
	s.Empty(s.deployment.Trunks(), "nothing is made in Stream")
	s.Empty(telnyx.inbound.E164, "telnyx is not told to send calls anywhere")
}
