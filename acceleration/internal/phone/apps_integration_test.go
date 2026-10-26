//go:build integration

package phone

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
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
	s.e164 = aNumber()

	vendor := &stub{vendor: "twilio", leg: StreamLegShared}
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

// aNumber is a number nobody in the suite's database holds yet.
func aNumber() string { return fmt.Sprintf("+1512%010d", rand.Int64N(10_000_000_000)) }

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

// onAnAllowlist is a number held on a vendor that cannot send a trunk password, and the
// addresses that vendor calls from.
type onAnAllowlist struct {
	e164       string
	signalling []string
}

// vendorOnAnAllowlist registers vonage, which authenticates by address, and gives the
// suite's customer a number on it.
func (s *AppsSuite) vendorOnAnAllowlist() onAnAllowlist {
	declared, ok := s.config().Lookup("vonage")
	s.Require().True(ok)
	s.Require().NotEmpty(declared.Signalling, "the test needs a vendor that declares its addresses")
	return onAnAllowlist{e164: s.numberOn("vonage"), signalling: declared.Signalling}
}

// numberOn registers a stub for the vendor next to the suite's own and gives the suite's
// customer a number on it.
func (s *AppsSuite) numberOn(vendor string) string {
	other := &stub{vendor: vendor}
	registry := NewRegistry(s.config())
	registry.Register(s.vendor.vendor, func() (Provider, error) { return s.vendor, nil })
	registry.Register(other.vendor, func() (Provider, error) { return other, nil })
	for _, name := range append(s.credentials(s.vendor.vendor), s.credentials(other.vendor)...) {
		s.T().Setenv(name, "set")
	}
	service, err := NewService(ServiceOptions{Registry: registry, Store: s.store, Apps: s.apps})
	s.Require().NoError(err)
	s.service = service

	e164 := aNumber()
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: other.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))
	return e164
}

// planStub is a vendor that fetches its call plan when the call is answered, and renders
// the bridge it is given into the plan so a test can read it back.
type planStub struct {
	stub
	rendered Bridge
}

func (p *planStub) Answer(bridge Bridge, _ string) (Plan, error) {
	p.rendered = bridge
	return Plan{ContentType: "text/plain", Body: []byte(bridge.Password)}, nil
}

func (s *AppsSuite) TestACallsTrunkHasAPasswordTheVendorIsGiven() {
	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	password, allowed := s.deployment.TrunkAuth(trunks[0])
	s.NotEmpty(password, "a trunk with no allowlist and no password lets anybody in")
	s.Empty(allowed)
	s.Equal(password, s.vendor.dialed.Bridge.Password, "the vendor answers the challenge with it")
}

func (s *AppsSuite) TestEveryCallsTrunkHasAPasswordOfItsOwn() {
	call := func() string {
		_, err := s.service.Call(s.ctx, CallRequest{
			Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
		})
		s.Require().NoError(err)
		return s.vendor.dialed.Bridge.Password
	}

	s.NotEqual(call(), call())
}

func (s *AppsSuite) TestAnAttachedNumbersTrunkHasAPasswordTheVendorIsGiven() {
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	password, allowed := s.deployment.TrunkAuth(attached.TrunkID)
	s.NotEmpty(password)
	s.Empty(allowed)
	s.Equal(password, s.vendor.inbound.Bridge.Password)
}

func (s *AppsSuite) TestACallOnAVendorThatCannotSendAPasswordHasItsAddressesAndNoPassword() {
	vonage := s.vendorOnAnAllowlist()

	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: vonage.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	password, allowed := s.deployment.TrunkAuth(trunks[0])
	s.Equal(vonage.signalling, allowed)
	s.Empty(password, "Stream would challenge a vendor that has no way to answer")
}

func (s *AppsSuite) TestACallPlanFetchedOnAnswerCarriesTheTrunksPassword() {
	plivo := &planStub{stub: stub{vendor: "plivo"}}
	registry := NewRegistry(s.config())
	registry.Register(plivo.vendor, func() (Provider, error) { return plivo, nil })
	for _, name := range s.credentials(plivo.vendor) {
		s.T().Setenv(name, "set")
	}
	service, err := NewService(ServiceOptions{
		Registry: registry, Store: s.store, Apps: s.apps, PublicURL: "https://router.example.test",
	})
	s.Require().NoError(err)
	e164 := aNumber()
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: plivo.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))

	_, err = service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: e164, To: "+15550001111",
	})
	s.Require().NoError(err)
	token := plivo.dialed.AnswerURL[strings.LastIndex(plivo.dialed.AnswerURL, "/")+1:]

	_, err = service.Answer(s.ctx, token)
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	password, _ := s.deployment.TrunkAuth(trunks[0])
	s.NotEmpty(password)
	s.Equal(password, plivo.rendered.Password)
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

func (s *AppsSuite) TestDetachingKeepsTheNumberOnItsTrunk() {
	s.ownApp()
	s.sealed()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	held := s.number()
	s.Nil(held.AttachedAt, "callers no longer reach the agent")
	s.Equal(attached.TrunkID, held.StreamTrunkID)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
	s.Equal([]string{attached.RouteID}, s.own.Rules())
	s.Equal([]string{s.e164}, s.own.TrunkNumbers(attached.TrunkID))
	s.Equal([]string{s.e164}, s.vendor.stopped, "the vendor stops sending callers to the trunk")
}

func (s *AppsSuite) TestDetachingANumberAttachedInTheDeploymentsApp() {
	// The deployment's app is stored as no app at all, and its trunk group is app zero.
	s.sealed()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	held := s.number()
	s.Nil(held.AttachedAt)
	s.Equal(attached.TrunkID, held.StreamTrunkID)
	s.Equal([]string{attached.TrunkID}, s.deployment.Trunks())
	s.Equal([]string{attached.RouteID}, s.deployment.Rules())
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(attached.TrunkID))
	s.Equal([]string{s.e164}, s.vendor.stopped)
}

func (s *AppsSuite) TestADetachedNumberStillPlacesCalls() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	_, err = s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
}

func (s *AppsSuite) TestDetachingANumberNotAttachedChangesNothing() {
	s.sealed()
	before := s.number()

	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	s.Equal(before, s.number())
	s.Equal([]string{s.e164}, s.vendor.stopped, "the vendor is told anyway, in case it still sends callers")
}

func (s *AppsSuite) TestDetachingAnotherCustomersNumberFails() {
	s.sealed()
	other := "customer-" + uuid.NewString()
	theirs := s.anotherNumber(other, s.vendor.vendor)
	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: other, E164: theirs})
	s.Require().NoError(err)

	err = s.service.Detach(s.ctx, s.customer, theirs)

	s.ErrorContains(err, "is not a number")
	held, err := s.store.Number(s.ctx, other, theirs)
	s.Require().NoError(err)
	s.NotNil(held.AttachedAt)
	s.Empty(s.vendor.stopped)
}

func (s *AppsSuite) TestAVendorThatRefusesLeavesTheNumberAttached() {
	s.sealed()
	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.vendor.err = errors.New("the vendor refused")

	s.Require().Error(s.service.Detach(s.ctx, s.customer, s.e164))
	s.NotNil(s.number().AttachedAt, "the vendor may still send callers, so the agent still answers")

	s.vendor.err = nil
	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))
	s.Nil(s.number().AttachedAt)
}

func (s *AppsSuite) TestAttachingADetachedNumberUsesTheSameTrunk() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))
	s.vendor.inbound = Inbound{}

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal(first.RouteID, again.RouteID)
	s.NotNil(s.number().AttachedAt)
	s.Equal(s.e164, s.vendor.inbound.E164, "the vendor is pointed back at the trunk")
}

func (s *AppsSuite) TestAttachingADetachedNumberWithoutAKeyWorks() {
	s.own.OneTrunkPerNumber()
	s.ownApp()
	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err, "Stream lets a number be on one trunk only")
	s.NotNil(s.number().AttachedAt)
	s.Equal([]string{again.TrunkID}, s.own.Trunks())
	s.Equal([]string{s.e164}, s.own.TrunkNumbers(again.TrunkID))
}

func (s *AppsSuite) TestDetachingLeavesALiveCallAlone() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	scope := store.AppScope{App: 4242}
	live, err := s.store.CallResourcesInApp(s.ctx, scope, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Require().Len(live, 1)

	s.Require().NoError(s.service.Detach(s.ctx, s.customer, s.e164))

	kept, err := s.store.CallResourcesInApp(s.ctx, scope, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Equal(live, kept)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
	s.Equal([]string{attached.RouteID}, s.own.Rules())

	s.Require().NoError(s.service.ReleaseCall(s.ctx, scope, placed.CallType, placed.CallID))
	ended, err := s.store.CallResourcesInApp(s.ctx, scope, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Empty(ended)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks(), "the number's own line outlives its calls")
}

func (s *AppsSuite) TestDetachingASipTrunkNumberChangesNothing() {
	sealer, err := auth.NewSealer("apps suite")
	s.Require().NoError(err)
	service, err := NewService(ServiceOptions{
		Registry: NewRegistry(s.config()), Store: s.store, Apps: s.apps, Sealer: sealer,
		SIPTrunks: &stub{vendor: SIPTrunkVendor},
	})
	s.Require().NoError(err)
	password := "trunk-s3cret"
	trunk, err := service.CreateSIPTrunk(s.ctx, s.customer, SIPTrunkSettings{
		Name: "main", Host: "trunk.example.com", Username: "trunk-user", Password: &password,
	})
	s.Require().NoError(err)
	e164 := aNumber()
	_, err = service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner: routing.Owner{CustomerID: s.customer}, TrunkID: trunk.ID, E164: e164, Country: "us",
	})
	s.Require().NoError(err)
	before, err := s.store.Number(s.ctx, s.customer, e164)
	s.Require().NoError(err)

	s.Require().NoError(service.Detach(s.ctx, s.customer, e164), "sip_trunk is no vendor of the registry")

	after, err := s.store.Number(s.ctx, s.customer, e164)
	s.Require().NoError(err)
	s.Equal(before, after)
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
	e164 := aNumber()
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

func (s *AppsSuite) TestANumberInAnotherFormIsNotBought() {
	_, err := s.service.Buy(s.ctx, Purchase{
		Vendor: s.vendor.vendor, E164: strings.TrimPrefix(aNumber(), "+"), Owner: routing.Owner{CustomerID: s.customer},
	})

	s.ErrorIs(err, store.ErrNotE164)
	s.Zero(s.vendor.ordered, "the vendor is not paid for a number that would not be recorded")
}

func (s *AppsSuite) TestAnotherFormOfANumberReachesNothingAtTheVendor() {
	// Another customer's row for the suite's number without its plus, as one could be written
	// before every write checked the form. The vendor would read it as the suite's number.
	other, variant := "customer-"+uuid.NewString(), strings.TrimPrefix(s.e164, "+")
	_, err := s.store.DB().ExecContext(s.ctx, "INSERT INTO phone_numbers (e164, vendor, country, customer_id)"+
		" VALUES (?, ?, 'US', ?)", variant, s.vendor.vendor, other)
	s.Require().NoError(err)

	s.ErrorIs(s.service.Release(s.ctx, other, variant), store.ErrNoNumber)
	_, err = s.service.Attach(s.ctx, Attachment{CustomerID: other, E164: variant})
	s.ErrorIs(err, store.ErrNoNumber)
	_, err = s.service.Call(s.ctx, CallRequest{Owner: routing.Owner{CustomerID: other}, From: variant, To: "+15550001111"})
	s.ErrorIs(err, store.ErrNoNumber)

	s.Empty(s.vendor.released)
	s.Zero(s.vendor.inbound)
	s.Zero(s.vendor.dialed)
	_, err = s.store.Number(s.ctx, s.customer, s.e164)
	s.NoError(err, "the number stays with who held it")
}

func (s *AppsSuite) TestAnAppStreamBlockedCannotBuyANumber() {
	s.apps.block(s.customer)

	_, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: "+15550002222", Owner: routing.Owner{CustomerID: s.customer}})

	s.ErrorIs(err, streamapp.ErrStreamAppDisconnected)
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
	s.Equal(lineCallTemplate, direct["call_id"], "each caller's call is named for the session answering them")
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

func (s *AppsSuite) callPlacedRows(placed Placed) []store.AgentLog {
	return s.eventRows(placed, "call_placed")
}

func (s *AppsSuite) eventRows(placed Placed, eventType string) []store.AgentLog {
	rows, err := s.store.AgentLogs(s.ctx, store.LogFilter{
		CustomerID: s.customer, SessionID: placed.SessionID, Limit: 250,
	})
	s.Require().NoError(err)
	var found []store.AgentLog
	for _, row := range rows {
		if row.EventType == eventType {
			found = append(found, row)
		}
	}
	return found
}

func (s *AppsSuite) TestAPlacedCallWritesACallPlacedRow() {
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	rows := s.callPlacedRows(placed)
	s.Require().Len(rows, 1)
	row, err := s.store.AgentLog(s.ctx, s.customer, rows[0].ID)
	s.Require().NoError(err)
	s.Equal(BridgeHost(s.vendor.dialed.Bridge.URI), row.Details["stream_sip_host"])
	s.NotEmpty(row.Details["stream_sip_host"])
	s.Equal("twilio", row.Details["vendor"])
	s.Equal(placed.VendorCallID, row.Details["vendor_call_id"])
	s.NotContains(row.Details, "customer_trunk_host", "the call went out through a vendor, not a customer's trunk")
	for key, value := range row.Details {
		s.NotContains(value, s.vendor.dialed.Bridge.Password, key)
	}
}

func (s *AppsSuite) TestASlowCallPlacedRowDoesNotFailTheCallOrLoseItsResourceRow() {
	// Holding the customer's log lock makes the call_placed write wait.
	locked, unlock := make(chan struct{}), make(chan struct{})
	go func() {
		_ = s.store.DB().RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			_, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, 731))", s.customer)
			close(locked)
			<-unlock
			return err
		})
	}()
	<-locked

	started := time.Now()
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	waited := time.Since(started)
	close(unlock)

	s.Require().NoError(err)
	s.Less(waited, 2*agentLogTimeout, "the reply waits for the row only up to its timeout")
	s.Empty(s.callPlacedRows(placed))
	resources, err := s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Len(resources, 1)
}

// trunkCall is a call placed through a customer's own trunk, with what its service logged and
// the vendor that was asked to place it.
type trunkCall struct {
	placed   Placed
	vendor   *stub
	logged   *bytes.Buffer
	password string
}

func (s *AppsSuite) callsThroughATrunk() trunkCall {
	sealer, err := auth.NewSealer("apps suite")
	s.Require().NoError(err)
	logged := &bytes.Buffer{}
	vendor := &stub{vendor: SIPTrunkVendor}
	service, err := NewService(ServiceOptions{
		Registry: NewRegistry(s.config()), Store: s.store, Apps: s.apps, Sealer: sealer,
		SIPTrunks: vendor, Logger: slog.New(slog.NewTextHandler(logged, nil)),
	})
	s.Require().NoError(err)
	password := "trunk-s3cret"
	trunk, err := service.CreateSIPTrunk(s.ctx, s.customer, SIPTrunkSettings{
		Name: "main", Host: "trunk.example.com", Username: "trunk-user", Password: &password,
	})
	s.Require().NoError(err)
	// A trunk number must be digits only, and stay unique across runs.
	e164 := fmt.Sprintf("+1555010%04d", time.Now().UnixNano()%10000)
	_, err = service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner: routing.Owner{CustomerID: s.customer}, TrunkID: trunk.ID, E164: e164, Country: "us",
	})
	s.Require().NoError(err)

	placed, err := service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: e164, To: "+15550001111",
	})
	s.Require().NoError(err)
	return trunkCall{placed: placed, vendor: vendor, logged: logged, password: password}
}

func (s *AppsSuite) TestACallThroughACustomersTrunkNamesTheTrunksHostOnly() {
	call := s.callsThroughATrunk()

	rows := s.callPlacedRows(call.placed)
	s.Require().Len(rows, 1)
	row, err := s.store.AgentLog(s.ctx, s.customer, rows[0].ID)
	s.Require().NoError(err)
	s.Equal("trunk.example.com", row.Details["customer_trunk_host"])
	s.Equal(SIPTrunkVendor, row.Details["vendor"])
	for key, value := range row.Details {
		s.NotContains(value, "trunk-user", key)
		s.NotContains(value, call.password, key)
	}
	s.Contains(call.logged.String(), "customer_trunk_host=trunk.example.com")
	s.NotContains(call.logged.String(), "trunk-user")
	s.NotContains(call.logged.String(), call.password)
}

func (s *AppsSuite) TestACallThroughACustomersTrunkKeepsWhereEachSidesAudioComesFrom() {
	call := s.callsThroughATrunk()
	s.Require().NotNil(call.vendor.dialed.Connected, "the bridge has nobody to tell")

	call.vendor.dialed.Connected(sipbridge.MediaAddrs{Customer: "198.51.100.7", Stream: "203.0.113.10"})

	rows := s.eventRows(call.placed, "bridge_media")
	s.Require().Len(rows, 1)
	row, err := s.store.AgentLog(s.ctx, s.customer, rows[0].ID)
	s.Require().NoError(err)
	s.Equal(map[string]any{"stream_media_addr": "203.0.113.10", "customer_media_addr": "198.51.100.7"}, row.Details)
}

func (s *AppsSuite) TestABridgeThatSawNoAddressWritesNoRow() {
	call := s.callsThroughATrunk()
	s.Require().NotNil(call.vendor.dialed.Connected)

	call.vendor.dialed.Connected(sipbridge.MediaAddrs{})

	s.Empty(s.eventRows(call.placed, "bridge_media"))
}

func (s *AppsSuite) TestACallThroughAVendorHasNoBridgeToHearFrom() {
	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111",
	})
	s.Require().NoError(err)

	s.Nil(s.vendor.dialed.Connected)
}

func (s *AppsSuite) TestACallTheVendorRefusedWritesNoCallPlacedRow() {
	s.vendor.err = errors.New("the vendor refused")

	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111", SessionID: "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3f",
	})
	s.Require().Error(err)

	s.Empty(s.callPlacedRows(Placed{SessionID: "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3f"}))
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
	password, _ := s.deployment.TrunkAuth(attached.TrunkID)
	s.Require().NotEmpty(password)
	s.Equal(password, bridge.Password, "the password the trunk was made with")
	s.NotContains(string(s.number().BridgePasswordSealed), password, "the password is stored sealed")
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
	e164 := aNumber()
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: telnyx.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))

	_, err = service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: e164})

	s.ErrorIs(err, ErrNoSealer)
	s.ErrorContains(err, "ROUTER_AUTH_KEK", "the error names what to set")
	s.Empty(s.deployment.Trunks(), "nothing is made in Stream")
	s.Empty(telnyx.inbound.E164, "telnyx is not told to send calls anywhere")
}

func (s *AppsSuite) TestATransfersTrunkHasAPasswordTheVendorIsGiven() {
	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550002222", CallID: "call-1",
	})
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	password, allowed := s.deployment.TrunkAuth(trunks[0])
	s.NotEmpty(password)
	s.Empty(allowed)
	s.Equal(password, s.vendor.dialed.Bridge.Password)
}

func (s *AppsSuite) TestATransferOnAVendorThatCannotSendAPasswordHasItsAddressesAndNoPassword() {
	vonage := s.vendorOnAnAllowlist()

	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: vonage.e164, To: "+15550002222", CallID: "call-1",
	})
	s.Require().NoError(err)

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	password, allowed := s.deployment.TrunkAuth(trunks[0])
	s.Equal(vonage.signalling, allowed, "without them the vendor's leg is challenged and cannot answer")
	s.Empty(password)
}

func (s *AppsSuite) TestATransferOnAVendorWithNoAddressesAndNoPasswordIsRefusedBeforeAnythingIsMade() {
	declared, ok := s.config().Lookup("sinch")
	s.Require().True(ok)
	s.Require().Empty(declared.Signalling)
	sinch := s.numberOn("sinch")

	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: sinch, To: "+15550002222", CallID: "call-1",
	})
	s.Require().ErrorContains(err, "signalling addresses")
	s.Empty(s.deployment.Trunks())
}

// sealed gives the suite's service a key to seal trunk passwords with, which lines need.
func (s *AppsSuite) sealed() {
	sealer, err := auth.NewSealer("apps suite")
	s.Require().NoError(err)
	s.service.sealer = sealer
}

// anotherNumber gives a customer one more number on a vendor.
func (s *AppsSuite) anotherNumber(customer, vendor string) string {
	e164 := aNumber()
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: e164, Vendor: vendor, Country: "US", CustomerID: customer, PurchasedAt: time.Now().UTC(),
	}))
	return e164
}

// lineFor puts a customer's number on a trunk in the app the customer makes lines in.
func (s *AppsSuite) lineFor(customer, e164 string) (TrunkLine, error) {
	held, err := s.store.Number(s.ctx, customer, e164)
	s.Require().NoError(err)
	stream, pin, err := s.service.streamFor(s.ctx, customer)
	s.Require().NoError(err)
	return s.service.line(s.ctx, stream, pin, held)
}

func (s *AppsSuite) TestTwoNumbersOfOneVendorShareATrunk() {
	s.ownApp()
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)

	first, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	other, err := s.lineFor(s.customer, second)
	s.Require().NoError(err)

	s.Equal([]string{first.TrunkID}, s.own.Trunks())
	s.Equal(first.TrunkID, other.TrunkID)
	s.Equal([]string{s.e164, second}, s.own.TrunkNumbers(first.TrunkID))
	s.Equal("sip:"+second+"@sip.example.test", other.Bridge.URI, "the address names the number")
	password, _ := s.own.TrunkAuth(first.TrunkID)
	s.Equal(password, other.Bridge.Password, "a number joining later is given the trunk's password")
}

func (s *AppsSuite) TestTheFiftyFirstNumberOpensASecondTrunk() {
	s.sealed()
	first, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	for range trunkCapacity - 1 {
		_, err := s.lineFor(s.customer, s.anotherNumber(s.customer, s.vendor.vendor))
		s.Require().NoError(err)
	}

	fiftyFirst, err := s.lineFor(s.customer, s.anotherNumber(s.customer, s.vendor.vendor))
	s.Require().NoError(err)

	s.Len(s.deployment.Trunks(), 2)
	s.NotEqual(first.TrunkID, fiftyFirst.TrunkID)
	s.Len(s.deployment.TrunkNumbers(first.TrunkID), trunkCapacity)
	s.Len(s.deployment.TrunkNumbers(fiftyFirst.TrunkID), 1)
}

func (s *AppsSuite) TestCustomersNeverShareATrunk() {
	s.sealed()
	other := "customer-" + uuid.NewString()
	theirs := s.anotherNumber(other, s.vendor.vendor)

	ours, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	their, err := s.lineFor(other, theirs)
	s.Require().NoError(err)

	s.NotEqual(ours.TrunkID, their.TrunkID)
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(ours.TrunkID))
	s.Equal([]string{theirs}, s.deployment.TrunkNumbers(their.TrunkID))
}

func (s *AppsSuite) TestVendorsNeverShareATrunk() {
	vonage := s.vendorOnAnAllowlist()
	s.sealed()

	twilio, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	other, err := s.lineFor(s.customer, vonage.e164)
	s.Require().NoError(err)

	s.NotEqual(twilio.TrunkID, other.TrunkID)
	_, allowed := s.deployment.TrunkAuth(other.TrunkID)
	s.Equal(vonage.signalling, allowed, "each vendor's trunk lets in that vendor")
}

func (s *AppsSuite) TestEachNumberHasARuleOfItsOwnThatReadsTheHeader() {
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	first, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	other, err := s.lineFor(s.customer, second)
	s.Require().NoError(err)

	s.NotEqual(first.RouteID, other.RouteID)
	for e164, routeID := range map[string]string{s.e164: first.RouteID, second: other.RouteID} {
		rule := s.deployment.Rule(routeID)
		s.Require().NotNil(rule)
		s.Equal(lineCallTemplate, rule["direct_routing_configs"].(map[string]any)["call_id"])
		s.Equal([]any{e164}, rule["called_numbers"])
		custom := rule["call_configs"].(map[string]any)["custom_data"].(map[string]any)
		s.Equal(e164, custom[NumberKey], "the call says which number it rang")
	}
	held := s.number()
	s.Equal(first.TrunkID, held.StreamTrunkID)
	s.Equal(first.RouteID, held.StreamRouteID)
}

func (s *AppsSuite) TestANumberAttachedBeforeSharedTrunksIsAdoptedAsItIs() {
	s.sealed()
	stream, pin, err := s.service.streamFor(s.ctx, s.customer)
	s.Require().NoError(err)
	trunkID, bridge, err := stream.CreateTrunk(s.ctx, Trunk{Name: "phone-" + s.e164, Numbers: []string{s.e164}})
	s.Require().NoError(err)
	routeID, err := stream.CreateRoute(s.ctx, Route{
		Name: "phone-" + s.e164, TrunkIDs: []string{trunkID}, CalledNumbers: []string{s.e164},
		CallID: "{{uuid}}", CallType: defaultCallType, Custom: map[string]string{NumberKey: s.e164},
	})
	s.Require().NoError(err)
	bridge, err = bridge.WithNumber(s.e164)
	s.Require().NoError(err)
	sealed, err := s.service.sealer.SealWithAAD(bridge.Password, bridgeAAD(s.customer, s.e164))
	s.Require().NoError(err)
	s.Require().NoError(s.store.AttachNumber(s.ctx, s.customer, s.e164, store.NumberAttachment{
		TrunkID: trunkID, RouteID: routeID, StreamAppPK: pin, CallType: defaultCallType,
		BridgeURI: bridge.URI, BridgeUsername: bridge.Username,
		BridgePasswordSealed: sealed, BridgePasswordKEKVersion: s.service.sealer.CurrentVersion(),
	}))

	line, err := s.lineFor(s.customer, s.e164)

	s.Require().NoError(err)
	s.Equal(trunkID, line.TrunkID)
	s.Equal(routeID, line.RouteID)
	s.Equal(bridge, line.Bridge)
	s.Equal([]string{trunkID}, s.deployment.Trunks(), "no new trunk")
	rule := s.deployment.Rule(routeID)
	direct := rule["direct_routing_configs"].(map[string]any)
	s.Equal(lineCallTemplate, direct["call_id"])
	s.Equal(defaultCallType, direct["call_type"])
	s.Equal([]any{s.e164}, rule["called_numbers"])
	s.Equal([]any{trunkID}, rule["trunk_ids"])
	custom := rule["call_configs"].(map[string]any)["custom_data"].(map[string]any)
	s.Equal(s.e164, custom[NumberKey], "the update keeps the number on the call")

	joined, err := s.lineFor(s.customer, s.anotherNumber(s.customer, s.vendor.vendor))
	s.Require().NoError(err)
	s.Equal(trunkID, joined.TrunkID, "the adopted trunk takes the customer's next number")
}

func (s *AppsSuite) TestTwoNumbersJoiningAtOnceMakeOneTrunk() {
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	var wait sync.WaitGroup
	for _, e164 := range []string{s.e164, second} {
		wait.Go(func() {
			_, err := s.lineFor(s.customer, e164)
			s.NoError(err)
		})
	}
	wait.Wait()

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	s.ElementsMatch([]string{s.e164, second}, s.deployment.TrunkNumbers(trunks[0]))
}

func (s *AppsSuite) TestARefusedRuleLeavesTheNumberOffTheTrunk() {
	s.sealed()
	first, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	s.deployment.RateLimit("/sip/inbound_routing_rules")

	_, err = s.lineFor(s.customer, second)

	s.Require().Error(err)
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(first.TrunkID))
	s.Equal([]string{first.RouteID}, s.deployment.Rules())
	held, err := s.store.Number(s.ctx, s.customer, second)
	s.Require().NoError(err)
	s.Empty(held.StreamTrunkID)
}

func (s *AppsSuite) TestARefusedRuleOnANewTrunkLeavesNoTrunk() {
	s.sealed()
	s.deployment.RateLimit("/sip/inbound_routing_rules")

	_, err := s.lineFor(s.customer, s.e164)

	s.Require().Error(err)
	s.Empty(s.deployment.Trunks())
	s.Empty(s.number().StreamTrunkID)
}

func (s *AppsSuite) TestALineNeedsAKeyToSealTheTrunksPassword() {
	_, err := s.lineFor(s.customer, s.e164)

	s.ErrorIs(err, ErrNoSealer)
	s.Empty(s.deployment.Trunks(), "nothing is made in Stream")
}

func (s *AppsSuite) TestOneNumberLinedTwiceAtOnceIsOnItsTrunkOnce() {
	s.ownApp()
	s.sealed()
	var wait sync.WaitGroup
	for range 2 {
		wait.Go(func() {
			_, err := s.lineFor(s.customer, s.e164)
			s.NoError(err)
		})
	}
	wait.Wait()

	trunks := s.own.Trunks()
	s.Require().Len(trunks, 1)
	s.Equal([]string{s.e164}, s.own.TrunkNumbers(trunks[0]))
	s.Len(s.own.Rules(), 1)
}

// callFrom places a call from one of the suite customer's numbers.
func (s *AppsSuite) callFrom(e164 string) (Placed, error) {
	return s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: e164, To: "+15550001111",
	})
}

func (s *AppsSuite) TestTwoCallsFromOneNumberAtOncePlaceBoth() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()

	_, first := s.callFrom(s.e164)
	_, second := s.callFrom(s.e164)

	s.Require().NoError(first)
	s.Require().NoError(second, "Stream lets a number be on one trunk only")
	s.Len(s.own.Trunks(), 1)
}

func (s *AppsSuite) TestACallFromAnAttachedNumberPlaces() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)

	_, err = s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
}

func (s *AppsSuite) TestSixtyNumbersPlacingCallsStayUnderTheTrunkCap() {
	s.ownApp()
	s.sealed()
	s.own.CapTrunks(25)
	numbers := []string{s.e164}
	for range 59 {
		numbers = append(numbers, s.anotherNumber(s.customer, s.vendor.vendor))
	}

	for _, e164 := range numbers {
		_, err := s.callFrom(e164)
		s.Require().NoError(err)
	}

	s.Len(s.own.Trunks(), 2)
}

func (s *AppsSuite) TestACallsCustomIsOnItsStreamCall() {
	s.ownApp()
	s.sealed()
	custom := map[string]string{"lead": "42"}

	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111", Custom: custom,
	})

	s.Require().NoError(err)
	made := s.own.Call(placed.CallType, placed.CallID)
	s.Require().NotNil(made, "the call the leg lands in is made before it is dialled")
	data, ok := made["data"].(map[string]any)
	s.Require().True(ok, "the call was created with data")
	s.Equal(map[string]any{"lead": "42"}, data["custom"])
}

func (s *AppsSuite) TestTheVendorIsGivenTheSessionAsTheCallID() {
	s.sealed()

	placed, err := s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Equal(placed.SessionID, s.vendor.dialed.CallID)
}

func (s *AppsSuite) TestACallOnAVendorWithoutTheHeaderGetsATrunkOfItsOwn() {
	plivo := s.numberOn("plivo")
	s.sealed()

	placed, err := s.callFrom(plivo)

	s.Require().NoError(err)
	s.Len(s.deployment.Trunks(), 1)
	s.Equal([]string{placed.CallID}, s.pinnedCalls(), "the rule names the call, as the leg cannot")
}

func (s *AppsSuite) TestWithoutAKeyACallStillPlacesOnATrunkOfItsOwn() {
	placed, err := s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Len(s.deployment.Trunks(), 1)
	s.Equal([]string{placed.CallID}, s.pinnedCalls())
	s.Empty(s.vendor.dialed.CallID)
}

// pinnedCalls are the calls the deployment app's rules send their legs into.
func (s *AppsSuite) pinnedCalls() []string {
	var calls []string
	for _, id := range s.deployment.Rules() {
		calls = append(calls, s.deployment.Rule(id)["direct_routing_configs"].(map[string]any)["call_id"].(string))
	}
	return calls
}

func (s *AppsSuite) TestAFailedDialLeavesNoCallResource() {
	s.sealed()
	s.vendor.err = errors.New("the vendor refused")
	session := uuid.NewString()

	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550001111", SessionID: session,
	})

	s.Require().Error(err)
	resources, err := s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, defaultCallType, session)
	s.Require().NoError(err)
	s.Empty(resources)
}

func (s *AppsSuite) TestEndingASharedCallKeepsItsTrunk() {
	s.sealed()
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)

	s.Require().NoError(s.service.ReleaseCall(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID))

	s.Len(s.deployment.Trunks(), 1, "the number's other calls use it")
	s.Len(s.deployment.Rules(), 1)
}

func (s *AppsSuite) TestATransferFromANumberWithALiveCallPlaces() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)

	_, err = s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550002222",
		CallID: placed.CallID, CallType: placed.CallType, StreamApp: placed.StreamApp,
	})

	s.Require().NoError(err)
	s.Len(s.own.Trunks(), 1)
	s.Equal(placed.CallID, s.vendor.dialed.CallID, "the human joins the call the agent is on")
}

func (s *AppsSuite) TestAFailedTransferFromTheCallsOwnNumberKeepsTheCallsRow() {
	s.sealed()
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.vendor.err = errors.New("the vendor refused")

	_, err = s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15550002222",
		CallID: placed.CallID, CallType: placed.CallType, StreamApp: placed.StreamApp,
	})

	s.Require().Error(err)
	resources, err := s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Len(resources, 1, "the call's own leg is still live")
}

func (s *AppsSuite) TestAttachingANumberThatAlreadyPlacedCallsKeepsItsTrunk() {
	s.ownApp()
	s.sealed()
	s.own.OneTrunkPerNumber()
	_, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	lined := s.number()

	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(lined.StreamTrunkID, attached.TrunkID)
	s.Equal(lined.StreamRouteID, attached.RouteID)
	s.Equal([]string{attached.TrunkID}, s.own.Trunks())
	s.Equal([]string{attached.RouteID}, s.own.Rules())
	s.Equal(s.e164, s.vendor.inbound.E164, "the vendor is told to send calls to the line")
	s.NotNil(s.number().AttachedAt)
}

func (s *AppsSuite) TestReleasingOneOfTwoNumbersKeepsTheTrunkForTheOther() {
	s.ownApp()
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	other, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: second})
	s.Require().NoError(err)
	s.Require().Equal(first.TrunkID, other.TrunkID)

	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Equal([]string{first.TrunkID}, s.own.Trunks())
	s.Equal([]string{second}, s.own.TrunkNumbers(first.TrunkID))
	s.Equal([]string{other.RouteID}, s.own.Rules())
}

func (s *AppsSuite) TestReleasingTheLastNumberDeletesTheTrunk() {
	s.sealed()
	_, err := s.callFrom(s.e164)
	s.Require().NoError(err)

	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Empty(s.deployment.Trunks())
	s.Empty(s.deployment.Rules())
	_, err = s.callFrom(s.anotherNumber(s.customer, s.vendor.vendor))
	s.Require().NoError(err, "the next number does not join the deleted trunk")
	s.Len(s.deployment.Trunks(), 1)
}

func (s *AppsSuite) TestATrunkStreamWouldNotDeleteIsKeptForTheReleaseToBeRetried() {
	s.sealed()
	_, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	trunk := s.number().StreamTrunkID
	s.deployment.RateLimit("/sip/inbound_trunks/" + trunk)

	s.Require().Error(s.service.Release(s.ctx, s.customer, s.e164))

	s.Equal([]string{trunk}, s.deployment.Trunks())
	s.Equal(trunk, s.number().StreamTrunkID, "the trunk and its password are still known here")
	s.Empty(s.vendor.released, "the number is not given back while its trunk lives on")

	s.deployment.LiftRateLimit("/sip/inbound_trunks/" + trunk)
	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Empty(s.deployment.Trunks())
	s.Equal([]string{s.e164}, s.vendor.released)
}

func (s *AppsSuite) TestANumberWhoseTrunkIsAlreadyGoneAtStreamIsReleased() {
	s.sealed()
	_, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	held := s.number()
	stream, err := s.service.streamToRemove(s.ctx, s.customer, held.StreamAppPK)
	s.Require().NoError(err)
	_, err = stream.Client().Video().DeleteSIPTrunk(s.ctx, held.StreamTrunkID, nil)
	s.Require().NoError(err)

	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	s.Equal([]string{s.e164}, s.vendor.released)
	group := store.TrunkGroup{CustomerID: s.customer, StreamAppPK: held.StreamAppPK, Vendor: held.Vendor}
	s.Require().NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *store.TrunkGroupTx) error {
		trunks, err := tx.Trunks()
		s.Empty(trunks)
		return err
	}))
}

func (s *AppsSuite) TestAFreedPlaceOnAFullTrunkIsTakenBeforeASiblingIsOpened() {
	s.sealed()
	first, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)
	for range trunkCapacity - 1 {
		_, err := s.lineFor(s.customer, s.anotherNumber(s.customer, s.vendor.vendor))
		s.Require().NoError(err)
	}
	s.Require().NoError(s.service.Release(s.ctx, s.customer, s.e164))

	added, err := s.lineFor(s.customer, s.anotherNumber(s.customer, s.vendor.vendor))

	s.Require().NoError(err)
	s.Equal([]string{first.TrunkID}, s.deployment.Trunks())
	s.Equal(first.TrunkID, added.TrunkID)
	s.Len(s.deployment.TrunkNumbers(first.TrunkID), trunkCapacity)
}

func (s *AppsSuite) TestReattachingAfterRegistrationTakesTheNumberOffItsSharedTrunk() {
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	other, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: second})
	s.Require().NoError(err)
	s.ownApp()

	moved, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal([]string{first.TrunkID}, s.deployment.Trunks())
	s.Equal([]string{second}, s.deployment.TrunkNumbers(first.TrunkID), "the other number keeps its trunk")
	s.Equal([]string{other.RouteID}, s.deployment.Rules())
	s.Equal([]string{moved.TrunkID}, s.own.Trunks())
	s.Equal([]string{s.e164}, s.own.TrunkNumbers(moved.TrunkID))
	s.Equal(int64(4242), s.number().StreamAppPK)
}

func (s *AppsSuite) TestReleasingANumberLinedSinceItWasReadTakesItOffItsTrunk() {
	// Release reads the number before it takes the lock: a call lining it in the
	// customer's new app meanwhile must not leave a released number on that trunk.
	stale := s.number()
	s.ownApp()
	s.sealed()
	second := s.anotherNumber(s.customer, s.vendor.vendor)
	other, err := s.lineFor(s.customer, second)
	s.Require().NoError(err)
	lined, err := s.lineFor(s.customer, s.e164)
	s.Require().NoError(err)

	s.Require().NoError(s.service.unline(s.ctx, stale, "released a number", true))

	s.Equal([]string{second}, s.own.TrunkNumbers(lined.TrunkID))
	s.Equal([]string{other.RouteID}, s.own.Rules())
	_, err = s.store.Number(s.ctx, s.customer, s.e164)
	s.ErrorIs(err, store.ErrNoNumber, "the number is released")
}

func (s *AppsSuite) TestAttachingANumberWhoseTrunkPasswordWasNotKeptGivesItATrunkOfItsOwn() {
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	_, err = s.store.DB().NewUpdate().Table("phone_numbers").Set("attached_at = NULL").
		Where("customer_id = ?", s.customer).Where("e164 = ?", s.e164).Exec(s.ctx)
	s.Require().NoError(err)
	s.sealed()

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.NotEqual(first.TrunkID, again.TrunkID)
	s.Equal([]string{again.TrunkID}, s.deployment.Trunks(), "the old trunk is deleted")
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(again.TrunkID))
	s.Equal([]string{again.RouteID}, s.deployment.Rules())
	s.NotEmpty(s.number().BridgePasswordSealed, "its new trunk's password is kept")
}

func (s *AppsSuite) TestAttachingANumberOnASharedTrunkAgainChangesNothing() {
	s.sealed()
	first, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.vendor.inbound = Inbound{}
	made := len(s.deployment.Requests("test"))
	s.Require().NotZero(made)

	again, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.Equal(first.TrunkID, again.TrunkID)
	s.Equal(first.RouteID, again.RouteID)
	s.Len(s.deployment.Requests("test"), made, "nothing is asked of Stream")
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(first.TrunkID))
	s.Empty(s.vendor.inbound.E164, "the vendor is not told anything new")
}

// backdate moves a call's rows back in time, as a call that has been up for hours has.
func (s *AppsSuite) backdate(callID string, age time.Duration) {
	_, err := s.store.DB().NewUpdate().Table("call_resources").
		Set("created_at = ?", time.Now().UTC().Add(-age)).
		Where("call_id = ?", callID).Exec(s.ctx)
	s.Require().NoError(err)
}

func (s *AppsSuite) TestAPerCallTrunkWhoseCallNeverEndedIsSweptAfterSixHours() {
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.Require().Len(s.deployment.Trunks(), 1)
	s.backdate(placed.CallID, 7*time.Hour)

	swept, err := s.service.SweepCallResources(s.ctx, time.Now().Add(-6*time.Hour))

	s.Require().NoError(err)
	s.Positive(swept)
	s.Empty(s.deployment.Trunks())
	s.Empty(s.deployment.Rules())
}

func (s *AppsSuite) TestAPerCallTrunkStreamWouldNotDeleteIsSweptAgainNextTime() {
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.Require().Len(s.deployment.Trunks(), 1)
	trunk := s.deployment.Trunks()[0]
	s.backdate(placed.CallID, 7*time.Hour)
	s.deployment.RateLimit("/sip/inbound_trunks/" + trunk)

	_, err = s.service.SweepCallResources(s.ctx, time.Now().Add(-6*time.Hour))

	s.Require().NoError(err)
	s.Equal([]string{trunk}, s.deployment.Trunks())
	rows, err := s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.NotEmpty(rows, "the row is what the next sweep finds the trunk by")

	s.deployment.LiftRateLimit("/sip/inbound_trunks/" + trunk)
	_, err = s.service.SweepCallResources(s.ctx, time.Now().Add(-6*time.Hour))

	s.Require().NoError(err)
	s.Empty(s.deployment.Trunks())
	rows, err = s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Empty(rows)
}

func (s *AppsSuite) TestAPerCallTrunkOfARecentCallIsNotSwept() {
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.backdate(placed.CallID, time.Hour)

	_, err = s.service.SweepCallResources(s.ctx, time.Now().Add(-6*time.Hour))

	s.Require().NoError(err)
	s.Len(s.deployment.Trunks(), 1)
}

func (s *AppsSuite) TestASharedCallWhoseCallNeverEndedLosesItsRowButNotItsLine() {
	s.sealed()
	placed, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.backdate(placed.CallID, 7*time.Hour)

	swept, err := s.service.SweepCallResources(s.ctx, time.Now().Add(-6*time.Hour))

	s.Require().NoError(err)
	s.Positive(swept)
	s.Len(s.deployment.Trunks(), 1, "the number's other calls use it")
	s.Len(s.deployment.Rules(), 1)
	rows, err := s.store.CallResourcesInApp(s.ctx, store.AppScope{App: 1, Unpinned: true}, placed.CallType, placed.CallID)
	s.Require().NoError(err)
	s.Empty(rows)
}

func (s *AppsSuite) TestMoreNumbersLiningAtOnceThanThePoolHasConnectionsAllLine() {
	s.sealed()
	s.store.DB().SetMaxOpenConns(3)
	s.T().Cleanup(func() {
		s.store.DB().SetMaxOpenConns(40)
		s.store.DB().SetMaxIdleConns(40)
	})
	numbers := []string{s.e164}
	for range 9 {
		numbers = append(numbers, s.anotherNumber(s.customer, s.vendor.vendor))
	}
	// A pool held by waiters deadlocks; the deadline turns that into a failure.
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	stream, pin, err := s.service.streamFor(ctx, s.customer)
	s.Require().NoError(err)

	var wait sync.WaitGroup
	for _, e164 := range numbers {
		wait.Go(func() {
			held, err := s.store.Number(ctx, s.customer, e164)
			if !s.NoError(err) {
				return
			}
			_, err = s.service.line(ctx, stream, pin, held)
			s.NoError(err)
		})
	}
	wait.Wait()

	trunks := s.deployment.Trunks()
	s.Require().Len(trunks, 1)
	s.ElementsMatch(numbers, s.deployment.TrunkNumbers(trunks[0]))
}

func (s *AppsSuite) TestANumberMarkedAttachedWithNoTrunkIsAttachedAgain() {
	s.sealed()
	_, err := s.store.DB().NewUpdate().Table("phone_numbers").Set("attached_at = ?", time.Now().UTC()).
		Where("customer_id = ?", s.customer).Where("e164 = ?", s.e164).Exec(s.ctx)
	s.Require().NoError(err)

	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})

	s.Require().NoError(err)
	s.NotEmpty(attached.TrunkID)
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(attached.TrunkID))
	s.Equal(attached.TrunkID, s.number().StreamTrunkID)
	s.Equal(s.e164, s.vendor.inbound.E164, "the vendor is told to send calls to the line")
}

func (s *AppsSuite) TestACallFromANumberAttachedInAnotherAppGoesOutOnATrunkOfItsOwn() {
	s.sealed()
	attached, err := s.service.Attach(s.ctx, Attachment{CustomerID: s.customer, E164: s.e164})
	s.Require().NoError(err)
	s.ownApp()
	s.own.OneTrunkPerNumber()

	placed, err := s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Equal([]string{s.e164}, s.deployment.TrunkNumbers(attached.TrunkID), "its callers still reach it")
	held := s.number()
	s.Equal(attached.TrunkID, held.StreamTrunkID)
	s.Zero(held.StreamAppPK)
	rules := s.own.Rules()
	s.Require().Len(rules, 1)
	s.Equal(placed.CallID, s.own.Rule(rules[0])["direct_routing_configs"].(map[string]any)["call_id"],
		"the call's own rule is pinned to it")
}

func (s *AppsSuite) TestACallFromANumberLinedInAnotherAppMovesItsLine() {
	s.sealed()
	_, err := s.callFrom(s.e164)
	s.Require().NoError(err)
	s.ownApp()

	_, err = s.callFrom(s.e164)

	s.Require().NoError(err)
	s.Empty(s.deployment.Trunks(), "the old app's trunk no longer lists it")
	s.Empty(s.deployment.Rules())
	trunks := s.own.Trunks()
	s.Require().Len(trunks, 1)
	s.Equal([]string{s.e164}, s.own.TrunkNumbers(trunks[0]))
	s.Equal(int64(4242), s.number().StreamAppPK)
}
