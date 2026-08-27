package phone

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type PhoneSuite struct {
	suite.Suite
	ctx context.Context
}

func TestPhone(t *testing.T) { suite.Run(t, new(PhoneSuite)) }

func (s *PhoneSuite) SetupTest() { s.ctx = context.Background() }

func (s *PhoneSuite) TestTheBuiltInVendorListDeclaresEveryVendorProperly() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	s.Len(config.Vendors, 11, "the sprint names eleven vendors")
	for _, vendor := range config.Vendors {
		s.NotEmpty(vendor.Capabilities, vendor.Vendor+" declares no capabilities")
		s.NotEmpty(vendor.Credentials, vendor.Vendor+" declares no credentials")
	}
}

func (s *PhoneSuite) TestEightOfElevenVendorsAreImplemented() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	var implemented []string
	for _, vendor := range config.Vendors {
		if vendor.Implemented {
			implemented = append(implemented, vendor.Vendor)
		}
	}

	s.ElementsMatch([]string{
		"twilio", "telnyx", "sinch", "bandwidth", "vonage", "bird", "didww", "plivo",
	}, implemented)
}

func (s *PhoneSuite) TestEveryImplementedVendorButDidwwCanPlaceACall() {
	// DIDWW sells numbers and has no call control API at all, so there is nothing to ask
	// it to dial with. A caller can see that before buying a number from it rather than
	// after. Attaching is still the narrower list, being inbound rather than outbound.
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for _, vendor := range config.Vendors {
		if !vendor.Implemented {
			s.Empty(vendor.Operations, vendor.Vendor+" claims operations it cannot do")
			continue
		}
		s.True(vendor.Does(OpSearch), vendor.Vendor+" cannot search")
		s.True(vendor.Does(OpBuy), vendor.Vendor+" cannot buy")
		s.True(vendor.Does(OpRelease), vendor.Vendor+" cannot release")

		s.Equal(vendor.Vendor != "didww", vendor.Does(OpDial),
			vendor.Vendor+" disagrees about dialling")

		attaches := vendor.Vendor == "twilio" || vendor.Vendor == "telnyx"
		s.Equal(attaches, vendor.Does(OpAttach), vendor.Vendor+" disagrees about attaching")
	}
}

func (s *PhoneSuite) TestEveryVendorThatCanDialSaysHowItGetsPastTheTrunk() {
	// A trunk with no password and no allowlist is open to anyone who learns its uri, so
	// which of the two a vendor uses is not something to leave undeclared.
	config, err := DefaultConfig()
	s.Require().NoError(err)

	byAddress := map[string]bool{
		"vonage": true, "bird": true, "sinch": true, "bandwidth": true,
	}
	for _, vendor := range config.Vendors {
		if !vendor.Does(OpDial) {
			continue
		}
		want := TrunkPassword
		if byAddress[vendor.Vendor] {
			want = TrunkAllowlist
		}
		s.Equal(want, vendor.Authenticates(), vendor.Vendor+" disagrees about trunk auth")
	}
}

func (s *PhoneSuite) TestAVendorAuthenticatingByAnUnknownMeansIsRejected() {
	_, err := parseConfig([]byte(`
vendors:
  - vendor: twilio
    capabilities: [voice]
    credentials: [TWILIO_ACCOUNT_SID]
    trunk_auth: vibes
`))

	s.ErrorContains(err, "neither")
}

func (s *PhoneSuite) TestAnImplementedVendorMustSayWhatItCanDo() {
	_, err := parseConfig([]byte(`
vendors:
  - vendor: twilio
    implemented: true
    capabilities: [voice]
    credentials: [TWILIO_ACCOUNT_SID]
`))

	s.ErrorContains(err, "declares no operations")
}

func (s *PhoneSuite) TestAVendorDeclaredTwiceIsRejected() {
	_, err := parseConfig([]byte(`
vendors:
  - vendor: twilio
    capabilities: [voice]
    credentials: [TWILIO_ACCOUNT_SID]
  - vendor: twilio
    capabilities: [voice]
    credentials: [TWILIO_ACCOUNT_SID]
`))

	s.ErrorContains(err, "declared twice")
}

func (s *PhoneSuite) TestAnUnknownVendorCannotBeOpened() {
	registry := NewRegistry(s.config())

	_, err := registry.Open("carrier-pigeon")

	s.ErrorContains(err, "not a known vendor")
}

func (s *PhoneSuite) TestADeclaredButUnimplementedVendorRefusesEveryOperation() {
	registry := NewRegistry(s.config())

	provider, err := registry.Open("sinch")
	s.Require().NoError(err, "an unimplemented vendor still resolves")

	s.Equal("sinch", provider.Vendor())
	_, err = provider.SearchNumbers(s.ctx, Search{Country: "US"})
	s.ErrorIs(err, ErrNotImplemented)
	s.ErrorContains(err, "sinch", "the error names the vendor that did nothing")

	_, err = provider.BuyNumber(s.ctx, Order{E164: "+15551234567"})
	s.ErrorIs(err, ErrNotImplemented)
	s.ErrorIs(provider.ReleaseNumber(s.ctx, "+15551234567"), ErrNotImplemented)
	s.ErrorIs(provider.ConfigureInbound(s.ctx, Inbound{}), ErrNotImplemented)
	_, err = provider.Dial(s.ctx, Outbound{})
	s.ErrorIs(err, ErrNotImplemented)
	s.ErrorIs(provider.SendDigits(s.ctx, "call-1", "1"), ErrNotImplemented)
}

func (s *PhoneSuite) TestOnlyWhatAKeypadHasCanBePressed() {
	s.NoError(ValidateDigits("1"))
	s.NoError(ValidateDigits("4123"))
	s.NoError(ValidateDigits("*0#"))
	s.NoError(ValidateDigits("1w2"), "a menu that asks for an extension after a beep needs a pause")
	s.NoError(ValidateDigits("1W2"), "the longer pause is spelt with a capital everywhere")
	s.NoError(ValidateDigits("ABCD"), "no handset has the fourth column but every vendor's api takes it")

	s.ErrorContains(ValidateDigits(""), "digits to press")
	s.ErrorContains(ValidateDigits("one"), "keypad can press")
	s.ErrorContains(ValidateDigits("press 1"), "keypad can press")
	s.ErrorContains(ValidateDigits("1E2"), "keypad can press", "the fourth column stops at D")
}

func (s *PhoneSuite) TestACallNamesOnlyTheTermsItWasActuallyPlacedOn() {
	bare := Outbound{From: "+15125551234", To: "+15550001111"}
	s.Empty(bare.Features(), "a call with no terms asks nothing of a vendor")

	full := Outbound{
		RingTimeout:   20 * time.Second,
		InitialDigits: "1234#",
		Headers:       map[string]string{"X-Ticket": "42"},
	}
	s.ElementsMatch(
		[]CallFeature{FeatureRingTimeout, FeatureInitialDigits, FeatureCustomHeaders},
		full.Features(),
	)
}

func (s *PhoneSuite) TestACallSaysWhichOfItsTermsAVendorCannotExpress() {
	call := Outbound{RingTimeout: 20 * time.Second, Headers: map[string]string{"X-Ticket": "42"}}
	provider := &stub{vendor: "twilio", undialable: []CallFeature{FeatureCustomHeaders}}

	s.Equal([]CallFeature{FeatureCustomHeaders}, call.Unsupported(provider))
	s.Empty(Outbound{RingTimeout: time.Second}.Unsupported(provider))
}

func (s *PhoneSuite) TestACallThatCannotBePlacedAsDescribedIsRefused() {
	bridge := Bridge{URI: "sip:trunk@sip.stream-io-api.com"}
	valid := Outbound{From: "+15125551234", To: "+15550001111", Bridge: bridge}
	s.NoError(valid.Validate())

	missing := valid
	missing.To = ""
	s.ErrorContains(missing.Validate(), "from and a to")

	unbridged := valid
	unbridged.Bridge = Bridge{}
	s.ErrorContains(unbridged.Validate(), "uri is required")

	backwards := valid
	backwards.RingTimeout = -time.Second
	s.ErrorContains(backwards.Validate(), "less than no time")

	words := valid
	words.InitialDigits = "extension four"
	s.ErrorContains(words.Validate(), "keypad can press")

	// Twilio stops at 32, so more than that is a call that would be placed with the
	// digits silently truncated at whichever vendor happened to answer.
	long := valid
	long.InitialDigits = strings.Repeat("1", maxInitialDigits+1)
	s.ErrorContains(long.Validate(), "than the 32")
}

func (s *PhoneSuite) TestALineThatWasNotPlacedFromHereCannotPressAnything() {
	// Pressing needs the vendor's own id for the leg, and an inbound call arrived at
	// Stream rather than being dialled from here, so there is no such id to name.
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)

	line := service.Line(LineOptions{From: "+15125551234", CallID: "support-line"})

	s.ErrorContains(line.SendDigits(s.ctx, "1"), "not placed from here")
}

func (s *PhoneSuite) TestACallSaysWhatItIsMissing() {
	// A call is a trunk and a routing rule as well as a leg at the vendor, so it needs
	// Stream even when the caller supplied everything else.
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)

	_, err = service.Call(s.ctx, CallRequest{From: "+15125551234", To: "+15550001111"})
	s.ErrorContains(err, "stream credentials")
}

func (s *PhoneSuite) TestAVendorThatCannotSendAPasswordWillNotDialWithoutAnAllowlist() {
	// Stream reads an empty allowlist as "accept everything" rather than "password only",
	// so a vendor that can send no password and has no declared addresses would be given
	// an open trunk. Refusing is the only safe answer.
	allowed, err := trunkAllowlist(Vendor{
		Vendor:     "vonage",
		TrunkAuth:  TrunkAllowlist,
		Signalling: []string{"216.147.0.0/18"},
	})
	s.Require().NoError(err)
	s.Equal([]string{"216.147.0.0/18"}, allowed)

	_, err = trunkAllowlist(Vendor{Vendor: "bandwidth", TrunkAuth: TrunkAllowlist})
	s.ErrorContains(err, "none are declared")

	// A vendor that can send the password needs no allowlist, and gets none: adding one
	// would refuse its calls from anywhere the list did not happen to mention.
	allowed, err = trunkAllowlist(Vendor{Vendor: "twilio", TrunkAuth: TrunkPassword})
	s.Require().NoError(err)
	s.Empty(allowed)
}

func (s *PhoneSuite) TestATransferSaysWhatItIsMissing() {
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)

	_, err = service.Transfer(s.ctx, TransferRequest{From: "+15125551234", To: "+15550001111"})
	s.ErrorContains(err, "stream credentials",
		"a transfer routes a leg into a call, which needs Stream")
}

func (s *PhoneSuite) TestAVendorIsOpenedOnceAndReused() {
	registry := NewRegistry(s.config())
	opened := 0
	registry.Register("twilio", func() (Provider, error) {
		opened++
		return NotImplemented("twilio"), nil
	})
	s.T().Setenv("TWILIO_ACCOUNT_SID", "sid")
	s.T().Setenv("TWILIO_AUTH_TOKEN", "token")

	first, err := registry.Open("twilio")
	s.Require().NoError(err)
	second, err := registry.Open("twilio")
	s.Require().NoError(err)

	s.Same(first, second, "a provider is a client, not a value")
	s.Equal(1, opened)
}

func (s *PhoneSuite) TestAVendorWithoutItsCredentialsSaysWhichAreMissing() {
	registry := NewRegistry(s.config())
	registry.Register("twilio", func() (Provider, error) { return NotImplemented("twilio"), nil })
	s.T().Setenv("TWILIO_ACCOUNT_SID", "sid")
	s.T().Setenv("TWILIO_AUTH_TOKEN", "")

	_, err := registry.Open("twilio")

	s.ErrorContains(err, "TWILIO_AUTH_TOKEN")
}

func (s *PhoneSuite) TestOnlyVendorsThatCanActuallyBeUsedAreAvailable() {
	registry := NewRegistry(s.config())
	registry.Register("twilio", func() (Provider, error) { return NotImplemented("twilio"), nil })
	registry.Register("telnyx", func() (Provider, error) { return NotImplemented("telnyx"), nil })
	s.T().Setenv("TWILIO_ACCOUNT_SID", "sid")
	s.T().Setenv("TWILIO_AUTH_TOKEN", "token")
	s.T().Setenv("TELNYX_API_KEY", "")

	s.Equal([]string{"twilio"}, registry.Available(), "telnyx has no key, so it cannot be used")
}

func (s *PhoneSuite) TestABridgeMustBeASipAddress() {
	s.ErrorContains(Bridge{}.Validate(), "uri is required")
	s.ErrorContains(Bridge{URI: "https://example.com"}.Validate(), "not a sip uri")
	s.NoError(Bridge{URI: "sip:trunk@sip.stream-io-api.com"}.Validate())
}

func (s *PhoneSuite) TestCapabilitiesAreCoveredOnlyWhenEveryOneIsPresent() {
	s.True(covers([]Capability{Voice, SMS}, []Capability{Voice}))
	s.True(covers([]Capability{Voice}, nil))
	s.False(covers([]Capability{Voice}, []Capability{Voice, MMS}))
}

func (s *PhoneSuite) TestOnlyTheFiltersUsedHaveToBeSupported() {
	search := Search{Country: "US", AdministrativeArea: "CO", Capabilities: []Capability{Voice}}

	s.ElementsMatch([]Filter{FilterCountry, FilterAdministrativeArea}, search.Filters(),
		"capabilities are checked on the results, not asked of the vendor")
}

func (s *PhoneSuite) TestSearchingEveryVendorMergesWhatTheyOfferCheapestFirst() {
	service := s.serviceWith(
		&stub{vendor: "twilio", offers: []Available{
			{E164: "+15125551234", Vendor: "twilio", MonthlyCostMicros: 2_000_000},
		}},
		&stub{vendor: "telnyx", offers: []Available{
			{E164: "+17195551234", Vendor: "telnyx", MonthlyCostMicros: 1_000_000},
		}},
	)

	offers, err := service.SearchAll(s.ctx, Search{Country: "US"})
	s.Require().NoError(err)

	s.Empty(offers.Skipped)
	s.Equal([]string{"+17195551234", "+15125551234"}, numbers(offers))
}

func (s *PhoneSuite) TestANumberWithNoQuotedPriceIsNotSortedAsIfItWereFree() {
	service := s.serviceWith(
		&stub{vendor: "twilio", offers: []Available{{E164: "+15125551234", Vendor: "twilio"}}},
		&stub{vendor: "telnyx", offers: []Available{
			{E164: "+17195551234", Vendor: "telnyx", MonthlyCostMicros: 1_000_000},
		}},
	)

	offers, err := service.SearchAll(s.ctx, Search{Country: "US"})
	s.Require().NoError(err)

	s.Equal([]string{"+17195551234", "+15125551234"}, numbers(offers),
		"twilio does not quote a price on a search, which is not the same as free")
}

func (s *PhoneSuite) TestAVendorThatCannotExpressAFilterIsSkippedRatherThanAsked() {
	// Dropping the filter would answer a search for Colorado with numbers from anywhere,
	// which reads as a result.
	elsewhere := &stub{vendor: "twilio", unsupported: []Filter{FilterAdministrativeArea}, offers: []Available{
		{E164: "+15125551234", Vendor: "twilio"},
	}}
	service := s.serviceWith(elsewhere, &stub{vendor: "telnyx", offers: []Available{
		{E164: "+17195551234", Vendor: "telnyx"},
	}})

	offers, err := service.SearchAll(s.ctx, Search{Country: "US", AdministrativeArea: "CO"})
	s.Require().NoError(err)

	s.Equal([]string{"+17195551234"}, numbers(offers))
	s.Require().Len(offers.Skipped, 1)
	s.Equal("twilio", offers.Skipped[0].Vendor)
	s.Contains(offers.Skipped[0].Reason, "administrative_area")
	s.Zero(elsewhere.searched, "a vendor that cannot answer is not asked")
}

func (s *PhoneSuite) TestANumberMissingACapabilityIsDroppedFromTheAnswer() {
	// Not every vendor can filter by capability, but all of them say what a number
	// carries, so the answer is checked here instead.
	service := s.serviceWith(&stub{vendor: "twilio", offers: []Available{
		{E164: "+15125551234", Vendor: "twilio", Capabilities: []Capability{Voice}},
		{E164: "+15125555678", Vendor: "twilio", Capabilities: []Capability{Voice, HDVoice}},
	}})

	offers, err := service.SearchAll(s.ctx, Search{
		Country:      "US",
		Capabilities: []Capability{Voice, HDVoice},
	})
	s.Require().NoError(err)

	s.Equal([]string{"+15125555678"}, numbers(offers))
}

func (s *PhoneSuite) TestAVendorThatFailsIsReportedRatherThanFailingTheWholeSearch() {
	service := s.serviceWith(
		&stub{vendor: "twilio", err: errors.New("twilio is having a day")},
		&stub{vendor: "telnyx", offers: []Available{{E164: "+17195551234", Vendor: "telnyx"}}},
	)

	offers, err := service.SearchAll(s.ctx, Search{Country: "US"})
	s.Require().NoError(err)

	s.Equal([]string{"+17195551234"}, numbers(offers))
	s.Require().Len(offers.Skipped, 1)
	s.Equal("twilio", offers.Skipped[0].Vendor)
	s.Contains(offers.Skipped[0].Reason, "having a day")
}

func (s *PhoneSuite) TestSearchingWithNoUsableVendorSaysSoRatherThanAnsweringEmpty() {
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)

	_, err = service.SearchAll(s.ctx, Search{Country: "US"})

	s.ErrorContains(err, "no vendor has the credentials")
}

// serviceWith returns a service whose registry resolves each stub under its own vendor name.
func (s *PhoneSuite) serviceWith(stubs ...*stub) *Service {
	registry := NewRegistry(s.config())
	for _, each := range stubs {
		provider := each
		registry.Register(provider.vendor, func() (Provider, error) { return provider, nil })
		for _, name := range s.credentials(provider.vendor) {
			s.T().Setenv(name, "set")
		}
	}

	service, err := NewService(ServiceOptions{Registry: registry})
	s.Require().NoError(err)
	return service
}

// credentials are what a vendor needs before the registry will consider it usable.
func (s *PhoneSuite) credentials(vendor string) []string {
	declared, ok := s.config().Lookup(vendor)
	s.Require().True(ok, vendor+" is not a declared vendor")
	return declared.Credentials
}

func numbers(offers Offers) []string {
	rendered := make([]string, 0, len(offers.Numbers))
	for _, number := range offers.Numbers {
		rendered = append(rendered, number.E164)
	}
	return rendered
}

// stub is a vendor with fixed inventory, which is what makes the fan-out's own behaviour
// observable without eleven live accounts.
type stub struct {
	vendor      string
	offers      []Available
	err         error
	unsupported []Filter
	// undialable are the call features this vendor cannot express.
	undialable []CallFeature
	// searched counts the times this vendor was actually asked, so a skip can be told
	// apart from an empty answer.
	searched int
	// dialed is the last call this vendor was asked to place, so a test can see what the
	// service passed on rather than only that it passed something.
	dialed Outbound
}

func (s *stub) SearchNumbers(context.Context, Search) ([]Available, error) {
	s.searched++
	if s.err != nil {
		return nil, s.err
	}
	return s.offers, nil
}

func (s *stub) BuyNumber(context.Context, Order) (Number, error) {
	return Number{Vendor: s.vendor}, s.err
}

func (s *stub) ReleaseNumber(context.Context, string) error { return s.err }

func (s *stub) ConfigureInbound(context.Context, Inbound) error { return s.err }

func (s *stub) Dial(_ context.Context, outbound Outbound) (Dialed, error) {
	s.dialed = outbound
	if s.err != nil {
		return Dialed{}, s.err
	}
	return Dialed{VendorCallID: s.vendor + "-call", Status: "dialing"}, nil
}

func (s *stub) SendDigits(context.Context, string, string) error { return s.err }

func (s *stub) Supports(filter Filter) bool { return !slices.Contains(s.unsupported, filter) }

func (s *stub) Dials(feature CallFeature) bool { return !slices.Contains(s.undialable, feature) }

func (s *stub) Vendor() string { return s.vendor }

func (s *stub) Client() *http.Client { return http.DefaultClient }

func (s *PhoneSuite) TestAFactoryThatFailsIsNotCached() {
	registry := NewRegistry(s.config())
	failures := 0
	registry.Register("twilio", func() (Provider, error) {
		failures++
		return nil, errors.New("no")
	})
	s.T().Setenv("TWILIO_ACCOUNT_SID", "sid")
	s.T().Setenv("TWILIO_AUTH_TOKEN", "token")

	_, first := registry.Open("twilio")
	_, second := registry.Open("twilio")

	s.Error(first)
	s.Error(second)
	s.Equal(2, failures, "a failure is retried, not remembered")
}

func (s *PhoneSuite) config() Config {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	return config
}
