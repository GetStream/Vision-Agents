package phone

import (
	"context"
	"errors"
	"testing"

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

func (s *PhoneSuite) TestTwilioAndTelnyxAreTheImplementedOnes() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	var implemented []string
	for _, vendor := range config.Vendors {
		if vendor.Implemented {
			implemented = append(implemented, vendor.Vendor)
		}
	}

	s.ElementsMatch([]string{"twilio", "telnyx"}, implemented)
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

	_, err = provider.BuyNumber(s.ctx, "+15551234567")
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

	s.ErrorContains(ValidateDigits(""), "digits to press")
	s.ErrorContains(ValidateDigits("one"), "keypad can press")
	s.ErrorContains(ValidateDigits("press 1"), "keypad can press")
}

func (s *PhoneSuite) TestALineThatWasNotPlacedFromHereCannotPressAnything() {
	// Pressing needs the vendor's own id for the leg, and an inbound call arrived at
	// Stream rather than being dialled from here, so there is no such id to name.
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)

	line := service.Line(LineOptions{From: "+15125551234", CallID: "support-line"})

	s.ErrorContains(line.SendDigits(s.ctx, "1"), "not placed from here")
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
