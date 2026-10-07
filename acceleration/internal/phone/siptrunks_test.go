package phone

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// trunkProvider stands in for the sip_trunk provider. Methods it does not override panic
// on the nil Provider, which is how a test proves they were never reached.
type trunkProvider struct {
	Provider
	digits []string
}

func (p *trunkProvider) Vendor() string { return SIPTrunkVendor }

func (p *trunkProvider) SendDigits(_ context.Context, vendorCallID, digits string) error {
	p.digits = append(p.digits, vendorCallID+":"+digits)
	return errors.New("phone: digits cannot be pressed on a call through a customer's own sip trunk")
}

func (s *PhoneSuite) TestSIPTrunksAreOffWithoutAKey() {
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config()), SIPTrunks: &trunkProvider{}})
	s.Require().NoError(err)

	password := "secret"
	_, err = service.CreateSIPTrunk(s.ctx, "acme", SIPTrunkSettings{
		Name: "main", Host: "trunk.example.com", Username: "agent", Password: &password,
	})
	s.ErrorIs(err, ErrSIPTrunksDisabled)
	_, err = service.SIPTrunks(s.ctx, "acme")
	s.ErrorIs(err, ErrSIPTrunksDisabled)
}

func (s *PhoneSuite) TestSIPTrunkIsNotAVendorAnyoneIsOffered() {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	registry := NewRegistry(config)

	_, declared := registry.Lookup(SIPTrunkVendor)
	s.False(declared, "the registry is what every customer is offered to buy from")
	s.NotContains(registry.Available(), SIPTrunkVendor)
	_, err = registry.Open(SIPTrunkVendor)
	s.EqualError(err, `phone: "sip_trunk" is not a known vendor`)
}

func (s *PhoneSuite) TestSIPTrunkSettingsAreCheckedInOnePass() {
	empty := ""
	settings := SIPTrunkSettings{
		Host: "sip:trunk.example.com:5060", Port: 70000, Transport: "sctp",
		Password: &empty, Codecs: []string{"OPUS"},
	}.normalized()

	err := settings.validate(false)
	s.ErrorIs(err, ErrInvalidSIPTrunk)
	s.EqualError(err, `sip_trunk: invalid: name is required; `+
		`host "sip:trunk.example.com:5060" must be a bare hostname, without sip: or a port; `+
		`port 70000 must be 1..65535; transport "sctp" must be udp, tcp or tls; `+
		`username is required; password cannot be empty; `+
		`codec "OPUS" is not supported, use PCMU, PCMA or G722`)

	s.EqualError(SIPTrunkSettings{Name: "main", Username: "agent"}.normalized().validate(true),
		"sip_trunk: invalid: host is required; password is required")
	password := "s3cret"
	s.NoError(SIPTrunkSettings{Name: "main", Host: "trunk.example.com", Username: "agent", Password: &password}.
		normalized().validate(true))
}

func (s *PhoneSuite) TestASIPTrunkCannotPointToAPrivateAddress() {
	password := "s3cret"
	for _, host := range []string{"10.0.0.1", "127.0.0.1", "169.254.169.254"} {
		err := SIPTrunkSettings{Name: "main", Host: host, Username: "agent", Password: &password}.
			normalized().validate(true)

		s.EqualError(err, `sip_trunk: invalid: host "`+host+`" is not a public address`)
	}
	s.NoError(SIPTrunkSettings{Name: "main", Host: "8.8.8.8", Username: "agent", Password: &password}.
		normalized().validate(true))
}

func (s *PhoneSuite) TestSIPTrunkSettingsFillTheirDefaults() {
	settings := SIPTrunkSettings{Name: "main", Host: " trunk.example.com ", Transport: "TCP", Codecs: []string{"pcma"}}.normalized()

	s.Equal(SIPTrunkSettings{
		Name: "main", Host: "trunk.example.com", Port: 5060, Transport: "tcp", Codecs: []string{"PCMA"},
	}, settings)
	s.Equal([]string{"PCMU", "PCMA"}, SIPTrunkSettings{}.normalized().Codecs)
}

func (s *PhoneSuite) TestDigitsOnACallThroughATrunkReachTheTrunkProvider() {
	trunks := &trunkProvider{}
	sealer, err := auth.NewSealer("phone suite")
	s.Require().NoError(err)
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config()), SIPTrunks: trunks, Sealer: sealer})
	s.Require().NoError(err)

	err = service.SendDigits(s.ctx, SIPTrunkVendor, "call-1", "12")
	s.EqualError(err, "phone: digits cannot be pressed on a call through a customer's own sip trunk")
	s.Equal([]string{"call-1:12"}, trunks.digits)
}

func (s *PhoneSuite) TestClosingWithoutATrunkProviderDoesNothing() {
	service, err := NewService(ServiceOptions{Registry: NewRegistry(s.config())})
	s.Require().NoError(err)
	s.NoError(service.Close(s.ctx))
}
