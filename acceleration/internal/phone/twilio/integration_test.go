//go:build integration

package twilio

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// Buying a number costs money and releasing it is not instant, so these only read.

type TwilioIntegrationSuite struct {
	suite.Suite
	provider *Provider
}

func TestTwilioIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TwilioIntegrationSuite))
}

func (s *TwilioIntegrationSuite) SetupSuite() {
	if os.Getenv(accountSIDEnvVar) == "" || os.Getenv(authTokenEnvVar) == "" {
		s.T().Skip(accountSIDEnvVar + " and " + authTokenEnvVar + " not set")
	}

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *TwilioIntegrationSuite) TestTwilioOffersVoiceNumbersInTheUS() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	offered, err := s.provider.SearchNumbers(ctx, phone.Search{
		Country:      "US",
		Limit:        5,
		Capabilities: []phone.Capability{phone.Voice},
	})
	s.Require().NoError(err)

	s.NotEmpty(offered, "twilio always has US numbers for sale")
	for _, number := range offered {
		s.Contains(number.Capabilities, phone.Voice, number.E164+" cannot take a call")
		s.Equal("US", number.Country)
	}
}

func (s *TwilioIntegrationSuite) TestANumberThisAccountDoesNotOwnIsReportedAsSuch() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := s.provider.ConfigureInbound(ctx, phone.Inbound{
		E164:   "+15005550006",
		Bridge: phone.Bridge{URI: "sip:nobody@sip.stream-io-api.com"},
	})

	s.ErrorContains(err, "not one of this account's numbers")
}
