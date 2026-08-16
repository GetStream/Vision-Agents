//go:build integration

package telnyx

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// Buying a number costs money, so these only read.

type TelnyxIntegrationSuite struct {
	suite.Suite
	provider *Provider
}

func TestTelnyxIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TelnyxIntegrationSuite))
}

func (s *TelnyxIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " not set")
	}

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *TelnyxIntegrationSuite) TestTelnyxOffersVoiceNumbersInTheUS() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	offered, err := s.provider.SearchNumbers(ctx, phone.Search{
		Country:      "US",
		Limit:        5,
		Capabilities: []phone.Capability{phone.Voice},
	})
	s.Require().NoError(err)

	s.NotEmpty(offered, "telnyx always has US numbers for sale")
	for _, number := range offered {
		s.Contains(number.Capabilities, phone.Voice, number.E164+" cannot take a call")
	}
}

func (s *TelnyxIntegrationSuite) TestANumberThisAccountDoesNotOwnIsReportedAsSuch() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := s.provider.ReleaseNumber(ctx, "+15005550006")

	s.ErrorContains(err, "not one of this account's numbers")
}
