//go:build integration

package telnyx

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// Buying a number costs money, so these only read, apart from the one call that rings a
// handset and says so.

const (
	// fromEnvVar names a number this account holds, which is what the handset sees.
	fromEnvVar = "PHONE_TEST_FROM"
	// toEnvVar names the handset to ring.
	toEnvVar = "PHONE_TEST_TO"
)

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

// TestColoradoHDVoiceNumbersAreOfferedAroundADollar is the sprint's validation search:
// somewhere specific, with features beyond voice, at a price worth knowing before buying.
// Nothing here buys, because the buy is the one step done by hand.
func (s *TelnyxIntegrationSuite) TestColoradoHDVoiceNumbersAreOfferedAroundADollar() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	offered, err := s.provider.SearchNumbers(ctx, phone.Search{
		Country:            "US",
		AdministrativeArea: "CO",
		Type:               phone.Local,
		Limit:              5,
		Capabilities:       []phone.Capability{phone.Voice, phone.HDVoice, phone.Emergency},
	})
	s.Require().NoError(err)
	s.Require().NotEmpty(offered, "telnyx sells local numbers in colorado")

	for _, number := range offered {
		s.Equal("CO", number.Region, number.E164+" is not in colorado")
		s.Equal(phone.Local, number.Type, number.E164+" is not a local number")
		s.Contains(number.Capabilities, phone.HDVoice, number.E164+" is not hd voice")
		s.Contains(number.Capabilities, phone.Emergency, number.E164+" cannot reach 911")

		// A local US number is about a dollar a month. Anything far outside that is a
		// price worth seeing before a buy rather than after.
		s.Greater(number.MonthlyCostMicros, int64(0), number.E164+" is quoted no price")
		s.Less(number.MonthlyCostMicros, int64(5_000_000), number.E164+" costs more than expected")
	}
}

// TestCallingAHandsetRingsItAndBridgesItToATrunk is the sprint's validation call, and the
// one test here that costs money and makes a telephone ring. It needs a number this account
// holds and somebody willing to answer, so it is skipped unless both are named.
//
// It goes as far as an automated test can: the handset rings, and Telnyx accepts every term
// the call was placed on. Whether the person who answers hears an agent depends on one being
// in the call, which is what the example does.
func (s *TelnyxIntegrationSuite) TestCallingAHandsetRingsItAndBridgesItToATrunk() {
	from, to := os.Getenv(fromEnvVar), os.Getenv(toEnvVar)
	if from == "" || to == "" {
		s.T().Skip(fromEnvVar + " and " + toEnvVar + " not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	stream, err := phone.NewStream(phone.StreamOptions{})
	s.Require().NoError(err)

	callID := "outbound-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	trunkID, bridge, err := stream.CreateTrunk(ctx, phone.Trunk{
		Name:    callID,
		Numbers: []string{from},
	})
	s.Require().NoError(err)
	defer func() {
		_, _ = stream.Client().Video().DeleteSIPTrunk(
			context.Background(), trunkID, &getstream.DeleteSIPTrunkRequest{})
	}()

	_, err = stream.CreateRoute(ctx, phone.Route{
		Name:          callID,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{from},
		CallID:        callID,
		CallType:      "default",
	})
	s.Require().NoError(err)

	placed, err := s.provider.Dial(ctx, phone.Outbound{
		From:        from,
		To:          to,
		Bridge:      bridge,
		RingTimeout: 20 * time.Second,
		Headers:     map[string]string{"X-Call-Id": callID},
	})
	s.Require().NoError(err)

	s.NotEmpty(placed.VendorCallID, "a placed call has a leg to hang up")
	s.T().Logf("ringing %s as %s; join call %s to hear it", to, placed.VendorCallID, callID)
}

func (s *TelnyxIntegrationSuite) TestANumberThisAccountDoesNotOwnIsReportedAsSuch() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := s.provider.ReleaseNumber(ctx, "+15005550006")

	s.ErrorContains(err, "not one of this account's numbers")
}
