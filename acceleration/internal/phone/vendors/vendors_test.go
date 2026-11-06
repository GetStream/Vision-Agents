package vendors

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/bandwidth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/bird"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/didww"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/plivo"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sinch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/twilio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vonage"
)

type VendorsSuite struct {
	suite.Suite
}

func TestVendors(t *testing.T) { suite.Run(t, new(VendorsSuite)) }

func (s *VendorsSuite) TestOnlyPlivoAndBandwidthFetchTheirCallPlanOnAnswer() {
	// Implementing AnswerRenderer is what makes the service park a plan and hand the
	// vendor a url for it, so which vendors do it decides which calls need this service to
	// be publicly reachable. Sinch is the near miss: it also has no inline plan on the
	// request in the usual sense, but its callout carries one as a string, so it needs no
	// url. Every other vendor bridges from the request itself.
	hosted := map[string]bool{"plivo": true, "bandwidth": true}

	for _, provider := range s.providers() {
		_, fetches := provider.(phone.AnswerRenderer)
		s.Equal(hosted[provider.Vendor()], fetches,
			provider.Vendor()+" disagrees about fetching its call plan")
	}
}

// A search a vendor cannot run as asked is refused before anything is sent, as a request the
// caller can change: the router answers it 400 in the vendor's words, not 500.
func (s *VendorsSuite) TestASearchAVendorCannotRunIsTheCallersToChange() {
	ctx := context.Background()
	for _, provider := range s.providers() {
		_, err := provider.SearchNumbers(ctx, phone.Search{})
		s.ErrorIs(err, phone.ErrInvalid, provider.Vendor()+": a search without a country")
	}
	unsold := phone.Search{Country: "US", Type: phone.NumberType("satellite")}
	both := phone.Search{Country: "US", Prefix: "512", Contains: "55"}
	refused := map[string][]phone.Search{
		"twilio":    {unsold},
		"sinch":     {unsold, both},
		"vonage":    {unsold, both},
		"bird":      {unsold},
		"plivo":     {unsold},
		"bandwidth": {{Country: "GB", AreaCode: "20"}, {Country: "US", AreaCode: "512", Type: phone.TollFree}},
	}
	for _, provider := range s.providers() {
		for _, search := range refused[provider.Vendor()] {
			_, err := provider.SearchNumbers(ctx, search)
			s.ErrorIs(err, phone.ErrInvalid, "%s: %+v", provider.Vendor(), search)
		}
	}
}

// providers builds one of each implemented vendor. The credentials are nonsense because
// nothing here places a call; they only have to be present.
func (s *VendorsSuite) providers() []phone.Provider {
	twilioProvider, err := twilio.New(twilio.Options{AccountSID: "AC1", AuthToken: "t"})
	s.Require().NoError(err)
	telnyxProvider, err := telnyx.New(telnyx.Options{APIKey: "k"})
	s.Require().NoError(err)
	sinchProvider, err := sinch.New(sinch.Options{ProjectID: "p", KeyID: "k", KeySecret: "s"})
	s.Require().NoError(err)
	bandwidthProvider, err := bandwidth.New(bandwidth.Options{
		AccountID: "a", Username: "u", Password: "p",
	})
	s.Require().NoError(err)
	vonageProvider, err := vonage.New(vonage.Options{APIKey: "k", APISecret: "s"})
	s.Require().NoError(err)
	// Bird reads the region out of its key, so this one has to look like a key.
	birdProvider, err := bird.New(bird.Options{AccessKey: "bk_us1_abc"})
	s.Require().NoError(err)
	didwwProvider, err := didww.New(didww.Options{APIKey: "k"})
	s.Require().NoError(err)
	plivoProvider, err := plivo.New(plivo.Options{AuthID: "MA1", AuthToken: "t"})
	s.Require().NoError(err)

	return []phone.Provider{
		twilioProvider, telnyxProvider, sinchProvider, bandwidthProvider,
		vonageProvider, birdProvider, didwwProvider, plivoProvider,
	}
}
