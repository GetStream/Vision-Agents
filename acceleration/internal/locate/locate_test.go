package locate

import (
	"context"
	"errors"
	"net/http"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/suite"
)

type LocateSuite struct {
	suite.Suite
}

func TestLocateSuite(t *testing.T) {
	suite.Run(t, new(LocateSuite))
}

func (s *LocateSuite) TestAnAddressInAPublishedRangeIsNamedByItsCloudAndRegion() {
	ranges, err := parseAWSRanges([]byte(`{"prefixes":[
		{"ip_prefix":"3.0.0.0/8","region":"GLOBAL"},
		{"ip_prefix":"3.80.0.0/12","region":"us-east-1"}]}`))
	s.Require().NoError(err)

	cloud, region := cloudOf(ranges, netip.MustParseAddr("3.84.1.2"), "AS14618 Amazon.com, Inc.")

	s.Equal("AWS", cloud)
	s.Equal("us-east-1", region)
}

func (s *LocateSuite) TestAnAddressOutsideThePublishedRangesFallsBackToItsNetworksName() {
	ranges, err := parseGoogleRanges([]byte(`{"prefixes":[{"ipv4Prefix":"34.0.0.0/15","scope":"us-east1"}]}`))
	s.Require().NoError(err)

	cloud, _ := cloudOf(ranges, netip.MustParseAddr("20.1.2.3"), "AS8075 Microsoft Corporation")
	s.Equal("Azure", cloud)

	cloud, _ = cloudOf(ranges, netip.MustParseAddr("168.86.128.1"), "AS20473 Twilio Inc.")
	s.Empty(cloud)
}

// refusing fails every request, as a client that cannot reach ipinfo does.
type refusing struct{}

func (refusing) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("connection refused")
}

func (s *LocateSuite) TestALookupThatFailsDoesNotNameTheToken() {
	locator := &Locator{client: &http.Client{Transport: refusing{}}, token: "ipinfo-secret", cache: map[string]Place{}}

	_, err := locator.Locate(context.Background(), netip.MustParseAddr("192.0.2.7"))

	s.Require().Error(err)
	s.Contains(err.Error(), "https://ipinfo.io/192.0.2.7/json", "the error still says what failed")
	s.NotContains(err.Error(), "ipinfo-secret")
}
