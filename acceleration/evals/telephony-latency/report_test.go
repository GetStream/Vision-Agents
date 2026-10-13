package main

import (
	"net/netip"
	"testing"

	"github.com/stretchr/testify/suite"
)

type ReportSuite struct {
	suite.Suite
}

func TestReportSuite(t *testing.T) {
	suite.Run(t, new(ReportSuite))
}

// call is a finished call between two vendors in a city with these stamp latencies.
func (s *ReportSuite) call(caller, callee, city string, forward ...float64) Result {
	return Result{
		Region:    "us-east1",
		Caller:    Line{Vendor: caller, City: city, E164: "+1" + caller},
		Callee:    Line{Vendor: callee, City: city, E164: "+1" + callee},
		ForwardMs: forward,
	}
}

func (s *ReportSuite) TestSplitRecoversWhatEachVendorAddsInEachRole() {
	// Built as 150 ms plus twilio +20 / telnyx -20 as caller and twilio -5 / telnyx +5 as
	// callee, so the fit has an exact answer to find.
	cells := map[Cell]float64{
		{"twilio", "twilio"}: 165,
		{"twilio", "telnyx"}: 175,
		{"telnyx", "twilio"}: 125,
		{"telnyx", "telnyx"}: 135,
	}

	shares := Split(cells)

	s.InDelta(150, shares.Baseline, 0.01)
	s.InDelta(20, shares.Caller["twilio"], 0.01)
	s.InDelta(-20, shares.Caller["telnyx"], 0.01)
	s.InDelta(-5, shares.Callee["twilio"], 0.01)
	s.InDelta(5, shares.Callee["telnyx"], 0.01)
	s.InDelta(0, shares.Residual, 0.01)
}

func (s *ReportSuite) TestSplitStillRecoversTheSharesWithACellMissing() {
	// Three vendors, the same additive build, and plivo never managed to call itself.
	caller := map[string]float64{"a": 30, "b": 0, "c": -30}
	callee := map[string]float64{"a": -10, "b": 0, "c": 10}
	cells := map[Cell]float64{}
	for from, out := range caller {
		for to, in := range callee {
			if from == "c" && to == "c" {
				continue
			}
			cells[Cell{from, to}] = 200 + out + in
		}
	}

	shares := Split(cells)

	s.InDelta(200, shares.Baseline, 0.5)
	for vendor := range caller {
		s.InDelta(caller[vendor], shares.Caller[vendor], 0.5, vendor)
		s.InDelta(callee[vendor], shares.Callee[vendor], 0.5, vendor)
	}
}

func (s *ReportSuite) TestAPairThatIsSlowOnlyTogetherShowsAsUnexplained() {
	cells := map[Cell]float64{
		{"twilio", "twilio"}: 150,
		{"twilio", "telnyx"}: 150,
		{"telnyx", "twilio"}: 150,
		{"telnyx", "telnyx"}: 250,
	}

	s.Greater(Split(cells).Residual, 20.0)
}

func (s *ReportSuite) TestACellIsTheMedianOfEachCallsMedian() {
	results := []Result{
		s.call("twilio", "telnyx", "Chicago", 100, 101, 400),
		s.call("twilio", "telnyx", "Chicago", 120, 121, 122),
		s.call("twilio", "telnyx", "Chicago", 140),
		{Caller: Line{Vendor: "twilio"}, Callee: Line{Vendor: "telnyx"}, Error: "dial: busy"},
	}

	stats := Summarise(results, Forward)

	s.Equal(Stats{Median: 121, P95: 140, Calls: 3, Failed: 1}, stats)
}

func (s *ReportSuite) TestTheReportShowsEachCitysMatrixAndTheVendorShares() {
	var results []Result
	for _, city := range []string{"Chicago", "Houston"} {
		results = append(results,
			s.call("twilio", "twilio", city, 165),
			s.call("twilio", "telnyx", city, 175),
			s.call("telnyx", "twilio", city, 125),
			s.call("telnyx", "telnyx", city, 135),
		)
	}
	results[1].CalleeMedia = "192.0.2.10:30000"
	places := map[string]Place{"192.0.2.10": {IP: "192.0.2.10", City: "Ashburn", Cloud: "AWS", CloudRegion: "us-east-1"}}

	report := Render(results, places)

	s.Contains(report, "## us-east1")
	s.Contains(report, "### Chicago")
	s.Contains(report, "### Houston")
	s.Contains(report, "| telnyx | 135 ms (p95 135, n=1) | 125 ms (p95 125, n=1) |")
	s.Contains(report, "| all calls | 150 ms | -20 ms | +20 ms | +5 ms | -5 ms | 0 ms |")
	s.Contains(report, "192.0.2.10 · Ashburn · AWS us-east-1")
}

func (s *ReportSuite) TestAnAddressInAPublishedRangeIsNamedByItsCloudAndRegion() {
	ranges, err := parseAWSRanges([]byte(`{"prefixes":[
		{"ip_prefix":"3.0.0.0/8","region":"GLOBAL"},
		{"ip_prefix":"3.80.0.0/12","region":"us-east-1"}]}`))
	s.Require().NoError(err)

	cloud, region := cloudOf(ranges, netip.MustParseAddr("3.84.1.2"), "AS14618 Amazon.com, Inc.")

	s.Equal("AWS", cloud)
	s.Equal("us-east-1", region)
}

func (s *ReportSuite) TestAnAddressOutsideThePublishedRangesFallsBackToItsNetworksName() {
	ranges, err := parseGoogleRanges([]byte(`{"prefixes":[{"ipv4Prefix":"34.0.0.0/15","scope":"us-east1"}]}`))
	s.Require().NoError(err)

	cloud, _ := cloudOf(ranges, netip.MustParseAddr("20.1.2.3"), "AS8075 Microsoft Corporation")
	s.Equal("Azure", cloud)

	cloud, _ = cloudOf(ranges, netip.MustParseAddr("168.86.128.1"), "AS20473 Twilio Inc.")
	s.Empty(cloud)
}
