package phone

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type ServiceSuite struct {
	suite.Suite
}

func TestService(t *testing.T) { suite.Run(t, new(ServiceSuite)) }

func (s *ServiceSuite) TestEachCallResultIsAnOutcome() {
	for result, want := range map[CallResult]outcome.Outcome{
		CallCompleted: outcome.Completed,
		CallBusy:      outcome.Busy,
		CallNoAnswer:  outcome.NoAnswer,
		CallFailed:    outcome.Failed,
		CallCanceled:  outcome.Canceled,
	} {
		got, err := outcomeOf(result)
		s.Require().NoError(err)
		s.Equal(want, got, string(result))
	}
}

func (s *ServiceSuite) TestANonFinalResultIsNoOutcome() {
	got, err := outcomeOf("")
	s.Require().NoError(err)

	s.Empty(got)
}

func (s *ServiceSuite) TestAnUnknownResultIsRefused() {
	_, err := outcomeOf(CallResult("voicemail"))

	s.ErrorContains(err, "voicemail")
}

// TestEveryVendorOfTheServiceHasItsCallSeriesAtZeroFromTheStart catches a vendor the service
// counts calls under whose first call makes a series that starts at 1, which increase() reads as
// no change: the registry's vendors and sip_trunk, with no call placed.
func (s *ServiceSuite) TestEveryVendorOfTheServiceHasItsCallSeriesAtZeroFromTheStart() {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	_, err = NewService(ServiceOptions{Registry: NewRegistry(config), Store: &store.Store{}})
	s.Require().NoError(err)

	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	has := map[string]bool{}
	for _, family := range families {
		if family.GetName() != "router_phone_call_outcomes_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["direction"] == "outbound" && labels["outcome"] == "completed" {
				has[labels["vendor"]] = true
			}
		}
	}
	vendors := []string{SIPTrunkVendor, "other"}
	for _, vendor := range config.Vendors {
		vendors = append(vendors, vendor.Vendor)
	}
	for _, vendor := range vendors {
		s.True(has[vendor], vendor)
	}
}
