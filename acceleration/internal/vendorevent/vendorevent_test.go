package vendorevent

import (
	"crypto/rand"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"
)

// TelnyxSuite is the level each Telnyx event type is counted at, and the name it is
// counted under.
type TelnyxSuite struct {
	suite.Suite
}

func TestTelnyxSuite(t *testing.T) {
	suite.Run(t, new(TelnyxSuite))
}

// A port-out takes a number away, so each of its events pages, whatever its status says.
func (s *TelnyxSuite) TestEveryPortOutEventPages() {
	for _, eventType := range []string{"portout.status_changed", "portout.new_comment", "portout.foc_date_changed"} {
		event, level := Telnyx(eventType, "pending")
		s.Equal(eventType, event)
		s.Equal(LevelPage, level, eventType)
	}
}

func (s *TelnyxSuite) TestAReachedMessagingSpendLimitPages() {
	event, level := Telnyx("messaging-profile.spend-limit-reached", "")
	s.Equal("messaging-profile.spend-limit-reached", event)
	s.Equal(LevelPage, level)
}

func (s *TelnyxSuite) TestAPortInIsCountedUnderItsNameAtInfo() {
	for _, eventType := range []string{"porting_order.status_changed", "porting_order.new_comment",
		"porting_order.split", "porting_order.deleted", "porting_order.messaging_changed"} {
		event, level := Telnyx(eventType, "")
		s.Equal(eventType, event)
		s.Equal(LevelInfo, level, eventType)
	}
}

func (s *TelnyxSuite) TestANumberOrderThatSucceededNeedsNobody() {
	event, level := Telnyx("number_order.complete", "success")
	s.Equal("number_order.complete", event)
	s.Equal(LevelInfo, level)
}

// Every status other than success leaves numbers unbought: failure and pending are the
// documented ones, and a missing or new status must not read as a success.
func (s *TelnyxSuite) TestANumberOrderThatDidNotSucceedWarns() {
	for _, status := range []string{"failure", "pending", "", "Success"} {
		event, level := Telnyx("number_order.complete", status)
		s.Equal("number_order.complete", event)
		s.Equal(LevelWarn, level, status)
	}
}

// Only a number order's status decides its level: a port-out whose status is success
// still pages.
func (s *TelnyxSuite) TestAStatusChangesTheLevelOfANumberOrderOnly() {
	_, level := Telnyx("portout.status_changed", "failure")
	s.Equal(LevelPage, level)
	_, level = Telnyx("porting_order.status_changed", "failure")
	s.Equal(LevelInfo, level)
}

// A type the map does not name is one label value, whatever Telnyx sent, so a vendor's new
// type or a near miss of a known one adds no series and pages nobody.
func (s *TelnyxSuite) TestATypeTheMapDoesNotNameIsCountedAsOther() {
	for _, eventType := range []string{"", "phone_number.deleted", "PORTOUT.STATUS_CHANGED",
		"portout.status_changed ", "portout", "message.received", rand.Text()} {
		event, level := Telnyx(eventType, "failure")
		s.Equal(Other, event, eventType)
		s.Equal(LevelInfo, level, eventType)
	}
}

// CountSuite is the counter a monitor reads.
type CountSuite struct {
	suite.Suite
}

func TestCountSuite(t *testing.T) {
	suite.Run(t, new(CountSuite))
}

// The vendor is unique to the test, so no other count moves the series it reads.
func (s *CountSuite) TestACountAddsOneToItsOwnSeriesOnly() {
	vendor := "test-" + rand.Text()
	page := testutil.ToFloat64(events.WithLabelValues(vendor, "portout.status_changed", "page"))
	warn := testutil.ToFloat64(events.WithLabelValues(vendor, "portout.status_changed", "warn"))

	Count(vendor, "portout.status_changed", LevelPage)

	s.Equal(1.0, testutil.ToFloat64(events.WithLabelValues(vendor, "portout.status_changed", "page"))-page)
	s.Equal(0.0, testutil.ToFloat64(events.WithLabelValues(vendor, "portout.status_changed", "warn"))-warn)
}

// The router's /metrics serves the default registry, and the monitor reads the counter by
// this name and these labels.
func (s *CountSuite) TestTheCounterIsServedUnderItsNameWithItsLabels() {
	vendor := "test-" + rand.Text()

	Count(vendor, Other, LevelInfo)

	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	var found map[string]string
	for _, family := range families {
		if family.GetName() != "vendor_account_events_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["vendor"] == vendor {
				found = labels
				s.Equal(1.0, metric.GetCounter().GetValue())
			}
		}
	}
	s.Equal(map[string]string{"vendor": vendor, "event": "other", "level": "info"}, found)
}
