package outcome

import (
	"context"
	"crypto/rand"
	"log/slog"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DroppedSuite is the rule that tells a dropped call from a completed one: the session of an
// answered call, as it was when the call's leg ended.
type DroppedSuite struct {
	suite.Suite
	ended time.Time
}

func TestDroppedSuite(t *testing.T) { suite.Run(t, new(DroppedSuite)) }

func (s *DroppedSuite) SetupTest() { s.ended = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC) }

// closed is a session that closed d after the leg ended (before it, when d is negative).
func (s *DroppedSuite) closed(d time.Duration, reason string) store.AgentSession {
	return store.AgentSession{ClosedAt: new(s.ended.Add(d)), EndReason: reason}
}

func (s *DroppedSuite) TestASessionStillOpenWhenThePhoneLeftDidNotDropTheCall() {
	s.False(leftOnTheLine(store.AgentSession{}, s.ended), "the caller or the network hung up first")
}

func (s *DroppedSuite) TestASessionStoppedBeforeThePhoneLeftDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(-time.Second, ""), s.ended))
}

func (s *DroppedSuite) TestASessionStoppedAsThePhoneLeftDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(0, ""), s.ended))
}

func (s *DroppedSuite) TestASessionThatClosedAfterThePhoneLeftDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(time.Millisecond, ""), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatHungUpDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "agent_hangup"), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatTransferredDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "transferred"), s.ended))
}

func (s *DroppedSuite) TestAnAgentThatLeftAVoicemailDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "voicemail"), s.ended))
}

func (s *DroppedSuite) TestASessionThatFollowedTheCallersHangUpDidNotDropTheCall() {
	s.False(leftOnTheLine(s.closed(-time.Second, "hangup"), s.ended))
}

func (s *DroppedSuite) TestASessionThatGaveUpWaitingBeforeThePhoneJoinedDroppedTheCall() {
	s.True(leftOnTheLine(s.closed(-time.Second, "start"), s.ended), "the caller answered an agent already gone")
	s.True(leftOnTheLine(s.closed(-time.Second, "unanswered"), s.ended))
}

// CarrierSuite is how a vendor's report of a call's carrier is labelled and when it counts.
type CarrierSuite struct {
	suite.Suite
	ended time.Time
}

func TestCarrierSuite(t *testing.T) { suite.Run(t, new(CarrierSuite)) }

func (s *CarrierSuite) SetupTest() { s.ended = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC) }

func (s *CarrierSuite) TestAResponseCodeOfRFC3261IsItsOwnLabel() {
	for code, label := range map[int]string{200: "200", 404: "404", 486: "486", 487: "487", 503: "503", 603: "603"} {
		got, ok := sipCodeLabel(code)
		s.True(ok, code)
		s.Equal(label, got)
	}
}

func (s *CarrierSuite) TestAnyOtherFinalCodeIsCountedAsItsClass() {
	for code, label := range map[int]string{299: "2xx", 399: "3xx", 429: "4xx", 599: "5xx", 607: "6xx", 699: "6xx"} {
		got, ok := sipCodeLabel(code)
		s.True(ok, code)
		s.Equal(label, got)
	}
}

func (s *CarrierSuite) TestACodeThatIsNoFinalResponseIsNotCounted() {
	for _, code := range []int{0, -486, 100, 180, 199, 700, 4860} {
		_, ok := sipCodeLabel(code)
		s.False(ok, code)
	}
}

func (s *CarrierSuite) TestACallAnsweredForUnderSixSecondsIsShort() {
	s.True(isShort(0))
	s.True(isShort(5999 * time.Millisecond))
	s.False(isShort(6 * time.Second))
	s.False(isShort(-time.Second), "a join dated after the end")
}

func (s *CarrierSuite) TestAnUndecidedCallHasNoFinalReport() {
	s.False(finalReportKept(Call{}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedWhileTheLegIsUpCameFromAReport() {
	s.True(finalReportKept(Call{Outcome: Busy, DecidedAt: new(s.ended)}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedBeforeTheEndCameFromAReport() {
	s.True(finalReportKept(Call{Outcome: Busy, DecidedAt: new(s.ended.Add(-time.Second)), EndedAt: new(s.ended)}))
}

func (s *CarrierSuite) TestAnOutcomeDecidedAtTheEndCameFromTheEnd() {
	s.False(finalReportKept(Call{Outcome: NoAnswer, DecidedAt: new(s.ended), EndedAt: new(s.ended)}))
	s.False(finalReportKept(Call{Outcome: Completed, DecidedAt: new(s.ended.Add(time.Second)), EndedAt: new(s.ended)}),
		"a join after the end")
}

// SeriesSuite is the series a router holds before its first call: the first call of a pod must
// move its series from 0 to 1, or increase() and rate() read no change. Each test names a vendor
// of its own, so it reads only the series it made.
type SeriesSuite struct {
	suite.Suite
	vendor string
	keeper *Keeper
}

func TestSeriesSuite(t *testing.T) { suite.Run(t, new(SeriesSuite)) }

func (s *SeriesSuite) SetupTest() {
	s.vendor = "vendor-" + rand.Text()
	s.keeper = NewKeeper(nil, time.Now, []string{s.vendor}, slog.New(slog.DiscardHandler))
}

// series is every phone series of vendor in the registry /metrics serves, by metric, direction
// and the label that tells them apart (outcome or code): a counter's value, a histogram's count.
func (s *SeriesSuite) series(vendor string) map[string]float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	got := map[string]float64{}
	for _, family := range families {
		if !strings.HasPrefix(family.GetName(), "router_phone_") {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["vendor"] != vendor {
				continue
			}
			key := family.GetName() + " " + labels["direction"] + " " + labels["outcome"] + labels["code"]
			if histogram := metric.GetHistogram(); histogram != nil {
				got[key] = float64(histogram.GetSampleCount())
			} else {
				got[key] = metric.GetCounter().GetValue()
			}
		}
	}
	return got
}

// wanted is every series a vendor the keeper names has before its first call, at 0.
func (s *SeriesSuite) wanted() map[string]float64 {
	want := map[string]float64{
		"router_phone_post_dial_delay_seconds outbound ": 0,
		"router_phone_short_calls_total inbound ":        0,
		"router_phone_short_calls_total outbound ":       0,
	}
	for _, outcome := range []string{"completed", "busy", "no_answer", "failed", "canceled", "unknown", "dropped", "refused"} {
		want["router_phone_call_outcomes_total outbound "+outcome] = 0
	}
	for _, outcome := range []string{"completed", "failed", "dropped"} {
		want["router_phone_call_outcomes_total inbound "+outcome] = 0
	}
	for _, code := range sipCodes {
		want["router_phone_call_sip_responses_total outbound "+strconv.Itoa(code)] = 0
	}
	for _, class := range []string{"2xx", "3xx", "4xx", "5xx", "6xx"} {
		want["router_phone_call_sip_responses_total outbound "+class] = 0
	}
	return want
}

func (s *SeriesSuite) TestAVendorTheKeeperNamesHasEverySeriesAtZeroBeforeItsFirstCall() {
	s.Equal(s.wanted(), s.series(s.vendor))
}

func (s *SeriesSuite) TestEveryKeeperHasTheOtherVendorsSeriesToo() {
	got := s.series(otherVendor)
	for series := range s.wanted() {
		s.Contains(got, series)
	}
}

func (s *SeriesSuite) TestTheFirstCallMovesExactlyItsSeriesFromZeroToOne() {
	want := s.wanted()
	want["router_phone_call_outcomes_total outbound busy"] = 1
	want["router_phone_call_sip_responses_total outbound 486"] = 1
	want["router_phone_post_dial_delay_seconds outbound "] = 1

	s.keeper.count(context.Background(), Key{}, s.vendor, outbound, Busy, nil, time.Now())
	s.keeper.countReport(Call{Vendor: s.vendor}, Report{Outcome: Busy, SIPCode: 486, PostDialDelay: 2 * time.Second}, time.Now())

	s.Equal(want, s.series(s.vendor))
}

func (s *SeriesSuite) TestAnInboundCallMovesItsOwnSeries() {
	want := s.wanted()
	want["router_phone_call_outcomes_total inbound failed"] = 1

	s.keeper.count(context.Background(), Key{}, s.vendor, inbound, Failed, nil, time.Now())

	s.Equal(want, s.series(s.vendor))
}

func (s *SeriesSuite) TestAVendorTheKeeperDoesNotNameIsCountedAsOtherAndMakesNoSeries() {
	raw := "raw-" + rand.Text()
	before := s.series(otherVendor)["router_phone_call_outcomes_total outbound failed"]

	s.keeper.count(context.Background(), Key{}, raw, outbound, Failed, nil, time.Now())

	s.Empty(s.series(raw))
	s.Equal(before+1, s.series(otherVendor)["router_phone_call_outcomes_total outbound failed"])
}
