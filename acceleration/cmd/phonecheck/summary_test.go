package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type SummarySuite struct{ suite.Suite }

func TestSummarySuite(t *testing.T) { suite.Run(t, new(SummarySuite)) }

func (s *SummarySuite) TestTheSummaryIsOneLine() {
	var out bytes.Buffer
	sum := summary{
		Direction: "outbound", Provider: "fake", SessionID: "sess-1", Outcome: "ok",
		Connect: 1500 * time.Millisecond, Talked: 20 * time.Second, Total: 25 * time.Second,
		AgentSFU: "sfu-a1", CallerSFU: "sfu-c.example.com", SFUsUsed: "2", VendorCallID: "CA0001",
	}

	sum.log(newLogger("text", &out))

	text := strings.TrimSpace(out.String())
	s.Equal(1, strings.Count(text, "\n")+1)
	s.Contains(text, "session_id=sess-1")
	s.Contains(text, "connect_ms=1500")
	s.Contains(text, "sfus_used=2")
	s.Contains(text, "agent_sfu=sfu-a1")
	s.Contains(text, "caller_sfu=sfu-c.example.com")
	s.Contains(text, "vendor_call_id=CA0001")
}

func (s *SummarySuite) TestAnInboundSummaryHasNoVendorCallID() {
	var out bytes.Buffer

	summary{Direction: "inbound", SessionID: "sess-1"}.log(newLogger("text", &out))

	s.NotContains(out.String(), "vendor_call_id")
}

func (s *SummarySuite) TestThePersonsCallLengthSitsNextToTalkedMs() {
	var out bytes.Buffer

	summary{Talked: 35 * time.Second, PersonCall: 27 * time.Second}.log(newLogger("text", &out))

	s.Contains(out.String(), "talked_ms=35000 person_call_s=27 total_ms=")
}

func (s *SummarySuite) TestAnUnknownPersonCallLengthIsLeftOut() {
	var out bytes.Buffer

	summary{Talked: 35 * time.Second}.log(newLogger("text", &out))

	s.NotContains(out.String(), "person_call_s")
}
