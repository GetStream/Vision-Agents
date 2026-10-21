package main

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

type LogRowsSuite struct {
	suite.Suite
	steps  *steps
	router *fakeRouter
}

func TestLogRowsSuite(t *testing.T) { suite.Run(t, new(LogRowsSuite)) }

func (s *LogRowsSuite) SetupTest() {
	s.steps = &steps{}
	s.router = &fakeRouter{steps: s.steps, rows: map[string]acceleration.AgentLog{}}
}

// addRows puts rows in the router's list (without details) and as whole rows by id.
func (s *LogRowsSuite) addRows(rows ...acceleration.AgentLog) {
	var list []acceleration.AgentLog
	for _, row := range rows {
		s.router.rows[row.Id] = row
		row.Details = nil
		list = append(list, row)
	}
	s.router.rowLists = [][]acceleration.AgentLog{list}
}

func logRow(id, eventType string, at time.Time, details map[string]any) acceleration.AgentLog {
	return acceleration.AgentLog{Id: id, EventType: eventType, OccurredAt: at, Details: &details}
}

func (s *LogRowsSuite) TestTheFirstJoinedRowGivesTheAgentsSFU() {
	at := time.Date(2026, 1, 1, 10, 0, 0, 0, time.UTC)
	s.addRows(
		logRow("2", "joined", at.Add(time.Minute), map[string]any{"sfu_edge": "later-edge", "sfu_host": "later.example.com"}),
		logRow("1", "joined", at, map[string]any{"sfu_edge": "first-edge", "sfu_host": "first.example.com"}),
	)

	got, err := readJoined(context.Background(), s.router, "sess")

	s.Require().NoError(err)
	s.Equal(joinedSites{SFUEdge: "first-edge", SFUHost: "first.example.com", JoinedAt: at}, got)
}

func (s *LogRowsSuite) TestNoJoinedRowIsAnError() {
	s.addRows(logRow("1", "left", time.Now(), nil))

	_, err := readJoined(context.Background(), s.router, "sess")

	s.Error(err)
}

func (s *LogRowsSuite) TestAnOutboundCallsSIPHostComesFromTheCallPlacedRow() {
	now := time.Now()
	s.addRows(
		logRow("1", "joined", now, map[string]any{"sfu_host": "sfu.example.com"}),
		logRow("2", "call_placed", now, map[string]any{"stream_sip_host": "sip.example.com", "vendor": "fake", "vendor_call_id": "CA0001"}),
		logRow("3", "call_sites", now, map[string]any{"sfus_used": "1"}),
	)
	ctx := context.Background()

	placed, found, err := readCallPlaced(ctx, s.router, "sess")
	s.Require().NoError(err)
	s.Require().True(found)
	sites, _, err := readCallSites(ctx, s.router, "sess", 0, time.Millisecond)
	s.Require().NoError(err)

	s.Equal(callPlacedRow{SIPHost: "sip.example.com", Vendor: "fake", VendorCallID: "CA0001"}, placed)
	s.Equal("sip.example.com", sipHostOf(person.Outbound, placed, sites))
}

func (s *LogRowsSuite) TestAnInboundCallsSIPHostComesFromTheCallSitesRow() {
	s.addRows(logRow("3", "call_sites", time.Now(), map[string]any{
		"sfus_used": "2", "sfu_locations": "a, b", "stream_session_id": "st1", "stream_sip_host": "sip.example.com",
	}))
	ctx := context.Background()

	placed, found, err := readCallPlaced(ctx, s.router, "sess")
	s.Require().NoError(err)
	s.False(found)
	sites, found, err := readCallSites(ctx, s.router, "sess", 0, time.Millisecond)
	s.Require().NoError(err)
	s.Require().True(found)

	s.Equal(callSitesRow{SFUsUsed: "2", SFULocations: "a, b", SIPHost: "sip.example.com", StreamSession: "st1"}, sites)
	s.Equal("sip.example.com", sipHostOf(person.Inbound, placed, sites))
}

func (s *LogRowsSuite) TestNoSIPHostAnywhereIsUnknownNotAnError() {
	s.Equal("", sipHostOf(person.Outbound, callPlacedRow{}, callSitesRow{}))
	s.Equal("", sipHostOf(person.Inbound, callPlacedRow{}, callSitesRow{}))
}

func (s *LogRowsSuite) TestTheCallSitesRowIsWaitedFor() {
	row := logRow("3", "call_sites", time.Now(), map[string]any{"sfus_used": "1"})
	s.router.rows[row.Id] = row
	s.router.rowLists = [][]acceleration.AgentLog{nil, nil, {{Id: "3", EventType: "call_sites"}}}

	got, found, err := readCallSites(context.Background(), s.router, "sess", time.Second, time.Millisecond)

	s.Require().NoError(err)
	s.True(found)
	s.Equal("1", got.SFUsUsed)
	s.Equal(3, s.router.listed)
}

func (s *LogRowsSuite) TestAMissingCallSitesRowIsUnknownNotAnError() {
	_, found, err := readCallSites(context.Background(), s.router, "sess", 20*time.Millisecond, time.Millisecond)

	s.NoError(err)
	s.False(found)
}

func (s *LogRowsSuite) TestTheLastPollIsAtTheDeadline() {
	s.router.rowLists = [][]acceleration.AgentLog{nil}
	start := time.Now()

	_, found, err := readCallSites(context.Background(), s.router, "sess", 30*time.Millisecond, time.Hour)

	s.NoError(err)
	s.False(found)
	s.Equal(2, s.router.listed, "one poll at the start, one at the deadline")
	s.Less(time.Since(start), time.Second)
}
