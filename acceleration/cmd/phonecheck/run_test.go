package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

const sessionID = "sess-1"

type RunSuite struct {
	suite.Suite
	steps    *steps
	router   *fakeRouter
	person   *fakePerson
	session  *fakeSession
	out      bytes.Buffer
	cfg      Config
	looked   *hostLog // hosts d.lookup was asked for
	resolved *hostLog // hosts the DoH fake was asked for
	doh      *httptest.Server
}

func TestRunSuite(t *testing.T) { suite.Run(t, new(RunSuite)) }

type hostLog struct {
	mu    sync.Mutex
	hosts []string
}

func (h *hostLog) add(host string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.hosts = append(h.hosts, host)
}

func (h *hostLog) all() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.hosts...)
}

func (s *RunSuite) SetupTest() {
	s.steps = &steps{}
	ended := make(chan struct{})
	s.person = &fakePerson{steps: s.steps, ended: ended}
	s.session = &fakeSession{steps: s.steps, id: sessionID, personEnded: ended}
	s.router = &fakeRouter{
		steps:   s.steps,
		numbers: []acceleration.PhoneNumber{attached(agentNumber, "twilio")},
		session: s.session,
		rows:    map[string]acceleration.AgentLog{},
	}
	s.addRows(
		logRow("1", "joined", time.Now(), map[string]any{"sfu_edge": "sfu-a1", "sfu_host": "sfu.example.com"}),
		logRow("2", "call_placed", time.Now(), map[string]any{"stream_sip_host": "sip.example.com", "vendor_call_id": "CA0001"}),
		logRow("3", "call_sites", time.Now(), map[string]any{"sfus_used": "2", "stream_sip_host": "sip.example.com"}),
	)
	s.out.Reset()
	s.cfg = Config{
		Direction: person.Outbound, Provider: "fake", Router: "http://router.example.com:8080",
		AgentNumber: agentNumber, PersonNumber: personNumber, Talk: time.Second, SIPLookupFrom: "us",
	}
	s.looked, s.resolved = &hostLog{}, &hostLog{}
	s.doh = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.resolved.add(r.URL.Query().Get("name"))
		_, _ = w.Write([]byte(`{"Answer":[{"type":1,"data":"192.0.2.10"}]}`))
	}))
}

func (s *RunSuite) TearDownTest() { s.doh.Close() }

func (s *RunSuite) addRows(rows ...acceleration.AgentLog) {
	var list []acceleration.AgentLog
	for _, row := range rows {
		s.router.rows[row.Id] = row
		row.Details = nil
		list = append(list, row)
	}
	s.router.rowLists = [][]acceleration.AgentLog{list}
}

func (s *RunSuite) inbound() {
	s.cfg.Direction = person.Inbound
	s.router.listener = &fakeListener{steps: s.steps, session: s.session}
}

func (s *RunSuite) deps() deps {
	return deps{
		router:   s.router,
		person:   s.person,
		locator:  fakeLocator{},
		resolver: sipResolver{base: s.doh.URL, client: s.doh.Client()},
		lookup: func(_ context.Context, host string) ([]netip.Addr, error) {
			s.looked.add(host)
			return []netip.Addr{netip.MustParseAddr("192.0.2.1")}, nil
		},
		logger: newLogger("json", &s.out),
	}
}

func (s *RunSuite) run() error {
	return runCall(context.Background(), s.cfg, s.deps())
}

func (s *RunSuite) lines() []map[string]any {
	var lines []map[string]any
	for _, text := range strings.Split(strings.TrimSpace(s.out.String()), "\n") {
		var line map[string]any
		s.Require().NoError(json.Unmarshal([]byte(text), &line), text)
		lines = append(lines, line)
	}
	return lines
}

func (s *RunSuite) linesSaying(msg string) []map[string]any {
	var found []map[string]any
	for _, line := range s.lines() {
		if line["msg"] == msg {
			found = append(found, line)
		}
	}
	return found
}

func (s *RunSuite) TestAnOutboundCallRingsThePersonAndWatchesBothSides() {
	s.Require().NoError(s.run())

	s.Equal([]string{"start_call", "person_wait", "session_wait", "hangup", "close"}, s.steps.calls())
}

func (s *RunSuite) TestAnInboundCallDialsOnlyOnceTheWorkerIsRegistered() {
	s.inbound()

	s.Require().NoError(s.run())

	s.Equal([]string{"listen", "dial", "agent_answer", "person_wait", "session_wait", "hangup", "stop", "close"}, s.steps.calls())
}

func (s *RunSuite) TestTeardownHangsUpThenClosesTheSession() {
	s.Require().NoError(s.run())

	calls := s.steps.calls()
	s.Equal([]string{"hangup", "close"}, calls[len(calls)-2:])
}

func (s *RunSuite) TestASessionThatAlreadyEndedClosesQuietly() {
	s.session.closeErr = websocket.ErrCloseSent

	s.Require().NoError(s.run())

	s.Empty(s.linesSaying("could not close the agent's session"))
}

func (s *RunSuite) TestASessionThatFailsToCloseIsLogged() {
	s.session.closeErr = errors.New("router unreachable")

	s.Require().NoError(s.run())

	s.Len(s.linesSaying("could not close the agent's session"), 1)
}

func (s *RunSuite) TestTheWaitForCallSitesIsAnnounced() {
	s.Require().NoError(s.run())

	lines := s.linesSaying("waiting for the router's call_sites row")
	s.Require().Len(lines, 1)
	s.InDelta(callSitesWait.Seconds(), lines[0]["max_wait_s"], 0)
}

func (s *RunSuite) TestTeardownRunsWhenTheRunIsCancelled() {
	s.person.stayOn = true
	s.person.waiting = make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)

	go func() { done <- runCall(ctx, s.cfg, s.deps()) }()
	<-s.person.waiting
	cancel()

	s.ErrorIs(<-done, context.Canceled)
	s.Contains(s.steps.calls(), "hangup")
	s.Contains(s.steps.calls(), "close")
}

func (s *RunSuite) TestTeardownHangsUpWhenTheAgentNeverStarted() {
	s.router.startErr = errors.New("the vendor refused the call")

	s.Error(s.run())

	s.Equal([]string{"start_call", "hangup"}, s.steps.calls())
}

func (s *RunSuite) TestNothingIsCalledWhenAPreconditionFails() {
	s.router.numbers = nil

	s.Error(s.run())

	s.False(s.steps.placedACall(), "steps: %v", s.steps.all())
}

func (s *RunSuite) TestAnInboundCallThatNeverArrivesNamesBothCauses() {
	s.inbound()
	s.router.listener.neverArrives = true
	s.shortenArrival()

	err := s.run()

	s.ErrorContains(err, "webhook")
	s.ErrorContains(err, "worker")
}

func (s *RunSuite) TestEveryLineCarriesTheCallID() {
	s.cfg.Region = "eu-west"

	s.Require().NoError(s.run())

	lines := s.lines()
	s.NotEmpty(lines)
	for _, line := range lines {
		s.NotEmpty(line["call_id"], "line: %v", line)
	}
}

func (s *RunSuite) TestTeardownStopsTheWorkerWhenNoCallArrived() {
	s.inbound()
	s.router.listener.neverArrives = true
	s.shortenArrival()

	s.Error(s.run())

	s.Equal([]string{"listen", "dial", "hangup", "stop"}, s.steps.calls())
}

func (s *RunSuite) TestTheSummaryIsWrittenOnEveryExit() {
	s.cfg.Region = "eu-west"
	s.router.startErr = errors.New("the vendor refused the call")
	s.person.hangupErr = errors.New("the vendor is down")

	s.Error(s.run())

	summaries := s.linesSaying("call summary")
	s.Require().Len(summaries, 1)
	s.Contains(summaries[0]["outcome"], "the vendor refused the call")
	s.Equal("eu-west", summaries[0]["region_label"])
	lines := s.lines()
	s.Equal("call summary", lines[len(lines)-1]["msg"], "the summary comes after teardown's lines")
	s.NotEmpty(s.linesSaying("could not hang up the person's call"))
}

func (s *RunSuite) TestTheRegionLabelIsLoggedAndChangesNothing() {
	s.Require().NoError(s.run())
	without := s.steps.calls()
	s.TearDownTest()
	s.SetupTest()
	s.cfg.Region = "eu-west"

	s.Require().NoError(s.run())

	s.Equal(without, s.steps.calls())
	labelled := false
	for _, line := range s.lines() {
		if line["msg"] != "call summary" && line["region_label"] == "eu-west" {
			labelled = true
		}
	}
	s.True(labelled, "a line with region_label eu-west")
	s.Require().NotEmpty(s.person.calls)
	s.Equal("eu-west", s.person.calls[0].RegionLabel)
}

func (s *RunSuite) TestTheCoreNeverResolvesThroughTheNetwork() {
	s.Require().NoError(s.run())

	s.ElementsMatch([]string{"sfu.example.com", "router.example.com"}, s.looked.all())
	s.Equal([]string{"sip.example.com"}, s.resolved.all())
}

func (s *RunSuite) shortenArrival() {
	was := inboundArrivalWait
	inboundArrivalWait = 50 * time.Millisecond
	s.T().Cleanup(func() { inboundArrivalWait = was })
}

func (s *RunSuite) TestHopsAreStillLoggedWhenThePersonsCallFailedToBeRead() {
	s.person.waitErr = errors.New("the vendor is down")

	err := s.run()

	s.ErrorContains(err, "the vendor is down")
	s.NotEmpty(s.linesSaying("hop"))
	s.NotEmpty(s.linesSaying("the call did not end cleanly; reading its sites anyway"))
	summaries := s.linesSaying("call summary")
	s.Require().Len(summaries, 1)
	s.Equal("sip.example.com", summaries[0]["sip_host"])
}

func (s *RunSuite) TestHopsAreNotReadOnceTheRunIsStopped() {
	s.person.stayOn = true
	s.person.waiting = make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)

	go func() { done <- runCall(ctx, s.cfg, s.deps()) }()
	<-s.person.waiting
	cancel()

	s.ErrorIs(<-done, context.Canceled)
	s.Empty(s.linesSaying("hop"))
}

func (s *RunSuite) TestTheConnectTimeLeavesOutTheChecks() {
	s.router.checkDelay = 200 * time.Millisecond

	s.Require().NoError(s.run())

	summaries := s.linesSaying("call summary")
	s.Require().Len(summaries, 1)
	s.Less(summaries[0]["connect_ms"], float64(200))
}

func (s *RunSuite) TestASlowJoinIsNotBlamedOnTheWebhook() {
	s.inbound()
	s.router.listener.joinErr = context.DeadlineExceeded

	err := s.run()

	s.ErrorIs(err, context.DeadlineExceeded)
	s.NotContains(err.Error(), "webhook")
}

func (s *RunSuite) TestAPersonSideThatIsNotReadyPlacesNothing() {
	s.person.readyErr = errors.New("TWILIO_PERSON_AUTH_TOKEN is not set")

	err := s.run()

	s.ErrorContains(err, "TWILIO_PERSON_AUTH_TOKEN is not set")
	s.False(s.steps.placedACall(), "steps: %v", s.steps.all())
	s.Equal([]string{"ready"}, s.steps.all(), "checked before the router is asked anything")
}

func (s *RunSuite) TestNoRunStartsOnceTheToolIsStopped() {
	s.cfg.Runs = 2
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := runCalls(ctx, s.cfg, s.deps())

	s.ErrorIs(err, context.Canceled)
	s.Empty(s.steps.all())
}
