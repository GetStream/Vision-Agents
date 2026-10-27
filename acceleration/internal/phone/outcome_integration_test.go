//go:build integration

package phone

import (
	"context"
	"errors"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// OutcomeSuite is what a placed call keeps for its outcome, in real Postgres: the row, the
// status URL its vendor is given, and what Stream says of its phone. The vendor is a stub named
// telnyx, and the service's clock is the suite's.
type OutcomeSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	redis    rueidis.Client
	service  *Service
	vendor   *stub
	clock    *meteringClock
	customer string
	e164     string
}

func TestOutcomeSuite(t *testing.T) { suite.Run(t, new(OutcomeSuite)) }

func (s *OutcomeSuite) SetupSuite() {
	dsn, address := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "phone_outcome"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.redis.Close)
}

func (s *OutcomeSuite) SetupTest() {
	s.vendor = &stub{vendor: "telnyx"}
	s.clock = &meteringClock{at: time.Now().UTC().Truncate(time.Second)}
	s.build(s.vendor, "https://router.example")
}

// build builds the service over a fresh customer whose number is at vendor, reachable at
// publicURL.
func (s *OutcomeSuite) build(vendor Provider, publicURL string) {
	list, err := pricing.NewList("vendor", prices.Vendor().Entries, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	recorder := routing.NewRecorder(routing.Phone, s.store, nil, list, slog.New(slog.DiscardHandler))
	s.T().Cleanup(recorder.Close)
	s.customer = "customer-" + uuid.NewString()
	s.e164 = aNumber()
	config, err := DefaultConfig()
	s.Require().NoError(err)
	registry := NewRegistry(config)
	registry.Register("telnyx", func() (Provider, error) { return vendor, nil })
	declared, _ := config.Lookup("telnyx")
	for _, name := range declared.Credentials {
		s.T().Setenv(name, "set")
	}
	deployment := chattest.NewServer(s.T())
	s.service, err = NewService(ServiceOptions{
		Registry: registry, Store: s.store, Recorder: recorder, PublicURL: publicURL, Now: s.clock.now,
		Apps:   &testApps{deployment: deployment, own: map[string]*chattest.Server{}, pins: map[string]int64{}, blocked: map[string]bool{}},
		Gate:   dlc.NewGate(s.store, s.redis, dlc.Sandbox{}, nil),
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: s.e164, Vendor: "telnyx", Country: "US", CustomerID: s.customer, PurchasedAt: s.clock.now(),
	}))
}

// place places a call from the suite's number.
func (s *OutcomeSuite) place() Placed {
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123",
	})
	s.Require().NoError(err)
	return placed
}

func (s *OutcomeSuite) TestAPlacedCallIsGivenItsStatusURL() {
	placed := s.place()

	statusURL := s.vendor.dialed.StatusURL
	token, ok := strings.CutPrefix(statusURL, "https://router.example/v1/phone/status/")
	s.Require().True(ok, statusURL)
	call, err := s.service.Outcomes().Read(s.ctx, s.customer, placed.CallID)
	s.Require().NoError(err)
	s.Equal("telnyx", call.Vendor)
	s.Equal(s.e164, call.Local)
	s.Equal("+15125550123", call.Remote)
	s.Equal(placed.CallType, call.CallType)
	s.True(call.Reports)
	s.Equal(placed.VendorCallID, call.VendorCallID)
	byToken, err := s.service.Outcomes().ByToken(s.ctx, token)
	s.Require().NoError(err)
	s.Equal(call.Key, byToken.Key, "the url names this call")
}

func (s *OutcomeSuite) TestWithoutAPublicURLNoStatusURLIsSent() {
	s.build(s.vendor, "")

	placed := s.place()

	s.Empty(s.vendor.dialed.StatusURL)
	call, err := s.service.Outcomes().Read(s.ctx, s.customer, placed.CallID)
	s.Require().NoError(err)
	s.False(call.Reports, "nothing can post to a router nobody can reach")
}

func (s *OutcomeSuite) TestAVendorThatReportsNothingGetsNoStatusURL() {
	s.build(silent{s.vendor}, "https://router.example")

	placed := s.place()

	s.Empty(s.vendor.dialed.StatusURL)
	call, err := s.service.Outcomes().Read(s.ctx, s.customer, placed.CallID)
	s.Require().NoError(err)
	s.False(call.Reports)
}

func (s *OutcomeSuite) TestADialThatFailsLeavesNoOutcome() {
	s.vendor.err = errors.New("telnyx is having a day")
	sessionID := uuid.NewString()

	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123", SessionID: sessionID,
	})
	s.Require().Error(err)

	_, err = s.service.Outcomes().Read(s.ctx, s.customer, sessionID)
	s.ErrorIs(err, outcome.ErrUnknownCall)
}

// placeReported places a call and returns it with the token its status URL carries.
func (s *OutcomeSuite) placeReported() (Placed, string) {
	placed := s.place()
	return placed, strings.TrimPrefix(s.vendor.dialed.StatusURL, "https://router.example/v1/phone/status/")
}

// report posts result and word to the status URL of the call token names.
func (s *OutcomeSuite) report(token string, result CallResult, word string) error {
	body := url.Values{"result": {string(result)}, "word": {word}}.Encode()
	return s.service.ReportStatus(s.ctx, token, "application/x-www-form-urlencoded", []byte(body))
}

// answered is the placed call's phone joining its Stream call, as the participant event says.
func (s *OutcomeSuite) answered(placed Placed) {
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, placed.CallType, placed.CallID, "session-1", "phone-1",
		"+15125550123", time.Time{}))
}

// left is the placed call's phone leaving its Stream call, as the participant event says.
func (s *OutcomeSuite) left(placed Placed) {
	s.Require().NoError(s.service.Left(s.ctx, everyApp, placed.CallType, placed.CallID, "phone-1", time.Time{}))
}

// ended is the Stream call ending, as session_ended says.
func (s *OutcomeSuite) ended(callType, callID string) {
	s.Require().NoError(s.service.EndLegs(s.ctx, everyApp, callType, callID, time.Time{}))
}

// transfer rings human into the call, from the suite's number.
func (s *OutcomeSuite) transfer(callID, human string) {
	_, err := s.service.Transfer(s.ctx, TransferRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: human, CallID: callID,
	})
	s.Require().NoError(err)
}

// arrived opens the inbound leg of a call to the suite's number, as a session_started that
// lists no caller does.
func (s *OutcomeSuite) arrived(callID string) {
	s.service.OpenLeg(s.ctx, routing.Owner{CustomerID: s.customer, Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound}},
		store.CallLeg{CallType: defaultCallType, CallID: callID, StreamSessionID: "session-1", Vendor: "telnyx",
			Direction: store.Inbound, LocalNumber: s.e164, Country: "US"}, true)
}

// read reads back how the placed call ended.
func (s *OutcomeSuite) read(callID string) outcome.Call {
	call, err := s.service.Outcomes().Read(s.ctx, s.customer, callID)
	s.Require().NoError(err)
	return call
}

func (s *OutcomeSuite) TestAVendorsBusyIsTheCallsOutcome() {
	placed, token := s.placeReported()

	s.Require().NoError(s.report(token, CallBusy, "busy"))

	call := s.read(placed.CallID)
	s.Equal(outcome.Busy, call.Outcome)
	s.Equal("busy", call.VendorStatus)
	s.True(call.NobodyAnswered())
}

func (s *OutcomeSuite) TestASecondFinalReportDoesNotOverwriteTheFirst() {
	placed, token := s.placeReported()
	s.Require().NoError(s.report(token, CallBusy, "busy"))

	s.Require().NoError(s.report(token, CallNoAnswer, "no-answer"))

	call := s.read(placed.CallID)
	s.Equal(outcome.Busy, call.Outcome)
	s.Equal("busy", call.VendorStatus)
}

func (s *OutcomeSuite) TestANoAnswerAfterThePhoneJoinedIsCompleted() {
	placed, token := s.placeReported()
	s.answered(placed)

	s.Require().NoError(s.report(token, CallNoAnswer, "no-answer"))

	call := s.read(placed.CallID)
	s.Equal(outcome.Completed, call.Outcome)
	s.False(call.NobodyAnswered())
}

func (s *OutcomeSuite) TestAReportBeforeTheSessionOpensIsStored() {
	sessionID := uuid.NewString()
	_, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123", SessionID: sessionID,
	})
	s.Require().NoError(err)
	token := strings.TrimPrefix(s.vendor.dialed.StatusURL, "https://router.example/v1/phone/status/")

	s.Require().NoError(s.report(token, CallFailed, "failed"))

	s.Equal(outcome.Failed, s.read(sessionID).Outcome)
}

func (s *OutcomeSuite) TestANonFinalReportDecidesNothing() {
	placed, token := s.placeReported()

	s.Require().NoError(s.report(token, "", "ringing"))

	call := s.read(placed.CallID)
	s.Empty(call.Outcome)
	s.Nil(call.DecidedAt)
	s.Nil(call.EndedAt)
	s.Equal("ringing", call.VendorStatus)
}

func (s *OutcomeSuite) TestABadTokenChangesNothing() {
	placed := s.place()

	err := s.report(uuid.NewString(), CallBusy, "busy")

	s.ErrorIs(err, ErrUnknownStatus)
	s.Empty(s.read(placed.CallID).Outcome)
}

func (s *OutcomeSuite) TestACallAnsweredAndEndedIsCompleted() {
	placed := s.place()
	s.clock.advance(20 * time.Second)
	s.answered(placed)
	s.clock.advance(61 * time.Second)

	s.left(placed)

	call := s.read(placed.CallID)
	s.Equal(outcome.Completed, call.Outcome)
	s.Require().NotNil(call.AnsweredAt)
	duration, ok := call.Duration()
	s.True(ok)
	s.Equal(61*time.Second, duration, "from the join, not the dial")
}

func (s *OutcomeSuite) TestACallNeverAnsweredIsNoAnswerWhenItEnds() {
	placed := s.place()
	s.clock.advance(time.Minute)

	s.ended(placed.CallType, placed.CallID)

	call := s.read(placed.CallID)
	s.Equal(outcome.NoAnswer, call.Outcome)
	s.True(call.NobodyAnswered())
}

func (s *OutcomeSuite) TestACallAtAVendorThatReportsNothingIsUnknownWhenUnanswered() {
	s.build(silent{s.vendor}, "https://router.example")
	placed := s.place()
	s.clock.advance(time.Minute)

	s.ended(placed.CallType, placed.CallID)

	s.Equal(outcome.Unknown, s.read(placed.CallID).Outcome)
}

func (s *OutcomeSuite) TestAVendorsBusyStaysWhenTheUnansweredCallEnds() {
	placed, token := s.placeReported()
	s.Require().NoError(s.report(token, CallBusy, "busy"))

	s.ended(placed.CallType, placed.CallID)

	call := s.read(placed.CallID)
	s.Equal(outcome.Busy, call.Outcome)
	s.NotNil(call.EndedAt)
}

func (s *OutcomeSuite) TestAVendorsBusyBecomesCompletedWhenThePhoneJoins() {
	placed, token := s.placeReported()
	s.Require().NoError(s.report(token, CallBusy, "busy"))

	s.answered(placed)

	call := s.read(placed.CallID)
	s.Equal(outcome.Completed, call.Outcome)
	s.Equal("busy", call.VendorStatus)
}

func (s *OutcomeSuite) TestATransferLegDoesNotChangeThePlacedCall() {
	placed := s.place()
	s.transfer(placed.CallID, "+15125550124")
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, placed.CallType, placed.CallID, "session-1", "human-1",
		"+15125550124", time.Time{}))
	s.clock.advance(time.Minute)

	s.Require().NoError(s.service.Left(s.ctx, everyApp, placed.CallType, placed.CallID, "human-1", time.Time{}))

	call := s.read(placed.CallID)
	s.Nil(call.AnsweredAt, "the human the agent transferred to is not the person the call rang")
	s.Nil(call.EndedAt)
	s.Empty(call.Outcome)
}

func (s *OutcomeSuite) TestAnInboundCallHasNoRow() {
	callID := "call-" + uuid.NewString()
	s.arrived(callID)
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, defaultCallType, callID, "session-1", "caller-1",
		"+15125550199", time.Time{}))
	s.clock.advance(time.Minute)

	s.ended(defaultCallType, callID)

	_, err := s.service.Outcomes().Read(s.ctx, s.customer, callID)
	s.ErrorIs(err, outcome.ErrUnknownCall)
}

func (s *OutcomeSuite) TestAStatusAfterTheTokenIsSweptIs404AndKeepsTheOutcome() {
	placed, token := s.placeReported()
	s.answered(placed)
	s.clock.advance(time.Minute)
	s.ended(placed.CallType, placed.CallID)
	s.clock.advance(statusTokenLifetime + time.Minute)

	swept, err := s.service.SweepStatusTokens(s.ctx)
	s.Require().NoError(err)
	s.Positive(swept)

	s.ErrorIs(s.report(token, CallBusy, "busy"), ErrUnknownStatus)
	call := s.read(placed.CallID)
	s.Equal(outcome.Completed, call.Outcome)
	s.NotNil(call.EndedAt)
}
