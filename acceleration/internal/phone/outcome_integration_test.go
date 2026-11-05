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
	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
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
	// asked is every (customer, session) the service asked to end because nobody answered.
	asked [][2]string
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
	s.redis, err = redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
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
	list, err := pricing.NewList("vendor", pricing.AllAtCost(prices.Vendor().Entries), slog.New(slog.DiscardHandler))
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
	s.asked = nil
	s.service.OnUnanswered(func(_ context.Context, customerID, sessionID string) {
		s.asked = append(s.asked, [2]string{customerID, sessionID})
	})
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

// placeDetecting places a call that asks for answering machine detection.
func (s *OutcomeSuite) placeDetecting() Placed {
	placed, err := s.service.Call(s.ctx, CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123", DetectMachine: true,
	})
	s.Require().NoError(err)
	return placed
}

// machineToken is the token the placed call's machine URL carries.
func (s *OutcomeSuite) machineToken() string {
	return strings.TrimPrefix(s.vendor.dialed.MachineURL, "https://router.example/v1/phone/machine/")
}

// told records every report the service passes to its machine hook.
func (s *OutcomeSuite) told() *[]string {
	var got []string
	s.service.OnMachine(func(_ context.Context, customerID, sessionID string, machine Machine) {
		got = append(got, customerID+" "+sessionID+" "+string(machine))
	})
	return &got
}

func (s *OutcomeSuite) TestACallThatAsksForDetectionIsGivenAMachineURL() {
	placed := s.placeDetecting()

	token, ok := strings.CutPrefix(s.vendor.dialed.MachineURL, "https://router.example/v1/phone/machine/")
	s.Require().True(ok, s.vendor.dialed.MachineURL)
	byToken, err := s.service.Outcomes().ByToken(s.ctx, token)
	s.Require().NoError(err)
	s.Equal(placed.CallID, byToken.CallID, "the url names this call")
}

func (s *OutcomeSuite) TestACallThatDoesNotAskForDetectionIsGivenNoMachineURL() {
	s.place()

	s.Empty(s.vendor.dialed.MachineURL)
}

func (s *OutcomeSuite) TestAVendorThatCannotDetectIsGivenNoMachineURL() {
	s.build(silent{s.vendor}, "https://router.example")

	s.placeDetecting()

	s.Empty(s.vendor.dialed.MachineURL)
}

func (s *OutcomeSuite) TestWithoutAPublicURLNoMachineURLIsSent() {
	s.build(s.vendor, "")

	s.placeDetecting()

	s.Empty(s.vendor.dialed.MachineURL)
}

func (s *OutcomeSuite) TestAMachineReportTellsTheHookWhoPickedUpWhichCall() {
	told := s.told()
	placed := s.placeDetecting()

	s.Require().NoError(s.service.Machine(s.ctx, s.machineToken(), "application/x-www-form-urlencoded", []byte("machine=greeting_ended")))

	s.Equal([]string{s.customer + " " + placed.CallID + " greeting_ended"}, *told)
}

func (s *OutcomeSuite) TestAMachineReportForAnUnknownTokenIsRefused() {
	told := s.told()

	for _, token := range []string{uuid.NewString(), "not-a-token", ""} {
		err := s.service.Machine(s.ctx, token, "application/x-www-form-urlencoded", []byte("machine=human"))
		s.ErrorIs(err, ErrUnknownStatus, token)
	}
	s.Empty(*told)
}

func (s *OutcomeSuite) TestAMachineReportThatCannotBeReadIsRefusedAndTellsNobody() {
	told := s.told()
	s.placeDetecting()

	err := s.service.Machine(s.ctx, s.machineToken(), "application/x-www-form-urlencoded", []byte("nothing=here"))

	s.ErrorIs(err, ErrUnreadableStatus)
	s.Empty(*told)
}

func (s *OutcomeSuite) TestAMachineReportToACallOfAVendorThatCannotDetectIsRefused() {
	s.placeDetecting()
	token := s.machineToken()
	s.build(silent{s.vendor}, "https://router.example")

	err := s.service.Machine(s.ctx, token, "application/x-www-form-urlencoded", []byte("machine=human"))

	s.ErrorIs(err, ErrUnknownStatus)
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

func (s *OutcomeSuite) TestACallCancelledBeforeItWasSentWasNeverPlaced() {
	ctx, cancel := context.WithCancel(s.ctx)
	cancel()

	_, err := s.service.Call(ctx, CallRequest{Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123"})

	s.ErrorIs(err, ErrNotPlaced)
	s.Empty(s.vendor.dialed.To, "the vendor was never asked")
}

func (s *OutcomeSuite) TestACallCancelledWhileTheVendorWasDialingMayHaveRung() {
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	s.build(&hangingUp{stub: s.vendor, cancel: cancel}, "https://router.example")

	_, err := s.service.Call(ctx, CallRequest{Owner: routing.Owner{CustomerID: s.customer}, From: s.e164, To: "+15125550123"})

	s.Require().Error(err)
	s.NotErrorIs(err, ErrNotPlaced, "the vendor had the request, so the phone may be ringing")
}

// hangingUp is a vendor whose dial is cut off by its caller going away mid-request.
type hangingUp struct {
	*stub
	cancel context.CancelFunc
}

func (h *hangingUp) Dial(ctx context.Context, outbound Outbound) (Dialed, error) {
	h.stub.dialed = outbound
	h.cancel()
	return Dialed{}, ctx.Err()
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

func (s *OutcomeSuite) TestATokenThatIsNotAUUIDChangesNothing() {
	placed := s.place()

	err := s.report("not-a-token", CallBusy, "busy")

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

// asksOnceToEnd checks a final result nobody answered asks once for the placed call's session
// to end.
func (s *OutcomeSuite) asksOnceToEnd(result CallResult, word string) {
	placed, token := s.placeReported()

	s.Require().NoError(s.report(token, result, word))

	s.Equal([][2]string{{s.customer, placed.SessionID}}, s.asked)
}

func (s *OutcomeSuite) TestABusyCallAsksForItsSessionToEnd() {
	s.asksOnceToEnd(CallBusy, "busy")
}

func (s *OutcomeSuite) TestANoAnswerCallAsksForItsSessionToEnd() {
	s.asksOnceToEnd(CallNoAnswer, "no-answer")
}

func (s *OutcomeSuite) TestAFailedCallAsksForItsSessionToEnd() {
	s.asksOnceToEnd(CallFailed, "failed")
}

func (s *OutcomeSuite) TestACanceledCallAsksForItsSessionToEnd() {
	s.asksOnceToEnd(CallCanceled, "canceled")
}

func (s *OutcomeSuite) TestACompletedCallAsksNothing() {
	_, token := s.placeReported()

	s.Require().NoError(s.report(token, CallCompleted, "completed"))

	s.Empty(s.asked)
}

func (s *OutcomeSuite) TestANonFinalReportAsksNothing() {
	_, token := s.placeReported()

	s.Require().NoError(s.report(token, "", "ringing"))

	s.Empty(s.asked)
}

func (s *OutcomeSuite) TestABusyAfterThePhoneJoinedAsksNothing() {
	placed, token := s.placeReported()
	s.answered(placed)

	s.Require().NoError(s.report(token, CallBusy, "busy"))

	s.Empty(s.asked)
}

func (s *OutcomeSuite) TestAnInboundCallAsksNothing() {
	callID := "call-" + uuid.NewString()
	s.arrived(callID)
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, defaultCallType, callID, "session-1", "caller-1",
		"+15125550199", time.Time{}))
	s.clock.advance(time.Minute)

	s.ended(defaultCallType, callID)

	s.Empty(s.asked)
}

// callsCounted is what router_phone_call_outcomes_total holds for telnyx, by direction and
// outcome, read off the registry /metrics serves. No other test in this package counts calls
// while a suite runs, so a test reads a change from it.
func (s *OutcomeSuite) callsCounted() map[[2]string]float64 { return s.callsCountedAt("telnyx") }

// callsCountedAt is callsCounted for vendor.
func (s *OutcomeSuite) callsCountedAt(vendor string) map[[2]string]float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	counted := map[[2]string]float64{}
	for _, family := range families {
		if family.GetName() != "router_phone_call_outcomes_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["vendor"] == vendor {
				counted[[2]string{labels["direction"], labels["outcome"]}] = metric.GetCounter().GetValue()
			}
		}
	}
	return counted
}

// countedSince is what the calls counted for telnyx grew by since before.
func (s *OutcomeSuite) countedSince(before map[[2]string]float64) map[[2]string]float64 {
	return s.countedAtSince("telnyx", before)
}

// countedAtSince is countedSince for vendor.
func (s *OutcomeSuite) countedAtSince(vendor string, before map[[2]string]float64) map[[2]string]float64 {
	grew := map[[2]string]float64{}
	for series, now := range s.callsCountedAt(vendor) {
		if now != before[series] {
			grew[series] = now - before[series]
		}
	}
	return grew
}

func (s *OutcomeSuite) TestABusyPlacedCallIsCountedOnceWhenItsCallEnds() {
	before := s.callsCounted()
	placed, token := s.placeReported()
	s.Require().NoError(s.report(token, CallBusy, "busy"))

	s.ended(placed.CallType, placed.CallID)
	s.ended(placed.CallType, placed.CallID)

	s.Equal(map[[2]string]float64{{"outbound", "busy"}: 1}, s.countedSince(before))
}

func (s *OutcomeSuite) TestAnAnsweredPlacedCallIsCountedOnceWhenItsPhoneLeavesAndTheCallEnds() {
	before := s.callsCounted()
	placed, token := s.placeReported()
	s.answered(placed)
	s.Require().NoError(s.report(token, CallCompleted, "completed"))
	s.clock.advance(time.Minute)

	s.left(placed)
	s.ended(placed.CallType, placed.CallID)

	s.Equal(map[[2]string]float64{{"outbound", "completed"}: 1}, s.countedSince(before))
}

func (s *OutcomeSuite) TestAnArrivingCallIsCountedInboundWhenItEnds() {
	before := s.callsCounted()
	callID := "call-" + uuid.NewString()
	s.arrived(callID)
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, defaultCallType, callID, "session-1", "caller-1",
		"+15125550199", time.Time{}))
	s.clock.advance(time.Minute)

	s.ended(defaultCallType, callID)
	s.ended(defaultCallType, callID)

	s.Equal(map[[2]string]float64{{"inbound", "completed"}: 1}, s.countedSince(before))
}

func (s *OutcomeSuite) TestATransferIsNotCountedBesideThePlacedCall() {
	placed := s.place()
	before := s.callsCounted()
	s.transfer(placed.CallID, "+15125550124")
	s.Require().NoError(s.service.Connect(s.ctx, everyApp, placed.CallType, placed.CallID, "session-1", "human-1",
		"+15125550124", time.Time{}))
	s.clock.advance(time.Minute)

	s.Require().NoError(s.service.Left(s.ctx, everyApp, placed.CallType, placed.CallID, "human-1", time.Time{}))

	s.Empty(s.countedSince(before))
}

// A join refused on its first delivery and delivered again after the call ended revives the leg,
// and the stored outcome becomes completed; the call was counted when its leg first ended, under
// what was known then, and is not counted again.
func (s *OutcomeSuite) TestAJoinDeliveredAgainAfterTheCallEndedDoesNotCountTheCallAgain() {
	before := s.callsCounted()
	placed := s.place()
	s.clock.advance(10 * time.Second)
	joined := s.clock.now()
	s.clock.advance(10 * time.Second)
	s.ended(placed.CallType, placed.CallID)

	s.Require().NoError(s.service.Connect(s.ctx, everyApp, placed.CallType, placed.CallID, "session-1", "phone-1",
		"+15125550123", joined))

	s.Equal(outcome.Completed, s.read(placed.CallID).Outcome)
	s.Equal(map[[2]string]float64{{"outbound", "no_answer"}: 1}, s.countedSince(before))
}

func (s *OutcomeSuite) TestAnArrivingCallItsAgentLeftOnTheLineIsCountedDropped() {
	before := s.callsCounted()
	callID := "call-" + uuid.NewString()
	s.arrived(callID)
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: callID, CustomerID: s.customer, CallType: defaultCallType, CallID: callID,
	}))
	s.Require().NoError(s.store.CloseSession(s.ctx, callID, s.clock.now(), ""))
	s.clock.advance(time.Minute)

	s.ended(defaultCallType, callID)

	s.Equal(map[[2]string]float64{{"inbound", "dropped"}: 1}, s.countedSince(before))
}

func (s *OutcomeSuite) TestACallOnACustomersTrunkIsCountedUnderTheTrunk() {
	before := s.callsCountedAt(SIPTrunkVendor)
	callID := "call-" + uuid.NewString()
	s.service.OpenLeg(s.ctx, routing.Owner{CustomerID: s.customer, Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound}},
		store.CallLeg{CallType: defaultCallType, CallID: callID, StreamSessionID: "session-1", Vendor: SIPTrunkVendor,
			Direction: store.Inbound, LocalNumber: s.e164, Country: "US"}, true)
	s.clock.advance(time.Minute)

	s.ended(defaultCallType, callID)

	s.Equal(map[[2]string]float64{{"inbound", "completed"}: 1}, s.countedAtSince(SIPTrunkVendor, before))
}
