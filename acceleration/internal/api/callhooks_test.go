//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// dropped is how long a hook that should wake nobody is given to prove it. A hook that
// dispatches does so before it answers, so this only has to outlast the answer.
const dropped = 200 * time.Millisecond

type CallHooksSuite struct {
	RouterSuite

	// e164 is the number rung in a test, and callID the call it arrives as. Both are
	// unique, because a number is held by one customer at a time.
	e164   string
	callID string
	// numberTags are the cost labels hold records the number with, and numberVendor the
	// vendor it is bought from.
	numberTags   map[string]string
	numberVendor string
	// stats stands in for Stream's call stats, so no test in the suite reaches Stream.
	stats *recordedStats
	// logged is what the router logged, at debug and up.
	logged *lockedLog
}

func TestCallHooksSuite(t *testing.T) {
	runSuite(t, new(CallHooksSuite))
}

func (s *CallHooksSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	// Stream Chat delivers messages too, which call events have nothing to do with.
	s.chatInboundKeys = []ed25519.PublicKey{public(ed25519.NewKeyFromSeed([]byte("the-current-delivery-key-seed-32")))}
	s.sessionTelephony = true
	s.RouterSuite.SetupSuite()
}

func (s *CallHooksSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.e164 = s.utils.number()
	s.callID = "phone-" + s.e164
	s.numberTags = nil
	s.numberVendor = "telnyx"
	s.stats = &recordedStats{asked: make(chan string, 8), answer: &getstream.QueryCallSessionParticipantStatsResponse{}}
	s.router.readCallStats = s.stats.read
	s.router.callStatsWaits = []time.Duration{0}
}

func (s *CallHooksSuite) TestAnArrivingCallReachesTheWorkerOfWhoeverHoldsTheNumber() {
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	select {
	case call := <-worker.Calls():
		s.Equal(s.callID, call.CallID)
		s.Equal("default", call.CallType)
		s.Equal(s.e164, call.CalledNumber, "the worker has to know which line rang")
		s.Equal("+15550001111", call.CallerNumber)
		s.Equal("support", call.Custom["line"])
		s.False(call.At.IsZero())
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestACallerWrittenWith00ReachesTheWorkerInE164() {
	s.Equal("+442079460018", s.callerReaching("sip-00442079460018"))
}

func (s *CallHooksSuite) TestACallerNotWrittenAsInternationalReachesTheWorkerAsItCame() {
	s.Equal("5550001111", s.callerReaching("sip-5550001111"),
		"a national number could be of any country, so it is not guessed at")
}

// Deployment mode acts on every delivery of a call event, as it always has: only a new message
// is recorded, for Stream Chat's delivery of it.
func (s *CallHooksSuite) TestACallDeliveredTwiceIsActedOnTwice() {
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))
	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	for range 2 {
		select {
		case <-worker.Calls():
		case <-time.After(settleFor):
			s.Fail("a delivery of the call was not acted on")
		}
	}
}

func (s *CallHooksSuite) TestACallToASuspendedAppIsNotAnsweredAndIsEnded() {
	s.hold("default", s.callID)
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.nothingReaches(worker.Calls(), "a suspended app's agent answered a call")
	s.Contains(s.logged.String(), "refused an arriving call to a suspended app; ending it")
	s.Eventually(func() bool { return strings.Contains(s.logged.String(), "could not end a refused arriving call") },
		5*time.Second, 20*time.Millisecond, "ended at Stream, which the suite's phone service has no credentials for")
}

// TestAnArrivingCallTheRouterRefusedIsCountedFailedWhenItsLegEnds catches a refused call, which
// no agent answered, counted completed when Stream's end of it arrives, and checks the refusal
// still bills the leg as base did: from the caller's arrival to the call's end.
func (s *CallHooksSuite) TestAnArrivingCallTheRouterRefusedIsCountedFailedWhenItsLegEnds() {
	// No other test of the package holds a number at this vendor, so its inbound series move
	// only here.
	s.numberVendor = "vonage"
	s.hold("default", s.callID)
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))
	failed, completed := s.inboundCalls("vonage", "failed"), s.inboundCalls("vonage", "completed")
	// Recent, so the legs are timed by the events, not by when they were delivered.
	arrived := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	ended := arrived.Add(30 * time.Second)

	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCallAt("default:"+s.callID, arrived)))
	s.Require().True(s.refused())
	s.Require().Eventually(func() bool {
		keys, err := s.live.Redis().Do(context.Background(), s.live.Redis().B().Keys().Pattern("phone:refused:*:"+s.callID+":session-1").Build()).AsStrSlice()
		return err == nil && len(keys) == 1
	}, settleFor, 10*time.Millisecond, "the refusal was never marked")
	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", fmt.Sprintf(
		`{"type":"call.session_ended","call_cid":"default:%s","session_id":"session-1","created_at":%q,`+
			`"call":{"cid":"default:%s","id":"%s","type":"default","custom":{}}}`,
		s.callID, ended.Format(time.RFC3339), s.callID, s.callID)))

	s.Equal(1.0, s.inboundCalls("vonage", "failed")-failed)
	s.Zero(s.inboundCalls("vonage", "completed") - completed)
	var legs []store.CallLeg
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&legs).Where("call_id = ?", s.callID).Scan(context.Background()))
	s.Require().Len(legs, 1)
	s.Equal(arrived, legs[0].StartedAt.UTC())
	s.Equal(ended, legs[0].EndedAt.UTC())
	s.Equal(store.LegEnded, legs[0].EndReason)
}

// TestALaterCallerInTheSameCallIsNotCountedRefused catches the refusal marked for the whole
// call: a number attached before per-caller calls puts every caller in the number's own call, each
// caller a Stream session with a leg of its own, so one refused caller made every later caller
// of the number count failed, answered or not.
func (s *CallHooksSuite) TestALaterCallerInTheSameCallIsNotCountedRefused() {
	s.numberVendor = "vonage"
	s.hold("default", s.callID)
	ctx := context.Background()
	s.Require().NoError(s.store.SuspendTelephony(ctx, &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))
	arrived := time.Now().UTC().Add(-2 * time.Minute).Truncate(time.Second)
	end := func(session string, at time.Time) {
		s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", fmt.Sprintf(
			`{"type":"call.session_ended","call_cid":"default:%s","session_id":%q,"created_at":%q,`+
				`"call":{"cid":"default:%s","id":"%s","type":"default","custom":{}}}`,
			s.callID, session, at.Format(time.RFC3339), s.callID, s.callID)))
	}
	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCallAt("default:"+s.callID, arrived)))
	s.Require().True(s.refused())
	s.Require().Eventually(func() bool {
		keys, err := s.live.Redis().Do(ctx, s.live.Redis().B().Keys().Pattern("phone:refused:*:"+s.callID+":*").Build()).AsStrSlice()
		return err == nil && len(keys) == 1
	}, settleFor, 10*time.Millisecond, "the refusal was never marked")
	end("session-1", arrived.Add(10*time.Second))
	s.Require().NoError(s.store.LiftTelephonySuspension(ctx, s.customerID()))

	failed, completed := s.inboundCalls("vonage", "failed"), s.inboundCalls("vonage", "completed")
	second := arrived.Add(time.Minute)
	body := strings.ReplaceAll(arrivingCallAt("default:"+s.callID, second), "session-1", "session-2")
	body = strings.ReplaceAll(body, `"s1"`, `"s2"`)
	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", body))
	end("session-2", second.Add(20*time.Second))

	var legs []store.CallLeg
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&legs).Where("call_id = ?", s.callID).Scan(ctx))
	s.Require().Len(legs, 2, "the second caller opened a leg of its own")
	s.Equal(1.0, s.inboundCalls("vonage", "completed")-completed, "the second caller counted completed")
	s.Zero(s.inboundCalls("vonage", "failed")-failed, "the second caller counted failed by the first caller's mark")
}

// inboundCalls is how many arriving calls at vendor were counted under outcome, read off the
// registry /metrics serves.
func (s *CallHooksSuite) inboundCalls(vendor, outcome string) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	want := map[string]string{"vendor": vendor, "direction": "inbound", "outcome": outcome}
	for _, family := range families {
		if family.GetName() != "router_phone_call_outcomes_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if maps.Equal(labels, want) {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func (s *CallHooksSuite) TestACallOnACustomNamedLineStillFindsItsOwner() {
	// A number attached to a call of its own is the case the stored binding exists for:
	// nothing about "the-support-line" says which number it belongs to.
	line := "line-" + s.utils.uuid()
	s.hold("support", line)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("support:"+line))

	select {
	case call := <-worker.Calls():
		s.Equal(line, call.CallID)
		s.Equal("support", call.CallType)
		s.Equal(s.e164, call.CalledNumber)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestAnotherCustomersWorkerIsNotGivenTheCall() {
	// Two customers' workers are two rotations, and a call is one customer's.
	s.hold("default", s.callID)
	somebodyElse, release := s.dispatch.Register(s.utils.uuid(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.nothingReaches(somebodyElse.Calls(), "a call was misrouted")
}

func (s *CallHooksSuite) TestACallOnANumberNobodyHoldsIsAcceptedAndDropped() {
	// Every video call in the app arrives at this hook too. Retrying would not make one
	// answerable, so it is accepted and nothing is woken.
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:standup-"+s.utils.uuid()))

	s.nothingReaches(worker.Calls(), "a video call was answered")
}

func (s *CallHooksSuite) TestACallOnAReleasedNumberIsNotAnswered() {
	// The number is gone, so whoever holds it now is not this customer.
	s.hold("default", s.callID)
	s.Require().NoError(s.store.ReleaseNumber(
		context.Background(), s.customerID(), s.e164, time.Now().UTC()))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.nothingReaches(worker.Calls(), "a released number was answered")
}

func (s *CallHooksSuite) TestAnUnsignedCallEventIsRefused() {
	status, _ := s.deliver("/v1/phone/hooks/stream", s.arriving("default:"+s.callID), "")

	s.Equal(http.StatusUnauthorized, status)
}

func (s *CallHooksSuite) TestACallEventSignedWithTheWrongSecretIsRefused() {
	body := s.arriving("default:" + s.callID)

	status, _ := s.deliver("/v1/phone/hooks/stream", body, sign(body, "somebody-elses-secret"))

	s.Equal(http.StatusUnauthorized, status)
}

// arriveRingingNumber delivers a call.session_started whose call says which number it rang.
func (s *CallHooksSuite) arriveRingingNumber() int {
	return s.arriveRingingNumberOn("default")
}

// refused reports whether the router logged refusing the test's call.
func (s *CallHooksSuite) refused() bool {
	return s.refusedCall(s.callID)
}

// refusedCall is refused for one call. It reads no field a later test sets, so a check still
// running when its test ends cannot race with the next one.
func (s *CallHooksSuite) refusedCall(callID string) bool {
	for _, line := range strings.Split(s.logged.String(), "\n") {
		if strings.Contains(line, "refused an arriving call") && strings.Contains(line, callID) {
			return true
		}
	}
	return false
}

// answerable names the call as Stream names a fresh number's caller's call (a uuid), which,
// unlike the suite's own name, a session can be named for.
func (s *CallHooksSuite) answerable() {
	s.callID = s.utils.uuid()
}

// answered waits for the agent answering the test's call, which joins after the hook answers.
func (s *CallHooksSuite) answered() *session.Session {
	var found *session.Session
	s.Require().Eventually(func() bool {
		var live bool
		found, live = s.router.sessions.Get(s.callID, session.Owner{CustomerID: s.customerID()})
		return live
	}, settleFor, 10*time.Millisecond, "no agent answered the call")
	return found
}

// arriveRingingNumberOn is arriveRingingNumber on a call of callType.
func (s *CallHooksSuite) arriveRingingNumberOn(callType string) int {
	return s.signedly("/v1/phone/hooks/stream", s.ringingNumber(callType))
}

// ringingNumber is the call.session_started of a call of callType that rang the number.
func (s *CallHooksSuite) ringingNumber(callType string) string {
	return strings.Replace(s.arriving(callType+":"+s.callID),
		`"custom": {"line": "support"}`, fmt.Sprintf(`"custom": {"line": "support", "phone_number": %q}`, s.e164), 1)
}

func (s *CallHooksSuite) TestAnArrivingCallTheRouterPlacedIsNotDispatched() {
	s.hold("default", s.callID)
	s.placed()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumber())

	s.nothingReaches(worker.Calls(), "a call the router placed was answered a second time")
}

func (s *CallHooksSuite) TestAnArrivingCallToANumberIsStillDispatched() {
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumber())

	select {
	case call := <-worker.Calls():
		s.Equal(s.callID, call.CallID)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestAnArrivingCallToANumberWithAnAgentIsAnsweredByIt() {
	s.answerable()
	s.numberTags = map[string]string{"line": "support", "team": "number"}
	s.hold("default", s.callID)
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("echo/local-model"),
		Tags: &map[string]string{"product": "voice", "team": "config"},
	}, &created))
	config := created.Id
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, config))

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("default"))

	found := s.answered()
	spec := found.Spec()
	s.Equal("default", spec.CallType, "the agent joins the call the caller is in")
	s.Equal(s.callID, spec.CallID)
	s.Equal(config, spec.ConfigID)
	s.True(spec.PersistConversation, "the conversation is kept, as a worker's session keeps it")
	s.Empty(spec.Caller.UserID, "a caller id is easy to fake, so it names no user")
	s.Require().NotNil(spec.Phone)
	s.Equal(s.e164, spec.Phone.Number)
	s.False(spec.Text)
	s.Equal("voice", spec.Tags["product"], "the config's tags are kept")
	s.Equal("support", spec.Tags["line"])
	s.Equal("number", spec.Tags["team"], "the number's tag wins where both name it")
}

// answeredWithCalls is the tools offered by the agent that answers a call to a number whose
// config has these calls settings.
func (s *CallHooksSuite) answeredWithCalls(calls *AgentCalls) []llm.Tool {
	s.answerable()
	s.hold("default", s.callID)
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("echo/local-model"), Calls: calls,
	}, &created))
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, created.Id))
	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("default"))
	return s.answered().ToolDefinitions()
}

func transferOf(tools []llm.Tool) (llm.Tool, bool) {
	for _, tool := range tools {
		if tool.Name == "transfer" {
			return tool, true
		}
	}
	return llm.Tool{}, false
}

func (s *CallHooksSuite) TestAConfigThatNeverSetCallsOffersTransferByNumber() {
	tool, ok := transferOf(s.answeredWithCalls(nil))

	s.Require().True(ok, "transfer is on unless a config turns it off")
	props, _ := tool.Parameters["properties"].(map[string]any)
	s.Contains(props, "to")
	s.NotContains(props, "destination")
}

func (s *CallHooksSuite) TestAConfigWithTransferOffOffersNoTransfer() {
	off := false

	_, ok := transferOf(s.answeredWithCalls(&AgentCalls{Transfer: &TransferSettings{Enabled: &off}}))

	s.False(ok)
}

func (s *CallHooksSuite) TestAConfigWithDestinationsOffersThemByNameAndNoNumber() {
	tool, ok := transferOf(s.answeredWithCalls(&AgentCalls{Transfer: &TransferSettings{Destinations: []TransferDestination{
		{Name: "Sales", Number: "+15550001111"},
		{Name: "Support", Number: "+15550002222"},
	}}}))

	s.Require().True(ok)
	props, _ := tool.Parameters["properties"].(map[string]any)
	s.NotContains(props, "to")
	destination, _ := props["destination"].(map[string]any)
	s.Require().NotNil(destination, "%v", tool.Parameters)
	s.ElementsMatch([]any{"Sales", "Support"}, destination["enum"])
}

// answeredOn is the session of the agent that answers a call to a number whose config
// talks on llm with these calls settings.
func (s *CallHooksSuite) answeredOn(llm string, calls *AgentCalls) *session.Session {
	s.answerable()
	s.hold("default", s.callID)
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo(llm), Calls: calls,
	}, &created))
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, created.Id))
	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("default"))
	return s.answered()
}

// shown reads a session as its backend does.
func (s *CallHooksSuite) shown(id string) Session {
	var shown Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &shown))
	return shown
}

// endedFor waits for the session id to show it ended for reason.
func (s *CallHooksSuite) endedFor(id string, reason SessionEndReason) {
	s.Eventually(func() bool {
		shown := s.shown(id)
		return shown.State == Ended && shown.EndReason != nil && *shown.EndReason == reason
	}, settleFor, 50*time.Millisecond, "the session never showed it ended for %s", reason)
}

// markedEnded reports whether the suite's Stream was asked to end the call for everybody.
func (s *CallHooksSuite) markedEnded(callType, callID string) bool {
	for _, request := range s.chat.Requests(suiteStreamKey) {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/video/call/"+callType+"/"+callID+"/mark_ended") {
			return true
		}
	}
	return false
}

func (s *CallHooksSuite) TestAnAgentThatHangsUpEndsThePhoneCall() {
	on := true
	found := s.answeredOn("hanging-up/hanging-up-model", &AgentCalls{EndCall: &EndCallSettings{Enabled: &on}})

	s.callerSays("that is all, thanks")

	s.Eventually(func() bool { return s.markedEnded("default", s.callID) }, settleFor, 20*time.Millisecond,
		"the phone call was left up after the agent hung up")
	s.endedFor(found.ID(), "agent_hangup")
}

func (s *CallHooksSuite) TestASessionShowsWhyItEnded() {
	found := s.answeredOn("echo/local-model", nil)
	s.Nil(s.shown(found.ID()).EndReason, "a running session gave a reason it ended")

	s.Require().True(s.router.sessions.EndUnanswered(s.customerID(), found.ID()))

	s.endedFor(found.ID(), "unanswered")
}

func (s *CallHooksSuite) TestAnArrivingCallDeliveredTwiceKeepsItsAgentOnTheLine() {
	s.answerable()
	s.hold("agent", s.callID)
	config := s.config("echo/local-model")
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, config))

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))
	s.answered()
	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	call := s.callID
	s.Never(func() bool { return s.refusedCall(call) }, dropped, 10*time.Millisecond,
		"a second delivery ended a call its agent is on")
	_, live := s.router.sessions.Get(s.callID, session.Owner{CustomerID: s.customerID()})
	s.True(live, "the agent answering the call was hung up on")
}

func (s *CallHooksSuite) TestACallToANumberWithAnAgentOnARouterWithNoSessionsIsRefused() {
	s.answerable()
	s.hold("agent", s.callID)
	config := s.config("echo/local-model")
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, config))
	sessions := s.router.sessions
	s.router.sessions = nil
	defer func() { s.router.sessions = sessions }()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	s.True(s.refused(), "a call nobody can answer is left ringing")
}

func (s *CallHooksSuite) TestARouterWithNoDispatchPoolStillAnswersWithTheAgent() {
	s.answerable()
	s.hold("agent", s.callID)
	config := s.config("echo/local-model")
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, config))
	pool := s.router.dispatch
	s.router.dispatch = nil
	defer func() { s.router.dispatch = pool }()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	s.answered()
}

func (s *CallHooksSuite) TestAnArrivingCallToANumberWithAnAgentIsNotGivenToAWaitingWorker() {
	s.answerable()
	s.hold("agent", s.callID)
	config := s.config("echo/local-model")
	s.Require().NoError(s.store.SetNumberAgent(context.Background(), s.customerID(), s.e164, config))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	s.nothingReaches(worker.Calls(), "a number with an agent was handed to a worker")
}

func (s *CallHooksSuite) TestAnArrivingCallToANumberWhoseAgentIsGoneIsRefused() {
	s.answerable()
	s.hold("agent", s.callID)
	config := s.config("echo/local-model")
	ctx := context.Background()
	s.Require().NoError(s.store.SetNumberAgent(ctx, s.customerID(), s.e164, config))
	s.Require().NoError(s.store.DeleteAgentConfig(ctx, s.customerID(), config))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumberOn("agent"))

	s.nothingReaches(worker.Calls(), "a number whose agent is gone fell back to a worker")
	s.Eventually(func() bool {
		return strings.Contains(s.logged.String(), "refused an arriving call to a number whose agent is gone")
	}, settleFor, 10*time.Millisecond)
	_, live := s.router.sessions.Get(s.callID, session.Owner{CustomerID: s.customerID()})
	s.False(live)
}

func (s *CallHooksSuite) TestAnArrivingCallToADetachedNumberIsNotDispatched() {
	// Detached, the number stays on its trunk for its calls out, so a caller can still reach
	// Stream until the vendor stops sending them.
	s.hold("default", s.callID)
	group := store.TrunkGroup{CustomerID: s.customerID(), Vendor: "telnyx"}
	s.Require().NoError(s.store.InTrunkGroup(context.Background(), group, func(_ context.Context, tx *store.TrunkGroupTx) error {
		return tx.Detach(s.e164)
	}))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arriveRingingNumber())

	s.nothingReaches(worker.Calls(), "a detached number was answered")
}

func (s *CallHooksSuite) TestATamperedCallEventIsRefused() {
	body := s.arriving("default:" + s.callID)
	signature := sign(body, suiteStreamSecret)
	tampered := strings.Replace(body, "+15550001111", "+15559998888", 1)

	status, _ := s.deliver("/v1/phone/hooks/stream", tampered, signature)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *CallHooksSuite) TestASessionEndedEventIsAccepted() {
	ended := fmt.Sprintf(`{"type":"call.session_ended","call_cid":"default:%s",`+
		`"session_id":"session-1","created_at":"2026-08-27T12:05:00Z",`+
		`"call":{"cid":"default:%s","id":"%s","type":"default","custom":{}}}`,
		s.callID, s.callID, s.callID)

	s.Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", ended))
}

func (s *CallHooksSuite) TestAnEventTypeThisVersionHasNeverHeardOfIsAccepted() {
	unknown := fmt.Sprintf(`{"type":"call.something_new","call_cid":"default:%s"}`, s.callID)

	s.Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", unknown),
		"a new event type must not look like an outage to Stream")
}

func (s *CallHooksSuite) TestSomethingThatIsNotACallEventIsRefused() {
	// Correctly signed, but there is no event in it to act on.
	s.Equal(http.StatusBadRequest, s.signedly("/v1/phone/hooks/stream", `{"not":"an event"}`))
}

func (s *CallHooksSuite) TestTheHookIgnoresWhoeverTheCallerClaimsToBe() {
	// Stream is not a customer, so a credential is neither required nor read. Sending one
	// changes nothing, which is what stops a caller thinking it scopes the hook.
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	body := s.arriving("default:" + s.callID)
	request := s.request("/v1/phone/hooks/stream", body, sign(body, suiteStreamSecret))
	request.Header.Set(CustomerHeader, "globex")
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Require().Equal(http.StatusOK, response.StatusCode)
	select {
	case call := <-worker.Calls():
		s.Equal(s.callID, call.CallID, "the number decided whose call it is")
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestAnEndedPlacedCallGetsACallSitesRow() {
	s.releasing()
	s.placed()
	s.stats.answer = agentAndCaller("sfu-a.example.com", "sfu-c.example.com")

	s.Require().Equal(http.StatusOK, s.end(`{}`))

	row := s.callSitesRow()
	s.Equal("2", row.Details["sfus_used"])
	s.Equal("sfu-a.example.com", row.Details["agent_current_sfu"])
	s.Equal("sfu-c.example.com", row.Details["caller_current_sfu"])
	line := s.sitesLine()
	s.Contains(line, "caller_current_sfu=sfu-c.example.com")
	s.NotContains(line, "+15550100002", "the caller's participant id names its number")
	s.Equal("session-1", row.Details["stream_session_id"])
	s.NotContains(row.Details, "stream_sip_host", "a placed call's SIP host is on its call_placed row")
	s.Equal("session-1", <-s.stats.asked)
	left, err := s.store.CallResourcesInApp(context.Background(), store.AppScope{Unpinned: true}, "default", s.callID)
	s.Require().NoError(err)
	s.Empty(left, "the call's resources are still released")
}

func (s *CallHooksSuite) TestAnEndedInboundCallsRowNamesTheNumbersSIPHost() {
	s.holdBridged("default", s.callID, "sip:+15550100001@sip.example.com")

	s.Require().Equal(http.StatusOK, s.end(fmt.Sprintf(`{"phone_number":%q}`, s.e164)))

	s.Equal("sip.example.com", s.callSitesRow().Details["stream_sip_host"])
}

func (s *CallHooksSuite) TestAnEndedCallRingingANumberNobodyHoldsIsWarnedAbout() {
	s.Require().Equal(http.StatusOK, s.end(fmt.Sprintf(`{"phone_number":%q}`, s.e164)))

	s.Contains(s.logged.String(), "could not tell whose number an ended call rang")
	s.NotContains(s.logged.String(), "no number "+s.e164+" is attached")
}

func (s *CallHooksSuite) TestAnEndedVideoCallGetsNoCallSitesRow() {
	s.Require().Equal(http.StatusOK, s.end(`{}`))

	select {
	case session := <-s.stats.asked:
		s.Failf("a video call's stats were read", "session %s", session)
	case <-time.After(dropped):
	}
}

func (s *CallHooksSuite) TestTheWebhookIsAnsweredBeforeTheStatsAreRead() {
	s.placed()
	s.stats.hold = make(chan struct{})

	s.Require().Equal(http.StatusOK, s.end(`{}`))

	select {
	case <-s.stats.asked:
		s.Fail("the stats were read before the webhook was answered")
	default:
	}
	close(s.stats.hold)
	s.Equal("session-1", <-s.stats.asked)
	s.callSitesRow()
}

// sitesLine is the router's "phone call sites" line for the suite's call.
func (s *CallHooksSuite) sitesLine() string {
	var found []string
	for line := range strings.Lines(s.logged.String()) {
		if strings.Contains(line, `msg="phone call sites"`) && strings.Contains(line, "session_id="+s.callID+" ") {
			found = append(found, line)
		}
	}
	s.Require().Len(found, 1)
	return found[0]
}

// recordedStats answers every stats read with one answer, and says which session it was
// asked for once that answer is given. hold, when set, keeps the answer back until closed.
type recordedStats struct {
	asked  chan string
	answer *getstream.QueryCallSessionParticipantStatsResponse
	hold   chan struct{}
}

func (r *recordedStats) read(ctx context.Context, _ *getstream.Stream, _, _, session string) (*getstream.QueryCallSessionParticipantStatsResponse, error) {
	if r.hold != nil {
		select {
		case <-r.hold:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	r.asked <- session
	return r.answer, nil
}

// releasing has the router delete an ended call's lines at the suite's Stream, as a
// deployment does. The suite's own phone service has no Stream, so it releases nothing.
func (s *CallHooksSuite) releasing() {
	bound, err := s.router.streamForApp(context.Background(), s.customerID(), 0, false)
	s.Require().NoError(err)
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	service, err := phone.NewService(phone.ServiceOptions{
		Registry: vendors.Registry(config),
		Store:    s.store,
		Gate:     s.gate,
		Stream:   phone.NewStreamFromClient(bound.Client),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	kept := s.router.phone
	s.router.phone = service
	s.T().Cleanup(func() { s.router.phone = kept })
}

// placed records the trunk an outbound call to the suite's call was given.
func (s *CallHooksSuite) placed() {
	s.Require().NoError(s.store.RecordCallResource(context.Background(), &store.CallResource{
		TrunkID: "trunk-" + s.utils.uuid(), RouteID: "route-" + s.utils.uuid(),
		CallType: "default", CallID: s.callID, CustomerID: s.customerID(),
	}))
}

// end delivers a signed call.session_ended for the suite's call, with the given custom data.
func (s *CallHooksSuite) end(custom string) int {
	return s.signedly("/v1/phone/hooks/stream", fmt.Sprintf(
		`{"type":"call.session_ended","call_cid":"default:%s","session_id":"session-1",`+
			`"created_at":"2026-08-27T12:05:00Z",`+
			`"call":{"cid":"default:%s","id":"%s","type":"default","custom":%s}}`,
		s.callID, s.callID, s.callID, custom))
}

// callSitesRow waits for the suite's call to get its call_sites row, and reads it whole.
func (s *CallHooksSuite) callSitesRow() *store.AgentLog {
	ctx := context.Background()
	var id int64
	s.Require().Eventually(func() bool {
		rows, err := s.store.AgentLogs(ctx, store.LogFilter{
			CustomerID: s.customerID(), SessionID: s.callID, Limit: 250,
		})
		s.Require().NoError(err)
		for _, row := range rows {
			if row.EventType == "call_sites" {
				id = row.ID
				return true
			}
		}
		return false
	}, settleFor, 50*time.Millisecond, "the ended call got no call_sites row")
	row, err := s.store.AgentLog(ctx, s.customerID(), id)
	s.Require().NoError(err)
	return row
}

// hold records the number for the suite's app and attaches it to a call.
func (s *CallHooksSuite) hold(callType, callID string) {
	s.holdBridged(callType, callID, "")
}

// holdBridged is hold, with the Stream trunk the number's calls go to.
func (s *CallHooksSuite) holdBridged(callType, callID, bridgeURI string) {
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164:        s.e164,
		Vendor:      s.numberVendor,
		Country:     "US",
		CustomerID:  s.customerID(),
		PurchasedAt: time.Now().UTC(),
		Tags:        s.numberTags,
	}))
	s.Require().NoError(s.store.AttachNumber(
		ctx, s.customerID(), s.e164, store.NumberAttachment{
			TrunkID: "trunk-" + s.utils.uuid(), CallType: callType, CallID: callID, BridgeURI: bridgeURI,
		}))
}

// arrive delivers a signed call.session_started for one call, as Stream would.
func (s *CallHooksSuite) arrive(cid string) int {
	return s.signedly("/v1/phone/hooks/stream", s.arriving(cid))
}

// callerReaching is the caller number a worker is handed for a call whose SIP participant is
// participantID.
func (s *CallHooksSuite) callerReaching(participantID string) string {
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := strings.Replace(s.arriving("default:"+s.callID), `"sip-+15550001111"`, fmt.Sprintf("%q", participantID), 1)
	s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", body))
	select {
	case call := <-worker.Calls():
		return call.CallerNumber
	case <-time.After(settleFor):
		s.FailNow("the call never reached the worker")
		return ""
	}
}

// arriving is what Stream sends when a caller lands in a call.
func (s *CallHooksSuite) arriving(cid string) string {
	return arrivingCall(cid)
}

func arrivingCall(cid string) string {
	return arrivingCallAt(cid, time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC))
}

// arrivingCallAt is arrivingCall made by Stream at at.
func arrivingCallAt(cid string, at time.Time) string {
	id := cid[strings.Index(cid, ":")+1:]
	callType := cid[:strings.Index(cid, ":")]
	return fmt.Sprintf(`{
  "type": "call.session_started",
  "call_cid": %q,
  "session_id": "session-1",
  "created_at": %q,
  "call": {
    "cid": %q,
    "id": %q,
    "type": %q,
    "custom": {"line": "support"},
    "session": {
      "id": "session-1",
      "participants": [
        {"user_session_id": "s1", "role": "user", "joined_at": "2026-08-27T12:00:00Z",
         "user": {"id": "sip-+15550001111"}}
      ]
    }
  }
}`, cid, at.UTC().Format(time.RFC3339Nano), cid, id, callType)
}

// nothingReaches fails when a call arrives on a channel that should stay empty.
func (s *CallHooksSuite) nothingReaches(calls <-chan dispatch.Call, message string) {
	select {
	case call := <-calls:
		s.Failf(message, "%s reached a worker", call.CallID)
	case <-time.After(dropped):
	}
}

// signedly delivers a body signed with the app secret, and answers with the status.
func (s *RouterSuite) signedly(path, body string) int {
	status, _ := s.deliver(path, body, sign(body, suiteStreamSecret))
	return status
}

// deliver posts a hook body with whatever signature it was given.
func (s *RouterSuite) deliver(path, body, signature string) (int, string) {
	response, err := s.server.Client().Do(s.request(path, body, signature))
	s.Require().NoError(err)
	defer response.Body.Close()

	answered, err := readAll(response)
	s.Require().NoError(err)
	return response.StatusCode, string(answered)
}

func (s *RouterSuite) request(path, body, signature string) *http.Request {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+path, bytes.NewReader([]byte(body)))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, signature)
	return request
}

// sign is the HMAC Stream signs a delivery with.
func sign(body, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(body))
	return hex.EncodeToString(mac.Sum(nil))
}
