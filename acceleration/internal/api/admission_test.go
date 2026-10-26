//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// AdmissionSuite runs the router as Stream's hosted one does about money: every organization's
// credit, status and spend hard limit are read from the real usage service, and the rows it
// acks let go of what was held against them.
type AdmissionSuite struct {
	RouterSuite
	// line is Telnyx ringing the suite's numbers, and logged what the router said of it.
	line   telnyxRinger
	logged *lockedLog
	// sharing is the Redis the router shares its holds through, behind a proxy a test can
	// take down.
	sharing *redisProxy
}

func TestAdmissionSuite(t *testing.T) {
	runSuite(t, new(AdmissionSuite))
}

func (s *AdmissionSuite) SetupSuite() {
	s.limited = true
	s.sharing, s.holds = newRedisProxy(s.T())
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.telnyxPublicKey, s.telnyxConnectionID = public, "1234567890"
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
	s.line = telnyxRinger{suite: &s.RouterSuite, private: private}
}

// SetupTest gives every test an app, and so an organization, of its own: what one test grants,
// spends or suspends is no other's.
func (s *AdmissionSuite) SetupTest() {
	s.usageFront.down.Store(false)
	s.sharing.down.Store(false)
	s.useApp(s.data.createApp())
}

func (s *AdmissionSuite) organization() string { return s.app.organization.ID }

// refused waits for the request to be refused once the organization's money has been read,
// and returns the status and the failure it was refused with. The request that warms an
// organization may be admitted, as a router that has read nothing yet admits.
func (s *AdmissionSuite) refused(method, path string, body any) (int, ErrorDetail) {
	var status int
	var failure ErrorDetail
	s.Require().Eventually(func() bool {
		var payload []byte
		status, payload = s.serverClient.call(method, path, body)
		if status < http.StatusBadRequest {
			return false
		}
		var answered ErrorResponse
		s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
		failure = answered.Error
		return status == http.StatusPaymentRequired || status == http.StatusForbidden
	}, 10*time.Second, 50*time.Millisecond, "%s %s is refused for money", method, path)
	return status, failure
}

func (s *AdmissionSuite) run(model string) CreateRunRequest {
	return CreateRunRequest{ConfigId: s.config(model), Input: "hello"}
}

func (s *AdmissionSuite) TestAnLLMRequestWithNoCreditIsAPaymentRequired() {
	status, failure := s.refused(http.MethodPost, "/v1/agents/runs", s.run("echo/echo-model"))

	s.Equal(http.StatusPaymentRequired, status)
	s.Equal(ErrorTypePaymentRequired, failure.Type)
	s.Equal(codeInsufficientCredits, failure.Code)
	s.Contains(failure.Message, "add credit")
	s.NotContains(failure.Message, "policy:", "the policy's words, without the package's")
}

func (s *AdmissionSuite) TestASearchWithNoCreditIsAPaymentRequired() {
	status, failure := s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	s.Equal(http.StatusPaymentRequired, status, "not the 400 a request the router cannot route is")
	s.Equal(codeInsufficientCredits, failure.Code)
}

func (s *AdmissionSuite) TestAnSTTSocketWithNoCreditIsRefusedWithTheEnvelope() {
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "warm"})

	listening := s.serverClient.opens("/v1/stt/stream")
	s.Require().NoError(listening.WriteJSON(frame{"type": "start", "target": "stub/stub-model"}))

	s.Require().NoError(listening.SetReadDeadline(time.Now().Add(settleFor)))
	var refusal map[string]any
	s.Require().NoError(listening.ReadJSON(&refusal))
	s.Equal("error", refusal["type"])
	s.Equal(codeInsufficientCredits, refusal["code"])
	s.EqualValues(ErrorTypePaymentRequired, refusal["error_type"])
	var closed *websocket.CloseError
	for {
		if _, _, err := listening.ReadMessage(); err != nil {
			s.Require().ErrorAs(err, &closed)
			break
		}
	}
	s.Equal(websocket.ClosePolicyViolation, closed.Code, "the caller's to fix, by paying")
}

// answersOnTheFreeModel asks the model the deployment runs itself, which costs nothing, for one
// response over the LLM socket and says whether it answered.
func (s *AdmissionSuite) answersOnTheFreeModel() {
	answering := s.serverClient.opens("/v1/llm/stream")
	defer answering.Close() //nolint:errcheck // the test is done with it
	s.Require().NoError(answering.WriteJSON(frame{"type": "start", "target": "echo/local-model"}))
	s.Equal("started", s.await(answering, "started")["type"])
	said := s.utils.uuid()
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1", "instructions": said,
		"messages": []frame{{"role": "user", "content": "hello"}}}))
	s.Equal(said, s.await(answering, "complete")["text"], "a model that costs nothing answers")
}

func (s *AdmissionSuite) TestAFreeModelAnswersWithNoCredit() {
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	s.answersOnTheFreeModel()
}

func (s *AdmissionSuite) TestAnAgentWhoseFlowControllerCostsMoneyIsRefused() {
	status, failure := s.refused(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{ConfigId: pointerTo(s.config("echo/local-model"))})

	s.Equal(http.StatusPaymentRequired, status, "every model a session opens is admitted, and its flow controller is priced")
	s.Equal(codeInsufficientCredits, failure.Code)
}

func (s *AdmissionSuite) TestASuspendedAccountIsForbidden() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.usage.Suspend(s.T(), s.organization(), "fraud review "+s.utils.uuid())

	status, failure := s.refused(http.MethodPost, "/v1/agents/runs", s.run("echo/local-model"))

	s.Equal(http.StatusForbidden, status, "a model that costs nothing too")
	s.Equal(ErrorTypePermission, failure.Type)
	s.Equal(codeAccountSuspended, failure.Code)
	s.Equal("this organization is suspended; contact support", failure.Message, "and never why")

	s.usage.Restore(s.T(), s.organization())
	s.Eventually(func() bool {
		return s.serverClient.do(http.MethodPost, "/v1/agents/runs", s.run("echo/local-model"), nil) == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond, "restored, it is admitted again")
}

func (s *AdmissionSuite) TestASpentLimitIsAPaymentRequiredNotAServerError() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.limitApp(usage.Daily, 10_000)

	// Searches cost $0.005 each, so the second spends the limit.
	status, failure := s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})

	s.Equal(http.StatusPaymentRequired, status, "a spent limit was a 500")
	s.Equal(ErrorTypePaymentRequired, failure.Type)
	s.Equal(codeSpendLimitReached, failure.Code)
	s.Contains(failure.Message, "the daily limit of $0.01 on app "+s.customerID()+" is spent; it resets at")
}

func (s *AdmissionSuite) TestAUsersSpentLimitIsTooManyRequestsWithRetryAfter() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	limit := store.Limit{
		Subject: usage.SubjectUser, Level: store.LevelApp, OrganizationID: s.organization(), AppID: s.customerID(),
		Interval: usage.Daily, LimitMicros: 1, UpdatedBy: "test",
	}
	s.Require().NoError(s.store.CreateLimit(s.T().Context(), &limit, store.LimitAudit{CustomerID: s.customerID(), Source: store.AuditSourceAPI}))
	spender := s.utils.uuid()
	s.usage.Spend(s.T(), s.organization(), 1_000, func(event *usage.Event) { event.AppID, event.UserID = s.customerID(), spender })
	asUser := s.signedIn(jwt.MapClaims{"user_id": spender}, auth.AuthTypeJWT, user, spender)

	var response *http.Response
	s.Require().Eventually(func() bool {
		response = asUser.raw(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		if response.StatusCode != http.StatusTooManyRequests {
			_ = response.Body.Close()
			return false
		}
		return true
	}, 10*time.Second, 50*time.Millisecond, "refused once the user's spend is read")
	defer response.Body.Close() //nolint:errcheck // a read-only body

	var answered ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	s.Equal(codeUserLimitReached, answered.Error.Code)
	s.Equal(ErrorTypeRateLimited, answered.Error.Type)
	s.Contains(answered.Error.Message, "is spent for user "+spender)
	retry, err := strconv.Atoi(response.Header.Get("Retry-After"))
	s.Require().NoError(err)
	s.Positive(retry)
	s.LessOrEqual(retry, 86_400, "the day's window ends within a day")
	another := s.utils.uuid()
	status, _ := s.signedIn(jwt.MapClaims{"user_id": another}, auth.AuthTypeJWT, user, another).
		call(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
	s.NotEqual(http.StatusTooManyRequests, status, "another user of the app spent nothing")
}

func (s *AdmissionSuite) TestAUserSpentForTheDayAndTheMonthRetriesAfterTheMonth() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		limit := store.Limit{
			Subject: usage.SubjectUser, Level: store.LevelApp, OrganizationID: s.organization(), AppID: s.customerID(),
			Interval: interval, LimitMicros: 1, UpdatedBy: "test",
		}
		s.Require().NoError(s.store.CreateLimit(s.T().Context(), &limit, store.LimitAudit{CustomerID: s.customerID(), Source: store.AuditSourceAPI}))
	}
	spender := s.utils.uuid()
	s.usage.Spend(s.T(), s.organization(), 1_000, func(event *usage.Event) { event.AppID, event.UserID = s.customerID(), spender })
	asUser := s.signedIn(jwt.MapClaims{"user_id": spender}, auth.AuthTypeJWT, user, spender)

	var response *http.Response
	s.Require().Eventually(func() bool {
		response = asUser.raw(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		if response.StatusCode != http.StatusTooManyRequests {
			_ = response.Body.Close()
			return false
		}
		return true
	}, 10*time.Second, 50*time.Millisecond, "refused once the user's spend is read")
	defer response.Body.Close() //nolint:errcheck // a read-only body

	retry, err := strconv.Atoi(response.Header.Get("Retry-After"))
	s.Require().NoError(err)
	_, monthEnd := usage.Window(usage.Monthly, time.Now())
	s.InDelta(time.Until(monthEnd).Seconds(), retry, 5, "tomorrow the month's limit still refuses, so the month's end")
}

func (s *AdmissionSuite) TestTheSpendHardLimitIsAPaymentRequired() {
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	daily := int64(1_000_000)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{SpendHardLimit: &usage.SpendHardLimit{DailyMicros: &daily}})
	s.usage.Spend(s.T(), s.organization(), 1_000_000)

	status, failure := s.refused(http.MethodPost, "/v1/agents/runs", s.run("echo/echo-model"))

	s.Equal(http.StatusPaymentRequired, status, "with $99 of credit left")
	s.Equal(codeSystemLimitReached, failure.Code)
	_, midnight := usage.Window(usage.Daily, time.Now())
	s.Equal(fmt.Sprintf("the organization's daily spend hard limit of $1.00 is reached; it resets at %s",
		midnight.Format(time.RFC3339)), failure.Message)

	s.answersOnTheFreeModel()
}

func (s *AdmissionSuite) TestTheLLMSessionStopsAnsweringWhenCreditRunsOut() {
	// One response of the counted model costs 1,000 micros: enough credit for one.
	s.usage.Credit(s.T(), s.organization(), 1_000)
	answering := s.serverClient.opens("/v1/llm/stream")
	s.Require().NoError(answering.WriteJSON(frame{"type": "start", "target": "counted/counted-model"}))
	s.Equal("started", s.await(answering, "started")["type"])
	// Opening the socket warmed the organization; two passes of the money loop read its credit.
	time.Sleep(3 * suiteFresh)

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1", "messages": []frame{{"role": "user", "content": "one"}}}))
	s.await(answering, "complete")
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r2", "messages": []frame{{"role": "user", "content": "two"}}}))
	refusal := s.await(answering, "error")

	s.Equal(codeInsufficientCredits, refusal["code"], "the first answer is held at once, with no read of the balance")
	s.EqualValues(ErrorTypePaymentRequired, refusal["error_type"])
}

func (s *AdmissionSuite) TestSpendCheckUnavailableIsA503() {
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	s.Require().Eventually(func() bool {
		status, _ := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "warm"})
		return status == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond, "a search is admitted once the credit is read")
	s.sharing.stop()
	defer s.sharing.down.Store(false)

	var response *http.Response
	s.Require().Eventually(func() bool {
		// Each search the router admits holds its estimate, which cannot be shared.
		response = s.serverClient.raw(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})
		if response.StatusCode != http.StatusServiceUnavailable {
			_ = response.Body.Close()
			return false
		}
		return true
	}, 10*time.Second, 20*time.Millisecond, "refused once a hold waited unshared")
	defer response.Body.Close() //nolint:errcheck // a read-only body

	var answered ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	s.Equal(ErrorTypeUnavailable, answered.Error.Type)
	s.Equal(codeSpendCheckUnavailable, answered.Error.Code)
	s.Contains(answered.Error.Message, "could not share spend with the other routers")
	s.Equal("5", response.Header.Get("Retry-After"))
	s.answersOnTheFreeModel()

	s.sharing.down.Store(false)
	s.Require().Eventually(func() bool {
		status, _ := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "back"})
		return status == http.StatusOK
	}, 5*time.Second, 20*time.Millisecond, "admitted again once a flush lands")
}

// otherRouter is a second router's enforcer over the same usage service and the same Redis,
// as another replica of the deployment.
func (s *AdmissionSuite) otherRouter() *policy.Enforcer {
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	client, err := usage.NewClient(usage.ClientOptions{URL: s.usage.URL, Token: s.usage.Token})
	s.Require().NoError(err)
	_, holds := newRedisProxy(s.T())
	other, err := policy.New(configs, policy.Settings{
		Usage: client, Holds: holds, Fresh: suiteFresh, SpendEvery: suiteFresh, SumEvery: suiteFresh,
		MaxStale: time.Minute, PendingTTL: time.Minute,
		FlushEvery: 10 * time.Millisecond, HoldRefresh: 50 * time.Millisecond, FlushMaxLag: suiteFlushMaxLag,
	}, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.T().Cleanup(other.Close)
	return other
}

func (s *AdmissionSuite) TestTwoRoutersStopAnLLMSessionTogether() {
	// One response of the counted model costs 1,000 micros.
	s.usage.Credit(s.T(), s.organization(), 1_000_000)
	other := s.otherRouter()
	other.Join(s.customerID(), s.organization(), usage.DefaultTier)
	answering := s.serverClient.opens("/v1/llm/stream")
	s.Require().NoError(answering.WriteJSON(frame{"type": "start", "target": "counted/counted-model"}))
	s.Equal("started", s.await(answering, "started")["type"])
	time.Sleep(3 * suiteFresh)
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1", "messages": []frame{{"role": "user", "content": "one"}}}))
	s.await(answering, "complete")

	// The other router's sessions take what credit is left.
	owner := routing.Owner{CustomerID: s.customerID(), Attribution: routing.Attribution{OrganizationID: s.organization()}}
	s.Require().NoError(other.Reserve(owner, uuid.Must(uuid.NewV7()), 999_000, time.Minute))
	time.Sleep(4 * 50 * time.Millisecond)
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r2", "messages": []frame{{"role": "user", "content": "two"}}}))
	refusal := s.await(answering, "error")

	s.Equal(codeInsufficientCredits, refusal["code"], "this router counts what the other holds, within a refresh")
	s.Error(other.Reserve(owner, uuid.Must(uuid.NewV7()), 1_000, time.Minute), "and the other counts what this one spent")
}

func (s *AdmissionSuite) TestCreatingASessionWarmsTheDecision() {
	config := s.config("echo/local-model")
	// Admitted or refused alike: what matters is that asking warmed the app and its organization.
	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{ConfigId: &config})
	s.Require().Contains([]int{http.StatusCreated, http.StatusPaymentRequired}, status)
	// Two passes of the money loop, with nothing else asked of this app.
	time.Sleep(3 * suiteFresh)
	s.usageFront.down.Store(true)

	started := time.Now()
	status, payload := s.serverClient.call(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	s.Less(time.Since(started), 200*time.Millisecond, "answered from memory with the usage service gone")
	s.Equal(http.StatusPaymentRequired, status, "with the balance the warmed account read: %s", payload)
}

// telephony is every way a customer starts paying a phone vendor: buying a number, placing a
// call and transferring one, each answering its status and failure.
func (s *AdmissionSuite) telephony() map[string]func() (int, ErrorDetail) {
	answer := func(path string, body any) func() (int, ErrorDetail) {
		return func() (int, ErrorDetail) {
			status, payload := s.serverClient.call(http.MethodPost, path, body)
			var answered ErrorResponse
			_ = json.Unmarshal(payload, &answered)
			return status, answered.Error
		}
	}
	return map[string]func() (int, ErrorDetail){
		"buy":      answer("/v1/phone/numbers", BuyNumberRequest{Vendor: "twilio", E164: "+15125551234"}),
		"call":     answer("/v1/phone/calls", PlaceCallRequest{From: "+15125551234", To: "+15550001111"}),
		"transfer": answer("/v1/phone/calls/transfer", TransferCallRequest{From: "+15125551234", To: "+15550001111", SessionId: s.data.storedSession()}),
	}
}

func (s *AdmissionSuite) TestAnOrganizationWithNoCreditCanNeitherBuyNorDialNorTransfer() {
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	for name, ask := range s.telephony() {
		status, failure := ask()
		s.Equal(http.StatusPaymentRequired, status, name)
		s.Equal(codeInsufficientCredits, failure.Code, name)
	}
}

func (s *AdmissionSuite) TestASuspendedOrganizationCanNeitherBuyNorDialNorTransfer() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.usage.Suspend(s.T(), s.organization(), "fraud review")
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	for name, ask := range s.telephony() {
		status, failure := ask()
		s.Equal(http.StatusForbidden, status, name)
		s.Equal(codeAccountSuspended, failure.Code, name)
		s.NotContains(failure.Message, "fraud", name)
	}
}

func (s *AdmissionSuite) TestACallOrATransferToACountryTheTierDoesNotAllowIsForbidden() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{Channels: map[string]usage.ChannelLimits{
		dlc.Voice: {Countries: []string{"US", "CA"}},
	}})
	transfer := func(to string) (int, string) {
		return s.serverClient.failure(http.MethodPost, "/v1/phone/calls/transfer",
			TransferCallRequest{From: "+15125551234", To: to, SessionId: s.data.storedSession()})
	}
	var status int
	var failure string
	s.Require().Eventually(func() bool {
		status, failure = transfer("+442079460000")
		return status == http.StatusForbidden
	}, 10*time.Second, 50*time.Millisecond, "the organization's limits are read")

	s.Contains(failure, "the default tier is not allowed to reach GB by voice")
	s.NotContains(failure, "+44", "the refusal does not name the number")
	status, failure = s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{From: "+15125551234", To: "+442079460000"})
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "not allowed to reach GB")
	status, failure = transfer("+15125550111")
	s.NotEqual(http.StatusForbidden, status, "a number of a country the tier allows passes the gate: %s", failure)
	s.NotEqual(http.StatusPaymentRequired, status, failure)
}

func (s *AdmissionSuite) TestWithCreditTelephonyGetsPastItsMoney() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"}, nil) == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")

	for name, ask := range s.telephony() {
		status, failure := ask()
		s.NotContains([]int{http.StatusPaymentRequired, http.StatusForbidden}, status, "%s: %s", name, failure.Message)
	}
}

func (s *AdmissionSuite) TestACallToANumberOfAnOrganizationThatMayNotPayIsRejectedUnanswered() {
	e164 := s.utils.number()
	s.line.attach(e164, "sip:"+e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	status, _, body := s.line.deliver(s.line.call(e164), true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, `<Response><Reject reason="rejected"/></Response>`,
		"Reject as the first verb leaves the call unanswered, so no second of it is billed")
	s.NotContains(body, "<Hangup/>", "a hang-up answers the call first")
	s.NotContains(body, "trunk-password")
	s.Contains(s.logged.String(), "may not pay for it")
	s.Contains(s.logged.String(), "code=insufficient_credits")

	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.Eventually(func() bool {
		_, _, body := s.line.deliver(s.line.call(e164), true)
		return strings.Contains(body, "sip:"+e164+"@trunk.sip.example.test")
	}, 10*time.Second, 50*time.Millisecond, "answered once it has credit")

	s.usage.Suspend(s.T(), s.organization(), "fraud review")
	s.Eventually(func() bool {
		_, _, body := s.line.deliver(s.line.call(e164), true)
		return strings.Contains(body, "<Reject")
	}, 10*time.Second, 50*time.Millisecond, "and rejected once suspended")
	s.Contains(s.logged.String(), "code=account_suspended")
}

func (s *AdmissionSuite) TestACallArrivingOverStreamForAnOrganizationThatMayNotPayIsNotAnswered() {
	for _, refused := range []string{codeInsufficientCredits, codeAccountSuspended} {
		s.Run(refused, func() {
			s.SetupTest()
			e164, callID := s.utils.number(), "phone-"+s.utils.uuid()
			ctx := context.Background()
			s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
				E164: e164, Vendor: "twilio", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
			}))
			s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), e164,
				store.NumberAttachment{TrunkID: "trunk-" + s.utils.uuid(), CallType: "default", CallID: callID}))
			if refused == codeAccountSuspended {
				s.usage.Credit(s.T(), s.organization(), 10_000_000)
				s.usage.Suspend(s.T(), s.organization(), "fraud review")
			}
			s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})
			worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
			defer release()

			s.Require().Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", arrivingCall("default:"+callID)))

			select {
			case call := <-worker.Calls():
				s.Failf("an agent answered", "%s reached a worker", call.CallID)
			case <-time.After(200 * time.Millisecond):
			}
			s.Contains(s.logged.String(), "may not pay for it; ending it")
			s.Eventually(func() bool { return strings.Contains(s.logged.String(), "could not end a refused arriving call") },
				5*time.Second, 20*time.Millisecond, "ended at Stream, which the suite's phone service has no credentials for")
			s.Contains(s.logged.String(), "code="+refused)
		})
	}
}

func (s *AdmissionSuite) TestACampaignCannotStartWhenItsOrganizationMayNotPay() {
	created := s.campaign(1)
	s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil)

	s.Equal(http.StatusPaymentRequired, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered))
	s.Equal(codeInsufficientCredits, answered.Error.Code)
	s.Equal(ContactStatePending, s.campaignContacts(created.Id)[0].State, "nobody was rung")
}

func (s *AdmissionSuite) TestACampaignRefusedMidwayPausesAndKeepsItsContacts() {
	for _, concurrency := range []int{1, 2} {
		for _, refused := range []string{codeInsufficientCredits, codeAccountSuspended, codeTelephonySuspended} {
			s.Run(fmt.Sprintf("%s at concurrency %d", refused, concurrency), func() {
				s.pausedMidway(refused, concurrency)
			})
		}
	}
}

// pausedMidway starts a campaign of concurrency whose one contact is held off until the
// organization is refused, and checks the refused call pauses it for good, with the contact
// given back to be rung.
func (s *AdmissionSuite) pausedMidway(refused string, concurrency int) {
	s.SetupTest()
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"}, nil) == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	created := s.campaign(concurrency)
	// Locking the contact's row holds the runner off it, as SKIP LOCKED claims, so the
	// campaign starts with credit and its call comes once it is refused.
	ctx := context.Background()
	locked, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	defer locked.Rollback() //nolint:errcheck // nothing was written
	_, err = locked.ExecContext(ctx, "SELECT id FROM campaign_contacts WHERE campaign_id = ? FOR UPDATE", created.Id)
	s.Require().NoError(err)
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil))

	switch refused {
	case codeAccountSuspended:
		s.usage.Suspend(s.T(), s.organization(), "fraud review")
		s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})
	case codeInsufficientCredits:
		s.usage.Credit(s.T(), s.organization(), -10_000_000)
		s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})
	default:
		s.Require().NoError(s.store.SuspendTelephony(ctx, &store.TelephonySuspension{
			CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
		}))
	}
	s.Require().NoError(locked.Rollback())

	read := func() (string, string) {
		var campaign Campaign
		s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id, nil, &campaign)
		return string(campaign.State), value(campaign.PausedReason)
	}
	s.Require().Eventually(func() bool { state, _ := read(); return state == string(CampaignStatePaused) },
		10*time.Second, 50*time.Millisecond, "paused at the refused call")
	s.Never(func() bool { state, reason := read(); return state != string(CampaignStatePaused) || reason != refused },
		500*time.Millisecond, 20*time.Millisecond, "and it stays paused, with why, whatever its other slots were doing")
	s.Equal(ContactStatePending, s.campaignContacts(created.Id)[0].State,
		"the contact nobody rang is given back, to be rung when the campaign is started again")

	switch refused {
	case codeAccountSuspended:
		s.usage.Restore(s.T(), s.organization())
	case codeInsufficientCredits:
		s.usage.Credit(s.T(), s.organization(), 10_000_000)
	default:
		s.Require().NoError(s.store.LiftTelephonySuspension(ctx, s.customerID()))
	}
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"}, nil) == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond, "paid again")
	var started Campaign
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, &started))
	s.Nil(started.PausedReason, "starting it again clears why it paused")
}

// campaign is a campaign of concurrency over a stored config with one person to ring.
func (s *AdmissionSuite) campaign(concurrency int) Campaign {
	var created Campaign
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns", map[string]any{
		"name": "campaign-" + s.utils.uuid(), "config_id": s.data.createAgentConfig().Id, "from_number": s.utils.number(),
		"concurrency": concurrency,
	}, &created))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts",
		map[string]any{"contacts": []map[string]any{{"to_number": s.utils.number()}}}, nil))
	return created
}

func (s *AdmissionSuite) campaignContacts(campaignID string) []Contact {
	var listed []Contact
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+campaignID+"/contacts", nil, &listed))
	return listed
}

func (s *AdmissionSuite) TestInlineSpeechWithNoCreditIsAPaymentRequired() {
	status, failure := s.refused(http.MethodPost, "/v1/tts/recordings",
		SpeechRequest{Text: "hello", Inline: pointerTo(true)})

	s.Equal(http.StatusPaymentRequired, status, "not a 202 with the refusal in its body")
	s.Equal(codeInsufficientCredits, failure.Code)
}

// ShadowAdmissionSuite is the admission suite's router with ROUTER_LIMITS_MODE=shadow.
type ShadowAdmissionSuite struct {
	RouterSuite
	logged *lockedLog
}

func TestShadowAdmissionSuite(t *testing.T) {
	runSuite(t, new(ShadowAdmissionSuite))
}

func (s *ShadowAdmissionSuite) SetupSuite() {
	s.limited, s.limitsMode = true, config.LimitsShadow
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *ShadowAdmissionSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ShadowAdmissionSuite) TestTelephonyIsAdmittedAndLoggedAsWouldRefuse() {
	organization := s.app.organization.ID
	s.Require().Eventually(func() bool {
		s.serverClient.do(http.MethodPost, "/v1/phone/numbers", BuyNumberRequest{Vendor: "twilio", E164: "+15125551234"}, nil)
		return strings.Contains(s.logged.String(), `msg="would refuse" organization=`+organization)
	}, 10*time.Second, 50*time.Millisecond, "the no-credit refusal is worked out and logged")

	for path, body := range map[string]any{
		"/v1/phone/numbers": BuyNumberRequest{Vendor: "twilio", E164: "+15125551234"},
		"/v1/phone/calls":   PlaceCallRequest{From: "+15125551234", To: "+15550001111"},
	} {
		status, failure := s.serverClient.failure(http.MethodPost, path, body)
		s.NotContains([]int{http.StatusPaymentRequired, http.StatusForbidden}, status, "%s is admitted: %s", path, failure)
	}
}

// TestAMoneyRefusalIsTheCodeNotTheText makes sure moneyError reads what the policy wrapped,
// however it was joined on the way.
func (s *AdmissionSuite) TestAMoneyRefusalIsTheCodeNotTheText() {
	_, ok := moneyError(errors.New("policy: no credit left"))
	s.False(ok, "a message that reads like one is not one")
	_, ok = moneyError(nil)
	s.False(ok)
	var answer APIError
	s.Require().True(errors.As(refusedRequest(errors.New("routing: unknown target")), &answer))
	s.True(strings.HasPrefix(answer.Code, "invalid_request"))
}

func (s *AdmissionSuite) TestAScheduledDeliveryRefusedForMoneyIsRecordedUnderItsCode() {
	unpaid := fmt.Errorf("%w: %w", dlc.ErrUnpaid, policy.ErrSuspended)

	s.Equal(codeAccountSuspended, deliveryReason(unpaid), "a call or text the gate refused")
	s.Equal(codeInsufficientCredits, deliveryReason(undeliveredFinally(scheduling.ReasonFailed, policy.ErrNoCredits)),
		"a session that could not open for it")
	s.Equal(scheduling.ReasonOnACall, deliveryReason(undelivered(scheduling.ReasonOnACall, errors.New("on a call"))))
}

func (s *AdmissionSuite) TestThePolicysCodesAreTheAPIs() {
	for _, refusal := range []error{policy.ErrSuspended, policy.ErrNoCredits, policy.ErrSystemLimit, policy.ErrBudgetSpent, policy.ErrUserLimit, policy.ErrSpendCheckUnavailable} {
		answer, ok := moneyError(refusal)
		s.Require().True(ok)
		s.Equal(answer.Code, policy.Code(refusal), "a campaign records the code the API answers")
	}
	s.Empty(policy.Code(errors.New("anything else")))
}
