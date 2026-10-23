//go:build integration

package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// AdmissionSuite runs the router as Stream's hosted one does about money: every organization's
// credit, status and spend hard limit are read from the real usage service, and the rows it
// acks let go of what was held against them.
type AdmissionSuite struct {
	RouterSuite
}

func TestAdmissionSuite(t *testing.T) {
	runSuite(t, new(AdmissionSuite))
}

func (s *AdmissionSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an app, and so an organization, of its own: what one test grants,
// spends or suspends is no other's.
func (s *AdmissionSuite) SetupTest() {
	s.usageFront.down.Store(false)
	s.useApp(s.data.createApp())
}

func (s *AdmissionSuite) organization() string { return s.app.organization.ID }

// config is the id of an agent config of the suite's app answered by model.
func (s *AdmissionSuite) config(model string) string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(
		http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
			Name: "agent-" + s.utils.uuid(), Llm: &model, Instructions: pointerTo("be brief"),
		}, &created))
	return created.Id
}

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

func (s *AdmissionSuite) TestASpentBudgetIsAPaymentRequiredNotAServerError() {
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/app",
		map[string]any{"budget": map[string]any{"limit_micros": 10_000, "interval": "daily"}}, nil))

	// Searches cost $0.005 each, so the second spends the budget.
	status, failure := s.refused(http.MethodPost, "/v1/search", SearchRequest{Query: "again"})

	s.Equal(http.StatusPaymentRequired, status, "a spent budget was a 500")
	s.Equal(ErrorTypePaymentRequired, failure.Type)
	s.Equal(codeSpendLimitReached, failure.Code)
	s.Contains(failure.Message, "the app's daily budget of $0.01 is spent; it resets at")
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

// TestAMoneyRefusalIsTheCodeNotTheText makes sure moneyError reads what the policy wrapped,
// however it was joined on the way.
func (s *AdmissionSuite) TestAMoneyRefusalIsTheCodeNotTheText() {
	_, ok := moneyError(errors.New("policy: no credit left"))
	s.False(ok, "a message that reads like one is not one")
	_, ok = moneyError(nil)
	s.False(ok)
	s.True(strings.HasPrefix(refusedRequest(errors.New("routing: unknown target")).Code, "invalid_request"))
}
