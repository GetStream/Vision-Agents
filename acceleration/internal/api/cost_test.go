//go:build integration

package api

import (
	"net/http"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// CostSuite runs the router as Stream's hosted one does about money, and reads what each
// answer says it cost against the rows the request recorded.
type CostSuite struct {
	RouterSuite
}

func TestCostSuite(t *testing.T) {
	runSuite(t, new(CostSuite))
}

func (s *CostSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an organization of its own, with credit.
func (s *CostSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.usage.Credit(s.T(), s.app.organization.ID, 100_000_000)
}

// answered sends the request as the client until it is answered with want, and returns the
// answer, its body read and closed.
func (s *RouterSuite) answered(client *testClient, method, path string, body any, want int) *http.Response {
	var response *http.Response
	s.Require().Eventually(func() bool {
		response = client.raw(method, path, body)
		_ = response.Body.Close()
		return response.StatusCode == want
	}, 10*time.Second, 50*time.Millisecond, "%s %s is answered %d", method, path, want)
	return response
}

// costOf is what the answer says it cost, failing when it says nothing.
func (s *RouterSuite) costOf(response *http.Response) int64 {
	header := response.Header.Get(headerCostMicros)
	s.Require().NotEmpty(header, "the answer says what it cost")
	micros, err := strconv.ParseInt(header, 10, 64)
	s.Require().NoError(err)
	return micros
}

func (s *CostSuite) TestASearchSaysWhatItCost() {
	since := time.Now().UTC()

	response := s.answered(s.serverClient, http.MethodPost, "/v1/search", SearchRequest{Query: "what it cost"}, http.StatusOK)

	rows := s.meteredRows(routing.Search, since, 1)
	s.Positive(rows[0].CostMicros, "a search is billed")
	s.Equal(rows[0].CostMicros, s.costOf(response), "what the row recorded")
	s.Empty(response.Header.Get(headerCostEstimated), "the vendor said what it used")
}

func (s *CostSuite) TestAnImageSaysWhatItCost() {
	since := time.Now().UTC()

	response := s.answered(s.serverClient, http.MethodPost, "/v1/image/generations", ImageGenerationRequest{
		Prompt: "a yellow watering can " + s.utils.uuid(), Options: &ImageOptions{Target: pointerTo("image-fast"), N: pointerTo(1)},
	}, http.StatusOK)

	rows := s.meteredRows(routing.Image, since, 1)
	s.EqualValues(40_000, rows[0].CostMicros)
	s.EqualValues(40_000, s.costOf(response))
}

func (s *CostSuite) TestAnEndUserSeesNoCost() {
	for name, client := range map[string]*testClient{"user": s.client, "guest": s.guestClient, "anonymous": s.anonymousClient} {
		response := s.answered(client, http.MethodPost, "/v1/search", SearchRequest{Query: "what it cost"}, http.StatusOK)

		for _, header := range []string{headerCostMicros, headerCostEstimated, headerBalanceMicros, headerBalanceWarning} {
			s.Empty(response.Header.Get(header), "%s: an end user is told nothing of money (%s)", name, header)
		}
	}
}

func (s *CostSuite) TestAnEstimatedAnswerSaysSo() {
	since := time.Now().UTC()

	// The echo model never says what a response used, so the router counts it.
	response := s.answered(s.serverClient, http.MethodPost, "/v1/agents/runs",
		CreateRunRequest{ConfigId: s.config("echo/echo-model"), Input: "hello"}, http.StatusOK)

	rows := s.meteredRows(routing.LLM, since, 1)
	s.Equal("true", rows[0].Attributes[routing.AttributeUsageEstimated])
	s.Equal(rows[0].CostMicros, s.costOf(response))
	s.Equal("true", response.Header.Get(headerCostEstimated))
}

func (s *CostSuite) TestAFreeAnswerSaysNothingWhenItRecordedNothing() {
	response := s.answered(s.serverClient, http.MethodGet, "/v1/policies/app", nil, http.StatusOK)

	s.Empty(response.Header.Get(headerCostMicros), "nothing was recorded")
}

func (s *CostSuite) TestAFailureSaysNothingOfMoney() {
	s.answered(s.serverClient, http.MethodGet, "/v1/policies/app", nil, http.StatusOK)

	response := s.answered(s.serverClient, http.MethodPost, "/v1/search", SearchRequest{}, http.StatusBadRequest)

	for _, header := range []string{headerCostMicros, headerCostEstimated, headerBalanceMicros, headerBalanceWarning} {
		s.Empty(response.Header.Get(header), "only a success and a refusal for money declare %s", header)
	}
}

func (s *CostSuite) TestAnAcceptedTurnSaysNoCost() {
	created := s.serverClient.createSession(textSession(nil))

	response := s.answered(s.serverClient, http.MethodPost, "/v1/agents/sessions/"+created.Id+"/responses",
		CreateResponseRequest{Text: "hello"}, http.StatusAccepted)

	s.Empty(response.Header.Get(headerCostMicros), "it answers before its turn has cost anything")
	s.Empty(response.Header.Get(headerCostEstimated))
	s.NotEmpty(response.Header.Get(headerBalanceMicros), "its balance is said all the same")
}

// responded opens an LLM socket as the client, asks the model for one response and returns its
// complete frame.
func (s *RouterSuite) responded(client *testClient, model string) map[string]any {
	answering := client.opens("/v1/llm/stream")
	defer answering.Close() //nolint:errcheck // the test is done with it
	s.Require().NoError(answering.WriteJSON(frame{"type": "start", "target": model}))
	s.await(answering, "started")
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1", "instructions": s.utils.uuid(),
		"messages": []frame{{"role": "user", "content": "hello"}}}))
	return s.await(answering, "complete")
}

func (s *CostSuite) TestAResponseCarriesItsCost() {
	since := time.Now().UTC()

	complete := s.responded(s.serverClient, "counted/counted-model")

	rows := s.meteredRows(routing.LLM, since, 1)
	s.Positive(rows[0].CostMicros)
	s.EqualValues(rows[0].CostMicros, complete["cost_micros"], "what its row recorded")
}

func (s *CostSuite) TestATranscriptCarriesItsCost() {
	since := time.Now().UTC()
	listening := s.serverClient.opens("/v1/stt/stream")
	defer listening.Close() //nolint:errcheck // the test is done with it
	s.Require().NoError(listening.WriteJSON(frame{"type": "start", "target": "en-low-latency"}))
	s.await(listening, "started")

	s.ears().emitter.Send(stt.Transcript{Text: "not yet", Mode: stt.ModeReplacement, Provider: "stub", Model: "stub-stt"})
	interim := s.await(listening, "transcript")
	s.ears().emitter.Send(stt.Transcript{Text: "a call costs a penny", Mode: stt.ModeFinal, AudioDurationMs: 60_000,
		Provider: "stub", Model: "stub-stt"})
	settled := s.await(listening, "transcript")

	rows := s.meteredRows(routing.STT, since, 1)
	s.NotContains(interim, "cost_micros", "a revision is no turn")
	s.Contains(settled, "cost_micros")
	s.EqualValues(rows[0].CostMicros, settled["cost_micros"], "what the turn's row recorded")
}

// The sockets that carry cost frames are a backend's: a device is refused them outright, low
// as its organization's balance is.
func (s *CostSuite) TestADeviceSocketGetsNoCost() {
	s.usage.Spend(s.T(), s.app.organization.ID, 99_000_000)
	for _, modality := range []string{"stt", "tts", "llm"} {
		for name, client := range map[string]*testClient{"user": s.client, "guest": s.guestClient, "anonymous": s.anonymousClient} {
			connection, status := client.watch("/v1/" + modality + "/stream")
			s.Nil(connection, "%s: %s", name, modality)
			s.Contains([]int{http.StatusUnauthorized, http.StatusForbidden}, status, "%s: %s", name, modality)
		}
	}
}

func (s *CostSuite) TestAnUtteranceCarriesItsCost() {
	since := time.Now().UTC()
	speaking := s.serverClient.opens("/v1/tts/stream")
	defer speaking.Close() //nolint:errcheck // the test is done with it
	s.Require().NoError(speaking.WriteJSON(frame{"type": "start", "target": "en-low-latency"}))
	s.await(speaking, "started")
	line := "your call is important, " + s.utils.uuid()
	s.Require().NoError(speaking.WriteJSON(frame{"type": "speak", "id": "s1", "text": line}))
	s.Require().Eventually(func() bool { return slices.Contains(s.voice.spoken(), line) }, settleFor, 10*time.Millisecond)

	s.speaking().emitter.Send(tts.SynthesisComplete{SynthesisID: "s1", Characters: int64(len(line)), AudioDurationMs: 2_000,
		Provider: "stub", Model: "stub-tts"})
	complete := s.await(speaking, "synthesis_complete")

	rows := s.meteredRows(routing.TTS, since, 1)
	s.Contains(complete, "cost_micros")
	s.EqualValues(rows[0].CostMicros, complete["cost_micros"], "what the utterance's row recorded")
}

// config is the id of an agent config of the suite's app answered by model.
func (s *RouterSuite) config(model string) string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(
		http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
			Name: "agent-" + s.utils.uuid(), Llm: &model, Instructions: pointerTo("be brief"),
		}, &created))
	return created.Id
}

// UnbilledCostSuite runs the router as a deployment that bills nobody: no usage service.
type UnbilledCostSuite struct {
	RouterSuite
}

func TestUnbilledCostSuite(t *testing.T) {
	runSuite(t, new(UnbilledCostSuite))
}

func (s *UnbilledCostSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *UnbilledCostSuite) TestARouterThatBillsNobodySaysTheCostAndNoBalance() {
	since := time.Now().UTC()

	response := s.answered(s.serverClient, http.MethodPost, "/v1/search", SearchRequest{Query: "what it cost"}, http.StatusOK)

	rows := s.meteredRows(routing.Search, since, 1)
	s.Equal(rows[0].CostMicros, s.costOf(response), "what the work cost is still the row's")
	s.Empty(response.Header.Get(headerBalanceMicros), "there is no credit to tell of")
	s.Empty(response.Header.Get(headerBalanceWarning))
}
