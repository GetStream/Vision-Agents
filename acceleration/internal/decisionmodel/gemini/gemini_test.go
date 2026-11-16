package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
)

// answered is the shape the Decisions API answers in, read off a real response.
const answered = `{"answers":{"heard":{"confidence":0.98,"noul":0.98,"type":"noul"}},
	"usage":{"total_tokens":219,"total_input_tokens":219,
	"input_tokens_by_modality":[{"modality":"text","tokens":219}],"total_output_tokens":0}}`

type GeminiSuite struct {
	suite.Suite
	ctx    context.Context
	server *httptest.Server
	client *Client

	path   string
	key    string
	body   map[string]any
	status int
	// respond is the JSON the stub answers with.
	respond string
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

func (s *GeminiSuite) SetupTest() {
	s.ctx = context.Background()
	s.status, s.respond = http.StatusOK, answered
	s.path, s.key, s.body = "", "", nil
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.path, s.key = r.URL.Path, r.Header.Get("x-goog-api-key")
		raw, _ := io.ReadAll(r.Body)
		s.body = map[string]any{}
		_ = json.Unmarshal(raw, &s.body)
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.respond))
	}))
	s.T().Cleanup(s.server.Close)

	client, err := New(Options{APIKey: "test-key", BaseURL: s.server.URL})
	s.Require().NoError(err)
	s.client = client
}

func (s *GeminiSuite) ask(state any, questions map[string]decisionmodel.Question) (decisionmodel.Result, error) {
	return s.client.Classify(s.ctx, decisionmodel.Request{State: state, Questions: questions})
}

func (s *GeminiSuite) heard() map[string]decisionmodel.Question {
	return map[string]decisionmodel.Question{"heard": decisionmodel.Noul("Did anyone speak?", "", "")}
}

func (s *GeminiSuite) question(id string) map[string]any {
	questions, ok := s.body["questions"].(map[string]any)
	s.Require().True(ok, "the request carried no questions")
	asked, ok := questions[id].(map[string]any)
	s.Require().True(ok, "the request did not ask %q", id)
	return asked
}

func (s *GeminiSuite) TestAKeyIsRequired() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})

	s.ErrorContains(err, apiKeyEnvVar)
}

func (s *GeminiSuite) TestTheQuestionsGoToTheDecisionsAPIWithGooglesKeyHeader() {
	_, err := s.ask("hello?", s.heard())
	s.Require().NoError(err)

	s.Equal("/v1beta/decisions", s.path)
	s.Equal("test-key", s.key)
	s.Equal(DefaultModel, s.body["model"])
	s.Equal("gemini", s.client.Provider())
}

func (s *GeminiSuite) TestTextStateIsTheInput() {
	_, err := s.ask("hello?", s.heard())
	s.Require().NoError(err)

	s.Equal("hello?", s.body["input"])
	s.NotContains(s.body, "state", "Google's field is input")
}

func (s *GeminiSuite) TestAStateWithPartsGoesAsItsJSON() {
	// An object in input is read as content blocks, so the parts a question names in
	// backticks have to arrive as text.
	_, err := s.ask(map[string]any{"policy": "SDK questions only", "message": "pizza?"}, s.heard())
	s.Require().NoError(err)

	s.JSONEq(`{"policy":"SDK questions only","message":"pizza?"}`, s.body["input"].(string))
}

func (s *GeminiSuite) TestANoulSaysWhatYesAndNoMeanInGooglesFields() {
	_, err := s.ask("press one for billing", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Is this a recorded menu?", "A recording listing options.", "A person talking."),
	})
	s.Require().NoError(err)

	s.Equal(map[string]any{
		"true_description":  "A recording listing options.",
		"false_description": "A person talking.",
	}, s.question("heard")["criteria"])
}

func (s *GeminiSuite) TestANoulWithNothingToSayAboutYesAndNoSendsNoCriteria() {
	_, err := s.ask("hello?", s.heard())
	s.Require().NoError(err)

	s.NotContains(s.question("heard"), "criteria")
}

func (s *GeminiSuite) TestChoicesAndScoresAreSentAsAsked() {
	s.respond = `{"answers":{
		"queue":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"other":0.1},"confidence":0.9},
		"severity":{"type":"score","score":1.4,"legend":{"0":"low","1":"mid","2":"high"},
			"probabilities":{"0":0.1,"1":0.4,"2":0.5},"confidence":0.5}},
		"usage":{"total_input_tokens":80}}`

	result, err := s.ask("charged twice", map[string]decisionmodel.Question{
		"queue":    decisionmodel.Choice("Which queue?", map[string]string{"billing": "Payments.", "other": ""}),
		"severity": decisionmodel.Score("How severe?", []string{"low", "mid", "high"}),
	})
	s.Require().NoError(err)

	s.Equal(map[string]any{"billing": "Payments.", "other": nil}, s.question("queue")["criteria"])
	s.Equal([]any{"low", "mid", "high"}, s.question("severity")["criteria"])
	s.Equal("billing", result.Answers["queue"].Chosen)
	s.InDelta(0.9, result.Answers["queue"].Probabilities["billing"], 0.001)
	s.InDelta(1.4, result.Answers["severity"].Level, 0.001)
	s.Equal("high", result.Answers["severity"].Legend["2"])
}

func (s *GeminiSuite) TestTheAnswerSaysWhatItReadAndWhichModelAnswered() {
	result, err := s.ask("hello?", s.heard())
	s.Require().NoError(err)

	s.InDelta(0.98, result.Answers["heard"].Yes, 0.001)
	s.EqualValues(219, result.Usage.InputTokens)
	s.Zero(result.Usage.OutputTokens)
	s.Equal(DefaultModel, result.Model, "Google names no version, and the model asked is one")
}

func (s *GeminiSuite) TestAnUnansweredQuestionIsAFailureRatherThanAZero() {
	_, err := s.ask("hello?", map[string]decisionmodel.Question{
		"heard":    decisionmodel.Noul("Did anyone speak?", "", ""),
		"shouting": decisionmodel.Noul("Is anyone shouting?", "", ""),
	})

	s.ErrorContains(err, "shouting")
}

func (s *GeminiSuite) TestABlockedInputIsAFailureRatherThanNoAnswers() {
	s.respond = `{"answers":{},"prompt_feedback":{"block_reason":"safety"}}`

	_, err := s.ask("hello?", s.heard())

	s.ErrorContains(err, "blocked (safety)")
}

func (s *GeminiSuite) TestNothingIsSentForAnEmptyState() {
	_, err := s.ask("  ", s.heard())

	s.ErrorContains(err, "empty")
	s.Empty(s.path)
}

func (s *GeminiSuite) TestARequestWithNoQuestionsIsNotSent() {
	_, err := s.ask("hello?", map[string]decisionmodel.Question{})

	s.ErrorContains(err, "at least one question")
	s.Empty(s.path)
}

func (s *GeminiSuite) TestFailuresMapOntoTheContract() {
	cases := map[int]error{
		http.StatusTooManyRequests:     decisionmodel.ErrRateLimited,
		http.StatusInternalServerError: decisionmodel.ErrUnavailable,
		http.StatusServiceUnavailable:  decisionmodel.ErrUnavailable,
		http.StatusGatewayTimeout:      decisionmodel.ErrUnavailable,
	}
	for status, want := range cases {
		s.status, s.respond = status, `{"error":{"message":"try later"}}`
		_, err := s.ask("hello?", s.heard())
		s.ErrorIs(err, want, "status %d", status)
	}

	// Out of prepaid credit, and a question the API refused: neither improves by waiting.
	for _, status := range []int{http.StatusPaymentRequired, http.StatusBadRequest} {
		s.status, s.respond = status, `{"error":{"message":"no"}}`
		_, err := s.ask("hello?", s.heard())
		s.ErrorContains(err, "gemini: the API returned")
		s.NotErrorIs(err, decisionmodel.ErrRateLimited)
		s.NotErrorIs(err, decisionmodel.ErrUnavailable)
	}
}

func (s *GeminiSuite) TestAnAPINobodyCanReachIsUnavailable() {
	client, err := New(Options{APIKey: "k", BaseURL: "http://127.0.0.1:1"})
	s.Require().NoError(err)

	_, err = client.Classify(s.ctx, decisionmodel.Request{State: "hello?", Questions: s.heard()})

	s.ErrorIs(err, decisionmodel.ErrUnavailable)
}
