package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// judge stands in for a classifier: it answers what it is told to, or fails the way it is
// told to.
type judge struct {
	name     string
	answered lcm.Result
	err      error

	mu    sync.Mutex
	asked []lcm.Request
}

func (j *judge) Classify(_ context.Context, request lcm.Request) (lcm.Result, error) {
	j.mu.Lock()
	j.asked = append(j.asked, request)
	j.mu.Unlock()
	if j.err != nil {
		return lcm.Result{}, j.err
	}
	return j.answered, nil
}

func (j *judge) Start(context.Context) error { return nil }
func (j *judge) Close() error                { return nil }
func (j *judge) Provider() string            { return j.name }
func (j *judge) Model() string               { return "stub" }

type ClassifySuite struct {
	suite.Suite
	quick   *judge
	careful *judge
	router  *lcmrouter.Router
	handler http.Handler
}

func TestClassifySuite(t *testing.T) {
	suite.Run(t, new(ClassifySuite))
}

func (s *ClassifySuite) SetupTest() {
	s.quick = &judge{name: "quick", answered: lcm.Result{
		Model: "judge-2026-09",
		Answers: map[string]lcm.Answer{
			"refund": {Type: lcm.TypeNoul, Yes: 0.91},
			"topic": {
				Type: lcm.TypeChoice, Chosen: "billing", Confidence: 0.8,
				Probabilities: map[string]float64{"billing": 0.9, "other": 0.1},
			},
			"urgency": {
				Type: lcm.TypeScore, Level: 1.4, Confidence: 0.6,
				Legend:        map[string]string{"0": "can wait", "1": "this week", "2": "today"},
				Probabilities: map[string]float64{"0": 0.1, "1": 0.4, "2": 0.5},
			},
		},
		Usage: lcm.Usage{InputTokens: 406, OutputTokens: 69},
	}}
	s.careful = &judge{name: "careful", answered: s.quick.answered}

	registry := lcmrouter.NewRegistry()
	registry.Register("quick", func(routing.Spec) (lcm.Provider, error) { return s.quick, nil })
	registry.Register("careful", func(routing.Spec) (lcm.Provider, error) { return s.careful, nil })
	router, err := lcmrouter.New(lcmrouter.Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{
					Provider: "quick", Model: "judge", Languages: []string{"en"},
					Realtime: true, Tier: routing.LowLatency,
				},
				{
					Provider: "careful", Model: "judge", Languages: []string{"en"},
					Tier: routing.HighQuality,
				},
			},
			Aliases: map[string]routing.Alias{
				"classify-fast": {RequireRealtime: true, Tier: routing.LowLatency},
			},
		},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.router = router

	authenticator, err := auth.New(auth.Proxy, nil)
	s.Require().NoError(err)
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LCM: router},
		Streams: &Streams{LCM: router},
		Auth:    authenticator,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
}

// allThree asks one question of each type about a support message.
const allThree = `{
	"state": "I was charged twice this month and nobody has answered my email.",
	"questions": {
		"refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
		"topic": {"type": "choice", "instructions": "What is this about?",
			"options": {"billing": "charges and invoices", "other": ""}},
		"urgency": {"type": "score", "instructions": "How soon does this need an answer?",
			"levels": ["can wait", "this week", "today"]}
	}
}`

func (s *ClassifySuite) classify(body string, headers ...string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Set(headers[i], headers[i+1])
	}
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}

func (s *ClassifySuite) result(recorder *httptest.ResponseRecorder) ClassifyResult {
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())
	var result ClassifyResult
	s.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), &result))
	return result
}

func (s *ClassifySuite) TestEachQuestionComesBackUnderItsOwnId() {
	result := s.result(s.classify(allThree))

	s.Equal("quick", result.Provider)
	s.Equal("judge-2026-09", result.Model, "the version that answered, not the alias")
	s.Require().Len(result.Answers, 3)

	refund := result.Answers["refund"]
	s.Equal(Noul, refund.Type)
	s.InDelta(0.91, *refund.Yes, 1e-9)
	s.Nil(refund.Confidence, "a noul's probability is the answer and has no confidence beside it")

	topic := result.Answers["topic"]
	s.Equal(Choice, topic.Type)
	s.Equal("billing", *topic.Chosen)
	s.InDelta(0.9, (*topic.Probabilities)["billing"], 1e-9)
	s.Nil(topic.Yes)

	urgency := result.Answers["urgency"]
	s.Equal(Score, urgency.Type)
	s.InDelta(1.4, *urgency.Level, 1e-9)
	s.Equal("today", (*urgency.Legend)["2"])

	s.Equal(ClassifyUsage{InputTokens: 406, OutputTokens: 69}, result.Usage)
}

func (s *ClassifySuite) TestEveryQuestionIsAskedInOneRequest() {
	s.result(s.classify(allThree))

	s.Require().Len(s.quick.asked, 1)
	asked := s.quick.asked[0]
	s.Equal("I was charged twice this month and nobody has answered my email.", asked.State)
	s.Len(asked.Questions, 3)
	s.Equal(lcm.Score("How soon does this need an answer?", []string{"can wait", "this week", "today"}),
		asked.Questions["urgency"])
}

func (s *ClassifySuite) TestAStateWithPartsIsPassedAsItWasSent() {
	s.result(s.classify(`{
		"state": {"message": "refund me", "channel": "email"},
		"questions": {"refund": {"type": "noul", "instructions": "Does ` + "`message`" + ` ask for money back?"}}
	}`))

	s.Require().Len(s.quick.asked, 1)
	s.Equal(map[string]any{"message": "refund me", "channel": "email"}, s.quick.asked[0].State)
}

func (s *ClassifySuite) TestATargetPicksTheClassifier() {
	result := s.result(s.classify(`{
		"target": "careful/judge",
		"state": "refund me",
		"questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}
	}`))

	s.Equal("careful", result.Provider)
	s.Empty(s.quick.asked)
}

func (s *ClassifySuite) TestAFailedJudgementIsAnErrorRatherThanAnEmptyAnswer() {
	s.quick.err = errors.New(`typesafe: "refund" was not answered`)

	recorder := s.classify(allThree)

	s.Equal(http.StatusBadRequest, recorder.Code)
	s.Contains(recorder.Body.String(), "was not answered")
}

func (s *ClassifySuite) TestAFailureWorthWaitingOutSaysSo() {
	cases := map[error]int{
		fmt.Errorf("typesafe: the API returned 429: %w", lcm.ErrRateLimited): http.StatusTooManyRequests,
		fmt.Errorf("typesafe: the API returned 529: %w", lcm.ErrUnavailable): http.StatusServiceUnavailable,
		errors.New("typesafe: the API returned 422: bad criteria"):           http.StatusBadRequest,
	}
	for failure, status := range cases {
		s.quick.err = failure

		recorder := s.classify(allThree)

		s.Equal(status, recorder.Code, failure.Error())
		s.Contains(recorder.Body.String(), failure.Error())
	}
}

func (s *ClassifySuite) TestATargetNobodyRoutesIsNotFound() {
	recorder := s.classify(`{"target": "nobody/nothing", "state": "hi",
		"questions": {"q": {"type": "noul", "instructions": "Is it?"}}}`)

	s.Equal(http.StatusNotFound, recorder.Code)
	s.Contains(recorder.Body.String(), "unknown target")
	s.Empty(s.quick.asked)
}

func (s *ClassifySuite) TestARequestThatCannotBeAnsweredIsRefusedBeforeAnythingIsAsked() {
	cases := map[string]string{
		"no state":        `{"questions": {"q": {"type": "noul", "instructions": "Is it?"}}}`,
		"blank state":     `{"state": "  ", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}`,
		"no questions":    `{"state": "hi", "questions": {}}`,
		"no instructions": `{"state": "hi", "questions": {"q": {"type": "noul", "instructions": " "}}}`,
		"unknown type":    `{"state": "hi", "questions": {"q": {"type": "maybe", "instructions": "Is it?"}}}`,
		"one option": `{"state": "hi", "questions": {"q": {"type": "choice", "instructions": "Which?",
			"options": {"only": ""}}}}`,
		"one level": `{"state": "hi", "questions": {"q": {"type": "score", "instructions": "How much?",
			"levels": ["some"]}}}`,
	}
	for name, body := range cases {
		recorder := s.classify(body)
		s.Equal(http.StatusBadRequest, recorder.Code, name)
	}
	s.Empty(s.quick.asked)
	s.Empty(s.careful.asked)
}

func (s *ClassifySuite) TestADeviceMayNotSpendOnTheCustomersAccount() {
	recorder := s.classify(allThree, auth.AuthTypeHeader, auth.AuthTypeJWT)

	s.Equal(http.StatusForbidden, recorder.Code)
	s.Empty(s.quick.asked)
}

func (s *ClassifySuite) TestADeploymentWithNoClassifierSaysSo() {
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LCM: s.router},
		Streams: &Streams{},
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(allThree))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, request)

	s.Equal(http.StatusNotFound, recorder.Code)
	s.Contains(recorder.Body.String(), "does not route classification")
}
