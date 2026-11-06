package llmrouter

import (
	"context"
	"crypto/rand"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// replayedLLM answers every request with the same events, whatever a provider sent.
type replayedLLM struct {
	*stubLLM
	events []llm.Event
}

func (r *replayedLLM) Create(context.Context, llm.ResponseParams) (*llm.Stream, error) {
	return llm.Replay(r.events...), nil
}

// ErrorMessageSuite asks a chat completions endpoint that answers 502 with a plain body, the
// way a self-hosted model's gateway does when the model behind it is down.
type ErrorMessageSuite struct {
	suite.Suite
	server *httptest.Server
	// body is what the endpoint answers the next request with.
	body atomic.Value
}

func TestErrorMessageSuite(t *testing.T) {
	suite.Run(t, new(ErrorMessageSuite))
}

func (s *ErrorMessageSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(s.body.Load().(string)))
	}))
}

func (s *ErrorMessageSuite) SetupTest() {
	s.body.Store("failed to connect to local service")
}

// timed asks provider for one response under the config's names and returns the timing
// the caller was told and the error Create returned.
func (s *ErrorMessageSuite) timed(provider Provider) (llm.CallTiming, error) {
	recorder := routing.NewRecorder(routing.LLM, nil, nil, nil, slog.Default())
	s.T().Cleanup(recorder.Close)
	session := newSession(provider, routing.ProviderConfig{Provider: "gemma", Model: "gemma-4"},
		routing.Owner{CustomerID: "acme"}, recorder, nil)
	var timings []llm.CallTiming
	stream, err := session.Create(context.Background(), llm.ResponseParams{
		ID: "flow-1", Purpose: "flow", TurnID: "turn-1", Input: prompt(),
		OnTiming: func(timing llm.CallTiming) { timings = append(timings, timing) },
	})
	if err == nil {
		_ = drain(stream)
	}
	s.Require().Len(timings, 1)
	return timings[0], err
}

// gateway is a chat completions provider behind the suite's endpoint.
func (s *ErrorMessageSuite) gateway() Provider {
	provider, err := Started(openaicompat.New(openaicompat.Options{
		Provider: "gemma", Model: "gemma-4", APIKey: "k", BaseURL: s.server.URL,
	}))
	s.Require().NoError(err)
	return provider
}

func (s *ErrorMessageSuite) TestAnEndpointThatAnswers502SaysWhatItAnswered() {
	timing, err := s.timed(s.gateway())

	s.Require().Error(err)
	s.False(timing.Success)
	s.Equal("create_failed", timing.ErrorCode)
	s.Contains(timing.ErrorMessage, "502 Bad Gateway")
	s.Contains(timing.ErrorMessage, "failed to connect to local service")
}

func (s *ErrorMessageSuite) TestAKeyTheEndpointEchoesIsRedactedAsTheRequestsTableRedactsIt() {
	token := "sk-test-" + rand.Text()
	s.body.Store("failed to connect to local service\nAuthorization: Bearer " + token + "\nX-Api-Key: " + token)

	timing, err := s.timed(s.gateway())

	s.Require().Error(err)
	s.NotContains(timing.ErrorMessage, token)
	s.NotContains(timing.ErrorMessage, "Bearer")
	s.Contains(timing.ErrorMessage, "failed to connect to local service")
	s.Contains(timing.ErrorMessage, "[redacted]")
	s.Equal(store.SafeLogText(err.Error()), timing.ErrorMessage, "the one rule the requests table stores it by")
}

func (s *ErrorMessageSuite) TestAStreamThatFailsMidAnswerSaysWhy() {
	token := "sk-test-" + rand.Text()
	timing, err := s.timed(&replayedLLM{stubLLM: newStubLLM(), events: []llm.Event{
		llm.OutputTextDelta{Delta: "Paris"},
		llm.ResponseFailed{Err: errors.New("stream reset by peer, Authorization: Bearer " + token)},
		llm.ResponseCompleted{Response: llm.Response{Status: llm.StatusFailed}},
	}})

	s.Require().NoError(err)
	s.False(timing.Success)
	s.Equal("provider_error", timing.ErrorCode)
	s.Contains(timing.ErrorMessage, "stream reset by peer")
	s.NotContains(timing.ErrorMessage, token)
}

func (s *ErrorMessageSuite) TestAFailureTheResponseRanToItsEndAfterSaysNothing() {
	timing, err := s.timed(&replayedLLM{stubLLM: newStubLLM(), events: []llm.Event{
		llm.ResponseFailed{Err: errors.New("a hiccup the provider recovered from")},
		llm.ResponseCompleted{Response: llm.Response{Status: llm.StatusIncomplete, IncompleteReason: llm.ReasonMaxOutputTokens}},
	}})

	s.Require().NoError(err)
	s.True(timing.Success)
	s.Empty(timing.ErrorCode)
	s.Empty(timing.ErrorMessage, "a served call's timing is the one it always was")
}

func (s *ErrorMessageSuite) TestACancelWithNoFailureSaysNothing() {
	timing, err := s.timed(&replayedLLM{stubLLM: newStubLLM(), events: []llm.Event{
		llm.ResponseCompleted{Response: llm.Response{Status: llm.StatusCancelled}},
	}})

	s.Require().NoError(err)
	s.Equal(routing.ErrorCancelled, timing.ErrorCode)
	s.Empty(timing.ErrorMessage)
}
