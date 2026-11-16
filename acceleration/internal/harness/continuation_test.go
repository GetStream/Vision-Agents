package harness

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func (s *HarnessSuite) TestAnOpenAIAccountWithZeroDataRetentionAnswersEveryTurn() {
	var mu sync.Mutex
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		var request struct {
			Previous string `json:"previous_response_id"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// What OpenAI answers an organisation that keeps nothing, so has nothing to continue.
		if request.Previous != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"message":"Previous response cannot be used for this organization due to Zero Data Retention.","type":"invalid_request_error","param":"previous_response_id","code":"unsupported_parameter"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Yes, until nine.\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"provider-reply\",\"status\":\"completed\"}}\n\n")
	}))
	s.T().Cleanup(server.Close)
	registry := llmrouter.NewRegistry()
	registry.Register("openai", func(routing.Spec) (llmrouter.Provider, error) {
		return llmrouter.Started(openai.New(openai.Options{APIKey: "test", Model: "gpt-6.1-sol", BaseURL: server.URL}))
	})
	config := stubConfig()
	config.Providers[0].Provider, config.Providers[0].Model = "openai", "gpt-6.1-sol"
	router, err := llmrouter.New(llmrouter.Options{Config: config, Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, llmrouter.Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)
	h, err := New(Options{Model: session})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = h.Close() })

	history := []llm.Message{{Role: llm.User, Content: "Is the pool open today?"}}
	stream, err := h.Respond(s.ctx, Turn{ID: "first", Instructions: "Answer as the front desk.", History: history})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)
	s.Require().Equal(llm.StatusCompleted, response.Status)
	h.Remember(response)

	history = append(history,
		llm.Message{Role: llm.Assistant, Content: response.OutputText},
		llm.Message{Role: llm.User, Content: "And tomorrow?"})
	stream, err = h.Respond(s.ctx, Turn{ID: "second", Instructions: "Answer as the front desk.", History: history})
	s.Require().NoError(err, "the second turn must not ask OpenAI to continue a response it never kept")
	response, err = llm.Collect(stream)
	s.Require().NoError(err)
	s.Equal(llm.StatusCompleted, response.Status)

	mu.Lock()
	defer mu.Unlock()
	s.Require().Len(bodies, 2)
	var sent struct {
		Input []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"input"`
	}
	s.Require().NoError(json.Unmarshal(bodies[1], &sent))
	s.Require().Len(sent.Input, 4, "the second turn carries the whole conversation")
	for i, want := range []struct{ role, text string }{
		{"developer", "Answer as the front desk."},
		{"user", "Is the pool open today?"},
		{"assistant", "Yes, until nine."},
		{"user", "And tomorrow?"},
	} {
		s.Equal(want.role, sent.Input[i].Role)
		s.Contains(string(sent.Input[i].Content), want.text)
	}
}

func (s *HarnessSuite) TestATokenLimitedToolCallDoesNotPoisonTheNextTurn() {
	var asked atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Previous string `json:"previous_response_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// The provider kept a call whose arguments never finished. There can be no
		// result for that call, so continuing its response is refused.
		if request.Previous != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"message":"No tool output found for function call call-cut-off.","type":"invalid_request_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(event string) { fmt.Fprintf(w, "data: %s\n\n", event) }
		if asked.Add(1) == 1 {
			send(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc-cut-off","call_id":"call-cut-off","name":"transfer"}}`)
			send(`{"type":"response.function_call_arguments.delta","item_id":"fc-cut-off","delta":"{\"to\":"}`)
			send(`{"type":"response.incomplete","response":{"id":"provider-cut-off","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"output_tokens":256}}}`)
			return
		}
		send(`{"type":"response.output_text.delta","delta":"Please confirm the number."}`)
		send(`{"type":"response.completed","response":{"id":"provider-recovered","status":"completed"}}`)
	}))
	s.T().Cleanup(server.Close)
	registry := llmrouter.NewRegistry()
	registry.Register("openai", func(routing.Spec) (llmrouter.Provider, error) {
		return llmrouter.Started(openai.New(openai.Options{APIKey: "test", Model: "gpt-6-sol", BaseURL: server.URL}))
	})
	config := stubConfig()
	config.Providers[0].Provider, config.Providers[0].Model = "openai", "gpt-6-sol"
	router, err := llmrouter.New(llmrouter.Options{Config: config, Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, llmrouter.Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)
	h, err := New(Options{Model: session, Tools: testTools()})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = h.Close() })
	history := []llm.Message{{Role: llm.User, Content: "transfer me"}}
	stream, err := h.Respond(s.ctx, Turn{ID: "cut-off", History: history})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)
	s.Equal(llm.StatusIncomplete, response.Status)
	s.Equal(llm.ReasonMaxOutputTokens, response.IncompleteReason)
	s.Empty(response.ToolCalls, "unfinished arguments must never be executed")
	s.Equal("provider-cut-off", response.ProviderResponseID)
	h.Remember(response)
	stream, err = h.Respond(s.ctx, Turn{ID: "recovery", History: history})
	s.Require().NoError(err, "recovery must not continue the provider's unfinished tool call")
	response, err = llm.Collect(stream)
	s.Require().NoError(err)
	s.Equal(llm.StatusCompleted, response.Status)
	s.Equal("Please confirm the number.", response.OutputText)
}

func (s *HarnessSuite) TestAFailedStoredReplyIsNotContinued() {
	s.assertUnfinishedReplyIsNotContinued(llm.StatusFailed)
}

func (s *HarnessSuite) TestACancelledStoredReplyIsNotContinued() {
	s.assertUnfinishedReplyIsNotContinued(llm.StatusCancelled)
}

func (s *HarnessSuite) assertUnfinishedReplyIsNotContinued(status llm.ResponseStatus) {
	s.build(false)
	s.fast.capabilities.Store = true
	history := []llm.Message{{Role: llm.User, Content: "find my booking"}}
	s.answer(Turn{ID: "lookup", Instructions: "be brief", History: history})
	s.harness.Remember(llm.Response{ID: "lookup", Status: status,
		Provider: "stub", Model: "stub-model", ProviderResponseID: "provider-unfinished"})
	s.answer(Turn{ID: "recover", Instructions: "be brief", History: history})
	sent := s.fast.requests()[1]
	s.Empty(sent.PreviousResponseID)
	s.Equal(history, sent.Input)
}
