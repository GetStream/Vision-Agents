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

func (s *HarnessSuite) TestAContinuedTurnSendsOpenAINothingTheModelWrote() {
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
		asked := len(bodies)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(event string) { fmt.Fprintf(w, "data: %s\n\n", event) }
		if asked == 1 {
			send(`{"type":"response.output_text.delta","delta":"Putting you through."}`)
			send(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc-1","call_id":"call-1","name":"transfer"}}`)
			send(`{"type":"response.function_call_arguments.delta","item_id":"fc-1","delta":"{\"to\":\"+15550001111\"}"}`)
			send(`{"type":"response.completed","response":{"id":"provider-transfer","status":"completed"}}`)
			return
		}
		send(`{"type":"response.output_text.delta","delta":"You are through."}`)
		send(`{"type":"response.completed","response":{"id":"provider-answer","status":"completed"}}`)
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
	ask := func(turn Turn) {
		stream, err := h.Respond(s.ctx, turn)
		s.Require().NoError(err)
		response, err := llm.Collect(stream)
		s.Require().NoError(err)
		s.Require().Equal(llm.StatusCompleted, response.Status)
		h.Remember(response)
	}
	type item struct {
		Type   string `json:"type"`
		Role   string `json:"role"`
		CallID string `json:"call_id"`
	}
	sent := func(request int) (string, []item) {
		mu.Lock()
		defer mu.Unlock()
		var body struct {
			Previous string `json:"previous_response_id"`
			Input    []item `json:"input"`
		}
		s.Require().NoError(json.Unmarshal(bodies[request], &body))
		return body.Previous, body.Input
	}

	history := []llm.Message{{Role: llm.User, Content: "put me through"}}
	ask(Turn{ID: "transfer", History: history})
	// The conversation keeps each reply as it was written, with the calls on it.
	history = append(history,
		llm.Message{Role: llm.Assistant, Content: "Putting you through.",
			ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "transfer", Arguments: `{"to":"+15550001111"}`}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "call-1", Content: "transferred"})
	ask(Turn{ID: "result", History: history, AfterTool: true})
	previous, input := sent(1)
	s.Equal("provider-transfer", previous)
	s.Equal([]item{{Type: "function_call_output", CallID: "call-1"}}, input,
		"the provider kept the reply and its call, so only the result is sent")

	history = append(history,
		llm.Message{Role: llm.Assistant, Content: "You are through."},
		llm.Message{Role: llm.User, Content: "thanks"})
	ask(Turn{ID: "thanks", History: history})
	previous, input = sent(2)
	s.Equal("provider-answer", previous)
	s.Equal([]item{{Role: "user"}}, input, "the provider kept the answer, so only what the caller said after it is sent")
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
