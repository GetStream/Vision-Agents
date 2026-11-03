package harness

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/meta"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// metaWire is a Meta endpoint that refuses a tool_choice other than auto, as the hosted API
// does, and keeps what each request carried.
type metaWire struct {
	mu       sync.Mutex
	requests []map[string]any
}

func (w *metaWire) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if json.NewDecoder(r.Body).Decode(&request) != nil {
		http.Error(rw, "bad request", http.StatusBadRequest)
		return
	}
	w.mu.Lock()
	w.requests = append(w.requests, request)
	w.mu.Unlock()
	if choice, ok := request["tool_choice"]; ok && choice != "auto" {
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(rw, `{"error":{"message":"only \"auto\" is supported for tool_choice","param":"tool_choice","type":"invalid_request_error"}}`)
		return
	}
	rw.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(rw, "data: {\"id\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"It is on its way.\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
}

func (w *metaWire) seen() []map[string]any {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]map[string]any(nil), w.requests...)
}

// buildOnMeta is build with the real Meta adapter as the fast model.
func (s *HarnessSuite) buildOnMeta(wire *metaWire) {
	server := httptest.NewServer(wire)
	s.T().Cleanup(server.Close)
	s.T().Setenv("META_API_KEY", "test")

	registry := llmrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return llmrouter.Started(meta.New(meta.Options{BaseURL: server.URL + "/v1"}))
	})
	router, err := llmrouter.New(llmrouter.Options{
		Config: stubConfig(), Registry: registry, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, llmrouter.Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	harness, err := New(Options{Model: session, Tools: s.tools, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.harness = harness
	s.events = collect(harness)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = harness.Close() })
}

func (s *HarnessSuite) metaReply(turn Turn) string {
	stream, err := s.harness.Respond(s.ctx, turn)
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)
	return response.OutputText
}

func (s *HarnessSuite) TestAMetaReplyThatMayCallNoToolIsSentWithoutToolsOrToolChoice() {
	for name, turn := range map[string]Turn{
		"answers":  {Answers: true},
		"no tools": {NoTools: true},
	} {
		s.Run(name, func() {
			s.tools = testTools()
			wire := &metaWire{}
			s.buildOnMeta(wire)

			turn.ID = "meta-" + name
			turn.Instructions = "be brief"
			turn.History = []llm.Message{{Role: llm.User, Content: "where is my order"}}
			s.Equal("It is on its way.", s.metaReply(turn))

			s.Require().Len(wire.seen(), 1)
			s.NotContains(wire.seen()[0], "tools")
			s.NotContains(wire.seen()[0], "tool_choice")
		})
	}
}

func (s *HarnessSuite) TestAMetaReplyOfferedToolsIsUnchanged() {
	s.tools = testTools()
	wire := &metaWire{}
	s.buildOnMeta(wire)

	s.metaReply(Turn{
		ID:           "meta-tools",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "where is my order"}},
	})

	s.Require().Len(wire.seen(), 1)
	s.Len(wire.seen()[0]["tools"], 2)
	s.NotContains(wire.seen()[0], "tool_choice")
}
