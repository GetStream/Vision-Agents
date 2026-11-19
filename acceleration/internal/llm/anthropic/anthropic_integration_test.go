//go:build integration

package anthropic

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

// AnthropicIntegrationSuite needs nothing of its own for the shared tests: Claude's thinking
// streams as empty blocks unless a display is asked for, so the suite sees a model that
// answers without reasoning.
type AnthropicIntegrationSuite struct {
	llmsuite.Suite
}

// Opus 5.5 is the one that takes a note mid-conversation as a system turn (systemTurnModels);
// the others take it on the user's side, so between them both ways are run.
func TestAnthropicIntegrationSuite(t *testing.T) {
	for _, model := range []string{"claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-2"} {
		t.Run(model, func(t *testing.T) {
			suite.Run(t, &AnthropicIntegrationSuite{Suite: llmsuite.Suite{
				New: func() llm.LLM {
					provider, err := New(Options{Model: model})
					require.NoError(t, err)
					return provider
				},
				Requires: []string{"ANTHROPIC_API_KEY"},
			}})
		})
	}
}

// TestTheNextRoundReadsThePrefixFromTheCache asks twice with the same long instructions and
// a conversation one round longer, and expects the second to be served mostly from cache.
// The instructions end on a random text, so the first round writes a prefix no earlier test
// or run has cached.
func (s *AnthropicIntegrationSuite) TestTheNextRoundReadsThePrefixFromTheCache() {
	instructions := "Answer in one word. " + strings.Repeat("The webhook retries a failed delivery three times, a minute apart. ", 300) + rand.Text()
	first := llm.ResponseParams{Instructions: instructions, Input: []llm.Message{{Role: llm.User, Content: "How many retries?"}}}
	answer, _ := s.Ask(first)
	s.Positive(answer.Usage.InputTokensDetails.CacheWriteTokens, "the first round writes the prefix")

	second := first
	second.Input = append(append([]llm.Message(nil), first.Input...),
		llm.Message{Role: llm.Assistant, Content: answer.OutputText},
		llm.Message{Role: llm.User, Content: "And how far apart?"})
	again, _ := s.Ask(second)

	s.Greater(again.Usage.InputTokensDetails.CachedTokens, again.Usage.InputTokens/2)
}

// TestThinkingIsReplayedThroughAToolRound sends a tool round's signed thinking back with its
// result. What Claude streamed is recorded on the wire, so the test knows whether there was
// thinking to keep without asking the adapter, and the turn sent back is compared with it
// block for block: some models take a moved block without a word, so an answer alone proves
// nothing about where it went.
func (s *AnthropicIntegrationSuite) TestThinkingIsReplayedThroughAToolRound() {
	wire := &recorder{}
	upstream, err := url.Parse("https://api.anthropic.com")
	s.Require().NoError(err)
	proxy := &httputil.ReverseProxy{Rewrite: func(r *httputil.ProxyRequest) {
		r.SetURL(upstream)
		r.Out.Header.Del("Accept-Encoding") // the recorder reads the stream as sent, not compressed
	}, ModifyResponse: wire.response}
	server := httptest.NewServer(wire.requests(proxy))
	defer server.Close()
	provider, err := New(Options{Model: s.LLM.Model(), BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	clock := llm.Tool{Name: "clock", Description: "Says the time in a city.", Parameters: map[string]any{
		"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"},
	}}
	question := llm.Message{Role: llm.User, Content: "What time is it in Paris? Think it over, then use the tool."}
	first := llm.ResponseParams{Instructions: "Use the tool to answer.", Input: []llm.Message{question}, Tools: []llm.Tool{clock}, Reasoning: llm.ReasoningParams{Effort: "high"}}
	called, _ := s.AskOn(provider, first)
	s.Require().NotEmpty(called.ToolCalls)
	written := wire.streamed(s.T())
	if !slices.ContainsFunc(written, func(block map[string]any) bool {
		return block["type"] == "redacted_thinking" || block["type"] == "thinking" && block["signature"] != ""
	}) {
		s.T().Skip("Claude called without signed thinking, so there is nothing to replay")
	}
	s.Require().NotEmpty(called.ReasoningBlocks, "Claude signed its thinking and the adapter dropped it")

	second := first
	second.Input = []llm.Message{question, {Role: llm.Assistant, Content: called.OutputText, ToolCalls: called.ToolCalls, ReasoningBlocks: called.ReasoningBlocks}}
	for _, call := range called.ToolCalls {
		second.Input = append(second.Input, llm.Message{Role: llm.ToolResult, ToolCallID: call.ID, Content: "14:05"})
	}
	answered, _ := s.AskOn(provider, second)

	sent, err := json.Marshal(written)
	s.Require().NoError(err)
	s.JSONEq(string(sent), wire.replayedTurn(s.T()), "the turn goes back block for block as Claude wrote it")
	s.Equal(llm.StatusCompleted, answered.Status)
	s.Contains(answered.OutputText, "14")
}

// recorder keeps each exchange between the adapter and the Messages API, with its status:
// the SDK retries a failed one, and only the ones that succeeded are the conversation.
type recorder struct {
	mu        sync.Mutex
	exchanges []exchange
}

type exchange struct {
	sent   []byte
	status int
	got    []byte
}

type sentKey struct{}

func (r *recorder) requests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(body))
		next.ServeHTTP(w, req.WithContext(context.WithValue(req.Context(), sentKey{}, body)))
	})
}

func (r *recorder) response(response *http.Response) error {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	response.Body = io.NopCloser(bytes.NewReader(body))
	sent, _ := response.Request.Context().Value(sentKey{}).([]byte)
	r.mu.Lock()
	r.exchanges = append(r.exchanges, exchange{sent: sent, status: response.StatusCode, got: body})
	r.mu.Unlock()
	return nil
}

// succeeded are the exchanges the API answered with a 200, in order.
func (r *recorder) succeeded(t *testing.T, want int) []exchange {
	r.mu.Lock()
	defer r.mu.Unlock()
	var ok []exchange
	for _, e := range r.exchanges {
		if e.status == http.StatusOK {
			ok = append(ok, e)
		}
	}
	require.Len(t, ok, want)
	return ok
}

// streamed is the content of the first successful response, rebuilt from its events as a request
// would carry it.
func (r *recorder) streamed(t *testing.T) []map[string]any {
	first := r.succeeded(t, 1)[0]
	var blocks []map[string]any
	inputs := map[int]*strings.Builder{}
	for _, line := range strings.Split(string(first.got), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type         string         `json:"type"`
			Index        int            `json:"index"`
			ContentBlock map[string]any `json:"content_block"`
			Delta        map[string]any `json:"delta"`
		}
		require.NoError(t, json.Unmarshal([]byte(data), &event))
		switch event.Type {
		case "content_block_start":
			// Only what a request carries back: a start can name more than that.
			kept := map[string][]string{"text": {"text"}, "thinking": {"thinking", "signature"}, "redacted_thinking": {"data"}, "tool_use": {"id", "name"}}
			block := map[string]any{"type": event.ContentBlock["type"]}
			for _, key := range kept[event.ContentBlock["type"].(string)] {
				block[key] = event.ContentBlock[key]
			}
			if block["type"] == "tool_use" {
				inputs[event.Index] = &strings.Builder{}
			}
			blocks = append(blocks, block)
		case "content_block_delta":
			block := blocks[len(blocks)-1]
			switch event.Delta["type"] {
			case "text_delta":
				block["text"] = block["text"].(string) + event.Delta["text"].(string)
			case "thinking_delta":
				block["thinking"] = block["thinking"].(string) + event.Delta["thinking"].(string)
			case "signature_delta":
				block["signature"] = event.Delta["signature"]
			case "input_json_delta":
				inputs[event.Index].WriteString(event.Delta["partial_json"].(string))
			}
		case "content_block_stop":
			if input, ok := inputs[event.Index]; ok {
				var parsed any = map[string]any{}
				if input.Len() > 0 {
					require.NoError(t, json.Unmarshal([]byte(input.String()), &parsed))
				}
				blocks[len(blocks)-1]["input"] = parsed
			}
		}
	}
	return blocks
}

// replayedTurn is the assistant turn of the request that continued successfully, without its cache breakpoints.
func (r *recorder) replayedTurn(t *testing.T) string {
	continued := r.succeeded(t, 2)[1]
	var request struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(continued.sent, &request))
	turn := request.Messages[1].Content
	for _, block := range turn {
		delete(block, "cache_control")
	}
	encoded, err := json.Marshal(turn)
	require.NoError(t, err)
	return string(encoded)
}
