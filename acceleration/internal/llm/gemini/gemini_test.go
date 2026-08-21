package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type GeminiSuite struct {
	suite.Suite

	// server speaks chat completions so what the provider puts on the wire can be read
	// back, which is the only place the reasoning setting is observable.
	server *httptest.Server
	// request is the decoded body of the last request served.
	request map[string]any
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

func (s *GeminiSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.request = nil

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		s.Require().NoError(err)
		s.Require().NoError(json.Unmarshal(raw, &s.request))

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		s.Require().True(ok)
		fmt.Fprint(w, "data: "+
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	s.T().Cleanup(s.server.Close)
}

// ask runs one completion against the test server and waits for it to settle.
func (s *GeminiSuite) ask(options Options) {
	options.APIKey = "k"
	options.BaseURL = s.server.URL

	provider, err := New(options)
	s.Require().NoError(err)
	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { provider.Close() })

	s.Require().NoError(provider.Respond(llm.Request{
		ID:       "c1",
		Messages: []llm.Message{{Role: llm.User, Content: "hello"}},
	}))

	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			if _, done := event.(llm.CompletionComplete); done {
				return
			}
		case <-deadline:
			s.FailNow("the completion never settled")
			return
		}
	}
}

func (s *GeminiSuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *GeminiSuite) TestADefaultModelIsUsedWhenNoneIsNamed() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.Equal(defaultModel, provider.Model())
}

func (s *GeminiSuite) TestModelIsSentUnqualified() {
	// Google's model ids carry no owner prefix, unlike the open-weight providers.
	provider, err := New(Options{APIKey: "k", Model: "gemini-3.6-flash"})
	s.Require().NoError(err)

	s.Equal("gemini-3.6-flash", provider.Model())
}

func (s *GeminiSuite) TestThinkingIsHeldAtItsMinimumForAConversation() {
	// Every Gemini 3 model thinks and none of them can be told not to, so the least it
	// will do is the most a live turn can afford. Left unset, Google picks the model's
	// own default, which on the Flash models is higher than this.
	s.ask(Options{})

	s.Equal(minimalEffort, s.request["reasoning_effort"])
}

func (s *GeminiSuite) TestMoreThinkingCanBeAskedForOffTheLivePath() {
	s.ask(Options{ReasoningEffort: "high"})

	s.Equal("high", s.request["reasoning_effort"])
}

func (s *GeminiSuite) TestReasoningIsNotClaimedBecauseGoogleDoesNotStreamIt() {
	// Google reports thinking as a token count and keeps the text, so there is nothing
	// for the session to separate out of the answer.
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.False(provider.Reasoning())
}
