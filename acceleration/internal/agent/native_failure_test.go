package agent

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
)

// A speech-to-speech model that failed is logged with the provider and model that failed,
// so a search for either finds the line: the failure is the event's, not the agent's.
func (s *AgentSuite) TestASpeechToSpeechFailureIsLoggedWithItsProviderAndModel() {
	for name, fatal := range map[string]bool{"session failed": true, "agent failure": false} {
		s.Run(name, func() {
			logs := &logLines{}
			agent, err := New(Options{
				CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2",
				Logger: slog.New(slog.NewJSONHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})),
			})
			s.Require().NoError(err)
			agent.ctx, agent.cancel = context.WithCancel(s.ctx)
			s.T().Cleanup(func() { agent.Close() })
			source := make(chan sts.Event, 1)
			s.T().Cleanup(func() { close(source) })
			ordered := make(chan sts.Event, 1)
			agent.pipe = newPipeline(agent.ctx, true)
			agent.pipe.running.Add(2)
			go agent.receiveSTS(agent.pipe, source, ordered)
			go agent.consumeSTS(agent.pipe, nil, ordered)

			source <- sts.Error{Provider: "gemini", Model: "live-model", Err: errors.New("502 Bad Gateway"), Fatal: fatal}

			message := "agent failure"
			if fatal {
				message = "the speech-to-speech session failed"
			}
			s.Eventually(func() bool { return len(logs.records(message)) == 1 }, 5*time.Second, time.Millisecond)
			line := logs.records(message)[0]
			s.Equal("gemini", line["provider"])
			s.Equal("live-model", line["model"])
		})
	}
}
