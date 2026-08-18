//go:build integration

package agent_test

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

const fullAgentTimeout = 12 * time.Minute

type FullAgentIntegrationSuite struct {
	suite.Suite
	ctx       context.Context
	callID    string
	ttsTarget string

	llm *llmrouter.Router
	stt *sttrouter.Router
	tts *ttsrouter.Router
}

func TestFullAgentIntegrationSuite(t *testing.T) {
	suite.Run(t, new(FullAgentIntegrationSuite))
}

func (s *FullAgentIntegrationSuite) SetupSuite() {
	for _, name := range []string{
		"STREAM_API_KEY",
		"PARAKEET_WS_URL",
		"BASETEN_API_KEY",
		"GEMMA_BASE_URL",
		"OPENAI_API_KEY",
	} {
		if os.Getenv(name) == "" {
			s.T().Skip(name + " not set")
		}
	}
	if os.Getenv("STREAM_API_SECRET") == "" && os.Getenv("STREAM_USER_TOKEN") == "" {
		s.T().Skip("STREAM_API_SECRET or STREAM_USER_TOKEN must be set")
	}
	switch {
	case os.Getenv("S2PRO_WS_URL") != "":
		s.ttsTarget = "s2pro/s2-pro"
	case os.Getenv("FISH_API_KEY") != "":
		s.ttsTarget = "fish/s2-pro"
	default:
		s.T().Skip("S2PRO_WS_URL or FISH_API_KEY must be set")
	}

	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), fullAgentTimeout)
	s.T().Cleanup(cancel)
	s.callID = fmt.Sprintf("sprint6-agent-%d", time.Now().UnixNano())

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	logger := slog.New(slog.DiscardHandler)
	s.llm, err = llmrouter.New(llmrouter.Options{
		Config: config[routing.LLM], Registry: llmrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.llm.Close)
	s.stt, err = sttrouter.New(sttrouter.Options{
		Config: config[routing.STT], Registry: sttrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.stt.Close)
	s.tts, err = ttsrouter.New(ttsrouter.Options{
		Config: config[routing.TTS], Registry: ttsrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.tts.Close)
}

func (s *FullAgentIntegrationSuite) edge(userID string) *streamedge.Edge {
	edge, err := streamedge.New(streamedge.Options{
		CallID: s.callID,
		User:   streamedge.User{ID: userID, Name: userID},
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(edge.Join(s.ctx))
	s.T().Cleanup(func() { _ = edge.Leave() })
	return edge
}

func (s *FullAgentIntegrationSuite) TestStreamParakeetFishGemmaAndSolHoldAConversation() {
	caller := s.edge("sprint6-caller")

	agentEdge, err := streamedge.New(streamedge.Options{
		CallID: s.callID,
		User:   streamedge.User{ID: "sprint6-agent", Name: "Sprint 6 Agent"},
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)

	skills, err := harness.DefaultSkills()
	s.Require().NoError(err)
	voiceAgent, err := agent.New(agent.Options{
		Edge: agentEdge,
		Instructions: "You are a concise voice assistant. For every arithmetic question, " +
			"say that you will calculate it and delegate the calculation with the think skill. " +
			"Never calculate it on the voice model.",
		CustomerID:     "sprint6-integration",
		AgentID:        "sprint6-agent",
		CallID:         s.callID,
		LLM:            s.llm,
		LLMTarget:      "gemma/gemma-4-E2B-it",
		SubagentTarget: "openai/gpt-5.6-sol",
		Skills:         skills,
		STT:            s.stt,
		STTTarget:      "parakeet/parakeet-tdt-0.6b-v3",
		TTS:            s.tts,
		TTSTarget:      s.ttsTarget,
		Duplex:         agent.DuplexOptions{Backchannel: true},
		Logger:         slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(voiceAgent.Join(s.ctx))
	s.T().Cleanup(func() { _ = voiceAgent.Close() })

	synthesis, err := s.tts.Start(s.ctx, ttsrouter.Request{
		CustomerID: "sprint6-integration",
		Target:     s.ttsTarget,
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = synthesis.Close() })

	output, err := s.stt.Start(s.ctx, sttrouter.Request{
		CustomerID: "sprint6-integration",
		Target:     "parakeet/parakeet-tdt-0.6b-v3",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = output.Close() })

	audioErrors := make(chan error, 1)
	go func() {
		for inbound := range caller.Audio() {
			if err := output.ProcessAudio(inbound.Audio, inbound.Participant); err != nil {
				audioErrors <- err
				return
			}
		}
	}()

	agentEvents := make(chan agent.Event, 128)
	go func() {
		for event := range voiceAgent.Events() {
			agentEvents <- event
		}
		close(agentEvents)
	}()

	s.Require().NoError(synthesis.Synthesize(tts.Request{
		ID:    "caller-question",
		Text:  "What is fifteen percent of eighty four point two?",
		Final: true,
	}))
	s.publishSynthesis(caller, synthesis)

	var transcript strings.Builder
	var delegated, settled bool
	for {
		select {
		case err := <-audioErrors:
			s.FailNowf("output audio error", "%v", err)
		case event, open := <-agentEvents:
			if !open {
				s.FailNow("agent stopped before answering")
			}
			switch typed := event.(type) {
			case agent.Delegated:
				delegated = typed.Skill == "think"
			case agent.TaskSettled:
				settled = typed.Text != "" && typed.Err == nil
			case agent.Error:
				s.FailNowf("agent error", "%s: %v", typed.Context, typed.Err)
			}
		case event, open := <-output.Events():
			if !open {
				s.FailNow("output transcriber stopped before answering")
			}
			switch typed := event.(type) {
			case stt.Transcript:
				if typed.Final() {
					transcript.WriteString(" ")
					transcript.WriteString(strings.ToLower(typed.Text))
				}
			case stt.Error:
				s.FailNowf("output transcription error", "%v", typed.Err)
			}
		case <-s.ctx.Done():
			s.FailNowf("full agent timed out", "transcript: %s", transcript.String())
		}
		if delegated && settled && strings.Contains(transcript.String(), "12") {
			return
		}
	}
}

func (s *FullAgentIntegrationSuite) publishSynthesis(
	caller *streamedge.Edge,
	synthesis *ttsrouter.Session,
) {
	for {
		select {
		case event := <-synthesis.Events():
			switch typed := event.(type) {
			case tts.AudioChunk:
				s.Require().NoError(caller.PublishAudio(typed.Audio))
			case tts.SynthesisComplete:
				return
			case tts.Error:
				s.FailNowf("caller synthesis error", "%v", typed.Err)
			}
		case <-s.ctx.Done():
			s.FailNow("caller synthesis timed out")
		}
	}
}
