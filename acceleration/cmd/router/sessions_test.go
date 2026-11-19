package main

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// TranscriptForSuite covers the transcript a session is written into in the app it is pinned to.
type TranscriptForSuite struct {
	suite.Suite
}

func TestTranscriptForSuite(t *testing.T) {
	suite.Run(t, new(TranscriptForSuite))
}

func (s *TranscriptForSuite) TestAVoiceTranscriptsChannelIsMadeInTheSessionsChannelTeam() {
	chat := chattest.NewServer(s.T())
	spec := session.Spec{AgentID: "front-desk", UserID: "agent", CustomerID: "customer-1", ChannelTeam: "red"}

	transcript, err := transcriptFor(false)(context.Background(), spec, streamapp.Bound{Client: chat.Client}, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.Require().NoError(transcript.Start(context.Background()))
	transcript.Close()

	created, ok := chat.Channel("front-desk")
	s.Require().True(ok)
	s.Equal("red", created["team"])
}

// EdgeForSuite covers the call edge a voice session joins with.
type EdgeForSuite struct {
	suite.Suite
}

func TestEdgeForSuite(t *testing.T) {
	suite.Run(t, new(EdgeForSuite))
}

// edge is the edge edgeFor joins a call with under phoneSettings.
func (s *EdgeForSuite) edge(phoneSettings config.Phone) *streamedge.Edge {
	factory, err := edgeFor(streamapp.NewClients(nil, streamapp.ClientsOptions{}), phoneSettings)
	s.Require().NoError(err)
	spec := session.Spec{CallID: "call-1", UserID: "agent"}
	bound := streamapp.Bound{Identity: streamapp.Identity{APIKey: "key", Secret: streamapp.NewSecret("secret")}}
	joined, err := factory(context.Background(), spec, bound, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	edge, ok := joined.(*streamedge.Edge)
	s.Require().True(ok)
	return edge
}

func (s *EdgeForSuite) TestWithEchoCancellationOnAPhoneCallersAudioIsCleanedAndNobodyElses() {
	edge := s.edge(config.Phone{EchoCancellation: true, EchoMaxDelay: time.Second})

	s.True(edge.EchoCancelled("sip-+15550100"))
	s.True(edge.EchoCancelled("sip-"), "a caller with no number is still on a phone")
	s.False(edge.EchoCancelled("alice"))
	s.False(edge.EchoCancelled("agent-sip-"))
}

func (s *EdgeForSuite) TestWithEchoCancellationOffNobodysAudioIsCleaned() {
	edge := s.edge(config.Phone{EchoMaxDelay: time.Second})

	s.False(edge.EchoCancelled("sip-+15550100"))
}

// AEC3 takes every delay in echodelay's bounds (echocancel's
// TestEveryLongestDelayInItsBoundsMakesACanceller), so a delay out of them is what a start
// refuses.
func (s *EdgeForSuite) TestAnEchoDelayOutOfItsBoundsIsRefusedAtStart() {
	_, err := edgeFor(streamapp.NewClients(nil, streamapp.ClientsOptions{}),
		config.Phone{EchoCancellation: true, EchoMaxDelay: 5 * time.Second})

	s.Error(err)
}
