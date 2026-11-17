package main

import (
	"context"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

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
