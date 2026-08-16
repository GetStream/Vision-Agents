//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	protoLogger "github.com/livekit/protocol/logger"
	lksdk "github.com/livekit/server-sdk-go/v2"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
)

// TranscribeIntegrationSuite runs the demo's own wiring against a real LiveKit room: one
// participant publishes speech, the bot subscribes and the transcripts come back through
// the router. This is the end-to-end path, Opus encode and decode included.
type TranscribeIntegrationSuite struct {
	suite.Suite

	url       string
	apiKey    string
	apiSecret string
	audio     stt.PcmData
	logger    *slog.Logger
}

func TestTranscribeIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TranscribeIntegrationSuite))
}

func (s *TranscribeIntegrationSuite) SetupSuite() {
	s.url = os.Getenv(urlEnvVar)
	s.apiKey = os.Getenv(apiKeyEnvVar)
	s.apiSecret = os.Getenv(apiSecretEnvVar)
	if s.url == "" || s.apiKey == "" || s.apiSecret == "" {
		s.T().Skipf("set %s, %s and %s to run this", urlEnvVar, apiKeyEnvVar, apiSecretEnvVar)
	}
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		s.T().Skip("set DEEPGRAM_API_KEY to run this")
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg is required to decode the test audio")
	}

	audio, err := testaudio.Load16kMono("mia.mp3")
	s.Require().NoError(err)
	s.audio = audio

	s.logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
}

func (s *TranscribeIntegrationSuite) TestTranscribesAParticipantSpeakingInARoom() {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	roomName := "stt-router-test-" + time.Now().Format("150405.000")

	sttRouter, cleanup, err := buildRouter(ctx, s.logger)
	s.Require().NoError(err)
	defer cleanup()

	collector := &transcriptCollector{}
	tracks := &trackSet{
		ctx:        ctx,
		router:     sttRouter,
		target:     "deepgram/flux-general-en",
		customerID: "transcribe-integration",
		logger:     s.logger,
		handle:     collector.collect,
	}
	defer tracks.closeAll()

	subscribed := make(chan struct{}, 1)
	bot, err := lksdk.ConnectToRoom(s.url, lksdk.ConnectInfo{
		APIKey:              s.apiKey,
		APISecret:           s.apiSecret,
		RoomName:            roomName,
		ParticipantIdentity: "stt-router-bot",
	}, &lksdk.RoomCallback{
		ParticipantCallback: lksdk.ParticipantCallback{
			OnTrackSubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, participant *lksdk.RemoteParticipant) {
				if track.Kind() != webrtc.RTPCodecTypeAudio {
					return
				}
				if err := tracks.add(track, participant); err != nil {
					s.logger.Error("could not transcribe", "error", err)
					return
				}
				select {
				case subscribed <- struct{}{}:
				default:
				}
			},
		},
	})
	s.Require().NoError(err)
	defer bot.Disconnect()

	speaker, err := lksdk.ConnectToRoom(s.url, lksdk.ConnectInfo{
		APIKey:              s.apiKey,
		APISecret:           s.apiSecret,
		RoomName:            roomName,
		ParticipantIdentity: "speaker",
	}, &lksdk.RoomCallback{})
	s.Require().NoError(err)
	defer speaker.Disconnect()

	track, err := lkmedia.NewPCMLocalTrack(s.audio.SampleRate, s.audio.Channels, protoLogger.GetLogger())
	s.Require().NoError(err)
	defer track.Close()

	_, err = speaker.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{Name: "speech"})
	s.Require().NoError(err)

	select {
	case <-subscribed:
	case <-ctx.Done():
		s.FailNow("the bot never subscribed to the published track")
	}

	// The track paces its own playout, so writing everything up front streams in real time.
	s.Require().NoError(track.WriteSample(s.audio.Samples))
	track.WaitForPlayout()

	// Trailing silence gives Deepgram the pause it needs to close the turn.
	s.Require().NoError(track.WriteSample(testaudio.Silence(1500).Samples))
	track.WaitForPlayout()

	s.Eventually(func() bool {
		return collector.finalText() != ""
	}, 20*time.Second, 250*time.Millisecond, "expected a final transcript of the published speech")

	transcript := strings.ToLower(collector.finalText())
	s.Contains(transcript, "mia", "the transcript should carry the words that were spoken")
	s.Equal("speaker", collector.speakerID(), "the transcript should be attributed to the publisher")
}

// transcriptCollector stands in for the terminal output, keeping what the demo would print.
type transcriptCollector struct {
	mu       sync.Mutex
	finals   []string
	partials int
	speaker  string
}

func (c *transcriptCollector) collect(session *sttrouter.Session, speaker stt.Participant) {
	for event := range session.Events() {
		transcript, ok := event.(stt.Transcript)
		if !ok {
			continue
		}

		c.mu.Lock()
		c.speaker = speaker.UserID
		if transcript.Final() {
			c.finals = append(c.finals, transcript.Text)
		} else {
			c.partials++
		}
		c.mu.Unlock()
	}
}

func (c *transcriptCollector) finalText() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.Join(c.finals, " ")
}

func (c *transcriptCollector) speakerID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.speaker
}
