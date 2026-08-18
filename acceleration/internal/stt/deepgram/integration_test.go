//go:build integration

package deepgram

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
)

type DeepgramIntegrationSuite struct {
	suite.Suite
	audio stt.PcmData
}

func TestDeepgramIntegrationSuite(t *testing.T) {
	suite.Run(t, new(DeepgramIntegrationSuite))
}

func (s *DeepgramIntegrationSuite) SetupSuite() {
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		s.T().Skip("DEEPGRAM_API_KEY not set")
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to decode the audio fixture")
	}

	audio, err := testaudio.Load16kMono("mia.mp3")
	s.Require().NoError(err)
	s.audio = audio
}

// transcribe streams the fixture followed by silence and collects events until the turn
// ends or the deadline passes.
func (s *DeepgramIntegrationSuite) transcribe(options Options) []stt.Event {
	provider, err := New(options)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	s.Require().NoError(provider.Start(ctx))
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				break
			}
		}
		collected <- events
	}()

	speaker := stt.Participant{ID: "test-user", UserID: "test-user"}
	// Real calls arrive in small chunks; Flux prefers roughly 80 ms of audio at a time.
	for _, chunk := range testaudio.Chunks(s.audio, 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}
	// Trailing silence is what makes Flux declare the turn over.
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}

	select {
	case events := <-collected:
		return events
	case <-time.After(20 * time.Second):
		s.FailNow("timed out waiting for a final transcript")
		return nil
	}
}

// TestAnIdleSessionOutlivesTheFluxPingDeadline is slow by nature: the deadline it is about
// is a minute long, so there is no shorter way to see the session survive it.
func (s *DeepgramIntegrationSuite) TestAnIdleSessionOutlivesTheFluxPingDeadline() {
	provider, err := New(Options{})
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	s.Require().NoError(provider.Start(ctx))
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	finals := make(chan stt.Transcript, 8)
	failures := make(chan error, 8)
	go func() {
		for event := range provider.Events() {
			switch typed := event.(type) {
			case stt.Transcript:
				if typed.Final() {
					finals <- typed
				}
			case stt.Error:
				failures <- typed.Err
			}
		}
	}()

	// Flux closes a session it has not been pinged on for 60 seconds, and the audio a
	// conversation sends does not count towards that.
	select {
	case err := <-failures:
		s.FailNowf("the idle session was closed", "%v", err)
	case <-time.After(75 * time.Second):
	}

	speaker := stt.Participant{ID: "test-user", UserID: "test-user"}
	for _, chunk := range testaudio.Chunks(s.audio, 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}

	select {
	case final := <-finals:
		s.NotEmpty(final.Text, "speech after a long pause is still transcribed")
	case err := <-failures:
		s.FailNowf("the session failed after the pause", "%v", err)
	case <-time.After(20 * time.Second):
		s.FailNow("nothing came back after the pause")
	}
}

func (s *DeepgramIntegrationSuite) TestTranscribesSpeechToAFinalRevision() {
	events := s.transcribe(Options{})

	var finals []stt.Transcript
	var sawConnected bool
	for _, event := range events {
		switch typed := event.(type) {
		case stt.Connected:
			sawConnected = true
		case stt.Transcript:
			if typed.Final() {
				finals = append(finals, typed)
			}
		case stt.Error:
			s.Failf("provider error", "%v", typed.Err)
		}
	}

	s.True(sawConnected, "should report the session becoming ready")
	s.Require().NotEmpty(finals, "should produce at least one final transcript")

	var text strings.Builder
	for _, final := range finals {
		text.WriteString(strings.ToLower(final.Text))
		text.WriteString(" ")
	}
	s.Contains(text.String(), "forgotten treasures")

	first := finals[0]
	s.Equal("test-user", first.Participant.UserID)
	s.Equal(ProviderName, first.Provider)
	s.Equal(DefaultModel, first.Model)
	s.Positive(first.AudioDurationMs, "final transcripts should report the audio window")
}
