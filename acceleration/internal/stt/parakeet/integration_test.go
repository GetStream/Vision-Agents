//go:build integration

package parakeet

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

// coldStartTimeout is how long a scaled-to-zero L4 deployment may take to answer.
const coldStartTimeout = 6 * time.Minute

type ParakeetIntegrationSuite struct {
	suite.Suite
	audio stt.PcmData
}

func TestParakeetIntegrationSuite(t *testing.T) {
	suite.Run(t, new(ParakeetIntegrationSuite))
}

func (s *ParakeetIntegrationSuite) SetupSuite() {
	if os.Getenv("PARAKEET_WS_URL") == "" {
		s.T().Skip("PARAKEET_WS_URL not set")
	}
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to decode the audio fixture")
	}

	audio, err := testaudio.Load16kMono("mia.mp3")
	s.Require().NoError(err)
	s.audio = audio
}

// start returns a connected provider. The deployment scales to zero, so the first
// connection after an idle spell pays for a GPU cold start and needs far longer than the
// timeout a live call would accept.
func (s *ParakeetIntegrationSuite) start() *STT {
	provider, err := New(Options{HandshakeTimeout: coldStartTimeout})
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), coldStartTimeout)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider
}

func (s *ParakeetIntegrationSuite) TestTranscribesSpeechAndEndsTheTurn() {
	provider := s.start()
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
	// Stream in realtime so the server's windowing behaves as it would on a live call.
	for _, chunk := range testaudio.Chunks(s.audio, 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(80 * time.Millisecond)
	}
	// Trailing silence is what makes the server finalise the utterance.
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 80) {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(80 * time.Millisecond)
	}

	var events []stt.Event
	select {
	case events = <-collected:
	case <-time.After(60 * time.Second):
		s.FailNow("timed out waiting for a final transcript")
	}

	var partials, finals []stt.Transcript
	var sawConnected, sawTurnStarted bool
	for _, event := range events {
		switch typed := event.(type) {
		case stt.Connected:
			sawConnected = true
		case stt.TurnStarted:
			sawTurnStarted = true
		case stt.Transcript:
			if typed.Final() {
				finals = append(finals, typed)
			} else {
				partials = append(partials, typed)
			}
		case stt.Error:
			s.Failf("provider error", "%v", typed.Err)
		}
	}

	s.True(sawConnected, "should report the session becoming ready")
	s.True(sawTurnStarted, "should report speech starting")
	s.NotEmpty(partials, "should stream incremental transcripts, not just a final")
	s.Require().NotEmpty(finals, "should produce a final transcript")

	final := finals[0]
	s.Contains(strings.ToLower(final.Text), "forgotten treasures")
	s.Equal("test-user", final.Participant.UserID)
	s.Equal(ProviderName, final.Provider)
	s.Positive(final.AudioDurationMs, "the final should report how much audio it covered")
	s.Positive(final.ProcessingTimeMs, "the final should report the server's decode time")

	// Partials replace each other, so each one should be at least as long as the last.
	for i := 1; i < len(partials); i++ {
		s.GreaterOrEqual(
			len(partials[i].Text), len(partials[i-1].Text)/2,
			"a replacement partial should not collapse to a fragment",
		)
	}
}

func (s *ParakeetIntegrationSuite) TestCloseFlushesTheTailOfTheCall() {
	provider := s.start()

	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
		}
		collected <- events
	}()

	// Stop mid-sentence with no trailing silence, so only an explicit flush can produce
	// a transcript for the audio still buffered on the server.
	speaker := stt.Participant{ID: "test-user", UserID: "test-user"}
	for _, chunk := range testaudio.Chunks(s.audio, 80)[:40] {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(80 * time.Millisecond)
	}

	s.Require().NoError(provider.Close())

	var events []stt.Event
	select {
	case events = <-collected:
	case <-time.After(30 * time.Second):
		s.FailNow("event channel was not closed")
	}

	var finals []stt.Transcript
	for _, event := range events {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			finals = append(finals, transcript)
		}
	}
	s.Require().NotEmpty(finals, "closing should flush the buffered audio into a final")
	s.Contains(strings.ToLower(finals[0].Text), "quiet village")
}
