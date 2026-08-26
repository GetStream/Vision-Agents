//go:build integration

package gemini

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

type GeminiIntegrationSuite struct {
	suite.Suite
	audio stt.PcmData
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, new(GeminiIntegrationSuite))
}

func (s *GeminiIntegrationSuite) SetupSuite() {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		s.T().Skip("GOOGLE_API_KEY not set")
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to decode the audio fixture")
	}

	audio, err := testaudio.Load16kMono("mia.mp3")
	s.Require().NoError(err)
	s.audio = audio
}

// start returns a connected provider.
func (s *GeminiIntegrationSuite) start(options Options) *STT {
	provider, err := New(options)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider
}

// speak streams the fixture in realtime, then enough silence for the server to decide the
// turn is over. Sending it faster than realtime would have the model see the whole clip
// at once, which is not how it behaves on a call.
func (s *GeminiIntegrationSuite) speak(provider *STT, chunks int) {
	speaker := stt.Participant{ID: "test-user", UserID: "test-user"}

	spoken := testaudio.Chunks(s.audio, 100)
	if chunks > 0 && chunks < len(spoken) {
		spoken = spoken[:chunks]
	}
	for _, chunk := range spoken {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(100 * time.Millisecond)
	}
}

func (s *GeminiIntegrationSuite) TestTranscribesSpeechAndSettlesTheTurn() {
	provider := s.start(Options{})
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

	s.speak(provider, 0)
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 100) {
		s.Require().NoError(provider.ProcessAudio(chunk, stt.Participant{ID: "test-user"}))
		time.Sleep(100 * time.Millisecond)
	}

	var events []stt.Event
	select {
	case events = <-collected:
	case <-time.After(60 * time.Second):
		s.FailNow("timed out waiting for a settled transcript")
	}

	var deltas, finals []stt.Transcript
	var sawConnected bool
	for _, event := range events {
		switch typed := event.(type) {
		case stt.Connected:
			sawConnected = true
		case stt.Transcript:
			if typed.Final() {
				finals = append(finals, typed)
			} else {
				deltas = append(deltas, typed)
			}
		case stt.Error:
			s.Failf("provider error", "%v", typed.Err)
		}
	}

	s.True(sawConnected, "should report the session becoming ready")
	s.NotEmpty(deltas, "should stream the words as they are heard, not only at the end")
	s.Require().NotEmpty(finals, "should settle the turn")

	final := finals[0]
	s.Contains(strings.ToLower(final.Text), "forgotten treasures")
	s.Equal("test-user", final.Participant.UserID)
	s.Equal(ProviderName, final.Provider)

	var appended strings.Builder
	for _, delta := range deltas {
		appended.WriteString(delta.Text)
	}
	s.Equal(final.Text, strings.TrimSpace(appended.String()),
		"the deltas appended must come to the same words as the final")
}

func (s *GeminiIntegrationSuite) TestCloseSettlesTheTailOfTheCall() {
	provider := s.start(Options{})

	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
		}
		collected <- events
	}()

	// Stop mid-sentence with no trailing silence, so only ending the audio stream can
	// settle what the server is still holding.
	s.speak(provider, 40)
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
	s.Require().NotEmpty(finals, "closing should settle the audio still being transcribed")
	s.Contains(strings.ToLower(finals[0].Text), "quiet village")
}

func (s *GeminiIntegrationSuite) TestAKeytermIsSpelledTheWayItWasGiven() {
	provider := s.start(Options{Keyterms: []string{"Mia"}})
	defer func() {
		s.Require().NoError(provider.Close())
	}()

	collected := make(chan stt.Transcript, 1)
	go func() {
		for event := range provider.Events() {
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				collected <- transcript
				return
			}
		}
	}()

	s.speak(provider, 0)
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 100) {
		s.Require().NoError(provider.ProcessAudio(chunk, stt.Participant{ID: "test-user"}))
		time.Sleep(100 * time.Millisecond)
	}

	select {
	case final := <-collected:
		s.Contains(final.Text, "Mia")
	case <-time.After(60 * time.Second):
		s.FailNow("timed out waiting for a settled transcript")
	}
}
