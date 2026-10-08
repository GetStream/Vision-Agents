//go:build integration

package gemini

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type GeminiIntegrationSuite struct {
	ttssuite.Suite
	model string
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"GOOGLE_API_KEY"},
		// A whole sentence goes up per request, so this is the wait before the model has
		// begun speaking it rather than a socket's first flush. It measures about a second.
		MaxTimeToFirstByte: 2_500,
		Interruptible:      true,
		Timeout:            90 * time.Second,
	}})
}

func TestGeminiFlashLiteIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{model: "gemini-3.8-flash-lite-tts", Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{Model: "gemini-3.8-flash-lite-tts"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"GOOGLE_API_KEY"},
		// Still a request per sentence, but it measures about half a second.
		MaxTimeToFirstByte: 1_500,
		Interruptible:      true,
		Timeout:            90 * time.Second,
	}})
}

// TestSpeaksInTheVoiceAndLanguageAskedFor is the one option worth a live request: the
// server refuses a voice it does not know, so a request that comes back as speech is one
// whose voice and language it accepted.
func (s *GeminiIntegrationSuite) TestSpeaksInTheVoiceAndLanguageAskedFor() {
	provider, err := New(Options{Model: s.model, Voice: "Kore", Language: "en-US"})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	complete, chunks := s.Say(provider, tts.Request{Text: ttssuite.Sentence, Final: true})

	s.Require().NotEmpty(chunks)
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.False(complete.Interrupted)

	complete, chunks = s.Say(provider, tts.Request{Text: "Bonjour, comment allez-vous aujourd'hui ?", Voice: "Puck", Language: "fr", Final: true})

	s.Require().NotEmpty(chunks)
	s.Greater(complete.AudioDurationMs, 1_000.0)
	s.False(complete.Interrupted)
}
