//go:build integration

package fish

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type FishIntegrationSuite struct {
	ttssuite.Suite
}

func TestFishIntegrationSuite(t *testing.T) {
	suite.Run(t, &FishIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"FISH_API_KEY"},
		// A whole utterance is synthesised per request rather than streamed from partial
		// text, so it takes longer to answer than the socket providers do.
		Timeout: 90 * time.Second,
	}})
}

// TestSynthesizesAtTheRequestedSampleRate is the one option worth spending a live request
// on: a wrong rate is not an error anywhere, it is speech at the wrong speed.
func (s *FishIntegrationSuite) TestSynthesizesAtTheRequestedSampleRate() {
	provider, err := New(Options{SampleRate: 44_100})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	complete, chunks := s.Say(provider, tts.Request{Text: ttssuite.Sentence, Final: true})

	s.Require().NotEmpty(chunks)
	s.Equal(44_100, chunks[0].Audio.SampleRate)
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.Less(complete.AudioDurationMs, 20_000.0)
}
