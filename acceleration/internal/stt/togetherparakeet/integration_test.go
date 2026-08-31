//go:build integration

package togetherparakeet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// TogetherParakeetIntegrationSuite inherits what every provider owes a call from sttsuite.
// Together's own examples send roughly 250ms of audio at a time; the suite's 100ms is
// closer to what a call delivers and is what this is held to.
type TogetherParakeetIntegrationSuite struct {
	sttsuite.Suite
}

func TestTogetherParakeetIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherParakeetIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"TOGETHER_API_KEY"},
		// Committing the buffer makes the server transcribe what it is still holding.
		SettlesOnClose: true,
	}})
}

// TestTheModelIsTheStreamingOneRatherThanTheBatchOne is worth an assertion because the two
// differ only by a suffix, and the batch model cannot serve a call. A session that opened
// on the wrong one would look like a slow provider rather than a misconfigured one.
func (s *TogetherParakeetIntegrationSuite) TestTheModelIsTheStreamingOneRatherThanTheBatchOne() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.Equal(DefaultModel, provider.Model())
	s.Contains(provider.Model(), "-realtime")
}
