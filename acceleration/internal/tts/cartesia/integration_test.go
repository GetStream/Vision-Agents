//go:build integration

package cartesia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type CartesiaIntegrationSuite struct {
	ttssuite.Suite
}

func TestCartesiaIntegrationSuite(t *testing.T) {
	suite.Run(t, &CartesiaIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"CARTESIA_API_KEY"},
		// The session is opened asking the server not to buffer, and this is what says it
		// listened: a buffering one would sit on the sentence instead of saying it.
		MaxTimeToFirstByte: 2_000,
		Interruptible:      true,
	}})
}
