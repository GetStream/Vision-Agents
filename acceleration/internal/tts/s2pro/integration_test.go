//go:build integration

package s2pro

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

// coldStartTimeout is how long a scaled-to-zero GPU deployment may take to answer.
const coldStartTimeout = 10 * time.Minute

type S2ProIntegrationSuite struct {
	ttssuite.Suite
}

func TestS2ProIntegrationSuite(t *testing.T) {
	suite.Run(t, &S2ProIntegrationSuite{Suite: ttssuite.Suite{
		// The deployment scales to zero, so the first connection after an idle spell pays
		// for a GPU cold start.
		New: func() ttssuite.Provider {
			provider, err := New(Options{HandshakeTimeout: coldStartTimeout})
			require.NoError(t, err)
			return provider
		},
		Requires:      []string{"S2PRO_WS_URL", "BASETEN_API_KEY"},
		Timeout:       coldStartTimeout,
		Interruptible: true,
	}})
}
