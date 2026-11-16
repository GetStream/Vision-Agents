//go:build integration

package minimax

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MiniMaxIntegrationSuite runs MiniMax M3.1 Flash, the model the router serves, through the
// suite every model vendor runs.
type MiniMaxIntegrationSuite struct {
	llmsuite.Suite
}

func TestMiniMaxIntegrationSuite(t *testing.T) {
	suite.Run(t, &MiniMaxIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{Model: "MiniMax-M3.1-Flash"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVars[0]},
	}})
}
