//go:build integration

package meta

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MetaIntegrationSuite runs Muse Spark through the suite every model vendor runs. It always
// thinks first and takes tool_choice auto and nothing else, which the suite reads from its
// capabilities.
type MetaIntegrationSuite struct {
	llmsuite.Suite
}

func TestMetaIntegrationSuite(t *testing.T) {
	suite.Run(t, &MetaIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
