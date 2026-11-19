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

// MetaIntegrationSuite runs Muse Spark and Muse Ember through the suite every model vendor
// runs. Both always think first and take tool_choice auto and nothing else, which the suite
// reads from their capabilities.
type MetaIntegrationSuite struct {
	llmsuite.Suite
}

func TestMetaIntegrationSuite(t *testing.T) {
	for _, model := range []string{DefaultModel, "muse-ember-1.0"} {
		t.Run(model, func(t *testing.T) {
			suite.Run(t, &MetaIntegrationSuite{Suite: llmsuite.Suite{
				New: func() llm.LLM {
					provider, err := New(Options{Model: model})
					require.NoError(t, err)
					return provider
				},
				Requires: []string{apiKeyEnvVar},
			}})
		})
	}
}
