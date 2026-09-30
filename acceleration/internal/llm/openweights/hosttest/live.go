//go:build integration

package hosttest

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// Live is the suite every language model is held to, pointed at one model on this host.
func Live(t *testing.T, host openweights.Host, build func(openweights.Options) (*openaicompat.LLM, error), model string) llmsuite.Suite {
	return llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := build(openweights.Options{Model: model})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{host.APIKeyEnvVar},
	}
}
