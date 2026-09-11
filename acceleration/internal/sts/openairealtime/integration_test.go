//go:build integration

package openairealtime

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/stssuite"
)

// realtimeSuite is the shared suite against one vendor reached over this protocol.
func realtimeSuite(t *testing.T, vendor Vendor) stssuite.Suite {
	return stssuite.Suite{
		New: func(ask stssuite.Ask) sts.STS {
			provider, err := New(Options{
				Vendor:          vendor,
				Instructions:    ask.Instructions,
				Tools:           ask.Tools,
				InputTranscript: true,
			})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{vendor.APIKeyEnvVar},
	}
}

type OpenAIIntegrationSuite struct {
	stssuite.Suite
}

func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &OpenAIIntegrationSuite{Suite: realtimeSuite(t, OpenAI)})
}

type XAIIntegrationSuite struct {
	stssuite.Suite
}

func TestXAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &XAIIntegrationSuite{Suite: realtimeSuite(t, XAI)})
}

type QwenIntegrationSuite struct {
	stssuite.Suite
}

func TestQwenIntegrationSuite(t *testing.T) {
	suite.Run(t, &QwenIntegrationSuite{Suite: realtimeSuite(t, Qwen)})
}
