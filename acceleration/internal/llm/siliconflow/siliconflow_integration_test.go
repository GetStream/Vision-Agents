//go:build integration

package siliconflow

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type SiliconFlowIntegrationSuite struct {
	llmsuite.Suite
}

func TestSiliconFlowIntegrationSuite(t *testing.T) {
	suite.Run(t, &SiliconFlowIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
