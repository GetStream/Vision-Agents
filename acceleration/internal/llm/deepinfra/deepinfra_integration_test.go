//go:build integration

package deepinfra

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type DeepInfraIntegrationSuite struct {
	llmsuite.Suite
}

func TestDeepInfraIntegrationSuite(t *testing.T) {
	suite.Run(t, &DeepInfraIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
