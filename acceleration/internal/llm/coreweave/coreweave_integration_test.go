//go:build integration

package coreweave

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type CoreWeaveIntegrationSuite struct {
	llmsuite.Suite
}

func TestCoreWeaveIntegrationSuite(t *testing.T) {
	suite.Run(t, &CoreWeaveIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
