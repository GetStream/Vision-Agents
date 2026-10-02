//go:build integration

package sailresearch

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type SailResearchIntegrationSuite struct {
	llmsuite.Suite
}

func TestSailResearchIntegrationSuite(t *testing.T) {
	suite.Run(t, &SailResearchIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
