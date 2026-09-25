//go:build integration

package sailresearch

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type SailResearchIntegrationSuite struct {
	hosttest.Live
}

func TestSailResearchIntegrationSuite(t *testing.T) {
	suite.Run(t, &SailResearchIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
