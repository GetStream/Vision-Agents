//go:build integration

package atlascloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type AtlasCloudIntegrationSuite struct {
	llmsuite.Suite
}

func TestAtlasCloudIntegrationSuite(t *testing.T) {
	suite.Run(t, &AtlasCloudIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/glm-5.3-flash")})
}
