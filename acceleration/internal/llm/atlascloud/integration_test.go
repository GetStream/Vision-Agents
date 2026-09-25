//go:build integration

package atlascloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type AtlasCloudIntegrationSuite struct {
	hosttest.Live
}

func TestAtlasCloudIntegrationSuite(t *testing.T) {
	suite.Run(t, &AtlasCloudIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/glm-5.3-flash",
	}})
}
