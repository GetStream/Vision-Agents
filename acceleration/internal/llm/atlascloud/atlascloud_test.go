package atlascloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type AtlasCloudSuite struct {
	hosttest.Unit
}

func TestAtlasCloudSuite(t *testing.T) {
	suite.Run(t, &AtlasCloudSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "xiaomi/mimo-v2.5",
		ThinkingModel: "zai-org/glm-5.3",
	}})
}
