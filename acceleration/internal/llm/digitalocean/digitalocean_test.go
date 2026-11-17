package digitalocean

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type DigitalOceanSuite struct {
	hosttest.Unit
}

func TestDigitalOceanSuite(t *testing.T) {
	suite.Run(t, &DigitalOceanSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "gemma-4-31B-it",
		ThinkingModel: "glm-5.3",
	}})
}
