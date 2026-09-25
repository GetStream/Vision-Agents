package sailresearch

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type SailResearchSuite struct {
	hosttest.Unit
}

func TestSailResearchSuite(t *testing.T) {
	suite.Run(t, &SailResearchSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "moonshotai/Kimi-K3",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
