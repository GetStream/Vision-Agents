package venice

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type VeniceSuite struct {
	hosttest.Unit
}

func TestVeniceSuite(t *testing.T) {
	suite.Run(t, &VeniceSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google-gemma-4-26b-a4b-it",
		ThinkingModel: "z-ai-glm-5-3",
	}})
}
