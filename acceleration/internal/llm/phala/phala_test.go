package phala

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type PhalaSuite struct {
	hosttest.Unit
}

func TestPhalaSuite(t *testing.T) {
	suite.Run(t, &PhalaSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google/gemma-4-31b-it",
		ThinkingModel: "z-ai/glm-5.3",
	}})
}
