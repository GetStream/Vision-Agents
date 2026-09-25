package deepinfra

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type DeepInfraSuite struct {
	hosttest.Unit
}

func TestDeepInfraSuite(t *testing.T) {
	suite.Run(t, &DeepInfraSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google/gemma-4-26B-A4B-it",
		ThinkingModel: "zai-org/GLM-5.3-Flash",
	}})
}
