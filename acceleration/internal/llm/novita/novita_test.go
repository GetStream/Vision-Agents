package novita

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type NovitaSuite struct {
	hosttest.Unit
}

func TestNovitaSuite(t *testing.T) {
	suite.Run(t, &NovitaSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google/gemma-4-26b-a4b-it",
		ThinkingModel: "zai-org/glm-5.3",
	}})
}
