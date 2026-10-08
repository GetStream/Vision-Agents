package fireworks

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type FireworksSuite struct {
	hosttest.Unit
}

func TestFireworksSuite(t *testing.T) {
	suite.Run(t, &FireworksSuite{hosttest.Unit{
		Host:  Host,
		New:   New,
		Model: "accounts/fireworks/models/gemma-4-26b-a4b-it",
		// The family survives the path and the dotless version, which is what decides
		// how the model is told not to think.
		ThinkingModel: "accounts/fireworks/models/glm-5p3",
	}})
}
