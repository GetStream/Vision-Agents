package together

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type TogetherSuite struct {
	hosttest.Unit
}

func TestTogetherSuite(t *testing.T) {
	suite.Run(t, &TogetherSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "MiniMaxAI/MiniMax-M3",
		ThinkingModel: "Qwen/Qwen3.8-2.4T-A95B",
	}})
}
