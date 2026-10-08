package baseten

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type BasetenSuite struct {
	hosttest.Unit
}

func TestBasetenSuite(t *testing.T) {
	suite.Run(t, &BasetenSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
