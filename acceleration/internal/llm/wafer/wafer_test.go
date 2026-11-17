package wafer

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type WaferSuite struct {
	hosttest.Unit
}

func TestWaferSuite(t *testing.T) {
	suite.Run(t, &WaferSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "GLM-5.2",
		ThinkingModel: "GLM-5.2",
	}})
}
