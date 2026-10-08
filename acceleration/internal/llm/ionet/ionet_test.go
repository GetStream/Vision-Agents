package ionet

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type IoNetSuite struct {
	hosttest.Unit
}

func TestIoNetSuite(t *testing.T) {
	suite.Run(t, &IoNetSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google/gemma-4-26b-a4b-it",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
