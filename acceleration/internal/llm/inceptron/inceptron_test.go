package inceptron

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type InceptronSuite struct {
	hosttest.Unit
}

func TestInceptronSuite(t *testing.T) {
	suite.Run(t, &InceptronSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "zai-org/GLM-5.3-Flash",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
