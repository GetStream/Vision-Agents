package nextbit

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type NextBitSuite struct {
	hosttest.Unit
}

func TestNextBitSuite(t *testing.T) {
	suite.Run(t, &NextBitSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "gemma4:26b-a4b",
		ThinkingModel: "glm:5.3-flash",
	}})
}
