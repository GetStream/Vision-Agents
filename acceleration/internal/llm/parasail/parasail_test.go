package parasail

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type ParasailSuite struct {
	hosttest.Unit
}

func TestParasailSuite(t *testing.T) {
	suite.Run(t, &ParasailSuite{hosttest.Unit{
		Host:  Host,
		New:   New,
		Model: "parasail-gemma-4-26b-a4b-it",
		// The deployment name still carries the family, which is what decides how the
		// model is told not to think.
		ThinkingModel: "parasail-glm-53",
	}})
}
