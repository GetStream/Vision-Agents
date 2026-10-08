package coreweave

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type CoreWeaveSuite struct {
	hosttest.Unit
}

func TestCoreWeaveSuite(t *testing.T) {
	suite.Run(t, &CoreWeaveSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "google/gemma-4-26B-A4B-it",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
