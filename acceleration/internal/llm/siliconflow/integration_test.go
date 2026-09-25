//go:build integration

package siliconflow

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type SiliconFlowIntegrationSuite struct {
	hosttest.Live
}

func TestSiliconFlowIntegrationSuite(t *testing.T) {
	suite.Run(t, &SiliconFlowIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
