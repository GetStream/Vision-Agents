//go:build integration

package deepinfra

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type DeepInfraIntegrationSuite struct {
	hosttest.Live
}

func TestDeepInfraIntegrationSuite(t *testing.T) {
	suite.Run(t, &DeepInfraIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
