//go:build integration

package wafer

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type WaferIntegrationSuite struct {
	hosttest.Live
}

func TestWaferIntegrationSuite(t *testing.T) {
	suite.Run(t, &WaferIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "GLM-5.2",
	}})
}
