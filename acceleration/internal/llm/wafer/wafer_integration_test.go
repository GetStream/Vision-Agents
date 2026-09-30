//go:build integration

package wafer

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type WaferIntegrationSuite struct {
	llmsuite.Suite
}

func TestWaferIntegrationSuite(t *testing.T) {
	suite.Run(t, &WaferIntegrationSuite{hosttest.Live(t, Host, New, "GLM-5.2")})
}
