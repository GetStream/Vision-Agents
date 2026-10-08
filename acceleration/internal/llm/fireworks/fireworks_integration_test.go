//go:build integration

package fireworks

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type FireworksIntegrationSuite struct {
	llmsuite.Suite
}

func TestFireworksIntegrationSuite(t *testing.T) {
	suite.Run(t, &FireworksIntegrationSuite{hosttest.Live(t, Host, New, "accounts/fireworks/models/glm-5p3-flash")})
}
