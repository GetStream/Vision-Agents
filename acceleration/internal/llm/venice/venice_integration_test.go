//go:build integration

package venice

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type VeniceIntegrationSuite struct {
	llmsuite.Suite
}

func TestVeniceIntegrationSuite(t *testing.T) {
	suite.Run(t, &VeniceIntegrationSuite{hosttest.Live(t, Host, New, "z-ai-glm-5-3-flash")})
}
