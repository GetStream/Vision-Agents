//go:build integration

package digitalocean

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type DigitalOceanIntegrationSuite struct {
	llmsuite.Suite
}

func TestDigitalOceanIntegrationSuite(t *testing.T) {
	suite.Run(t, &DigitalOceanIntegrationSuite{hosttest.Live(t, Host, New, "glm-5.3-flash")})
}
