//go:build integration

package digitalocean

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type DigitalOceanIntegrationSuite struct {
	hosttest.Live
}

func TestDigitalOceanIntegrationSuite(t *testing.T) {
	suite.Run(t, &DigitalOceanIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "glm-5.3-flash",
	}})
}
