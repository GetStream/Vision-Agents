//go:build integration

package venice

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type VeniceIntegrationSuite struct {
	hosttest.Live
}

func TestVeniceIntegrationSuite(t *testing.T) {
	suite.Run(t, &VeniceIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "z-ai-glm-5-3-flash",
	}})
}
