//go:build integration

package fireworks

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type FireworksIntegrationSuite struct {
	hosttest.Live
}

func TestFireworksIntegrationSuite(t *testing.T) {
	suite.Run(t, &FireworksIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "accounts/fireworks/models/glm-5p3-flash",
	}})
}
