//go:build integration

package nextbit

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type NextBitIntegrationSuite struct {
	hosttest.Live
}

func TestNextBitIntegrationSuite(t *testing.T) {
	suite.Run(t, &NextBitIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "glm:5.3-flash",
	}})
}
