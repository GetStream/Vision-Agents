//go:build integration

package nextbit

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type NextBitIntegrationSuite struct {
	llmsuite.Suite
}

func TestNextBitIntegrationSuite(t *testing.T) {
	suite.Run(t, &NextBitIntegrationSuite{hosttest.Live(t, Host, New, "glm:5.3-flash")})
}
