//go:build integration

package phala

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type PhalaIntegrationSuite struct {
	llmsuite.Suite
}

func TestPhalaIntegrationSuite(t *testing.T) {
	suite.Run(t, &PhalaIntegrationSuite{hosttest.Live(t, Host, New, "z-ai/glm-5.3-flash")})
}
