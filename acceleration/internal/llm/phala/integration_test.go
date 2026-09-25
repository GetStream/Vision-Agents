//go:build integration

package phala

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type PhalaIntegrationSuite struct {
	hosttest.Live
}

func TestPhalaIntegrationSuite(t *testing.T) {
	suite.Run(t, &PhalaIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "z-ai/glm-5.3-flash",
	}})
}
