//go:build integration

package parasail

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type ParasailIntegrationSuite struct {
	hosttest.Live
}

func TestParasailIntegrationSuite(t *testing.T) {
	suite.Run(t, &ParasailIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "parasail-glm-53-flash",
	}})
}
