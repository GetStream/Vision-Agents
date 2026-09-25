//go:build integration

package novita

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type NovitaIntegrationSuite struct {
	hosttest.Live
}

func TestNovitaIntegrationSuite(t *testing.T) {
	suite.Run(t, &NovitaIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/glm-5.3-flash",
	}})
}
