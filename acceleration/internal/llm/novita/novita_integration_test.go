//go:build integration

package novita

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type NovitaIntegrationSuite struct {
	llmsuite.Suite
}

func TestNovitaIntegrationSuite(t *testing.T) {
	suite.Run(t, &NovitaIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/glm-5.3-flash")})
}
