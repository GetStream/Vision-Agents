//go:build integration

package parasail

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type ParasailIntegrationSuite struct {
	llmsuite.Suite
}

func TestParasailIntegrationSuite(t *testing.T) {
	suite.Run(t, &ParasailIntegrationSuite{hosttest.Live(t, Host, New, "parasail-glm-53-flash")})
}
