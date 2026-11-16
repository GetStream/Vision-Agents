//go:build integration

package baseten

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type BasetenIntegrationSuite struct {
	llmsuite.Suite
}

func TestBasetenIntegrationSuite(t *testing.T) {
	suite.Run(t, &BasetenIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
