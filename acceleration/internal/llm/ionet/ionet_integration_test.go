//go:build integration

package ionet

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type IoNetIntegrationSuite struct {
	llmsuite.Suite
}

func TestIoNetIntegrationSuite(t *testing.T) {
	suite.Run(t, &IoNetIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
