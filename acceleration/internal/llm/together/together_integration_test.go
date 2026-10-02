//go:build integration

package together

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type TogetherIntegrationSuite struct {
	llmsuite.Suite
}

func TestTogetherIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
