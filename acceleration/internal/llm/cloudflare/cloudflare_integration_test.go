//go:build integration

package cloudflare

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type CloudflareIntegrationSuite struct {
	llmsuite.Suite
}

func TestCloudflareIntegrationSuite(t *testing.T) {
	suite.Run(t, &CloudflareIntegrationSuite{hosttest.Live(t, Host, New, "@cf/zai-org/glm-5.3-flash")})
}
