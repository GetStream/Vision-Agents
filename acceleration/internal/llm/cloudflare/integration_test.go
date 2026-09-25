//go:build integration

package cloudflare

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type CloudflareIntegrationSuite struct {
	hosttest.Live
}

func TestCloudflareIntegrationSuite(t *testing.T) {
	suite.Run(t, &CloudflareIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "@cf/zai-org/glm-5.3-flash",
	}})
}
