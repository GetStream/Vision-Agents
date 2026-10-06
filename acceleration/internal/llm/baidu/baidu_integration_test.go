//go:build integration

package baidu

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type BaiduIntegrationSuite struct {
	llmsuite.Suite
}

func TestBaiduIntegrationSuite(t *testing.T) {
	suite.Run(t, &BaiduIntegrationSuite{hosttest.Live(t, Host, New, "glm-5.3-flash-intl")})
}
