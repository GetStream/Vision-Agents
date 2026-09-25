//go:build integration

package baidu

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type BaiduIntegrationSuite struct {
	hosttest.Live
}

func TestBaiduIntegrationSuite(t *testing.T) {
	suite.Run(t, &BaiduIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "glm-5.3-flash-intl",
	}})
}
