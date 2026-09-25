package baidu

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type BaiduSuite struct {
	hosttest.Unit
}

func TestBaiduSuite(t *testing.T) {
	suite.Run(t, &BaiduSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "kimi-k2.6",
		ThinkingModel: "glm-5.3-intl",
	}})
}
