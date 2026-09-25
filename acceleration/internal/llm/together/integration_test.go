//go:build integration

package together

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type TogetherIntegrationSuite struct {
	hosttest.Live
}

func TestTogetherIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
