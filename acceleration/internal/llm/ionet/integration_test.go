//go:build integration

package ionet

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type IoNetIntegrationSuite struct {
	hosttest.Live
}

func TestIoNetIntegrationSuite(t *testing.T) {
	suite.Run(t, &IoNetIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
