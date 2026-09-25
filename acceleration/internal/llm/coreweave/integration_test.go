//go:build integration

package coreweave

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type CoreWeaveIntegrationSuite struct {
	hosttest.Live
}

func TestCoreWeaveIntegrationSuite(t *testing.T) {
	suite.Run(t, &CoreWeaveIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
