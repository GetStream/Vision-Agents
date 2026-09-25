//go:build integration

package inceptron

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type InceptronIntegrationSuite struct {
	hosttest.Live
}

func TestInceptronIntegrationSuite(t *testing.T) {
	suite.Run(t, &InceptronIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "zai-org/GLM-5.3-Flash",
	}})
}
