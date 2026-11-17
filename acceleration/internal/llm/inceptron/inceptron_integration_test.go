//go:build integration

package inceptron

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type InceptronIntegrationSuite struct {
	llmsuite.Suite
}

func TestInceptronIntegrationSuite(t *testing.T) {
	suite.Run(t, &InceptronIntegrationSuite{hosttest.Live(t, Host, New, "zai-org/GLM-5.3-Flash")})
}
