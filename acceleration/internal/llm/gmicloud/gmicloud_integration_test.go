//go:build integration

package gmicloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type GMICloudIntegrationSuite struct {
	llmsuite.Suite
}

func TestGMICloudIntegrationSuite(t *testing.T) {
	suite.Run(t, &GMICloudIntegrationSuite{hosttest.Live(t, Host, New, "deepseek-ai/DeepSeek-V4-Flash")})
}
