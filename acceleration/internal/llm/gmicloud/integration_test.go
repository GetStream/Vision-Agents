//go:build integration

package gmicloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type GMICloudIntegrationSuite struct {
	hosttest.Live
}

func TestGMICloudIntegrationSuite(t *testing.T) {
	suite.Run(t, &GMICloudIntegrationSuite{hosttest.Live{
		Host:  Host,
		New:   New,
		Model: "deepseek-ai/DeepSeek-V4-Flash",
	}})
}
