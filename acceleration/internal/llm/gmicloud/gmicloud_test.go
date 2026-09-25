package gmicloud

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type GMICloudSuite struct {
	hosttest.Unit
}

func TestGMICloudSuite(t *testing.T) {
	suite.Run(t, &GMICloudSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "XiaomiMiMo/MiMo-V2.5",
		ThinkingModel: "zai-org/GLM-5.3",
	}})
}
