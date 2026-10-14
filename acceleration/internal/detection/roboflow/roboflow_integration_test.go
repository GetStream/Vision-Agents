//go:build integration

package roboflow

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/detectionsuite"
)

// RoboflowIntegrationSuite runs YOLOv8 nano, a hosted model, through detectionsuite.
type RoboflowIntegrationSuite struct {
	detectionsuite.Suite
}

func TestRoboflowIntegrationSuite(t *testing.T) {
	suite.Run(t, &RoboflowIntegrationSuite{Suite: detectionsuite.Suite{
		New: func() detection.Provider {
			provider, err := New(Options{Model: "coco/3"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{APIKeyEnvVar},
	}})
}

// RoboflowFacesIntegrationSuite runs the built-in face Workflow, which finds faces only
// and returns a vector for each.
type RoboflowFacesIntegrationSuite struct {
	detectionsuite.Suite
}

func TestRoboflowFacesIntegrationSuite(t *testing.T) {
	suite.Run(t, &RoboflowFacesIntegrationSuite{Suite: detectionsuite.Suite{
		New: func() detection.Provider {
			provider, err := New(Options{Model: ModelFaces})
			require.NoError(t, err)
			return provider
		},
		Requires:  []string{APIKeyEnvVar},
		FacesOnly: true,
		Embeds:    true,
	}})
}
