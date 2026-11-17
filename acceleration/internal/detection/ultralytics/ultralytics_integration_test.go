//go:build integration

package ultralytics

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/detectionsuite"
)

// UltralyticsIntegrationSuite inherits what every detection provider owes from
// detectionsuite. Ultralytics' own models need no key on the shared API, so nothing is
// required.
type UltralyticsIntegrationSuite struct {
	detectionsuite.Suite
}

func TestUltralyticsIntegrationSuite(t *testing.T) {
	suite.Run(t, &UltralyticsIntegrationSuite{Suite: detectionsuite.Suite{
		New: func() detection.Provider {
			provider, err := New(Options{Model: "yolo26/yolo26n"})
			require.NoError(t, err)
			return provider
		},
	}})
}
