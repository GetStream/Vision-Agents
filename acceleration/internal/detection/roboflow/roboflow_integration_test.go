//go:build integration

package roboflow

import (
	"encoding/base64"
	"encoding/json"
	"os"
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
		Requires:      []string{APIKeyEnvVar},
		FacesOnly:     true,
		Embeds:        true,
		ReportsBilled: true,
	}})
}

// TestRoboflowListsTheModelCallsAFaceIsBilledFor: the face Workflow is billed by the calls
// Roboflow lists, so one face is the detector and one CLIP call. Without the list a
// Workflow would be billed by a guess.
func TestRoboflowListsTheModelCallsAFaceIsBilledFor(t *testing.T) {
	if os.Getenv(APIKeyEnvVar) == "" {
		t.Skipf("%s not set", APIKeyEnvVar)
	}
	face, err := os.ReadFile("../detectionsuite/testdata/face.jpg")
	require.NoError(t, err)
	provider, err := New(Options{Model: ModelFaces})
	require.NoError(t, err)

	var answer workflowResponse
	header, err := provider.post(t.Context(), "application/json", func(key string) (string, []byte, error) {
		body, err := json.Marshal(map[string]any{
			"api_key":       key,
			"inputs":        map[string]any{"image": map[string]string{"type": "base64", "value": base64.StdEncoding.EncodeToString(face)}},
			"specification": facesSpecification,
		})
		return provider.baseURL + "/workflows/run", body, err
	}, &answer)

	require.NoError(t, err)
	require.Len(t, answer.Outputs, 1)
	require.Len(t, answer.Outputs[0].Embeddings, 1, "the fixture is one face")
	require.Equal(t, 2, modelCalls(header))
}
