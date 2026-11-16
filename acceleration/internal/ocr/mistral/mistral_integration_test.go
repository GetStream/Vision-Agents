//go:build integration

package mistral

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr/ocrsuite"
)

// MistralIntegrationSuite inherits what every OCR provider owes from ocrsuite.
type MistralIntegrationSuite struct {
	ocrsuite.Suite
}

func TestMistralIntegrationSuite(t *testing.T) {
	suite.Run(t, &MistralIntegrationSuite{Suite: ocrsuite.Suite{
		New:      func() (ocr.Provider, error) { return New(Options{Model: "mistral-ocr-4-1"}) },
		Requires: []string{APIKeyEnvVar},
	}})
}
