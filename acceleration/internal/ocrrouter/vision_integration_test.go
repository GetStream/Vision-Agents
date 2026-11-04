//go:build integration

package ocrrouter

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr/ocrsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// VisionIntegrationSuite holds the vision models in the router's OCR config to what every
// OCR provider owes, each built as the router builds it.
type VisionIntegrationSuite struct {
	ocrsuite.Suite
}

func TestVisionIntegrationSuite(t *testing.T) {
	for _, model := range []struct{ provider, model, key string }{
		{openai.ProviderName, "gpt-6-luna", "OPENAI_API_KEY"},
		{gemini.ProviderName, "gemini-3.8-flash", "GOOGLE_API_KEY"},
	} {
		t.Run(model.model, func(t *testing.T) {
			suite.Run(t, &VisionIntegrationSuite{Suite: ocrsuite.Suite{
				New: func() (ocr.Provider, error) {
					return DefaultRegistry().Build(model.provider, routing.Spec{Model: model.model})
				},
				Requires: []string{model.key},
				ByTokens: true,
			}})
		})
	}
}
