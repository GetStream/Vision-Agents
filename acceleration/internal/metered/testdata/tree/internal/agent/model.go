// Package agent builds a vendor of its own, which no router meters.
package agent

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
)

// model is built here rather than by the router's registry.
var model, _ = openai.New(openai.Options{})
