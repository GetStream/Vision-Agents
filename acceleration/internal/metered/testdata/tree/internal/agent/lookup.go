// Package agent looks things up in a knowledge base of its own, which nothing meters.
package agent

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
)

// lookups is a knowledge backend held outside the metering layer.
type lookups struct {
	base knowledge.Backend
}

// open builds the vendor itself rather than taking a store from internal/metered.
func open() (*lookups, error) {
	base, err := turbopuffer.New(turbopuffer.Options{})
	return &lookups{base: base}, err
}
