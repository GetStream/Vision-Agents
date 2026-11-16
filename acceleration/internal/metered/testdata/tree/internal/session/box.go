// Package session makes a sandbox of its own, which nothing meters.
package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
)

// box builds the provider itself rather than opening one from internal/metered.
func box() (sandbox.Sandbox, error) {
	return daytona.New(daytona.Options{})
}
