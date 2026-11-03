package core

import (
	"fmt"
	"path"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Offered reports whether a server's tool is in an allowlist of names and path.Match
// patterns. An empty allowlist offers every tool.
func Offered(allowed []string, tool string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, pattern := range allowed {
		if matched, _ := path.Match(pattern, tool); matched {
			return true
		}
	}
	return false
}

// CheckToolPatterns refuses an allowlist entry path.Match cannot read, which would
// otherwise match nothing and hide a tool without saying why.
func CheckToolPatterns(allowed []string) error {
	for _, pattern := range allowed {
		if _, err := path.Match(pattern, ""); err != nil {
			return stack.Wrap(fmt.Errorf("%q is not a tool name or pattern", pattern))
		}
	}
	return nil
}
