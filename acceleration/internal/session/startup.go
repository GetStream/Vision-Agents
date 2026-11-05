package session

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// newStartup times the start-up of the session spec opens, asked for at began. A text session
// joins no call, so it has none.
func newStartup(spec Spec, began time.Time) *agent.Startup {
	if spec.Text {
		return nil
	}
	kind := agent.StartupWeb
	if spec.Phone != nil {
		kind = agent.StartupPhone
	}
	return agent.NewStartup(kind, began)
}
