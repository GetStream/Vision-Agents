package session

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// newStartup times the start-up of the session spec opens, asked for at began. Only a session
// on a call the manager's Edge opens has one. A text session joins no call. A session on an
// edge its caller hands in (spec.Edge: an audio socket, a simulation's loopback) has no
// connection whose coming up ends a start-up, so none of its stages would be the call's.
func newStartup(spec Spec, began time.Time) *agent.Startup {
	if spec.Text || spec.Edge != nil {
		return nil
	}
	kind := agent.StartupWeb
	if spec.Phone != nil {
		kind = agent.StartupPhone
	}
	return agent.NewStartup(kind, began)
}
