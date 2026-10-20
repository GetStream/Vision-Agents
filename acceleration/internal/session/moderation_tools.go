package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/moderator"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// moderationTools offers a session of the app's moderator agent the reads of its moderation
// data. Only a process the customer runs is offered them, since an end user's device may
// open a session on any of the app's agents and the queue is every user's reports. Only an
// app of the customer's own is read: the deployment's shared app holds every other
// customer's moderation data too. A name the caller declared stays the caller's.
func moderationTools(spec Spec, stream streamapp.Bound, declared []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
	if spec.Role != store.RoleAgentModerator || spec.CallerKind != auth.KindServer || stream.Client == nil || !stream.Identity.Registered {
		return declared, next
	}
	taken := map[string]bool{}
	for _, tool := range declared {
		taken[tool.Name] = true
	}
	var offered []harness.Tool
	for _, tool := range moderator.Tools() {
		if !taken[tool.Name] {
			offered = append(offered, tool)
		}
	}
	if len(offered) == 0 {
		return declared, next
	}
	return append(declared, offered...), moderator.NewRunner(stream.Client, offered, next)
}
