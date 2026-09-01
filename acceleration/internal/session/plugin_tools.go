package session

import (
	"context"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// attachPlugins opens the MCP servers this config is logged into and returns their tools.
// A server that will not start is skipped so a broken Slack login does not refuse the call.
func attachPlugins(ctx context.Context, spec Spec, db *store.Store, logger *slog.Logger) (*plugins.Runtime, []harness.Tool) {
	if db == nil || spec.ConfigID == "" {
		return nil, nil
	}
	conns, err := db.ConnectedPlugins(ctx, spec.CustomerID, spec.ConfigID)
	if err != nil {
		logger.Warn("not loading plugin connections", "config", spec.ConfigID, "error", err)
		return nil, nil
	}
	if len(conns) == 0 {
		return nil, nil
	}

	wanted := make([]plugins.Connection, 0, len(conns))
	for _, conn := range conns {
		plugin, ok := plugins.Lookup(conn.PluginID)
		if !ok {
			continue
		}
		endpoint, err := plugin.Endpoint(conn.InstanceURL)
		if err != nil {
			logger.Warn("plugin has no endpoint", "plugin", conn.PluginID, "error", err)
			continue
		}
		token := conn.AccessToken
		if conn.ExpiresAt != nil && time.Until(*conn.ExpiresAt) < time.Minute && conn.RefreshToken != "" {
			auth := &plugins.Auth{}
			refreshed, err := auth.Refresh(ctx, conn.TokenEndpoint, conn.ClientID, conn.RefreshToken)
			if err != nil {
				logger.Warn("could not refresh a plugin token", "plugin", conn.PluginID, "error", err)
			} else {
				token = refreshed.AccessToken
				conn.AccessToken = refreshed.AccessToken
				conn.RefreshToken = refreshed.RefreshToken
				conn.ExpiresAt = refreshed.ExpiresAt
				if err := db.SavePluginConnection(ctx, &conn); err != nil {
					logger.Warn("could not store a refreshed plugin token", "plugin", conn.PluginID, "error", err)
				}
			}
		}
		wanted = append(wanted, plugins.Connection{
			PluginID:    conn.PluginID,
			Endpoint:    endpoint,
			AccessToken: token,
		})
	}

	runtime, tools, failures := plugins.Open(ctx, wanted, nil)
	for _, failure := range failures {
		logger.Warn("plugin did not connect", "error", failure)
	}
	return runtime, tools
}

// pluginRunner runs prefixed MCP tools itself and hands everything else to the caller bridge.
type pluginRunner struct {
	mcp  *plugins.Runtime
	next agent.ToolRunner
}

func (r *pluginRunner) Run(ctx context.Context, call llm.ToolCall) (string, error) {
	if r.mcp != nil && r.mcp.Owns(call.Name) {
		return r.mcp.Call(ctx, call)
	}
	if r.next != nil {
		return r.next.Run(ctx, call)
	}
	return "", errUnknownTool(call.Name)
}

func errUnknownTool(name string) error {
	return &toolError{name: name}
}

type toolError struct{ name string }

func (e *toolError) Error() string {
	return "session: " + e.name + " is not a tool this session can run"
}
