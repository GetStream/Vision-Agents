package plugins

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/getsentry/sentry-go"
)

// DeprecatedUse is the message every use of the plugin system logs at warn. It is the same
// on every path, so the uses left can be counted before the system is removed for
// connectors; the path attribute says which use it was. Nothing secret is logged with it.
const DeprecatedUse = "deprecated plugin path used"

// ReportUse logs a use of the plugin system on path at warn, with args as the log's
// key-value pairs, and reports it to Sentry, which keeps it after the log is gone. Each path,
// and via where a site names one, is one Sentry issue; args are its tags.
func ReportUse(ctx context.Context, logger *slog.Logger, path string, args ...any) {
	logger.Warn(DeprecatedUse, append([]any{"path", path}, args...)...)

	event := sentry.NewEvent()
	event.Level = sentry.LevelWarning
	event.Message = DeprecatedUse
	event.Fingerprint = []string{DeprecatedUse, path}
	event.Tags = map[string]string{"path": path}
	for i := 0; i+1 < len(args); i += 2 {
		event.Tags[fmt.Sprint(args[i])] = tagValue(fmt.Sprint(args[i+1]))
	}
	if via, ok := event.Tags["via"]; ok {
		event.Fingerprint = append(event.Fingerprint, via)
	}
	// Set, so the hub's scope attaches no request of its own: the login callback's query
	// carries the OAuth code and state, which Sentry's default scrubbing keeps
	// (https://develop.sentry.dev/sdk/foundations/client/data-collection/#sensitive-denylist).
	event.Request = &sentry.Request{}
	hub := sentry.GetHubFromContext(ctx)
	if hub == nil {
		hub = sentry.CurrentHub()
	}
	hub.CaptureEvent(event)
}

// maxTagValue is the longest tag value Sentry keeps, in characters; it drops a tag whose
// value is longer (https://docs.sentry.io/platforms/go/enriching-events/tags/).
const maxTagValue = 200

func tagValue(v string) string {
	if r := []rune(v); len(r) > maxTagValue {
		return string(r[:maxTagValue])
	}
	return v
}

// The paths a use of the plugin system is logged under, as the path attribute.
const (
	// PathLogin is a plugin login started: by the app from the dashboard, or by an end user
	// asked in the conversation.
	PathLogin = "login"
	// PathCallback is a provider redirecting a plugin login back to the router.
	PathCallback = "callback"
	// PathConfigSave is a config saved with plugins or plugin_events.
	PathConfigSave = "config_save"
	// PathSessionTools is a session starting with plugins, logged once per session.
	PathSessionTools = "session_tools"
	// PathEventDelivery is a plugin's MCP server delivering to a subscription.
	PathEventDelivery = "event_delivery"
	// PathClientSet is a plugin's OAuth client set for a config.
	PathClientSet = "client_set"
	// PathClientDelete is a plugin's OAuth client dropped from a config.
	PathClientDelete = "client_delete"
	// PathDisconnect is a plugin's login dropped and its name taken off a config.
	PathDisconnect = "disconnect"
	// PathListConfig is a config's plugin logins listed.
	PathListConfig = "list_config"
)

// The two ways a config reaches a server through a plugin login, as the via attribute: a
// catalog plugin named under plugins, or an MCP server named by URL under mcp_servers. Only
// the first goes when the catalog does; the second is a use of the plugin login machinery by
// a field that stays, so the two are counted apart.
const (
	ViaPlugins    = "plugins"
	ViaMCPServers = "mcp_servers"
)

// Via is how a config named the plugin id: a catalog plugin, or else an MCP server. A server
// may not be named as a catalog id (mcpServersComplaint), so the catalog tells them apart.
func Via(id string) string {
	if _, ok := Lookup(id); ok {
		return ViaPlugins
	}
	return ViaMCPServers
}
