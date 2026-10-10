package plugins

// DeprecatedUse is the message every use of the plugin system logs at warn. It is the same
// on every path, so the uses left can be counted before the system is removed for
// connectors; the path attribute says which use it was. Nothing secret is logged with it.
const DeprecatedUse = "deprecated plugin path used"

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
