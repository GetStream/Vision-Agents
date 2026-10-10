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
)
