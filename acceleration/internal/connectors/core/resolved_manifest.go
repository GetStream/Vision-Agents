package core

// ResolvedManifest is the manifest, resolved for one connection's inputs and captured values. It is
// what a scheme reads instead of switching on a connector id. Manifest.Resolve builds one.
type ResolvedManifest struct {
	ConnectorID string
	// Revision is the definition revision the connection pinned, so a manifest change
	// reaches a connection only through a reconnect.
	Revision int
	Scheme   string
	// Endpoints are resolved URLs by role: authorize, token, refresh, revoke, issuer,
	// api_base, mcp, resource. A role whose template needs a value not captured yet is
	// absent.
	Endpoints map[string]string
	// Inputs are what the connection was created with, defaults applied: shop, instance,
	// region, tenant.
	Inputs map[string]string
	// Metadata are values captured at connect time: instance_url, realm_id, team_id.
	Metadata map[string]string
	// Hooks maps a hook point to the registered hook name the manifest asked for.
	Hooks map[string]string

	Client          ClientPolicy
	AuthorizeParams map[string]string
	TokenParams     map[string]string
	Scopes          ScopePolicy
	// Identity and Capture are the rules Apply reads a consent's callback and token
	// response with.
	Identity  []string
	Capture   []CaptureRule
	Refresh   RefreshPolicy
	RateLimit RateLimitRule
	// Sources are the tool sources the connector offers. A ToolSource finds its own by Kind
	// and reads its endpoint from Endpoints under the role the rule names.
	Sources []SourceRule
	// Channel is the manifest's channel block, or nil when the connector is no inbound
	// channel. It is the Manifest's own, not a copy, as Client and Scopes are: Reply only reads
	// it.
	Channel *ChannelRule
	// vars are the manifest's vars, which a reply template may name.
	vars map[string]Var
}
