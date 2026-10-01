package core

// Profile is the manifest, resolved for one connection's inputs and captured values. It is
// what a scheme reads instead of switching on a connector id.
//
// Only the fields whose shape is plain data are here. The policy fields (scopes, refresh,
// identity, capture, client, rate limit) arrive with the manifest model, which owns their
// types.
type Profile struct {
	ConnectorID string
	// Revision is the definition revision the connection pinned, so a manifest change
	// reaches a connection only through a reconnect.
	Revision int
	Scheme   string
	// Endpoints are resolved URLs by role: authorize, token, refresh, revoke, issuer,
	// api_base.
	Endpoints map[string]string
	// Inputs are what the connection was created with: shop, instance, region, tenant.
	Inputs map[string]string
	// Metadata are values captured at connect time: instance_url, realm_id, team_id.
	Metadata map[string]string
	// Hooks maps a hook point to the registered hook name the manifest asked for.
	Hooks map[string]string
}
