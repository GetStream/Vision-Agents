package stream

import (
	"encoding/json"
	"time"
)

// ConnectorScopeRequired is a connector_scope_required event: a connector tool call the
// provider refused because the caller's own connection lacks access it asked for, and the
// step-up consent the router began for it. Open LaunchURL in a popup and post it
// HandoffToken, as for a consent the backend begins. The old grant keeps working until the
// step-up succeeds, and the same call works afterwards in the same session.
type ConnectorScopeRequired struct {
	// Name is the binding's alias.
	Name         string `json:"name"`
	ConnectorID  string `json:"connector_id"`
	ConnectionID string `json:"connection_id"`
	// Scopes are what the provider asked for, empty for a claims challenge.
	Scopes          []string  `json:"scopes"`
	AuthorizationID string    `json:"authorization_id"`
	LaunchURL       string    `json:"launch_url"`
	HandoffToken    string    `json:"handoff_token"`
	ExpiresAt       time.Time `json:"expires_at"`
}

// ConnectorScopeRequired reads the event as a connector_scope_required, reporting false for
// any other kind.
func (e Event) ConnectorScopeRequired() (ConnectorScopeRequired, bool) {
	var asked ConnectorScopeRequired
	if e.Kind != "connector_scope_required" {
		return asked, false
	}
	raw, err := json.Marshal(e.Frame)
	if err != nil {
		return asked, false
	}
	return asked, json.Unmarshal(raw, &asked) == nil
}
